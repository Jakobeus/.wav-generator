#!/usr/bin/env python3
"""
Adversarial tone transfer — multi-critic, hybrid reverb generator.

  G = HybridG: PackedWaveNet dry path (drive/EQ/cab, ~85ms, exported as .nam)
      + ParamFX: a reverb and a delay described by a few dozen interpretable
      parameters (RT60 per band, band levels, pre-delay, early reflections,
      wet; delay time, feedback, damping, mix), rendered to one 1.5s impulse
      response and convolved with the dry signal.
      out = dry + fx(dry)
      Why parametric: the earlier free 1.5s FIR tails learned a bank of
      non-decaying tuned resonators (a drone that filled the ref's spectral
      bands for the critics) instead of a reverb. A noise-based reverb with
      per-band decay must decay and cannot be tonal, so that cheat is closed;
      the dry path has to earn the tone itself. The learned values are printed
      with every probe so the space can be rebuilt with real plugins.
      Identity anchors (envelope/pitch) apply to the DRY PATH ONLY —
      tails are allowed to outlast the input envelope by design.

  Critics (training-time scaffolding, discarded after):
    MPD  : waveform folded at periods {2,3,5,7,11} — periodicity/phase
    MSD  : raw waveform avg-pooled {1x,2x,4x} — waveform shape + dynamics
    STFT : spectrograms at n_fft {512,2048,8192} + mel-scaled — spectral texture
    LONG : WaveDisc(4) + SpecDisc(4096) on the full 1.5s window — tail grading
    (short-view critics see a random internal 0.34s subcrop per call)

  Losses: hinge adversarial (summed) + feature matching + dry-path envelope
  and pitch-consistency (EQ-blind, see pitch_loss) + HF guard above the
  critic cutoff. DiffAugment (gain/polarity/shift/mild EQ tilt) on real+fake,
  then both are low-passed at the ref's measured cutoff (--critic-lpf) so
  bandwidth can't be used as a tell.
  G is fed --lead-in extra samples of DI before every chunk and critics only
  see the last CHUNK, so fake chunks carry reverb history like real crops do.

Usage:
  python adv_train.py --di di1.wav di2.wav --ref wet_stem.wav \
      --modeldir runs/x/model_adv [--init-ckpt pair.ckpt] [--steps 20000]
      [--device cuda] [--critics mpd msd stft long]
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 48000
CHUNK = 73728          # ~1.536s @ 48k — covers the tail
SHORT = 16384          # ~0.34s subcrop for texture critics
TAIL = 73728           # learned reverb field length
TARGET_RMS = 10 ** (-18 / 20)
sn = nn.utils.spectral_norm


def load_mono(path):
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = torch.from_numpy(x.mean(1))
    if sr != SR:
        from scipy import signal
        x = torch.from_numpy(signal.resample_poly(x.numpy(), SR, sr).astype(np.float32))
    return x


def measure_cutoff(x, drop_db=60.0):
    """Highest frequency where the ref's averaged power spectrum is still within
    drop_db of its peak — i.e. the codec/resampler brick-wall of the ref stem.
    Rounded down to 100Hz."""
    n = len(x) // CHUNK
    frames = x[:n * CHUNK].view(n, CHUNK) * torch.hann_window(CHUNK)
    psd = torch.fft.rfft(frames).abs().pow(2).mean(0)
    psd = F.avg_pool1d(psd[None, None], 33, 1, padding=16)[0, 0]     # ~100Hz smoothing
    db = 10 * torch.log10(psd + 1e-20)
    f = torch.fft.rfftfreq(CHUNK, 1 / SR)
    above = f[db > db.max() - drop_db]
    return float(torch.floor(above.max() / 100) * 100)


def rand_crop(x, n):
    i = np.random.randint(0, max(1, len(x) - n + 1))
    seg = x[i:i + n]
    return F.pad(seg, (0, n - len(seg)))


def augment(x, tilt_db=0.0, lpf_hz=0.0, hpf_hz=0.0):
    """DiffAugment: same family of transforms on real and fake.
    gain ±6dB, polarity flip, circular shift — invariants D must learn.
    tilt_db>0 adds a random spectral slope ±tilt_db (20Hz-20kHz, pivot ~632Hz):
    mild values stop D memorizing the ref's exact EQ fingerprint; large values
    would dissolve the spectral target itself — keep it small.
    lpf_hz>0 zeroes everything above lpf_hz on real AND fake before any critic
    sees them. The ref is a 44.1k lossy-sourced demucs stem (nothing above its
    codec cutoff), the DIs are native 48k with content/noise to 24k — without
    this the critics can separate real from fake on bandwidth alone and never
    need to learn tone.
    hpf_hz>0 zeroes everything below hpf_hz the same way (sub-bass rumble/DC
    offset carries no tone information worth judging)."""
    B, T = x.shape
    g = torch.rand(B, 1, device=x.device) * 12 - 6
    x = x * (10 ** (g / 20))
    x = x * torch.where(torch.rand(B, 1, device=x.device) < 0.5, -1.0, 1.0)
    shifts = torch.randint(0, T, (B,), device=x.device)
    idx = (torch.arange(T, device=x.device).unsqueeze(0) + shifts.unsqueeze(1)) % T
    x = torch.gather(x, 1, idx)
    if tilt_db > 0 or lpf_hz > 0 or hpf_hz > 0:
        X = torch.fft.rfft(x)
        f = torch.fft.rfftfreq(T, 1 / SR, device=x.device)
        if tilt_db > 0:
            u = torch.log(f.clamp(min=20) / 20) / np.log(20000 / 20)   # 0..1 across band
            s = (torch.rand(B, 1, device=x.device) * 2 - 1) * tilt_db
            X = X * (10 ** (s * (u - 0.5) * 2 / 20))
        if lpf_hz > 0:
            X = X * (f <= lpf_hz)
        if hpf_hz > 0:
            X = X * (f >= hpf_hz)
        x = torch.fft.irfft(X, n=T)
    return x


def _mean_feats(feats):
    return [f.mean(dim=(2, 3)) if f.dim() == 4 else f.mean(dim=2) for f in feats]


def fft_fir(x, h):
    """Causal FIR via FFT: y[t] = sum_k h[k] x[t-k]. Fully differentiable —
    tap gradients collect output error arriving at their delay."""
    n = x.shape[-1] + h.shape[-1] - 1
    nfft = 1 << (n - 1).bit_length()
    y = torch.fft.irfft(torch.fft.rfft(x, nfft) * torch.fft.rfft(h, nfft), nfft)
    return y[..., :x.shape[-1]]


# ----------------------------- generator -----------------------------

def bandlimit(x, lo_hz=0.0, hi_hz=0.0):
    """Brick-wall band-limit G's OUTPUT (not just the critics' view). Anything
    the critics are blind to (below lo_hz / above hi_hz) is removed at the
    source, so G has nothing to gain by dumping energy/DC there."""
    if not lo_hz and not hi_hz:
        return x
    T = x.shape[-1]
    f = torch.fft.rfftfreq(T, 1 / SR, device=x.device)
    m = torch.ones_like(f)
    if lo_hz:
        m = m * (f >= lo_hz)
    if hi_hz:
        m = m * (f <= hi_hz)
    return torch.fft.irfft(torch.fft.rfft(x) * m, n=T)


def _sig_range(p, lo, hi):
    """Unconstrained parameter -> value in [lo, hi]."""
    return lo + (hi - lo) * torch.sigmoid(p)


def _inv_sig_range(v, lo, hi):
    u = (v - lo) / (hi - lo)
    return float(np.log(u / (1 - u)))


class ParamFX(nn.Module):
    """Reverb + delay with a few dozen interpretable parameters, rendered to one
    TAIL-long impulse response per forward and applied by FFT convolution.

    Reverb: fixed band-passed noise (one band per octave) x per-band exponential
    decay (learned RT60 per band), learned per-band level, pre-delay, a handful
    of early reflections, and a wet level. It must decay and cannot be tonal,
    so it cannot turn into the resonator bank a free FIR learned.
    Delay: time, feedback, damping low-pass, mix (N_REP repeats).
    Time parameters are learned as linear-phase shifts in the frequency domain,
    so they are continuous and differentiable."""

    BANDS = (40, 125, 250, 500, 1000, 2000, 4000, 8000, 12000)   # Hz edges, 8 bands
    N_ER = 8                                                     # early reflections
    N_REP = 8                                                    # delay repeats
    PREDELAY = (0.0, 120.0)                                      # ms
    ER_T = (1.0, 80.0)                                           # ms
    DELAY = (50.0, 700.0)                                        # ms
    FB = (0.0, 0.85)
    DAMP = (500.0, 12000.0)                                      # Hz

    def __init__(self, tail=TAIL):
        super().__init__()
        self.tail = tail
        K = len(self.BANDS) - 1
        gen = torch.Generator().manual_seed(0)
        W = torch.fft.rfft(torch.randn(tail, generator=gen))
        f = torch.fft.rfftfreq(tail, 1 / SR)
        noise = []
        for lo, hi in zip(self.BANDS[:-1], self.BANDS[1:]):
            n = torch.fft.irfft(W * ((f >= lo) & (f < hi)), n=tail)
            noise.append(n / n.pow(2).mean().sqrt())
        self.register_buffer("noise", torch.stack(noise))                  # (K, tail), unit RMS each
        self.register_buffer("t", torch.arange(tail, dtype=torch.float32) / SR)
        self.nfft = 1 << (4 * tail - 1).bit_length()                        # room for N_REP*DELAY without wrap
        self.register_buffer("f", torch.fft.rfftfreq(self.nfft, 1 / SR))
        # reverb
        self.log_t60 = nn.Parameter(torch.zeros(K))                         # ln(seconds); 0 -> 1.0 s per band
        self.band_db = nn.Parameter(torch.zeros(K))
        self.p_predelay = nn.Parameter(torch.tensor(_inv_sig_range(20.0, *self.PREDELAY)))
        self.p_er_t = nn.Parameter(torch.linspace(_inv_sig_range(8.0, *self.ER_T),
                                                  _inv_sig_range(60.0, *self.ER_T), self.N_ER))
        self.er_g = nn.Parameter(torch.randn(self.N_ER, generator=gen) * 0.02)
        self.wet = nn.Parameter(torch.tensor(0.3))
        # delay
        self.p_delay = nn.Parameter(torch.tensor(_inv_sig_range(350.0, *self.DELAY)))
        self.p_fb = nn.Parameter(torch.tensor(_inv_sig_range(0.3, *self.FB)))
        self.p_damp = nn.Parameter(torch.tensor(_inv_sig_range(4000.0, *self.DAMP)))
        self.mix = nn.Parameter(torch.tensor(0.2))

    # readable values
    def t60(self):          return self.log_t60.exp()
    def predelay_ms(self):  return _sig_range(self.p_predelay, *self.PREDELAY)
    def er_ms(self):        return _sig_range(self.p_er_t, *self.ER_T)
    def delay_ms(self):     return _sig_range(self.p_delay, *self.DELAY)
    def feedback(self):     return _sig_range(self.p_fb, *self.FB)
    def damp_hz(self):      return _sig_range(self.p_damp, *self.DAMP)

    def _shift(self, sec):                       # pure delay as a linear phase in the rfft domain
        return torch.exp(-2j * np.pi * self.f * sec)

    def impulse_response(self):
        # reverb body: noise bands x exponential decay (-60 dB at T60), unit energy, pre-delayed
        dec = torch.exp(-6.908 * self.t[None, :] / self.t60()[:, None])      # (K, tail)
        body = (10 ** (self.band_db[:, None] / 20) * self.noise * dec).sum(0)
        body = body / body.pow(2).sum().sqrt().clamp(min=1e-9)
        R = torch.fft.rfft(body, self.nfft) * self._shift(self.predelay_ms() / 1000)
        # early reflections
        ER = (self.er_g[:, None] * self._shift(self.er_ms()[:, None] / 1000)).sum(0)
        # delay: damped, fed-back repeats
        damp = 1 / torch.sqrt(1 + (self.f / self.damp_hz()) ** 2)
        tau, fb = self.delay_ms() / 1000, self.feedback()
        D = torch.zeros_like(R)
        for n in range(1, self.N_REP + 1):
            D = D + (fb ** (n - 1)) * (damp ** n) * self._shift(n * tau)
        H = self.wet * (R + ER) + self.mix * D
        return torch.fft.irfft(H, n=self.nfft)[: self.tail]

    def describe(self):
        t60 = " ".join(f"{v:.2f}" for v in self.t60().tolist())
        db = " ".join(f"{v:+.0f}" for v in self.band_db.tolist())
        er = " ".join(f"{t:.0f}" for t in self.er_ms().tolist())
        return (f"reverb wet {float(self.wet):+.2f} predelay {float(self.predelay_ms()):.0f}ms "
                f"RT60[s]/band {t60} | level[dB]/band {db} | ER[ms] {er}\n"
                f"      delay {float(self.delay_ms()):.0f}ms fb {float(self.feedback()):.2f} "
                f"damp {float(self.damp_hz()):.0f}Hz mix {float(self.mix):+.2f}")


class HybridG(nn.Module):
    """dry: PackedWaveNet (wrapped) = amp/EQ/cab, exported as .nam.
    fx: ParamFX reverb+delay applied to the dry signal.  out = dry + fx(dry)."""

    def __init__(self, wavenet, sub, tail=TAIL):
        super().__init__()
        self.dry = wavenet
        self.sub = sub
        self.fx = ParamFX(tail)
        self.band = (0.0, 0.0)                  # (hpf_hz, lpf_hz) applied to dry and out; set by trainer, saved in .pt

    def forward(self, x):                       # x: (B,T) DI -> (out, dry)
        raw = self.dry(x)[:, self.sub]          # raw WaveNet output (what the .nam plays)
        self.raw_dc = raw.mean(-1)              # informational (dc column in the log)
        dry = bandlimit(raw, *self.band)        # everything downstream sees the clean dry
        out = dry + fft_fir(dry, self.fx.impulse_response())
        out = bandlimit(out, *self.band)
        return out, dry

    def describe(self):
        return self.fx.describe()


# ----------------------------- critics -----------------------------

class ShortView(nn.Module):
    """Wraps a critic: sees a random internal SHORT subcrop per call —
    texture judgment at 0.34s inside the 1.5s batch window."""

    def __init__(self, d, view=SHORT):
        super().__init__()
        self.d, self.view = d, view

    def forward(self, x, return_feats=False):
        if x.shape[-1] > self.view:
            i = torch.randint(0, x.shape[-1] - self.view + 1, (1,)).item()
            x = x[..., i:i + self.view]
        return self.d(x, return_feats=return_feats)


class PeriodDisc(nn.Module):
    """Fold waveform at period p — periodicity/phase police."""

    def __init__(self, p):
        super().__init__()
        self.p = p
        chans = [1, 32, 64, 128, 256]
        self.convs = nn.ModuleList(
            sn(nn.Conv2d(a, b, (5, 1), stride=(3, 1), padding=(2, 0)))
            for a, b in zip(chans, chans[1:]))
        self.head = sn(nn.Conv2d(256, 1, (3, 1), padding=(1, 0)))

    def forward(self, x, return_feats=False):
        B, T = x.shape
        pad = (-T) % self.p
        if pad:
            x = F.pad(x, (0, pad))
        h = x.view(B, 1, -1, self.p)
        feats = []
        for c in self.convs:
            h = F.leaky_relu(c(h), 0.2)
            feats.append(h)
        logit = self.head(h).mean(dim=(2, 3)).squeeze(-1)
        return (logit, _mean_feats(feats)) if return_feats else logit


class WaveDisc(nn.Module):
    """Raw waveform at decimation scale s — waveform shape + dynamics."""

    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        chans = [1, 32, 64, 128, 256]
        self.convs = nn.ModuleList(
            sn(nn.Conv1d(a, b, 15, stride=4, padding=7))
            for a, b in zip(chans, chans[1:]))
        self.head = sn(nn.Conv1d(256, 1, 3, padding=1))

    def forward(self, x, return_feats=False):
        if self.scale > 1:
            x = F.avg_pool1d(x.unsqueeze(1), 4, self.scale).squeeze(1)
        h = x.unsqueeze(1)
        feats = []
        for c in self.convs:
            h = F.leaky_relu(c(h), 0.2)
            feats.append(h)
        logit = self.head(h).mean(dim=2).squeeze(-1)
        return (logit, _mean_feats(feats)) if return_feats else logit


class SpecDisc(nn.Module):
    """Spectrogram critic at n_fft resolution."""

    def __init__(self, n_fft):
        super().__init__()
        self.n_fft = n_fft
        self.register_buffer("win", torch.hann_window(n_fft))
        chans = [1, 32, 64, 128, 256]
        self.convs = nn.ModuleList(
            sn(nn.Conv2d(a, b, 5, stride=2, padding=2))
            for a, b in zip(chans, chans[1:]))
        self.head = sn(nn.Conv2d(256, 1, 1))

    def forward(self, x, return_feats=False):
        S = torch.stft(x, self.n_fft, self.n_fft // 4,
                       window=self.win, return_complex=True)
        h = torch.log(S.abs() + 1e-6).unsqueeze(1)
        feats = []
        for c in self.convs:
            h = F.leaky_relu(c(h), 0.2)
            feats.append(h)
        logit = self.head(h).mean(dim=(2, 3)).squeeze(-1)
        return (logit, _mean_feats(feats)) if return_feats else logit


class MelSpecDisc(nn.Module):
    """Perceptual-scale critic — mel bins weight frequency like the ear."""

    def __init__(self):
        super().__init__()
        import torchaudio.transforms as T
        self.mel = T.MelSpectrogram(
            sample_rate=SR, n_fft=2048, hop_length=512, win_length=2048,
            n_mels=128, f_min=60, f_max=16000, power=2.0)
        chans = [1, 32, 64, 128, 256]
        self.convs = nn.ModuleList(
            sn(nn.Conv2d(a, b, 5, stride=2, padding=2))
            for a, b in zip(chans, chans[1:]))
        self.head = sn(nn.Conv2d(256, 1, 1))

    def forward(self, x, return_feats=False):
        h = torch.log(self.mel(x) + 1e-6).unsqueeze(1)
        feats = []
        for c in self.convs:
            h = F.leaky_relu(c(h), 0.2)
            feats.append(h)
        logit = self.head(h).mean(dim=(2, 3)).squeeze(-1)
        return (logit, _mean_feats(feats)) if return_feats else logit


def build_critics(kind):
    if kind == "mpd":
        return [ShortView(PeriodDisc(p)) for p in (2, 3, 5, 7, 11)]
    if kind == "msd":
        return [ShortView(WaveDisc(s)) for s in (1, 2, 4)]
    if kind == "stft":
        return [ShortView(SpecDisc(n)) for n in (512, 2048, 8192)] \
            + [ShortView(MelSpecDisc())]
    if kind == "long":
        # full-window tail graders: gross dynamics + fine spectral decay
        return [WaveDisc(4), SpecDisc(4096)]
    raise ValueError(kind)


def env_loss(a, b):
    """Envelope L1 at two frame sizes — timing/dynamics anchor (dry path)."""
    l = 0.0
    for w in (128, 1024):
        ea = F.avg_pool1d(a.abs().unsqueeze(1), w, w).squeeze(1)
        eb = F.avg_pool1d(b.abs().unsqueeze(1), w, w).squeeze(1)
        l = l + F.l1_loss(torch.log(ea + 1e-7), torch.log(eb + 1e-7))
    return l


def _whitened_power(x, lifter=40):
    """Power spectrum with its smooth envelope (EQ/tone colour) removed by
    cepstral liftering: quefrencies < `lifter` samples (< ~0.83ms @ 48k, far
    below the shortest guitar period at 1200Hz) are zeroed. Harmonic peak
    positions — i.e. pitch — survive; spectral tilt does not."""
    T = x.shape[-1]
    logp = torch.log(torch.fft.rfft(x).abs() ** 2 + 1e-8)
    ceps = torch.fft.irfft(logp, n=T)
    keep = torch.ones(T, device=x.device)
    keep[:lifter] = 0
    keep[T - lifter + 1:] = 0
    return torch.exp(torch.fft.rfft(ceps * keep).real.clamp(max=30.0))


def pitch_loss(x, y):
    """F0-consistency, detector-free: normalized autocorrelation profiles
    encode the period — L1 over the guitar lag range (80-1200Hz) penalizes
    pitch drift/octave jumps. Differentiable; harmonic-content changes only
    soften autocorr peaks, they don't move them.

    Review note (Devin): the original version of this claim was wrong. By
    Wiener-Khinchin, autocorrelation is the inverse FFT of the power spectrum,
    so the normalized profile carries the full spectral envelope. Any EQ
    change (cutting lows, boosting upper mids) reshapes the profile at lags
    40-600 and is penalized as if it were pitch drift — the term fought the
    exact tone change the critics ask for. Fixed by whitening the spectrum
    (_whitened_power) before the inverse FFT, so only harmonic spacing
    (pitch) is compared."""
    T = x.shape[-1]
    ax = torch.fft.irfft(_whitened_power(x), n=T)[:, :600]
    ay = torch.fft.irfft(_whitened_power(y), n=T)[:, :600]
    ax = ax / ax[:, :1].clamp(min=1e-9)
    ay = ay / ay[:, :1].clamp(min=1e-9)
    return F.l1_loss(ax[:, 40:], ay[:, 40:])


def hinge(real_logit, fake_logit):
    return F.relu(1 - real_logit).mean() + F.relu(1 + fake_logit).mean()


def save_hybrid(ema, modeldir, basename):
    """Full hybrid -> .pt (needs HybridG to load); dry PackedWaveNet -> .nam."""
    torch.save({"state_dict": ema.state_dict(), "kind": "hybrid",
                "sr": SR, "sub": 1, "band": list(ema.band)}, modeldir / f"{basename}.pt")
    try:
        ema.dry.sample_rate = SR
        ema.dry.export(modeldir, basename=basename)
        print(f"  saved {modeldir}/{basename}.pt + {basename}.nam (dry path)", flush=True)
    except Exception as e:
        print(f"  saved {modeldir}/{basename}.pt  (.nam export failed, non-fatal: {e})", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--di", nargs="+", required=True, help="clean DI wavs")
    p.add_argument("--ref", required=True, help="wet reference stem")
    p.add_argument("--modeldir", required=True)
    p.add_argument("--init-ckpt", default=None, help="warm-start G dry path")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lam-content", type=float, default=50.0)
    p.add_argument("--lam-fm", type=float, default=2.0)
    p.add_argument("--lam-pitch", type=float, default=5.0,
                   help="EQ-blind autocorrelation F0-consistency weight on the dry path (0 disables)")
    p.add_argument("--aug-tilt", type=float, default=2.0,
                   help="random EQ tilt in dB applied to real+fake (0 disables)")
    p.add_argument("--lam-hf", type=float, default=0.0,
                   help="penalty when G puts more energy above --critic-lpf than the "
                        "ref does (critics can't see that region); 0 disables (default)")
    p.add_argument("--lam-dc", type=float, default=0.0,
                   help="optional penalty on the raw WaveNet's DC offset. Off by default: the dry "
                        "path is band-limited (no DC) before anything uses it, and the NAM plugin "
                        "runs a DC blocker on the .nam's output anyway")
    p.add_argument("--fx-lr", type=float, default=1e-3,
                   help="learning rate for the ParamFX reverb/delay parameters (WaveNet stays at 1e-4)")
    p.add_argument("--grad-diag-every", type=int, default=500,
                   help="every N steps print the gradient norm each G loss term "
                        "alone puts on G's weights (0 disables)")
    p.add_argument("--critic-lpf", default="auto",
                   help="low-pass real+fake at this Hz before every critic so "
                        "bandwidth can't be used to tell them apart; "
                        "'auto' measures the ref's cutoff, 0 disables")
    p.add_argument("--critic-hpf", type=float, default=40.0,
                   help="high-pass real+fake at this Hz before every critic "
                        "(0 disables)")
    p.add_argument("--lead-in", type=int, default=TAIL,
                   help="extra DI samples fed to G before each chunk so the reverb "
                        "field has history; critics see only the last CHUNK (0 disables)")
    p.add_argument("--device", default=None)
    p.add_argument("--critics", nargs="+",
                   default=["mpd", "msd", "stft", "long"],
                   choices=["mpd", "msd", "stft", "long"])
    p.add_argument("--probe", default=None)
    p.add_argument("--probe-every", type=int, default=500)
    p.add_argument("--probe-start", type=float, default=0.0, help="seconds into --probe to start")
    p.add_argument("--probe-len", type=float, default=0.0, help="seconds of --probe to render (0 = whole file)")
    p.add_argument("--export-every", type=int, default=2000)
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).parent / "RemFx"))
    sys.path.insert(0, str(Path(__file__).parent / "RemFx" / "umx"))
    modeldir = Path(args.modeldir)
    modeldir.mkdir(parents=True, exist_ok=True)
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from nam.train import core
    cfg = core._get_packed_model_config()
    cls = core._get_lightning_module_cls(cfg)
    lm = cls.init_from_config(cfg)
    if args.init_ckpt:
        lm = cls.load_from_checkpoint(args.init_ckpt, **cls.parse_config(cfg))
        print(f"G dry path warm-started from {args.init_ckpt}")
    SUB = 1
    G = HybridG(lm.net, SUB).to(dev)
    G.train()
    ema = copy.deepcopy(G).eval()               # EMA -> probe + export
    for q in ema.parameters():
        q.requires_grad_(False)

    critics = nn.ModuleList(
        [d for kind in args.critics for d in build_critics(kind)]).to(dev)
    names = [f"{type(getattr(d,'d',d)).__name__}"
             f"{getattr(getattr(d,'d',d),'p',getattr(getattr(d,'d',d),'scale',getattr(getattr(d,'d',d),'n_fft','')))}"
             for d in critics]
    print(f"critics ({len(critics)}): {names}")

    di = [load_mono(f) for f in args.di]
    ref = load_mono(args.ref)
    norm = lambda x: x * (TARGET_RMS / x.pow(2).mean().add(1e-12).sqrt())
    di = [norm(x) for x in di]
    ref = norm(ref)
    print(f"device {dev} | DI: {[f'{len(x)/SR:.1f}s' for x in di]}  ref: {len(ref)/SR:.1f}s")
    if args.critic_lpf == "auto":
        args.critic_lpf = measure_cutoff(ref)
        print(f"critic low-pass: auto -> {args.critic_lpf:.0f} Hz (ref cutoff)")
    else:
        args.critic_lpf = float(args.critic_lpf)
        print(f"critic low-pass: {args.critic_lpf:.0f} Hz" if args.critic_lpf else "critic low-pass: off")
    print(f"critic high-pass: {args.critic_hpf:.0f} Hz" if args.critic_hpf else "critic high-pass: off")
    G.band = ema.band = (args.critic_hpf, args.critic_lpf)
    print(f"G output band-limited to {args.critic_hpf:.0f}-{args.critic_lpf:.0f} Hz (same band the critics hear)")
    f_bins = torch.fft.rfftfreq(CHUNK, 1 / SR).to(dev)
    hf_mask = (f_bins > args.critic_lpf).float() if args.critic_lpf else torch.zeros_like(f_bins)
    def hf_frac(x):                              # share of energy above the critic cutoff
        P = torch.fft.rfft(x).abs() ** 2
        return (P * hf_mask).sum(1) / P.sum(1).clamp(min=1e-12)
    ref_hf = hf_frac(ref[:len(ref) // CHUNK * CHUNK].view(-1, CHUNK).to(dev)).mean().item()
    print(f"ref energy above cutoff: {100 * ref_hf:.4f}% "
          + (f"(G penalized for exceeding this, lam-hf {args.lam_hf})" if args.lam_hf and args.critic_lpf
             else "(hf penalty OFF; critics judge it directly)"))

    fx_params = list(G.fx.parameters())
    fx_ids = {id(q) for q in fx_params}
    opt_g = torch.optim.Adam([
        {"params": [q for q in G.parameters() if id(q) not in fx_ids], "lr": 1e-4},
        {"params": fx_params, "lr": args.fx_lr},     # ~38 reverb/delay knobs: let them move
    ], betas=(0.5, 0.9))
    print(f"G lr 1e-4 | fx lr {args.fx_lr:g}")
    opt_d = torch.optim.Adam(critics.parameters(), 1e-4, betas=(0.5, 0.9))
    d_loss = torch.tensor(0.0)
    accs = torch.zeros(len(critics))
    run = {}                                  # 100-step running sums for the log line
    def acc_log(**kv):
        for k, v in kv.items():
            run[k] = run.get(k, 0.0) + (v.detach().cpu() if torch.is_tensor(v) else v)

    for step in range(1, args.steps + 1):
        xb = torch.stack([rand_crop(di[i], CHUNK + args.lead_in)
                          for i in np.random.randint(0, len(di), args.batch)]).to(dev)
        rb = torch.stack([rand_crop(ref, CHUNK) for _ in range(args.batch)]).to(dev)

        # --- D step ---
        with torch.no_grad():
            fake = G(xb)[0][:, -CHUNK:]
        d_loss = torch.zeros((), device=dev)
        for i, D in enumerate(critics):
            d_real = D(augment(rb, args.aug_tilt, args.critic_lpf, args.critic_hpf))
            d_fake = D(augment(fake.detach(), args.aug_tilt, args.critic_lpf, args.critic_hpf))
            d_loss = d_loss + hinge(d_real, d_fake)
            accs[i] = ((d_real > 0).float().mean() + (d_fake < 0).float().mean()) / 2
        opt_d.zero_grad(); d_loss.backward(); opt_d.step()

        # --- G step ---
        fake, dry = G(xb)
        fake, dry, xc = fake[:, -CHUNK:], dry[:, -CHUNK:], xb[:, -CHUNK:]
        fa = augment(fake, args.aug_tilt, args.critic_lpf, args.critic_hpf)
        ra = augment(rb, args.aug_tilt, args.critic_lpf, args.critic_hpf)
        adv = torch.zeros((), device=dev)
        fm = torch.zeros((), device=dev)
        for D in critics:
            logit, f_fake = D(fa, return_feats=True)
            with torch.no_grad():
                _, f_real = D(ra, return_feats=True)
            adv = adv - logit.mean()
            fm = fm + sum(F.l1_loss(ff.mean(0), fr.mean(0))
                          for ff, fr in zip(f_fake, f_real))
        cont = env_loss(dry, xc)                # anchors on DRY path only
        pit = pitch_loss(xc, dry) if args.lam_pitch else torch.tensor(0.0)
        hf = F.relu(torch.log(hf_frac(fake) + 1e-9) - np.log(ref_hf + 1e-9)).mean() \
            if args.lam_hf and args.critic_lpf else torch.tensor(0.0, device=dev)
        dc = G.raw_dc.abs().mean() / TARGET_RMS      # raw dry-path DC offset, relative to nominal level
        terms = {"adv": adv, "fm": args.lam_fm * fm, "cont": args.lam_content * cont,
                 "pit": args.lam_pitch * pit, "hf": args.lam_hf * hf, "dc": args.lam_dc * dc}
        g_loss = sum(terms.values())

        if args.grad_diag_every and step % args.grad_diag_every == 0:
            # how hard each weighted term alone pushes G's weights (L2 norm of its gradient)
            norms = {}
            for k, v in terms.items():
                if v.requires_grad:
                    gs = torch.autograd.grad(v, list(G.parameters()), retain_graph=True, allow_unused=True)
                    norms[k] = torch.sqrt(sum((g ** 2).sum() for g in gs if g is not None)).item()
            print("  G grad-norm per term:", " ".join(f"{k}:{n:.3g}" for k, n in norms.items()), flush=True)
        opt_g.zero_grad(); g_loss.backward(); opt_g.step()
        with torch.no_grad():                    # EMA update
            for qe, q in zip(ema.parameters(), G.parameters()):
                qe.mul_(0.999).add_(q.detach(), alpha=0.001)
        acc_log(d_loss=d_loss, adv=adv, fm=fm, cont=cont, pit=pit, hf=hf, dc=dc, accs=accs.clone(), n=1)

        if step % 100 == 0:                      # values are means over the last 100 steps
            n = run.pop("n")
            m = {k: v / n for k, v in run.items()}
            acc_s = " ".join(f"{nm}:{a:.2f}" for nm, a in zip(names, m["accs"]))
            print(f"{step:6d}  d_loss {m['d_loss']:.3f}  adv {m['adv']:+.3f}  "
                  f"fm {m['fm']:.3f}  cont {m['cont']:.3f}  pit {m['pit']:.3f}  hf {m['hf']:.3f}  dc {m['dc']:.3f}  | {acc_s}",
                  flush=True)
            run.clear()

        if args.probe and step % args.probe_every == 0:
            try:
                ema.eval()
                with torch.no_grad():
                    px = load_mono(args.probe)
                    s0 = int(args.probe_start * SR)
                    px = px[s0:s0 + int(args.probe_len * SR)] if args.probe_len else px[s0:]
                    px = norm(px).to(dev)
                    py, pdy = ema(px[None, :])
                    py = py[0].cpu().numpy()
                    pdy = pdy[0].cpu().numpy()
                sf.write(str(modeldir / "probe.wav"), py, SR)
                sf.write(str(modeldir / "probe_dry.wav"), pdy, SR)
                X = np.abs(np.fft.rfft(px.cpu().numpy())) ** 2
                Y = np.abs(np.fft.rfft(py)) ** 2
                fr = np.fft.rfftfreq(len(px), 1 / SR)
                bands = lambda X: {f"{lo}-{hi}": float(X[(fr >= lo) & (fr < hi)].sum() / X.sum())
                                   for lo, hi in [(80, 300), (300, 1200), (1200, 4000)]}
                print("  probe bands in :", {k: round(v, 3) for k, v in bands(X).items()})
                print("  probe bands out:", {k: round(v, 3) for k, v in bands(Y).items()})
                print("  fx:", ema.describe(), flush=True)
            except Exception as e:
                print(f"  probe failed (non-fatal): {e}")

        if step % args.export_every == 0:
            save_hybrid(ema, modeldir, f"adv_step{step}")

    ema.eval()
    save_hybrid(ema, modeldir, "adv_final")
    print(f"done -> {modeldir}/adv_final.pt (full hybrid, render with apply_hybrid.py) "
          f"+ adv_final.nam (dry path only: drive/EQ/cab, no reverb — NAM plugin)")


if __name__ == "__main__":
    main()
