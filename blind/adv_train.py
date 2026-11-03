#!/usr/bin/env python3
"""
Adversarial tone transfer — multi-critic version (HiFi-GAN-style).

  G: PackedWaveNet (waveform->waveform, same net as the NAM trainer, so the
     result still exports to .nam). Optional warm-start from a pair-trained
     .ckpt. Training-time scaffolding only — critics are discarded after.

  Critic families (each blind to what the others police):
    MPD  : waveform folded at periods {2,3,5,7,11} — periodicity/phase
    MSD  : raw waveform avg-pooled {1x,2x,4x} — waveform shape + dynamics
    STFT : spectrograms at n_fft {512,2048,8192} — spectral texture, 3 resolutions

  Losses: hinge adversarial (summed over critics) + feature matching on each
  critic's intermediate layers + multi-scale envelope content loss.
  DiffAugment (gain/polarity/circular-shift) on real+fake — no EQ tilt:
  spectral envelope is the feature D is SUPPOSED to police.

Usage:
  python adv_train.py --di di1.wav di2.wav --ref wet_stem.wav \
      --modeldir runs/x/model_adv [--init-ckpt pair.ckpt] [--steps 20000]
      [--device cuda] [--critics mpd msd stft]
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
CHUNK = 16384  # ~0.34s @ 48k
TARGET_RMS = 10 ** (-18 / 20)
sn = nn.utils.spectral_norm


def load_mono(path):
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = torch.from_numpy(x.mean(1))
    if sr != SR:
        from scipy import signal
        x = torch.from_numpy(signal.resample_poly(x.numpy(), SR, sr).astype(np.float32))
    return x


def rand_crop(x, n):
    i = np.random.randint(0, max(1, len(x) - n + 1))
    seg = x[i:i + n]
    return F.pad(seg, (0, n - len(seg)))


def augment(x, tilt_db=0.0):
    """DiffAugment: same family of transforms on real and fake.
    gain ±6dB, polarity flip, circular shift — invariants D must learn.
    tilt_db>0 adds a random spectral slope ±tilt_db (20Hz-20kHz, pivot ~632Hz):
    mild values stop D memorizing the ref's exact EQ fingerprint; large values
    would dissolve the spectral target itself — keep it small."""
    B, T = x.shape
    g = torch.rand(B, 1, device=x.device) * 12 - 6
    x = x * (10 ** (g / 20))
    x = x * torch.where(torch.rand(B, 1, device=x.device) < 0.5, -1.0, 1.0)
    shifts = torch.randint(0, T, (B,), device=x.device)
    idx = (torch.arange(T, device=x.device).unsqueeze(0) + shifts.unsqueeze(1)) % T
    x = torch.gather(x, 1, idx)
    if tilt_db > 0:
        X = torch.fft.rfft(x)
        f = torch.fft.rfftfreq(T, 1 / SR, device=x.device).clamp(min=20)
        u = torch.log(f / 20) / np.log(20000 / 20)            # 0..1 across band
        s = (torch.rand(B, 1, device=x.device) * 2 - 1) * tilt_db
        x = torch.fft.irfft(X * (10 ** (s * (u - 0.5) * 2 / 20)), n=T)
    return x


def _mean_feats(feats):
    return [f.mean(dim=(2, 3)) if f.dim() == 4 else f.mean(dim=2) for f in feats]


class PeriodDisc(nn.Module):
    """HiFi-GAN MPD: fold waveform at period p — catches periodicity/phase
    artifacts that spectral critics cannot see."""

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
        h = x.view(B, 1, -1, self.p)              # fold: (B,1,T/p,p)
        feats = []
        for c in self.convs:
            h = F.leaky_relu(c(h), 0.2)
            feats.append(h)
        logit = self.head(h).mean(dim=(2, 3)).squeeze(-1)
        return (logit, _mean_feats(feats)) if return_feats else logit


class WaveDisc(nn.Module):
    """MSD-style: raw waveform at decimation scale s — waveform shape (s=1)
    through gross envelope/dynamics (s=4)."""

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
    """Spectrogram critic at n_fft resolution — small n_fft polices
    transient/time detail, large n_fft polices harmonic/freq detail."""

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


def build_critics(kind):
    if kind == "mpd":
        return [PeriodDisc(p) for p in (2, 3, 5, 7, 11)]
    if kind == "msd":
        return [WaveDisc(s) for s in (1, 2, 4)]
    if kind == "stft":
        return [SpecDisc(n) for n in (512, 2048, 8192)]
    raise ValueError(kind)


def env_loss(a, b):
    """Amplitude-envelope L1 at multiple frame sizes — keeps timing/dynamics."""
    l = 0.0
    for w in (128, 1024):
        ea = F.avg_pool1d(a.abs().unsqueeze(1), w, w).squeeze(1)
        eb = F.avg_pool1d(b.abs().unsqueeze(1), w, w).squeeze(1)
        l = l + F.l1_loss(torch.log(ea + 1e-7), torch.log(eb + 1e-7))
    return l


def pitch_loss(x, y):
    """F0-consistency, detector-free: normalized autocorrelation profiles
    encode the period — L1 over the guitar lag range (80-1200Hz) penalizes
    pitch drift/octave jumps. Differentiable; harmonic-content changes only
    soften autocorr peaks, they don't move them."""
    T = x.shape[-1]
    ax = torch.fft.irfft(torch.fft.rfft(x).abs() ** 2, n=T)[:, :600]
    ay = torch.fft.irfft(torch.fft.rfft(y).abs() ** 2, n=T)[:, :600]
    ax = ax / ax[:, :1].clamp(min=1e-9)
    ay = ay / ay[:, :1].clamp(min=1e-9)
    return F.l1_loss(ax[:, 40:], ay[:, 40:])


def hinge(real_logit, fake_logit):
    return F.relu(1 - real_logit).mean() + F.relu(1 + fake_logit).mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--di", nargs="+", required=True, help="clean DI wavs")
    p.add_argument("--ref", required=True, help="wet reference stem")
    p.add_argument("--modeldir", required=True)
    p.add_argument("--init-ckpt", default=None, help="warm-start G from .ckpt")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lam-content", type=float, default=50.0)
    p.add_argument("--lam-fm", type=float, default=2.0)
    p.add_argument("--lam-pitch", type=float, default=5.0,
                   help="autocorrelation F0-consistency weight (0 disables)")
    p.add_argument("--aug-tilt", type=float, default=2.0,
                   help="random EQ tilt in dB applied to real+fake (0 disables)")
    p.add_argument("--device", default=None)
    p.add_argument("--critics", nargs="+", default=["mpd", "msd", "stft"],
                   choices=["mpd", "msd", "stft"])
    p.add_argument("--probe", default=None,
                   help="clean wav to render through G every 500 steps")
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
        print(f"G warm-started from {args.init_ckpt}")
    G = lm.net.to(dev)
    G.train()
    ema = copy.deepcopy(G).eval()               # EMA copy -> probe + export
    for q in ema.parameters():
        q.requires_grad_(False)

    critics = nn.ModuleList(
        [d for kind in args.critics for d in build_critics(kind)]).to(dev)
    names = [f"{type(d).__name__}{getattr(d, 'p', getattr(d, 'scale', getattr(d, 'n_fft', '')))}"
             for d in critics]
    print(f"critics ({len(critics)}): {names}")

    di = [load_mono(f) for f in args.di]
    ref = load_mono(args.ref)
    norm = lambda x: x * (TARGET_RMS / x.pow(2).mean().add(1e-12).sqrt())
    di = [norm(x) for x in di]
    ref = norm(ref)
    print(f"device {dev} | DI: {[f'{len(x)/SR:.1f}s' for x in di]}  ref: {len(ref)/SR:.1f}s")

    opt_g = torch.optim.Adam(G.parameters(), 1e-4, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(critics.parameters(), 1e-4, betas=(0.5, 0.9))
    SUB = 1  # packed submodel (channels_8)
    d_loss = torch.tensor(0.0)
    accs = torch.zeros(len(critics))

    for step in range(1, args.steps + 1):
        xb = torch.stack([rand_crop(di[i], CHUNK)
                          for i in np.random.randint(0, len(di), args.batch)]).to(dev)
        rb = torch.stack([rand_crop(ref, CHUNK) for _ in range(args.batch)]).to(dev)

        # --- D step: every critic sees augmented real + fake ---
        with torch.no_grad():
            fake = G(xb)[:, SUB]
        d_loss = torch.zeros((), device=dev)
        for i, D in enumerate(critics):
            d_real = D(augment(rb, args.aug_tilt))
            d_fake = D(augment(fake.detach(), args.aug_tilt))
            d_loss = d_loss + hinge(d_real, d_fake)
            accs[i] = ((d_real > 0).float().mean() + (d_fake < 0).float().mean()) / 2
        opt_d.zero_grad(); d_loss.backward(); opt_d.step()

        # --- G step: fool every critic + match their features + keep playing ---
        fake = G(xb)[:, SUB]
        fa = augment(fake, args.aug_tilt)
        ra = augment(rb, args.aug_tilt)
        adv = torch.zeros((), device=dev)
        fm = torch.zeros((), device=dev)
        for D in critics:
            logit, f_fake = D(fa, return_feats=True)
            with torch.no_grad():
                _, f_real = D(ra, return_feats=True)
            adv = adv - logit.mean()
            fm = fm + sum(F.l1_loss(ff.mean(0), fr.mean(0))
                          for ff, fr in zip(f_fake, f_real))
        cont = env_loss(fake, xb)
        pit = pitch_loss(xb, fake) if args.lam_pitch else torch.tensor(0.0)
        g_loss = adv + args.lam_fm * fm + args.lam_content * cont \
            + args.lam_pitch * pit
        opt_g.zero_grad(); g_loss.backward(); opt_g.step()
        with torch.no_grad():                    # EMA update
            for qe, q in zip(ema.parameters(), G.parameters()):
                qe.mul_(0.999).add_(q.detach(), alpha=0.001)

        if step % 100 == 0:
            acc_s = " ".join(f"{n}:{a:.2f}" for n, a in zip(names, accs))
            print(f"{step:6d}  d_loss {d_loss.item():.3f}  adv {adv.item():+.3f}  "
                  f"fm {fm.item():.3f}  cont {cont.item():.3f}  pit {pit.item():.3f}  | {acc_s}",
                  flush=True)

        if args.probe and step % 500 == 0:
            try:
                ema.eval()
                with torch.no_grad():
                    px = norm(load_mono(args.probe)).to(dev)
                    py = ema(px[None, :])[0, SUB].cpu().numpy()
                sf.write(str(modeldir / "probe.wav"), py, SR)
                X = np.abs(np.fft.rfft(px.cpu().numpy())) ** 2
                Y = np.abs(np.fft.rfft(py)) ** 2
                fr = np.fft.rfftfreq(len(px), 1 / SR)
                bands = lambda X: {f"{lo}-{hi}": float(X[(fr >= lo) & (fr < hi)].sum() / X.sum())
                                   for lo, hi in [(80, 300), (300, 1200), (1200, 4000)]}
                print("  probe bands in :", {k: round(v, 3) for k, v in bands(X).items()})
                print("  probe bands out:", {k: round(v, 3) for k, v in bands(Y).items()})
            except Exception as e:
                print(f"  probe failed (non-fatal): {e}")

        if step % 2000 == 0:
            ema.sample_rate = SR
            ema.export(modeldir, basename=f"adv_step{step}")
            print(f"  saved {modeldir}/adv_step{step}.nam", flush=True)

    ema.eval()
    ema.sample_rate = SR
    ema.export(modeldir, basename="adv_final")
    print(f"done -> {modeldir}/adv_final.nam")


if __name__ == "__main__":
    main()
