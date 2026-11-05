#!/usr/bin/env python3
"""
Adversarial tone transfer — multi-critic, hybrid reverb generator.

  G = HybridG: PackedWaveNet dry path (drive/EQ/cab, ~85ms) + parallel
      reverb field:
        tail1 = learned FIR  (static, time-invariant room)
        tail2 = input-gated FIR driven by a frame-level transformer gate
                (nonlinear reverb: gated/ducked/selective tails; degenerates
                to a second linear tail if modulation is unused)
      out = dry + w1·tail1 + w2·tail2
      Identity anchors (envelope/pitch) apply to the DRY PATH ONLY —
      tails are allowed to outlast the input envelope by design.

  Critics (training-time scaffolding, discarded after):
    MPD  : waveform folded at periods {2,3,5,7,11} — periodicity/phase
    MSD  : raw waveform avg-pooled {1x,2x,4x} — waveform shape + dynamics
    STFT : spectrograms at n_fft {512,2048,8192} + mel-scaled — spectral texture
    LONG : WaveDisc(4) + SpecDisc(4096) on the full 1.5s window — tail grading
    (short-view critics see a random internal 0.34s subcrop per call)

  Losses: hinge adversarial (summed) + feature matching + dry-path envelope
  and pitch-consistency. DiffAugment (gain/polarity/shift/mild EQ tilt)
  on real+fake.

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


def rand_crop(x, n):
    i = np.random.randint(0, max(1, len(x) - n + 1))
    seg = x[i:i + n]
    return F.pad(seg, (0, n - len(seg)))


def augment(x, tilt_db=0.0):
    """DiffAugment on real and fake alike. gain ±6dB / polarity flip /
    circular shift; tilt_db>0 adds a random spectral slope ±tilt_db."""
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
        u = torch.log(f / 20) / np.log(20000 / 20)
        s = (torch.rand(B, 1, device=x.device) * 2 - 1) * tilt_db
        x = torch.fft.irfft(X * (10 ** (s * (u - 0.5) * 2 / 20)), n=T)
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

class HybridG(nn.Module):
    """Waveshaper + parallel reverb field. dry: PackedWaveNet (wrapped).
    tail1: learned FIR (static). tail2: FIR over input gated per-frame by a
    small transformer context net (nonlinear/selective tails)."""

    FRAME = 480  # 10ms gate frames

    def __init__(self, wavenet, sub, tail=TAIL, d_model=128, n_layers=2):
        super().__init__()
        self.dry = wavenet
        self.sub = sub
        # FIR taps: tiny decaying noise — zero-init deadlocks gradients
        t = torch.arange(tail, dtype=torch.float32)
        env = torch.exp(-3.0 * t / tail)
        self.h1 = nn.Parameter(torch.randn(tail) * 1e-3 * env)
        self.h2 = nn.Parameter(torch.randn(tail) * 1e-3 * env)
        self.w1 = nn.Parameter(torch.tensor(0.3))
        self.w2 = nn.Parameter(torch.tensor(0.3))
        # gate context: frame encoder -> transformer -> per-frame gate
        self.enc = nn.Conv1d(1, d_model, self.FRAME, stride=self.FRAME // 2)
        self.pos = nn.Parameter(torch.randn(1, 4096, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, 4, d_model * 4, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, n_layers)
        self.gate = nn.Linear(d_model, 1)

    def forward(self, x):                       # x: (B,T) DI -> (out, dry)
        B, T = x.shape
        dry = self.dry(x)[:, self.sub]          # nonlinear tone path
        e = self.enc(dry.unsqueeze(1)).transpose(1, 2)        # (B,F,d)
        e = e + self.pos[:, :e.shape[1]]
        g = torch.sigmoid(self.gate(self.tr(e)))              # (B,F,1)
        g = F.interpolate(g.transpose(1, 2), size=T).squeeze(1)  # (B,T)
        tail1 = fft_fir(dry, self.h1)
        tail2 = fft_fir(dry * (2 * g), self.h2)               # input-gated tail
        out = dry + self.w1 * tail1 + self.w2 * tail2
        return out, dry


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


def pitch_loss(x, y):
    """F0-consistency via normalized autocorrelation profiles (80-1200Hz)."""
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
    p.add_argument("--init-ckpt", default=None, help="warm-start G dry path")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lam-content", type=float, default=50.0)
    p.add_argument("--lam-fm", type=float, default=2.0)
    p.add_argument("--lam-pitch", type=float, default=5.0)
    p.add_argument("--aug-tilt", type=float, default=2.0)
    p.add_argument("--device", default=None)
    p.add_argument("--critics", nargs="+",
                   default=["mpd", "msd", "stft", "long"],
                   choices=["mpd", "msd", "stft", "long"])
    p.add_argument("--probe", default=None)
    p.add_argument("--probe-every", type=int, default=500)
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

    opt_g = torch.optim.Adam(G.parameters(), 1e-4, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(critics.parameters(), 1e-4, betas=(0.5, 0.9))
    d_loss = torch.tensor(0.0)
    accs = torch.zeros(len(critics))

    for step in range(1, args.steps + 1):
        xb = torch.stack([rand_crop(di[i], CHUNK)
                          for i in np.random.randint(0, len(di), args.batch)]).to(dev)
        rb = torch.stack([rand_crop(ref, CHUNK) for _ in range(args.batch)]).to(dev)

        # --- D step ---
        with torch.no_grad():
            fake, _ = G(xb)
        d_loss = torch.zeros((), device=dev)
        for i, D in enumerate(critics):
            d_real = D(augment(rb, args.aug_tilt))
            d_fake = D(augment(fake.detach(), args.aug_tilt))
            d_loss = d_loss + hinge(d_real, d_fake)
            accs[i] = ((d_real > 0).float().mean() + (d_fake < 0).float().mean()) / 2
        opt_d.zero_grad(); d_loss.backward(); opt_d.step()

        # --- G step ---
        fake, dry = G(xb)
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
        cont = env_loss(dry, xb)                # anchors on DRY path only
        pit = pitch_loss(xb, dry) if args.lam_pitch else torch.tensor(0.0)
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

        if args.probe and step % args.probe_every == 0:
            try:
                ema.eval()
                with torch.no_grad():
                    px = norm(load_mono(args.probe)).to(dev)
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
            except Exception as e:
                print(f"  probe failed (non-fatal): {e}")

        if step % args.export_every == 0:
            path = modeldir / f"adv_step{step}.pt"
            torch.save({"state_dict": ema.state_dict(), "kind": "hybrid",
                        "sr": SR, "sub": SUB}, path)
            print(f"  saved {path}", flush=True)

    ema.eval()
    torch.save({"state_dict": ema.state_dict(), "kind": "hybrid",
                "sr": SR, "sub": SUB}, modeldir / "adv_final.pt")
    print(f"done -> {modeldir}/adv_final.pt  (hybrid net — NOT .nam; needs adv_train.HybridG to load)")


if __name__ == "__main__":
    main()
