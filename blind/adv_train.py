#!/usr/bin/env python3
"""
Adversarial tone transfer — train a PackedWaveNet generator to make clean
DI sound like a wet reference WITHOUT aligned pairs.

  G: PackedWaveNet (deterministic waveform->waveform, same as the NAM trainer,
     so the result still exports to .nam). Optional warm-start from a
     pair-trained .ckpt.
  D: mel-spectrogram CNN — classifies ~0.34s chunks as real-wet vs generated.
  Losses:
    GAN  : hinge loss, D on random crops of ref stem vs G(DI crops)
    cont : multi-scale envelope L1 between G output and its DI input —
           preserves played notes/dynamics while the D term reshapes timbre.

Usage:
  python adv_train.py --di my_di1.wav my_di2.wav --ref wet_stem.wav \
      --modeldir runs/x/model_adv [--init-ckpt pair_trained.ckpt] [--steps 20000]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 48000
CHUNK = 16384  # ~0.34s @ 48k — small enough to batch, big enough for D
TARGET_RMS = 10 ** (-18 / 20)


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


class MelDisc(nn.Module):
    """2D CNN on log-mel — learns the reference's spectral/harmonic signature."""

    def __init__(self):
        super().__init__()
        import torchaudio.transforms as T
        self.mel = T.MelSpectrogram(
            sample_rate=SR, n_fft=2048, hop_length=512, win_length=2048,
            n_mels=128, f_min=60, f_max=16000, power=2.0)
        chans = [1, 32, 64, 128, 256]
        self.convs = nn.ModuleList(
            nn.Conv2d(a, b, 5, stride=2, padding=2)
            for a, b in zip(chans, chans[1:]))
        self.head = nn.Conv2d(256, 1, 1)

    def forward(self, x):
        m = torch.log(self.mel(x) + 1e-6)          # (B, 128, T')
        h = m.unsqueeze(1)                          # (B, 1, 128, T')
        for c in self.convs:
            h = F.leaky_relu(c(h), 0.2)
        return self.head(h).mean(dim=(2, 3)).squeeze(-1)  # (B,) logits


def env_loss(a, b):
    """Amplitude-envelope L1 at multiple frame sizes — keeps timing/dynamics."""
    l = 0.0
    for w in (128, 1024):
        ea = F.avg_pool1d(a.abs().unsqueeze(1), w, w).squeeze(1)
        eb = F.avg_pool1d(b.abs().unsqueeze(1), w, w).squeeze(1)
        l = l + F.l1_loss(torch.log(ea + 1e-7), torch.log(eb + 1e-7))
    return l


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--di", nargs="+", required=True, help="clean DI wavs")
    p.add_argument("--ref", required=True, help="wet reference stem")
    p.add_argument("--modeldir", required=True)
    p.add_argument("--init-ckpt", default=None, help="warm-start G from .ckpt")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lam-content", type=float, default=50.0)
    p.add_argument("--probe", default=None,
                   help="clean wav to render through G every 500 steps")
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).parent / "RemFx"))
    sys.path.insert(0, str(Path(__file__).parent / "RemFx" / "umx"))
    modeldir = Path(args.modeldir)
    modeldir.mkdir(parents=True, exist_ok=True)

    from nam.train import core
    cfg = core._get_packed_model_config()
    cls = core._get_lightning_module_cls(cfg)
    lm = cls.init_from_config(cfg)
    if args.init_ckpt:
        lm = cls.load_from_checkpoint(args.init_ckpt, **cls.parse_config(cfg))
        print(f"G warm-started from {args.init_ckpt}")
    G = lm.net
    G.train()
    D = MelDisc()

    di = [load_mono(f) for f in args.di]
    ref = load_mono(args.ref)
    norm = lambda x: x * (TARGET_RMS / x.pow(2).mean().add(1e-12).sqrt())
    di = [norm(x) for x in di]
    ref = norm(ref)
    print(f"DI: {[f'{len(x)/SR:.1f}s' for x in di]}  ref: {len(ref)/SR:.1f}s")

    opt_g = torch.optim.Adam(G.parameters(), 1e-4, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), 1e-4, betas=(0.5, 0.9))
    SUB = 1  # packed submodel (channels_8)

    for step in range(1, args.steps + 1):
        xb = torch.stack([rand_crop(di[i], CHUNK)
                          for i in np.random.randint(0, len(di), args.batch)])
        rb = torch.stack([rand_crop(ref, CHUNK) for _ in range(args.batch)])

        # --- D step (hinge) ---
        with torch.no_grad():
            fake = G(xb)[:, SUB]
        d_real, d_fake = D(rb), D(fake.detach())
        d_loss = (F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean())
        opt_d.zero_grad(); d_loss.backward(); opt_d.step()

        # --- G step ---
        fake = G(xb)[:, SUB]
        adv = -D(fake).mean()
        cont = env_loss(fake, xb)
        g_loss = adv + args.lam_content * cont
        opt_g.zero_grad(); g_loss.backward(); opt_g.step()

        if step % 100 == 0:
            acc = ((d_real > 0).float().mean() + (d_fake < 0).float().mean()) / 2
            print(f"{step:6d}  d_loss {d_loss.item():.3f}  adv {adv.item():+.3f}  "
                  f"cont {cont.item():.3f}  D acc {acc.item():.2f}", flush=True)

        if args.probe and step % 500 == 0:
            G.eval()
            with torch.no_grad():
                px = load_mono(args.probe)
                px = norm(px)
                py = G(px[None, :])[0, SUB].numpy()
            G.train()
            sf.write(str(modeldir / "probe.wav"), py, SR)
            X = np.abs(np.fft.rfft(px.numpy())) ** 2
            Y = np.abs(np.fft.rfft(py)) ** 2
            fr = np.fft.rfftfreq(len(px), 1 / SR)
            bands = lambda X: {f"{lo}-{hi}": float(X[(fr >= lo) & (fr < hi)].sum() / X.sum())
                               for lo, hi in [(80, 300), (300, 1200), (1200, 4000)]}
            print("  probe bands in :", {k: round(v, 3) for k, v in bands(X).items()})
            print("  probe bands out:", {k: round(v, 3) for k, v in bands(Y).items()})

        if step % 2000 == 0:
            G.eval()
            G.sample_rate = SR
            path = modeldir / f"adv_step{step}.nam"
            G.export(path)
            print(f"  saved {path}", flush=True)
            G.train()

    G.eval()
    G.sample_rate = SR
    G.export(modeldir / "adv_final.nam")
    print(f"done -> {modeldir}/adv_final.nam")


if __name__ == "__main__":
    main()
