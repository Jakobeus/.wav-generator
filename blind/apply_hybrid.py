#!/usr/bin/env python3
"""
Chunked whole-song inference for the hybrid (waveshaper + reverb) generator.

Loads a checkpoint saved by adv_train.py ({"kind": "hybrid", ...}) and renders
arbitrary-length audio by streaming context windows: each fresh W samples of
output are produced from the last CTX+W samples of input — tails overlap
automatically because every window sees its own history.

  python apply_hybrid.py --model runs/fme_full/model_adv/adv_final.pt \
      --in di_song.wav --out toned.wav [--device cuda]
"""
import argparse
import sys
from pathlib import Path

import torch

from adv_train import CHUNK, TAIL, HybridG, load_mono, TARGET_RMS  # noqa
import soundfile as sf

CTX = TAIL + 8192  # reverb field + dry-path context margin


def build(dev):
    from nam.train import core
    cfg = core._get_packed_model_config()
    cls = core._get_lightning_module_cls(cfg)
    lm = cls.init_from_config(cfg)
    return HybridG(lm.net, 1).to(dev).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="adv_stepN.pt / adv_final.pt")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dry-out", default=None,
                   help="also write the amp-only (dry path, no reverb) render here")
    p.add_argument("--device", default=None)
    p.add_argument("--norm-input", action="store_true",
                   help="normalize input to -18dBFS like training")
    args = p.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    G = build(dev)
    sd = torch.load(args.model, map_location=dev, weights_only=True)
    G.load_state_dict(sd["state_dict"])
    G.band = tuple(sd.get("band", (0.0, 0.0)))
    print(f"loaded {args.model} on {dev}  (output band {G.band[0]:.0f}-{G.band[1]:.0f} Hz)")

    x = load_mono(args.inp)
    if args.norm_input:
        x = x * (TARGET_RMS / x.pow(2).mean().add(1e-12).sqrt())
    W, T = CHUNK, len(x)
    y = torch.zeros(T)
    yd = torch.zeros(T)
    with torch.no_grad():
        for s in range(0, T, W):
            lo, hi = s - CTX, min(T, s + W)
            seg = x[max(0, lo):hi]
            if lo < 0:                       # left-pad missing context
                seg = torch.cat([torch.zeros(-lo), seg])
            seg = seg.to(dev)
            out, dry = G(seg.unsqueeze(0))
            n = min(W, T - s)
            y[s:s + n] = out[0, -n:].cpu()
            yd[s:s + n] = dry[0, -n:].cpu()
    sf.write(args.out, y.numpy(), 48000)
    print(f"wrote {args.out} ({T/48000:.1f}s)")
    if args.dry_out:
        sf.write(args.dry_out, yd.numpy(), 48000)
        print(f"wrote {args.dry_out} (dry path only)")


if __name__ == "__main__":
    main()
