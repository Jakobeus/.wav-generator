#!/usr/bin/env python3
"""
Prepare an aligned input/output pair for NAM training.

- mono, resample to 48k
- cross-correlate to find the lag between dry and wet, shift to align
  (wet trails dry by the chain's latency)
- trim both to common length
- writes pair/input.wav (dry) and pair/output.wav (wet)

Usage: python align.py <dry.wav> <wet.wav> <outdir>
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

SR = 48000


def load_mono_48k(path):
    x, sr = sf.read(path, always_2d=True)
    x = x.mean(axis=1).astype(np.float64)
    if sr != SR:
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g)
    peak = np.abs(x).max()
    if peak > 0:
        x *= 0.95 / peak  # leave headroom — NAM flags |y|>=1.0 as clipped
    return x


def best_lag(dry, wet, max_lag_s=0.25):
    """Return samples wet trails dry (positive => wet is later)."""
    n = min(len(dry), len(wet), int(max_lag_s * SR))
    d, w = dry[: 10 * SR], wet[: 10 * SR + n]
    corr = fftconvolve(w, d[::-1], mode="valid")
    return int(np.argmax(np.abs(corr)))


def main():
    dry, wet, outdir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    outdir.mkdir(parents=True, exist_ok=True)
    d = load_mono_48k(dry)
    w = load_mono_48k(wet)

    lag = best_lag(d, w)
    if lag >= 0:
        d, w = d[: len(w) - lag], w[lag:]
    else:
        d, w = d[-lag:], w[: len(d) + lag]
    n = min(len(d), len(w))
    d, w = d[:n], w[:n]
    print(f"lag {lag} samples ({lag / SR * 1000:.1f} ms), pair length {n / SR:.1f}s")

    sf.write(outdir / "input.wav", d.astype(np.float32), SR, subtype="FLOAT")
    sf.write(outdir / "output.wav", w.astype(np.float32), SR, subtype="FLOAT")
    print(f"wrote {outdir}/input.wav + {outdir}/output.wav")


if __name__ == "__main__":
    main()
