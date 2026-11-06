"""Inspect the learned reverb tails in a hybrid .pt: decay shape + spectral peakiness.
Usage: python tail_check.py runs/fme_full/model_adv_v6/adv_step4000.pt"""
import sys
import numpy as np
import torch

SR = 48000
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=True)["state_dict"]
w1, w2 = float(sd["w1"]), float(sd["w2"])
print(f"w1 {w1:+.3f}  w2 {w2:+.3f}")
for name, w in (("h1", w1), ("h2", w2)):
    h = sd[name].numpy().astype(np.float64)
    T = len(h)
    print(f"\n{name}: {T / SR:.2f}s, {T} taps, peak |tap| {np.abs(h).max():.4f}, "
          f"effective gain (|w|*rms*sqrt(T)) {abs(w) * np.sqrt((h ** 2).sum()):.3f}")
    # decay: RMS per 100 ms window, dB rel. first window
    win = SR // 10
    seg = h[: T // win * win].reshape(-1, win)
    rms = np.sqrt((seg ** 2).mean(1)) + 1e-12
    db = 20 * np.log10(rms / rms[0])
    print("  decay (dB per 100ms window):", " ".join(f"{d:+.0f}" for d in db))
    # spectrum: peaks above the local median
    H = np.abs(np.fft.rfft(h)) ** 2
    f = np.fft.rfftfreq(T, 1 / SR)
    Hdb = 10 * np.log10(H + 1e-20)
    from scipy.ndimage import median_filter
    base = median_filter(Hdb, size=201)
    prom = Hdb - base
    flat = np.exp(np.mean(np.log(H[(f > 60) & (f < 12000)] + 1e-20))) / np.mean(H[(f > 60) & (f < 12000)])
    print(f"  spectral flatness 60Hz-12k: {flat:.3f}  (1.0 = white/noise-like reverb, ->0 = a few resonant lines)")
    idx = np.argsort(prom)[::-1]
    picked, shown = [], 0
    for i in idx:
        if f[i] < 40 or f[i] > 12000:
            continue
        if any(abs(f[i] - p) < 15 for p in picked):
            continue
        picked.append(f[i])
        shown += 1
        if shown >= 12:
            break
    print("  strongest resonances (Hz, dB above local median):",
          " ".join(f"{p:.0f}({prom[np.argmin(np.abs(f - p))]:+.0f})" for p in sorted(picked)))
    # how much of the FIR's energy sits in its 20 strongest 10-Hz bins
    order = np.argsort(H)[::-1]
    print(f"  energy share of top-20 bins ({20 * SR / T:.1f} Hz total width): {100 * H[order[:20]].sum() / H.sum():.1f}%")
