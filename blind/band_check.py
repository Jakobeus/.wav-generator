"""Where does a wav's energy sit? Usage: python band_check.py file.wav [file2.wav ...]"""
import sys
import numpy as np
import soundfile as sf

for n in sys.argv[1:]:
    x, sr = sf.read(n, dtype="float32")
    x = x if x.ndim == 1 else x.mean(1)
    P = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    tot = P.sum()

    def b(lo, hi):
        return 100 * P[(f >= lo) & (f < hi)].sum() / tot

    print(f"{n}: dc {x.mean():.4f} rms {np.sqrt((x ** 2).mean()):.4f} peak {np.abs(x).max():.3f}")
    print("  <40:%.1f  40-80:%.1f  80-300:%.1f  300-1200:%.1f  1200-4k:%.1f  4k-12k:%.1f  >12k:%.1f"
          % (b(0, 40), b(40, 80), b(80, 300), b(300, 1200), b(1200, 4000), b(4000, 12000), b(12000, sr)))
