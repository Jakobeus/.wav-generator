# blind — per-tone blind capture pipeline

Given a song with an effected guitar, build a playable NAM model of that tone
by estimating the dry signal and training on the (estimated dry → wet) pair.

## Setup

    ./setup.sh          # venv + deps (demucs, audio-separator, neural-amp-modeler)

## Usage

    ./pipeline.sh song.wav                  # full route (de-effect = separator)
    DRY=my_di.wav ./pipeline.sh song.wav bypass   # supply your own dry
    ./pipeline.sh song.wav separator 100    # more epochs for a keeper

Or step by step:

    ./separate.sh song.wav                  # → separated/.../guitar.wav (wet ref)
    .venv/bin/python deeffect.py separator wet.wav est_dry.wav [model]
    .venv/bin/python deeffect.py remfx wet.wav est_dry.wav [chain]
    #   chain = comma list: distortion,compressor,reverb,chorus,delay
    #   ckpts auto-load from remfx_ckpts/ (Zenodo record 8218621)
    .venv/bin/python align.py est_dry.wav wet.wav pairdir
    .venv/bin/python train.py pairdir modeldir --epochs 40

## Pipeline

| Step | Tool | Notes |
|---|---|---|
| stem separation | Demucs `htdemucs_6s` | isolates the guitar stem — the wet reference |
| de-effect | `deeffect.py` | `bypass` = supply dry (validates rest of pipeline). `separator` = an audio-separator model (de-reverb/echo/denoise — **not** true distortion inversion, and smears timing: ~0.25 corr). `remfx` = pretrained RemFx removers (mhrice/RemFx, 48kHz): `demucs_*` for distortion/compressor, `dcunet_*` for reverb/chorus/delay — output stays sample-aligned (~0.99 corr) |
| align | `align.py` | mono, 48k, cross-correlation latency fix, common trim |
| train | `nam.train.core.train` | `ignore_checks=True` since input isn't the official sweep |

## What to expect

- **Amp/cab/EQ/dynamics** — what the model can actually learn: static nonlinear
  transform + linear filtering. This is the bulk of "tone."
- **Reverb/delay/modulation** — long-memory or time-varying FX get smeared
  or dropped. Expected failure, not a bug.
- **De-effect quality is everything.** A clean distortion-free stem trained as
  "dry = wet" produces a useless identity model. The pipeline is only as good
  as the estimated dry — swap in RemFx/Estonia weights for distorted material.
- **Input level matters.** Level-match your DI to the estimated dry (amp
  drive response is level-dependent); `align.py` peak-normalizes both sides,
  which approximates it.
- CPU training: 40 epochs on a small model ≈ coffee, not overnight.
