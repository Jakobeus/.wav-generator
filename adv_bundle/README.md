# Adversarial Guitar Tone Transfer — Handoff Guide

## The goal, in one paragraph

Make my clean guitar DI sound like a specific reference recording's tone
(a Mickey Darling song, "feed my ego"). No aligned dry/wet pair exists —
the reference is a finished record — so a supervised `dry→wet` model was
impossible. Instead: **GAN-style training**. A waveform generator (`G`)
transforms my DI; a committee of discriminators (`D`) learns to tell real
reference excerpts from G's output; G is trained to fool them while losses
keep my playing intact. The critics are training scaffolding — **only G
ships**, exported as a `.nam` file (Neural Amp Modeler format, loads in
the NAM plugin / real-time use).

## What's in this bundle

```
adv_bundle/
├── adv_train.py          # the trainer (same file as ../blind/adv_train.py)
├── clips/
│   ├── di_nig_a_2-3.wav      60s DI (FL Studio, dead dry — no FX on insert)
│   ├── di_lost_memories.wav  20s DI
│   ├── di_internet_broke.wav 84s DI
│   └── probe_di.wav          5s DI slice — listening test input only
├── ref/ref_wet.wav       175s guitar stem (htdemucs_6s) — the "real" pile
├── ckpt/warmstart.ckpt   pair-trained PackedWaveNet (optional warm start)
└── README.md             this file
```

Note: `ref_wet.wav` is the demucs guitar stem — already "wet" (amp+FX).
The naming just means "the wet reference," it is NOT clean.

## Environment (Linux/WSL2, RTX 4070)

```bash
python -m venv .venv && .venv/bin/pip install -U pip
.venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install neural-amp-modeler==0.13.0 soundfile scipy numpy
# verify: .venv/bin/python -c "import torch; print(torch.cuda.is_available())"  # must be True
```

## Run it

```bash
cd wav-generator/blind
../.venv/bin/python adv_train.py \
    --di ../adv_bundle/clips/di_nig_a_2-3.wav \
         ../adv_bundle/clips/di_lost_memories.wav \
         ../adv_bundle/clips/di_internet_broke.wav \
    --ref ../adv_bundle/ref/ref_wet.wav \
    --modeldir runs/fme_full/model_adv \
    --init-ckpt ../adv_bundle/ckpt/warmstart.ckpt \
    --steps 20000 --device cuda --batch 32 \
    --probe ../adv_bundle/clips/probe_di.wav
```

(adjust python path to wherever the venv lives; `--init-ckpt` is optional —
drop it for a cold start. On the 4070 expect ~10–30ms/step, so 20k steps
is well under an hour.)

## What the trainer does (architecture)

- **G** = PackedWaveNet (~22.8K params, the NAM capture net — ~85ms
  receptive field). Input: random 16384-sample (0.34s) crops of the DI
  files; output: transformed waveform. Warm-started if `--init-ckpt`.
- **11 critics**, each sees the same 0.34s chunk, different representation:
  - `PeriodDisc{2,3,5,7,11}` — folds waveform at period p; catches
    periodicity/phase fakes (the "metallic fizz" police)
  - `WaveDisc{1,2,4}` — raw waveform at 1×/2×/4× decimation; waveform
    shape + dynamics
  - `SpecDisc{512,2048,8192}` — spectrograms at 3 resolutions; spectral
    texture (small n_fft = transients, large = fine frequency detail)
  - all conv layers have spectral normalization
- **Losses (G)**: hinge adversarial (summed over critics) + `fm`
  feature-matching on critics' intermediate layers + `cont` envelope L1
  (playing dynamics preserved) + `pit` autocorrelation-F0 L1 (pitch kept).
- **DiffAugment** on every chunk before D: gain ±6dB / polarity flip /
  circular shift / ±2dB EQ tilt — needed because the real pile is only
  175s; kills cheap memorization paths.
- **EMA**: a 0.999-decay weight copy is used for probe + `.nam` exports.

## Reading the log line

```
step  d_loss   adv      fm      cont    pit    | per-critic accs
 500  13.7     +6.10    2.24    0.14    0.04   | PeriodDisc2:0.50 ... SpecDisc8192:0.94
```

- `adv` — summed across critics; positive = critics still winning. Wants
  to descend over thousands of steps. It will NOT hit 0 cleanly — healthy
  GANs oscillate.
- `fm` — feature-matching gap; wants to trend down (output features
  converging on reference stats).
- `cont` — envelope loss; low = your playing preserved. Should stay < ~0.2.
- `pit` — pitch-drift; should stay small (< ~0.1). Climbing = G is
  detuning notes — raise `--lam-pitch`.
- per-critic accuracy — THE diagnostic. 0.5 = fooled, 1.0 = sees through:
  - PeriodDisc ~0.5 throughout is expected (warm-start already has good
    periodicity; pitch term protects it).
  - Spec/Wave critics starting 0.8–0.95 and descending = G learning.
  - **All critics pinned at 1.00** = critics are memorizing the 175s ref
    → raise `--aug-tilt` to 4–6, or drop `--critics` to a subset.
  - **All critics at 0.50 very early** = critics too weak / G output
    already in-domain — verify by EAR on probe.wav, don't trust numbers.

## Outputs

- `modeldir/probe.wav` — probe_di.wav through the EMA generator, refreshed
  every `--probe-every` (default 500) steps. **Listen to this** — the real
  metric is whether your DI starts sounding like it went through his rig.
- `modeldir/adv_step{N}.nam` — every `--export-every` (2000) steps.
- `adv_final.nam` at the end. Load in the NAM plugin; feed it your DI.

## Knobs (when to turn which)

| Symptom | Turn |
|---|---|
| All critics 1.00 (memorizing) | `--aug-tilt 4`–`6`, or `--critics mpd stft` |
| adv stuck high, fm flat | D too strong: fewer critics or lower their LR in code |
| Output loses your playing/dynamics | `--lam-content` up (default 50) |
| Output drifts in pitch | `--lam-pitch` up (default 5) |
| Tone morph too timid | `--lam-content` down toward 10–20 |
| Tone morph too aggressive | `--lam-content` up toward 100 |
| Want to isolate which critic matters | `--critics mpd` / `msd` / `stft` subsets |

## Design history — dead ends already explored (don't repeat)

- **RemFx pretrained effect-removal**: removes named pedalboard FX but
  cannot un-voice an amp/cab — output was still wet. ESR 0.116 paired
  model learned a hollow delta (proved via clean-DI inference test).
- **Spectral "devoicing" EQ-inversion of the stem**: moved energy balance
  but never produced a true DI — pair ESR ~0.17 was learnability, not truth.
- **Estonia Mel-Denoiser + BigVGAN recovery net**: wired but abandoned —
  same fundamental problem (no true dry signal recoverable).
- **Single mel-critic on contaminated/short ref**: pinned at 1.00 accuracy;
  fixed via multi-critic + augmentation.
- **Probe file `c_chord_md.wav`**: had voice in it — deleted; use
  `probe_di.wav` only.

The adversarial path was chosen precisely because it never needs the dry
signal that couldn't be recovered.

## Verification after training

1. Listen to `probe.wav` evolution — does the DI clip gain the reference's
   mid-forward, driven character without losing note identity?
2. Spectral sanity: probe band energy should move toward the stem's
   profile (~4/79/17 in the 80–300/300–1200/1.2–4k bands).
3. Load `adv_final.nam` in the NAM plugin, play DI through it — the ear
   is the metric, losses are advisory.
4. If it plateaus toneless: the productive lever is MORE clean reference
   (other songs by the same artist, separated the same way — same rig =
   in-domain data), not more steps.

## Known limits

- G's ~85ms receptive field can't learn long reverb tails/tempo delays —
  it captures drive/EQ/cab/dynamics character, which is where the tone lives.
- The reference is one song's worth of notes — the "tone palette" is
  bounded by his register/content; `--aug-tilt` + more references fight this.
- 0.34s critics judge texture+dynamics, not musical structure — by design.
