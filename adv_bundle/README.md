# adversarial tone-transfer bundle

Everything `blind/adv_train.py` needs on a GPU box. Copy this dir anywhere,
then inside the cloned wav-generator repo:

```bash
# 1. env (cuda build of torch — cu130 matches what's here, cu124 also fine)
python -m venv .venv && .venv/bin/pip install -U pip
.venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
.venv/bin/pip install neural-amp-modeler==0.13.0 soundfile scipy numpy

# 2. run (from wav-generator/blind/)
.venv/bin/python adv_train.py \
    --di  <bundle>/clips/di_nig_a_2-3.wav <bundle>/clips/di_lost_memories.wav <bundle>/clips/di_internet_broke.wav \
    --ref <bundle>/ref/ref_wet.wav \
    --modeldir runs/fme_full/model_adv \
    --init-ckpt <bundle>/ckpt/warmstart.ckpt \
    --steps 20000 --device cuda --batch 32 \
    --probe <bundle>/clips/probe_di.wav
```

- `clips/` — DI training inputs (all 164s used via random 0.34s crops) + probe clip
- `ref/ref_wet.wav` — 26.7s effected-guitar reference (the discriminator's "real" pile)
- `ckpt/warmstart.ckpt` — pair-trained devoiced model, warm-starts G (optional, drop `--init-ckpt` for cold start)
- probe.wav + adv_step*.nam land in `--modeldir`; the .nam is the plugin-usable model

Knobs: `--critics mpd msd stft` (subset to ablate), `--lam-content` (playing fidelity vs tone morph), `--lam-fm`, `--lam-pitch` (F0-consistency, autocorrelation-based), `--aug-tilt` (random EQ tilt in dB on real+fake — 2 default; raise toward 4–6 if critics memorize the ref fingerprint, 0 disables).
