#!/usr/bin/env python3
"""
Estimate the dry (pre-effects) signal from a wet guitar stem.

Modes:
  bypass <dry.wav>   - you supply the dry signal (amp-sim pair, own recording,
                       or an already-clean stem). Validates the rest of the
                       pipeline without the ill-posed step.
  separator [model]  - run an audio-separator model (e.g. a de-reverb model).
                       Honest caveat: covers reverb/echo/denoise removal;
                       true distortion inversion needs RemFx or the
                       Estonia repo's trained model (see REMFX mode stub).
  remfx [chain]      - run RemFx pretrained de-effect models. chain is a
                       comma-separated subset of distortion,compressor,reverb,
                       chorus,delay (default: distortion). Checkpoints live in
                       remfx_ckpts/ (Zenodo record 8218621).
  devoice <target>   - zero-phase spectral whitening of a remfx-style dry
                       estimate toward the envelope of <target> (your own
                       clean DI). Removes amp/cab voicing the removers leave.

Usage: python deeffect.py <mode> <wet_in.wav> <dry_out.wav> [args]
"""
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
VENV_PY = HERE / ".venv" / "bin" / "python"
VENV_SEP = HERE / ".venv" / "bin" / "audio-separator"


def bypass(wet, out, dry_path):
    shutil.copyfile(dry_path, out)
    print(f"bypass: using supplied dry {dry_path}")


def separator(wet, out, model):
    import re
    outdir = HERE / "_deeffected"
    outdir.mkdir(exist_ok=True)
    subprocess.run(
        [str(VENV_SEP), str(wet), "-m", model, "--output_dir", str(outdir),
         "--output_format", "WAV"],
        check=True,
    )
    produced = sorted(outdir.rglob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not produced:
        sys.exit("separator produced no wavs — check model name (audio-separator -l)")
    # models emit "<input>_(<stem>)_<model>.wav" — keep the DRY stem, not the
    # removed effect ('(noreverb)', '(dry)'…), never '(reverb)'/'(no dry)'…
    keep = re.compile(r"no[ _-]?reverb|no[ _-]?echo|no[ _-]?noise|\(dry|\(clean", re.I)
    choice = ([p for p in produced if keep.search(p.stem)] or produced)[0]
    shutil.copyfile(choice, out)
    print(f"separator({model}): {choice.name} -> {out}")


REMFX_DIR = HERE / "RemFx"
REMFX_SR = 48000
REMFX_CHUNK = 262144  # 5.46s @ 48k, from cfg/exp/remfx_all.yaml

_EFFECTS = {
    "distortion": ("demucs", "demucs_distortion_aug.ckpt"),
    "compressor": ("demucs", "demucs_compressor_aug.ckpt"),
    "reverb": ("dcunet", "dcunet_reverb_aug.ckpt"),
    "chorus": ("dcunet", "dcunet_chorus_aug.ckpt"),
    "delay": ("dcunet", "dcunet_delay_aug.ckpt"),
}


def _remfx_model(arch, ckpt_path):
    import torch
    from remfx.models import DemucsModel, DCUNetModel
    if arch == "demucs":
        m = DemucsModel(
            sample_rate=REMFX_SR, sources=["mixture"], audio_channels=1,
            nfft=4096, channels=48,
        )
    else:
        m = DCUNetModel(
            sample_rate=REMFX_SR, num_bins=1025,
            architecture="Large-DCUNet-20", stft_kernel_size=512,
            fix_length_mode="pad",
        )
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    # ckpt wraps the network as RemFX.model.* — strip that prefix, drop metrics
    sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    m.load_state_dict(sd, strict=False)
    return m.eval()


def _infer_long(model, x, chunk=REMFX_CHUNK, overlap=REMFX_CHUNK // 4):
    """OLA over long inputs: windows of `chunk`, `overlap` crossfaded."""
    import torch
    T = x.shape[-1]
    if T <= chunk:
        return model.sample(x)
    out = torch.zeros_like(x)
    weight = torch.zeros_like(x)
    win = torch.ones(chunk)
    win[:overlap] = torch.linspace(0, 1, overlap)
    win[-overlap:] = torch.linspace(1, 0, overlap)
    for s in range(0, T, chunk - overlap):
        seg = x[..., s:s + chunk]
        n = seg.shape[-1]
        if n < chunk:
            seg = torch.nn.functional.pad(seg, (0, chunk - n))
        y = model.sample(seg)
        out[..., s:s + n] += y[..., :n] * win[:n]
        weight[..., s:s + n] += win[:n]
    return out / weight.clamp(min=1e-8)


def devoice(est_dry, out, target_di):
    """Invert the long-term spectral envelope toward a DI-domain target.

    RemFx removes named effects but leaves amp/cab voicing (a big static EQ
    curve) in the 'dry' estimate, so the tone model never learns it. This
    stage whitens est_dry toward the spectral envelope of `target_di` —
    ideally the user's own clean DI, so inference inputs are in-domain.

    Zero-phase (filtfilt) filtering keeps sample alignment with the wet stem.
    """
    import numpy as np
    import soundfile as sf
    from scipy import signal

    MAX_DB = 12.0  # per-iteration clamp; filtfilt applies twice -> ±24 dB/iter

    d, sr = sf.read(str(est_dry), dtype="float32", always_2d=True)
    t, sr_t = sf.read(str(target_di), dtype="float32", always_2d=True)
    d, t = d.mean(1), t.mean(1)
    if sr_t != sr:
        t = signal.resample_poly(t, sr, sr_t)

    nper = 8192
    f, Pt = signal.welch(t, sr, nperseg=nper)
    flog = np.log(f[1:])
    grid = np.linspace(flog[0], flog[-1], 512)
    gf = np.exp(grid)
    band = (gf >= 60) & (gf <= 16000)

    # smooth PSD over ~1/3-octave in log-f — note-content differences between
    # clips must not become EQ moves
    k = int(512 * (1 / 3) * np.log10(2) / np.log10(gf[-1] / gf[0]) * 2)
    k = max(k, 9) | 1
    kern = np.hanning(k) / np.hanning(k).sum()

    def smooth_db(x):
        _, P = signal.welch(x, sr, nperseg=nper)
        y = np.interp(grid, flog, np.log(np.maximum(P[1:], 1e-14)))
        return np.convolve(y, kern, "same")  # dB-domain-smoothed log PSD

    Pt_db = smooth_db(t)
    y = d.copy()
    for it in range(5):
        resid_db = Pt_db - smooth_db(y)
        resid_db[~band] = 0.0
        # filtfilt applies the FIR twice -> applied change = 2*corr; request
        # half the residual so each iteration moves ~all the way to target
        corr = np.clip(resid_db / 2.0, -MAX_DB, MAX_DB)
        if np.abs(resid_db[band]).max() < 1.0:
            print(f"devoice: converged after {it} iterations")
            break
        freq_pts = np.concatenate([[0.0], gf, [sr / 2]])
        resp_db = np.concatenate([[0.0], corr, [0.0]])
        taps = signal.firwin2(4097, np.clip(freq_pts, 0, sr / 2) / (sr / 2),
                            10 ** (resp_db / 20))
        y = signal.filtfilt(taps, 1.0, y)
    resid_db = Pt_db - smooth_db(y)
    print(f"devoice: residual |err| max {np.abs(resid_db[band]).max():.1f} dB "
          f"(target {target_di}) -> {out}")

    sf.write(str(out), np.clip(y, -1, 1), sr)


def remfx(wet, out, chain):
    import sys
    import types
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    for p in (str(REMFX_DIR), str(REMFX_DIR / "umx")):
        if p not in sys.path:
            sys.path.insert(0, p)
    # classifier-only deps — unused for fixed chains, stub if absent
    for mod in ("hearbaseline", "hearbaseline.vggish", "hearbaseline.wav2vec2",
                "wav2clip_hear", "panns_hear"):
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = types.ModuleType(mod)

    names = chain or ["distortion"]
    for name in names:
        if name not in _EFFECTS:
            sys.exit(f"remfx: unknown effect {name} — {sorted(_EFFECTS)}")
    ckpts = [(name,) + _EFFECTS[name] for name in names]
    missing = [f for _, _, f in ckpts if not (HERE / "remfx_ckpts" / f).exists()]
    if missing:
        sys.exit(f"remfx: missing checkpoints in {HERE/'remfx_ckpts'}: {missing}")

    data, orig_sr = sf.read(str(wet), dtype="float32", always_2d=True)
    x = torch.from_numpy(data.T).mean(0, keepdim=True)  # mono — models are 1-channel
    sr, T = orig_sr, x.shape[-1]
    if sr != REMFX_SR:
        x = torchaudio.functional.resample(x, sr, REMFX_SR)
        sr = REMFX_SR
    x = x.unsqueeze(0)  # (1, 1, T)
    with torch.no_grad():
        for name, arch, fname in ckpts:
            print(f"remfx: {name} ({fname}) ...", flush=True)
            model = _remfx_model(arch, HERE / "remfx_ckpts" / fname)
            x = _infer_long(model, x)
            del model
    y = x.squeeze(0)
    if sr != orig_sr:
        y = torchaudio.functional.resample(y, sr, orig_sr)
    y = y[..., :T] if y.shape[-1] >= T else torch.nn.functional.pad(y, (0, T - y.shape[-1]))
    sf.write(str(out), y.squeeze(0).clamp(-1, 1).numpy(), orig_sr)
    print(f"remfx: {'->'.join(names)} -> {out}")


if __name__ == "__main__":
    mode = sys.argv[1]
    wet, out = Path(sys.argv[2]), Path(sys.argv[3])
    out.parent.mkdir(parents=True, exist_ok=True)
    if mode == "bypass":
        bypass(wet, out, sys.argv[4])
    elif mode == "separator":
        separator(wet, out, sys.argv[4] if len(sys.argv) > 4 else "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt")
    elif mode == "remfx":
        chain = sys.argv[4].split(",") if len(sys.argv) > 4 else None
        remfx(wet, out, chain)
    elif mode == "devoice":
        devoice(wet, out, sys.argv[4])
    else:
        sys.exit(f"unknown mode {mode} — bypass | separator | remfx | devoice")
