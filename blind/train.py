#!/usr/bin/env python3
"""
Train a NAM model on an arbitrary (dry, wet) pair — the "blind per-tone"
step. core.train() insists on the official NAM calibration sweep (it
hash-checks the input file), so this drives the same internals directly:
PackedWaveNet model, paired dataset, lightning trainer. The .nam export
happens automatically in the checkpoint callback.

Usage: python train.py <pairdir> <modeldir> [--epochs 40] [--ny 8192]

Expects pairdir/input.wav (dry) + pairdir/output.wav (wet) from align.py.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # headless — no tkinter GUI backend


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pairdir")
    p.add_argument("modeldir")
    p.add_argument("--epochs", type=int, default=40,
                   help="default 40 (quick). 100+ for a keeper model.")
    p.add_argument("--ny", type=int, default=2048,
                   help="output window length in samples — the dataset slices "
                        "non-overlapping windows, so smaller ny = more data "
                        "per epoch on short clips (default 2048)")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    pairdir, modeldir = Path(args.pairdir), Path(args.modeldir)
    modeldir.mkdir(parents=True, exist_ok=True)
    x_path, y_path = str(pairdir / "input.wav"), str(pairdir / "output.wav")

    import soundfile as sf
    import pytorch_lightning as pl
    from nam.train import core

    # Hold out the last ~15% of the clip for validation. The official configs
    # do the same thing — fixed sample offsets — just at sweep-specific spots.
    n = sf.info(x_path).frames
    val_start = int(n * 0.85)
    data_config = {
        "train": {"ny": args.ny, "stop_samples": val_start},
        # no silent point to split at — responses leak across the boundary a
        # bit (same caveat Proteus inputs have); ESR is still meaningful.
        "validation": {"ny": None, "start_samples": val_start,
                       "require_input_pre_silence": False},
        "common": {
            "x_path": x_path,
            "y_path": y_path,
            "delay": 0,  # align.py already removed the chain's latency
            "allow_unequal_lengths": True,
        },
        "joint": [
            {"name": "nam.data.normalize_joint_dataset_output",
             "kwargs": {"level_rms_dbfs": -18.0}}
        ],
    }
    model_config = core._get_packed_model_config()
    learning_config = {
        "train_dataloader": {"batch_size": args.batch_size, "shuffle": True,
                             "pin_memory": True, "drop_last": True,
                             "num_workers": 0},
        "val_dataloader": {},
        "trainer": {"max_epochs": args.epochs},
    }

    is_packed = model_config["net"]["name"] == "PackedWaveNet"
    lightning_cls = core._get_lightning_module_cls(model_config)
    model = lightning_cls.init_from_config(model_config)
    train_dl, val_dl = core._get_dataloaders(data_config, learning_config, model)
    sample_rate = train_dl.dataset.sample_rate
    model.net.sample_rate = sample_rate

    callbacks = core.get_callbacks(None, packed=is_packed)
    packed_best = next(
        (c for c in callbacks if isinstance(c, core._PackedBestCheckpoint)), None
    )
    trainer = pl.Trainer(
        callbacks=callbacks,
        default_root_dir=str(modeldir),
        **learning_config["trainer"],
    )
    print(f"training on {n / sample_rate:.1f}s pair "
          f"(train to {val_start / sample_rate:.1f}s, val after)")
    try:
        with core._filter_warnings("ignore", category=core._PossibleUserWarning):
            trainer.fit(model, train_dl, val_dl)
    except KeyboardInterrupt:
        print("\nTraining interrupted — exporting best checkpoint anyway.")
    finally:
        best = trainer.checkpoint_callback.best_model_path
        if best:
            model = lightning_cls.load_from_checkpoint(
                best, **lightning_cls.parse_config(model_config)
            )
        model.cpu()
        model.eval()
        model.net.sample_rate = sample_rate
        core._handshake_dataloaders(model, train_dl, val_dl)
        if is_packed and packed_best is not None:
            core._apply_packed_best_checkpoints(
                model, model_config, packed_best.checkpoint_paths
            )
        esr = core._plot(model, val_dl.dataset, silent=True,
                         filepath=str(modeldir / "model"))
        print(f"validation ESR: {esr:.4g}  (<0.02 ≈ indistinguishable)")
        for dl in (train_dl, val_dl):
            try:
                dl.dataset.teardown()
            except Exception:
                pass

    nams = sorted(modeldir.rglob("*.nam"), key=lambda p: p.stat().st_mtime, reverse=True)
    if nams:
        print("models:")
        for m in nams:
            print(f"  {m}")
    else:
        print(f"no .nam found under {modeldir} — check lightning_logs/checkpoints")


if __name__ == "__main__":
    main()
