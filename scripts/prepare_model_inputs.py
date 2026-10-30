#!/usr/bin/env python3
"""Write train/validation arrays from the subject-split IMU windows.

LIMU inputs are converted from the public archive's g-valued acceleration to
m/s^2 so LIMU's unchanged preprocessing step (divide acceleration by 9.8)
returns g-scale values like its pretrained pipeline. HART inputs remain
separate, per-axis windows already z-scored per recording and axis.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


WINDOW_ROOT = Path("Data/windowed/public_stroke")
OUTPUT_ROOT = Path("Data/model_inputs/public_stroke")


def prepare_model(model: str, source_name: str, input_spec: dict) -> None:
    model_root = WINDOW_ROOT / model
    windows = np.load(model_root / source_name, mmap_mode="r")
    manifest = pd.read_csv(model_root / "manifest.csv")
    if len(windows) != len(manifest):
        raise ValueError(f"{model}: window array and manifest row counts differ")
    if manifest["split"].isna().any():
        raise ValueError(f"{model}: manifest contains unassigned windows")

    manifest = manifest.copy()
    manifest.insert(0, "source_window_index", np.arange(len(manifest)))
    out = OUTPUT_ROOT / model
    out.mkdir(parents=True, exist_ok=True)

    for split in ("train", "validation"):
        mask = manifest["split"].eq(split).to_numpy()
        values = np.asarray(windows[mask], dtype=np.float32).copy()
        if model == "LIMU-BERT-X":
            # Public archive accelerometer values are in g. Convert to m/s^2
            # because LIMU's original preprocessing divides these axes by 9.8.
            values[:, :, :3] *= np.float32(9.8)
        if not np.isfinite(values).all():
            raise ValueError(f"{model} {split}: non-finite model inputs found")
        np.save(out / f"{split}.npy", values)
        manifest.loc[mask].to_csv(out / f"{split}_manifest.csv", index=False)

    (out / "input_spec.json").write_text(json.dumps(input_spec, indent=2) + "\n")


def main() -> None:
    prepare_model(
        "LIMU-BERT-X",
        "windows_20hz.npy",
        {
            "sampling_rate_hz": 20,
            "window_shape": [20, 6],
            "channel_order": ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"],
            "stored_acceleration_units": "m/s^2; LIMU preprocessing divides these channels by 9.8",
            "stored_gyroscope_units": "source units, unchanged",
            "training_objective": "self-supervised masked reconstruction",
            "participant_split": "80% train / 20% validation; no test set",
        },
    )
    prepare_model(
        "HART",
        "windows_30hz_single_channel.npy",
        {
            "sampling_rate_hz": 30,
            "window_shape": [120, 1],
            "window_seconds": 4,
            "input_mode": "one sensor axis per window; axis name is in manifest channel column",
            "normalization": "per-recording, per-axis z-score",
            "training_objective": "self-supervised masked time/frequency reconstruction",
            "participant_split": "80% train / 20% validation; no test set",
        },
    )
    print(f"Prepared train/validation inputs under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
