#!/usr/bin/env python3
"""Prepare six-channel public-stroke windows for HART adaptation.

This follows HART/TSPulse's multivariate input format: one 120 x 6 window per
time interval, with each axis standardized per recording before windowing.
The existing participant-level split is reused unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESAMPLED_ROOT = ROOT / "Data/preprocessed/public_stroke/30Hz"
SPLIT_FILE = ROOT / "Data/windowed/public_stroke/participant_split.csv"
WINDOW_ROOT = ROOT / "Data/windowed/public_stroke/HART"
INPUT_ROOT = ROOT / "Data/model_inputs/public_stroke/HART"
CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")
WINDOW_LENGTH = 120
HOP_LENGTH = 30


def main() -> None:
    split_frame = pd.read_csv(SPLIT_FILE)
    split_lookup = {
        (row.subject_id, row.cohort): (row.canonical_subject_id, row.split)
        for row in split_frame.itertuples(index=False)
    }
    files = [
        path
        for cohort in ("Data_long", "Data_Stroke")
        for path in sorted((RESAMPLED_ROOT / cohort).glob("*_lowback.csv"))
    ]
    if not files:
        raise RuntimeError(f"No 30 Hz low-back recordings found under {RESAMPLED_ROOT}")

    windows: list[np.ndarray] = []
    manifest_rows: list[dict[str, object]] = []
    for path in files:
        frame = pd.read_csv(path)
        missing = [column for column in CHANNELS if column not in frame]
        if missing:
            raise ValueError(f"{path}: missing channels {missing}")
        values = frame.loc[:, CHANNELS].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: non-finite IMU values")

        subject_id = path.name.split("_", 1)[0]
        cohort = path.parent.name
        canonical_id, split = split_lookup[(subject_id, cohort)]
        means = values.mean(axis=0)
        stds = np.maximum(values.std(axis=0), np.float32(1e-5))
        standardized = (values - means) / stds
        recording_id = path.stem
        for start in range(0, len(standardized) - WINDOW_LENGTH + 1, HOP_LENGTH):
            windows.append(standardized[start : start + WINDOW_LENGTH].copy())
            manifest_rows.append({
                "subject_id": subject_id,
                "canonical_subject_id": canonical_id,
                "cohort": cohort,
                "split": split,
                "recording_id": recording_id,
                "source_file": str(path.relative_to(RESAMPLED_ROOT)),
                "window_start_s": start / 30.0,
                "channel_order": ",".join(CHANNELS),
            })

    array = np.stack(windows).astype(np.float32, copy=False)
    manifest = pd.DataFrame(manifest_rows)
    if len(array) != len(manifest):
        raise RuntimeError("HART joint window array and manifest row counts differ")
    WINDOW_ROOT.mkdir(parents=True, exist_ok=True)
    np.save(WINDOW_ROOT / "windows_30hz_6channel.npy", array)
    manifest.to_csv(WINDOW_ROOT / "manifest.csv", index=False)

    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        selected = manifest["split"].eq(split).to_numpy()
        split_array = array[selected]
        if not np.isfinite(split_array).all():
            raise RuntimeError(f"Non-finite values in HART joint {split} windows")
        np.save(INPUT_ROOT / f"{split}.npy", split_array)
        manifest.loc[selected].to_csv(INPUT_ROOT / f"{split}_manifest.csv", index=False)

    spec = {
        "sampling_rate_hz": 30,
        "window_shape": [120, 6],
        "window_seconds": 4,
        "hop_samples": 30,
        "hop_seconds": 1,
        "channel_order": list(CHANNELS),
        "normalization": "per-recording, per-axis z-score",
        "training_objective": "self-supervised HART/TSPulse masked reconstruction",
        "participant_split": "existing 80% train / 20% validation split; no test set",
        "joint_multichannel_input": True,
    }
    (INPUT_ROOT / "input_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    counts = manifest.groupby("split").size().to_dict()
    print(
        f"HART joint: {len(array):,} windows from {len(files)} low-back recordings; "
        f"shape {array.shape}; split counts {counts}"
    )


if __name__ == "__main__":
    main()
