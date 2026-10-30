#!/usr/bin/env python3
"""Create low-back IMU windows for LIMU-BERT-X and HART.

Inputs are the resampled public stroke CSV copies under Data/preprocessed.
The script writes NumPy arrays plus one metadata CSV per model. It does not
change the HART or LIMU model repositories or the source recordings.

LIMU-BERT-X: 20 Hz, non-overlapping windows of 20 rows x 6 axes.
HART: 30 Hz, 120-row windows of one axis at a time, 30-row hop (75% overlap).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")


def source_files(root: Path, rate: str) -> list[Path]:
    paths = []
    for cohort in ("Data_long", "Data_Stroke"):
        paths.extend(sorted((root / rate / cohort).glob("*_lowback.csv")))
    return paths


def metadata_for(path: Path, root: Path, window_start_s: float, **extra) -> dict:
    return {
        "subject_id": path.name.split("_", 1)[0],
        "cohort": path.parent.name,
        "recording_id": path.stem,
        "source_file": str(path.relative_to(root)),
        "window_start_s": window_start_s,
        **extra,
    }


def read_channels(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    missing = [name for name in CHANNELS if name not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing sensor columns {missing}")
    values = frame.loc[:, CHANNELS].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: sensor values contain NaN or infinity")
    return values


def make_limu_windows(root: Path, output_root: Path) -> tuple[int, int]:
    files = source_files(root, "20Hz")
    windows = []
    rows = []
    for path in files:
        values = read_channels(path)
        for start in range(0, len(values) - 19, 20):
            windows.append(values[start:start + 20])
            rows.append(metadata_for(path, root, start / 20.0))

    if not windows:
        raise RuntimeError("No LIMU windows were created")
    out = output_root / "LIMU-BERT-X"
    out.mkdir(parents=True, exist_ok=True)
    array = np.stack(windows).astype(np.float32, copy=False)
    np.save(out / "windows_20hz.npy", array)
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    return len(files), len(array)


def make_hart_windows(root: Path, output_root: Path) -> tuple[int, int]:
    files = source_files(root, "30Hz")
    windows = []
    rows = []
    window_length = 120
    hop = 30

    for path in files:
        values = read_channels(path)
        for channel_index, channel_name in enumerate(CHANNELS):
            series = values[:, channel_index]
            mean = float(np.mean(series))
            std = max(float(np.std(series)), 1e-5)
            # Match HART's original per-recording, per-channel standardisation.
            series = (series - mean) / std
            for start in range(0, len(series) - window_length + 1, hop):
                windows.append(series[start:start + window_length, None])
                rows.append(metadata_for(
                    path,
                    root,
                    start / 30.0,
                    channel=channel_name,
                    normalization_mean=mean,
                    normalization_std=std,
                ))

    if not windows:
        raise RuntimeError("No HART windows were created")
    out = output_root / "HART"
    out.mkdir(parents=True, exist_ok=True)
    array = np.stack(windows).astype(np.float32, copy=False)
    np.save(out / "windows_30hz_single_channel.npy", array)
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    return len(files), len(array)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resampled-root",
        type=Path,
        default=Path("Data/preprocessed/public_stroke"),
        help="Root containing 20Hz/ and 30Hz/ resampled files",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("Data/windowed/public_stroke"),
        help="Destination for model arrays and metadata manifests",
    )
    args = parser.parse_args()

    limu_files, limu_count = make_limu_windows(args.resampled_root, args.output_root)
    hart_files, hart_count = make_hart_windows(args.resampled_root, args.output_root)
    print(f"LIMU-BERT-X: {limu_count:,} windows from {limu_files} low-back recordings; shape (N, 20, 6)")
    print(f"HART: {hart_count:,} single-channel windows from {hart_files} low-back recordings; shape (N, 120, 1)")
    print(f"Saved arrays and manifests under {args.output_root}")


if __name__ == "__main__":
    main()
