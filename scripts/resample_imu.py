#!/usr/bin/env python3
"""Create model-rate CSV copies of IMU recordings.

Examples for the public stroke CSVs (recorded at about 104 Hz):

    python scripts/resample_imu.py input.csv output_limu.csv \
        --input-fs 104 --target-fs 20
    python scripts/resample_imu.py input.csv output_hart.csv \
        --input-fs 104 --target-fs 30
    python scripts/resample_imu.py Data_long resampled/20hz/Data_long \
        --input-fs 104 --target-fs 20

The input is never modified. The output contains only the selected signal
channels and a new time_s column starting at zero.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly


DEFAULT_CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")


class InsufficientSamplesError(ValueError):
    """Raised when a source file cannot form a time series."""


def resample_csv(
    input_path: Path,
    output_path: Path,
    input_fs: float,
    target_fs: int,
    channels: tuple[str, ...],
) -> int:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different.")
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")

    frame = pd.read_csv(input_path)
    missing = [channel for channel in channels if channel not in frame.columns]
    if missing:
        raise ValueError(f"Missing channel columns in {input_path}: {missing}")

    signal = frame.loc[:, channels].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    if signal.shape[0] < 2:
        raise InsufficientSamplesError("fewer than two sensor samples")
    if not np.isfinite(signal).all():
        raise ValueError("IMU channels contain missing or non-finite values; clean them first.")

    # Polyphase resampling applies a low-pass FIR filter before changing the rate.
    ratio = Fraction(target_fs / input_fs).limit_denominator(10000)
    resampled = resample_poly(signal, up=ratio.numerator, down=ratio.denominator, axis=0)
    output = pd.DataFrame(resampled, columns=channels)
    output.insert(0, "time_s", np.arange(len(output), dtype=np.float64) / target_fs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return len(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path, help="Source CSV or folder of CSVs; left unchanged")
    parser.add_argument("output_path", type=Path, help="New CSV path or destination folder")
    parser.add_argument("--input-fs", type=float, required=True, help="Source sampling rate in Hz")
    parser.add_argument("--target-fs", type=int, choices=(20, 30), required=True,
                        help="20 Hz for LIMU-BERT-X or 30 Hz for HART")
    parser.add_argument("--channels", default=",".join(DEFAULT_CHANNELS),
                        help="Comma-separated signal columns in time order")
    parser.add_argument("--resume", action="store_true",
                        help="In folder mode, leave existing outputs and continue")
    parser.add_argument("--skip-empty", action="store_true",
                        help="In folder mode, skip and report files with fewer than two samples")
    args = parser.parse_args()

    channels = tuple(name.strip() for name in args.channels.split(",") if name.strip())
    if not channels:
        parser.error("--channels must name at least one column")

    if args.input_fs <= 0:
        parser.error("--input-fs must be greater than zero")

    if args.input_path.is_file():
        count = resample_csv(args.input_path, args.output_path, args.input_fs,
                             args.target_fs, channels)
        print(f"Saved {count} samples at {args.target_fs} Hz: {args.output_path}")
        return

    if not args.input_path.is_dir():
        parser.error(f"Input path does not exist: {args.input_path}")

    csv_paths = sorted(args.input_path.rglob("*.csv"))
    if not csv_paths:
        parser.error(f"No CSV files found under {args.input_path}")

    total_samples = 0
    completed = 0
    skipped = []
    for source in csv_paths:
        relative = source.relative_to(args.input_path)
        destination = args.output_path / relative
        if args.resume and destination.exists():
            completed += 1
            continue
        try:
            count = resample_csv(source, destination, args.input_fs,
                                 args.target_fs, channels)
        except InsufficientSamplesError as exc:
            if not args.skip_empty:
                raise
            skipped.append((source, str(exc)))
            continue
        total_samples += count
        completed += 1

    print(f"Processed or resumed {completed} files at {args.target_fs} Hz; "
          f"{total_samples:,} newly written samples.")
    if skipped:
        print(f"Skipped {len(skipped)} empty files:")
        for path, reason in skipped:
            print(f"  {path}: {reason}")


if __name__ == "__main__":
    main()
