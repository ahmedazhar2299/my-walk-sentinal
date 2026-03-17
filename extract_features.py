#!/usr/bin/env python3

import argparse
from pathlib import Path

from imu_features import (
    extract_dataset_features,
    load_config,
    run_validation_checks,
)
from imu_features.utils import plot_debug_signal


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract one-row-per-patient-date IMU features from activity CSV files."
    )
    parser.add_argument("--dataset-root", default="Data", help="Root folder containing patient/date folders.")
    parser.add_argument("--output-csv", default="features_dataset.csv", help="Output feature CSV path.")
    parser.add_argument("--config-json", default=None, help="Optional JSON file with config overrides.")
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable Butterworth low-pass denoising.",
    )
    parser.add_argument(
        "--walk-distance-m",
        type=float,
        default=None,
        help="Distance in meters used for walking_speed = distance / walk_duration.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce console logging.")
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip post-extraction validation checks.",
    )
    parser.add_argument(
        "--debug-file",
        default=None,
        help="Optional: run a debug plot on one CSV file instead of full extraction.",
    )
    parser.add_argument(
        "--debug-signal",
        default="motion_acc",
        help="Signal for debug plotting (default: motion_acc).",
    )
    parser.add_argument(
        "--debug-peak-mode",
        choices=["steps", "transition"],
        default=None,
        help="Optional peak overlay mode for debug plot.",
    )
    parser.add_argument(
        "--debug-plot-out",
        default=None,
        help="Optional path to save debug figure (PNG). If omitted, shows plot interactively.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config_json)
    config.dataset_root = args.dataset_root
    config.output_csv = args.output_csv
    if args.no_filter:
        config.filtering.enabled = False
    if args.walk_distance_m is not None:
        config.walk_distance_m = args.walk_distance_m

    if args.debug_file:
        peaks = plot_debug_signal(
            csv_path=args.debug_file,
            config=config,
            signal_name=args.debug_signal,
            peak_mode=args.debug_peak_mode,
            output_path=args.debug_plot_out,
        )
        print(f"[INFO] Debug plot complete. Detected peaks: {len(peaks)}")
        return

    df = extract_dataset_features(
        dataset_root=Path(config.dataset_root),
        config=config,
        output_csv=config.output_csv,
        save_csv=True,
        verbose=not args.quiet,
    )

    if not args.skip_checks:
        checks = run_validation_checks(df)
        if checks:
            print("[VALIDATION] Potential issues:")
            for item in checks:
                print(f" - {item}")
        else:
            print("[VALIDATION] Basic checks passed.")


if __name__ == "__main__":
    main()
