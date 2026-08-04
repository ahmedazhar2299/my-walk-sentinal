from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from vicon_processing import activity_from_trial_name, summarize_vicon_trial, trial_sort_key


HERE = Path(__file__).resolve().parent
DEFAULT_VICON_ROOT = HERE.parent / "Data_Sensors" / "Vicon_CSV_Files"
DEFAULT_OUTPUT = HERE / "Vicon - Annotations_long.csv"


def visit_label_from_folder(name):
    digits = "".join(ch for ch in str(name) if ch.isdigit())
    return f"V{int(digits):02d}" if digits else str(name)


def natural_trial_number(name):
    match = re.search(r"(\d+)", str(name))
    return int(match.group(1)) if match else 9999


def find_target_trials(vicon_root):
    vicon_root = Path(vicon_root)
    rows = []
    for csv_path in sorted(vicon_root.rglob("*.csv")):
        trial_folder = csv_path.parent.name
        if trial_folder.lower().endswith("_x"):
            continue
        activity = activity_from_trial_name(trial_folder)
        if activity is None:
            continue
        try:
            visit_dir = csv_path.parents[1]
            subject_dir = csv_path.parents[2]
        except IndexError:
            continue
        if not subject_dir.name.startswith("SRS"):
            continue
        if not visit_dir.name.lower().startswith("visit"):
            continue
        rows.append(
            {
                "subject_id": subject_dir.name,
                "visit_id": visit_label_from_folder(visit_dir.name),
                "visit_folder": visit_dir.name,
                "activity": activity,
                "trial_name": trial_folder,
                "csv_path": csv_path,
            }
        )
    return pd.DataFrame(rows)


def build_annotations(vicon_root=DEFAULT_VICON_ROOT):
    trials = find_target_trials(vicon_root)
    if trials.empty:
        return pd.DataFrame()

    trials = trials.sort_values(["subject_id", "visit_id", "activity", "csv_path"])
    trials = trials.copy()
    trials["trial_id"] = trials["trial_name"].map(natural_trial_number)

    rows = []
    for _, trial in trials.iterrows():
        row = {
            "subject_id": trial["subject_id"],
            "visit_id": trial["visit_id"],
            "activity": trial["activity"],
            "trial_id": int(trial["trial_id"]),
            "trial_name": trial["trial_name"],
        }
        try:
            summary = summarize_vicon_trial(trial["csv_path"])
            row.update(summary)
        except Exception as exc:
            print(
                f"[WARN] {trial['subject_id']} {trial['visit_id']} "
                f"{trial['activity']} {trial['trial_name']}: {exc}"
            )
            row.update(
                {
                    "start_time_s": np.nan,
                    "end_time_s": np.nan,
                    "duration_s": np.nan,
                    "step_count": np.nan,
                    "cadence_steps_min": np.nan,
                    "mean_step_time_s": np.nan,
                    "time_to_peak_s": np.nan,
                }
            )
        rows.append(row)

    columns = [
        "subject_id",
        "visit_id",
        "activity",
        "trial_id",
        "trial_name",
        "start_time_s",
        "end_time_s",
        "duration_s",
        "step_count",
        "cadence_steps_min",
        "mean_step_time_s",
        "time_to_peak_s",
    ]
    return pd.DataFrame(rows)[columns]


def main():
    parser = argparse.ArgumentParser(description="Rebuild long-format Vicon annotations from raw marker CSV trials.")
    parser.add_argument("--vicon-root", type=Path, default=DEFAULT_VICON_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    annotations = build_annotations(args.vicon_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    annotations.to_csv(args.output, index=False)

    print(f"Wrote {args.output}")
    print(f"Rows: {len(annotations)}")
    if not annotations.empty:
        print("\nRows by activity")
        print(annotations["activity"].value_counts().to_string())


if __name__ == "__main__":
    main()
