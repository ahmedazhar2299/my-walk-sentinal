#!/usr/bin/env python3
"""Build trial-level laboratory feature datasets from Vicon and XSENS recordings.

The Vicon and XSENS exports are intentionally separate.  Each row represents
one detected task trial.  XSENS has an additional ``sensor`` column so that
the right shank, left shank, trunk, and sacrum estimates remain available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VISUALIZER_DIR = PROJECT_ROOT / "Sensors_Visualizers"
for path in (PROJECT_ROOT, VISUALIZER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from imu_features.config import load_config
from rebuild_xsens_results import SENSORS
from rebuild_xsens_trial_results import (
    ACTIVITY_SPECS,
    find_activity_dirs,
    natural_trial_number,
    normalize_summary,
    summarize_sensor,
    visit_id_from_folder,
)
from rebuild_vicon_annotations import build_annotations


ACTIVITIES = ("walk", "left_turn", "right_turn", "sit_to_stand")


def empty_task_fields() -> dict[str, float | str]:
    """Return a stable, wide schema with blank values for other activities."""
    fields: dict[str, float | str] = {}
    for activity in ACTIVITIES:
        fields[f"{activity}_trial_name"] = ""
        fields[f"{activity}_start_s"] = np.nan
        fields[f"{activity}_end_s"] = np.nan
        fields[f"{activity}_duration_s"] = np.nan
    fields.update(
        {
            "walk_step_count": np.nan,
            "walk_cadence_steps_min": np.nan,
            "walk_mean_step_time_s": np.nan,
            "left_turn_step_count": np.nan,
            "right_turn_step_count": np.nan,
        }
    )
    return fields


def feature_row(
    subject_id: str,
    visit_id: str,
    trial_id: int,
    activity: str,
    trial_name: str,
    values: dict,
    sensor: str | None = None,
) -> dict:
    row = {
        "subject_id": subject_id,
        "visit_id": visit_id,
        "activity": activity,
        "trial_id": trial_id,
    }
    if sensor is not None:
        row["sensor"] = sensor
    row.update(empty_task_fields())
    row[f"{activity}_trial_name"] = trial_name
    row[f"{activity}_start_s"] = values.get("start_time_s", np.nan)
    row[f"{activity}_end_s"] = values.get("end_time_s", np.nan)
    row[f"{activity}_duration_s"] = values.get("duration_s", np.nan)

    if activity == "walk":
        row["walk_step_count"] = values.get("step_count", np.nan)
        row["walk_cadence_steps_min"] = values.get("cadence_steps_min", np.nan)
        row["walk_mean_step_time_s"] = values.get("mean_step_time_s", np.nan)
    elif activity == "left_turn":
        row["left_turn_step_count"] = values.get("step_count", np.nan)
    elif activity == "right_turn":
        row["right_turn_step_count"] = values.get("step_count", np.nan)
    return row


def build_vicon_features(vicon_root: Path) -> pd.DataFrame:
    annotations = build_annotations(vicon_root)
    rows = []
    for _, trial in annotations.iterrows():
        rows.append(
            feature_row(
                subject_id=trial["subject_id"],
                visit_id=trial["visit_id"],
                trial_id=int(trial["trial_id"]),
                activity=trial["activity"],
                trial_name=trial["trial_name"],
                values=trial.to_dict(),
            )
        )
    return pd.DataFrame(rows)


def build_xsens_features(xsens_root: Path) -> pd.DataFrame:
    config = load_config()
    config.filtering.enabled = True
    rows = []
    subject_dirs = sorted(path for path in xsens_root.iterdir() if path.is_dir() and path.name.startswith("SRS"))
    for subject_dir in subject_dirs:
        visit_dirs = sorted(path for path in subject_dir.iterdir() if path.is_dir() and path.name.lower().startswith("visit"))
        for visit_dir in visit_dirs:
            for activity, spec in ACTIVITY_SPECS.items():
                for trial_dir in find_activity_dirs(visit_dir, spec["prefixes"]):
                    for sensor in SENSORS:
                        try:
                            summary, _ = summarize_sensor(trial_dir, sensor, activity, config)
                        except Exception as exc:
                            print(f"[WARN] {subject_dir.name} {visit_dir.name} {trial_dir.name} {sensor}: {exc}")
                            summary = None
                        rows.append(
                            feature_row(
                                subject_id=subject_dir.name,
                                visit_id=visit_id_from_folder(visit_dir.name),
                                trial_id=natural_trial_number(trial_dir),
                                activity=activity,
                                trial_name=trial_dir.name,
                                values=normalize_summary(summary, activity),
                                sensor=sensor,
                            )
                        )
    return pd.DataFrame(rows)


def ordered_columns(include_sensor: bool) -> list[str]:
    columns = ["subject_id", "visit_id", "activity", "trial_id"]
    if include_sensor:
        columns.append("sensor")
    for activity in ACTIVITIES:
        columns.extend(
            [
                f"{activity}_trial_name",
                f"{activity}_start_s",
                f"{activity}_end_s",
                f"{activity}_duration_s",
            ]
        )
        if activity == "walk":
            columns.extend(
                [
                    "walk_step_count",
                    "walk_cadence_steps_min",
                    "walk_mean_step_time_s",
                ]
            )
        elif activity in {"left_turn", "right_turn"}:
            columns.append(f"{activity}_step_count")
    return columns


def write_features(df: pd.DataFrame, output: Path, include_sensor: bool) -> None:
    columns = ordered_columns(include_sensor)
    output.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        df = pd.DataFrame(columns=columns)
    else:
        df = df.reindex(columns=columns).sort_values(
            ["subject_id", "visit_id", "activity", "trial_id"] + (["sensor"] if include_sensor else [])
        )
        numeric = df.select_dtypes(include="number").columns
        df.loc[:, numeric] = df[numeric].round(3)
    df.to_csv(output, index=False, float_format="%.3f")
    print(f"Wrote {output} ({len(df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Vicon and XSENS trial-level laboratory feature CSV files.")
    parser.add_argument("--vicon-root", type=Path, default=PROJECT_ROOT / "Data/lab/Vicon_CSV_Files")
    parser.add_argument("--xsens-root", type=Path, default=PROJECT_ROOT / "Data/lab/XSENS_synchronized")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "feature_dataset_lab")
    args = parser.parse_args()

    vicon = build_vicon_features(args.vicon_root)
    write_features(vicon, args.output_dir / "Vicon/vicon_features.csv", include_sensor=False)

    xsens = build_xsens_features(args.xsens_root)
    write_features(xsens, args.output_dir / "XSENS/xsens_features.csv", include_sensor=True)


if __name__ == "__main__":
    main()
