#!/usr/bin/env python3
"""Build wide, visit-level laboratory feature datasets from Vicon and XSENS.

Each row combines the first, second, or third available trial from every task
for one participant visit. XSENS values remain separate by sensor within the
same row, while the Vicon reference export contains one shared set of task
features.
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
    normalize_summary,
    summarize_sensor,
    visit_id_from_folder,
)
from rebuild_vicon_annotations import build_annotations


ACTIVITIES = ("walk", "left_turn", "right_turn", "sit_to_stand")
SENSOR_LABELS = {
    "right": "right_shank",
    "left": "left_shank",
    "trunk": "trunk",
    "sacrum": "sacrum",
}


def metric_names(activity: str) -> tuple[str, ...]:
    metrics = ("start_s", "end_s", "duration_s")
    if activity == "walk":
        return metrics + ("step_count", "cadence_steps_min", "mean_step_time_s")
    if activity in {"left_turn", "right_turn"}:
        return metrics + ("step_count",)
    return metrics


def empty_row(subject_id: str, visit_id: str, trial_id: int, include_sensors: bool) -> dict:
    row: dict[str, float | str | int] = {
        "subject_id": subject_id,
        "visit_id": visit_id,
        "trial_id": trial_id,
    }
    for activity in ACTIVITIES:
        row[f"{activity}_trial_name"] = ""
        for metric in metric_names(activity):
            if include_sensors:
                for sensor in SENSORS:
                    row[f"{SENSOR_LABELS[sensor]}_{activity}_{metric}"] = np.nan
            else:
                row[f"{activity}_{metric}"] = np.nan
    return row


def add_task_values(row: dict, activity: str, trial_name: str, values: dict, sensor: str | None = None) -> None:
    row[f"{activity}_trial_name"] = trial_name
    prefix = f"{SENSOR_LABELS[sensor]}_" if sensor else ""
    value_map = {
        "start_s": values.get("start_time_s", np.nan),
        "end_s": values.get("end_time_s", np.nan),
        "duration_s": values.get("duration_s", np.nan),
        "step_count": values.get("step_count", np.nan),
        "cadence_steps_min": values.get("cadence_steps_min", np.nan),
        "mean_step_time_s": values.get("mean_step_time_s", np.nan),
    }
    for metric in metric_names(activity):
        row[f"{prefix}{activity}_{metric}"] = value_map[metric]


def wide_vicon_features(vicon_root: Path) -> pd.DataFrame:
    annotations = build_annotations(vicon_root)
    if annotations.empty:
        return pd.DataFrame()

    annotations = annotations.sort_values(["subject_id", "visit_id", "activity", "trial_id", "trial_name"]).copy()
    annotations["visit_trial_id"] = annotations.groupby(["subject_id", "visit_id", "activity"]).cumcount() + 1
    rows: dict[tuple[str, str, int], dict] = {}
    for _, trial in annotations.iterrows():
        key = (trial["subject_id"], trial["visit_id"], int(trial["visit_trial_id"]))
        row = rows.setdefault(key, empty_row(*key, include_sensors=False))
        add_task_values(row, trial["activity"], trial["trial_name"], trial.to_dict())
    return pd.DataFrame(rows.values())


def wide_xsens_features(xsens_root: Path) -> pd.DataFrame:
    config = load_config()
    config.filtering.enabled = True
    rows: dict[tuple[str, str, int], dict] = {}
    subject_dirs = sorted(path for path in xsens_root.iterdir() if path.is_dir() and path.name.startswith("SRS"))
    for subject_dir in subject_dirs:
        visit_dirs = sorted(path for path in subject_dir.iterdir() if path.is_dir() and path.name.lower().startswith("visit"))
        for visit_dir in visit_dirs:
            visit_id = visit_id_from_folder(visit_dir.name)
            for activity, spec in ACTIVITY_SPECS.items():
                for trial_id, trial_dir in enumerate(find_activity_dirs(visit_dir, spec["prefixes"]), start=1):
                    key = (subject_dir.name, visit_id, trial_id)
                    row = rows.setdefault(key, empty_row(*key, include_sensors=True))
                    for sensor in SENSORS:
                        try:
                            summary, _ = summarize_sensor(trial_dir, sensor, activity, config)
                        except Exception as exc:
                            print(f"[WARN] {subject_dir.name} {visit_dir.name} {trial_dir.name} {sensor}: {exc}")
                            summary = None
                        add_task_values(
                            row,
                            activity,
                            trial_dir.name,
                            normalize_summary(summary, activity),
                            sensor=sensor,
                        )
    return pd.DataFrame(rows.values())


def ordered_columns(include_sensors: bool) -> list[str]:
    columns = ["subject_id", "visit_id", "trial_id"]
    for activity in ACTIVITIES:
        columns.append(f"{activity}_trial_name")
        for metric in metric_names(activity):
            if include_sensors:
                columns.extend(f"{SENSOR_LABELS[sensor]}_{activity}_{metric}" for sensor in SENSORS)
            else:
                columns.append(f"{activity}_{metric}")
    return columns


def write_features(df: pd.DataFrame, output: Path, include_sensors: bool) -> None:
    columns = ordered_columns(include_sensors)
    output.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        df = pd.DataFrame(columns=columns)
    else:
        df = df.reindex(columns=columns).sort_values(["subject_id", "visit_id", "trial_id"])
        numeric = df.select_dtypes(include="number").columns
        df.loc[:, numeric] = df[numeric].round(3)
    df.to_csv(output, index=False, float_format="%.3f")
    print(f"Wrote {output} ({len(df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build visit-level Vicon and XSENS laboratory feature CSV files.")
    parser.add_argument("--vicon-root", type=Path, default=PROJECT_ROOT / "Data/lab/Vicon_CSV_Files")
    parser.add_argument("--xsens-root", type=Path, default=PROJECT_ROOT / "Data/lab/XSENS_synchronized")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "feature_dataset_lab")
    args = parser.parse_args()

    write_features(
        wide_vicon_features(args.vicon_root),
        args.output_dir / "Vicon/vicon_features.csv",
        include_sensors=False,
    )
    write_features(
        wide_xsens_features(args.xsens_root),
        args.output_dir / "XSENS/xsens_features.csv",
        include_sensors=True,
    )


if __name__ == "__main__":
    main()
