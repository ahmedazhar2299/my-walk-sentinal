from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from imu_features.config import load_config

from rebuild_xsens_results import (
    SENSORS,
    add_walking_sacrum_qc,
    load_sensor,
    sts_summary,
    turn_summary,
    walking_summary,
)


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = PROJECT_ROOT / "Data_Sensors" / "XSENS_synchronized"
DEFAULT_OUTPUT_DIR = HERE / "XSENS_results"

ACTIVITY_SPECS = {
    "walk": {
        "prefixes": ("NW",),
        "output": "walking_results.csv",
    },
    "left_turn": {
        "prefixes": ("TurnL", "TURNL"),
        "output": "left_turn_results.csv",
    },
    "right_turn": {
        "prefixes": ("TurnR", "TURNR"),
        "output": "right_turn_results.csv",
    },
    "sit_to_stand": {
        "prefixes": ("STS",),
        "output": "sit_to_stand_results.csv",
    },
}

GENERIC_FEATURES = (
    "start_time_s",
    "end_time_s",
    "duration_s",
    "step_count",
    "cadence_steps_min",
    "mean_step_time_s",
    "ten_step_start_time_s",
    "ten_step_end_time_s",
    "ten_step_duration_s",
    "ten_step_count",
    "ten_step_cadence_steps_min",
    "ten_step_mean_step_time_s",
    "peak_angular_velocity",
    "mean_angular_velocity",
    "flexion_peak",
    "extension_peak",
    "time_to_peak_acc_s",
    "peak_acc",
    "peak_gyro",
)


def natural_trial_number(path: Path) -> int:
    match = re.search(r"(\d+)", path.name)
    return int(match.group(1)) if match else 9999


def visit_id_from_folder(name: str) -> str:
    digits = "".join(ch for ch in str(name) if ch.isdigit())
    return f"V{int(digits):02d}" if digits else str(name)


def find_activity_dirs(visit_dir: Path, prefixes: tuple[str, ...]) -> list[Path]:
    if not visit_dir.exists():
        return []
    prefixes = tuple(prefix.lower() for prefix in prefixes)
    dirs = [
        child
        for child in visit_dir.iterdir()
        if child.is_dir() and any(child.name.lower().startswith(prefix) for prefix in prefixes)
    ]
    return sorted(dirs, key=lambda p: (natural_trial_number(p), p.name.lower()))


def normalize_summary(summary: dict | None, activity: str) -> dict:
    out = {feature: np.nan for feature in GENERIC_FEATURES}
    if not summary:
        return out

    if activity == "walk":
        out.update(
            {
                "start_time_s": summary.get("ten_step_start", np.nan),
                "end_time_s": summary.get("ten_step_end", np.nan),
                "duration_s": summary.get("ten_step_duration", np.nan),
                "step_count": summary.get("ten_step_count", np.nan),
                "cadence_steps_min": summary.get("ten_step_cadence", np.nan),
                "mean_step_time_s": summary.get("ten_step_mean_step_time", np.nan),
                "ten_step_start_time_s": summary.get("ten_step_start", np.nan),
                "ten_step_end_time_s": summary.get("ten_step_end", np.nan),
                "ten_step_duration_s": summary.get("ten_step_duration", np.nan),
                "ten_step_count": summary.get("ten_step_count", np.nan),
                "ten_step_cadence_steps_min": summary.get("ten_step_cadence", np.nan),
                "ten_step_mean_step_time_s": summary.get("ten_step_mean_step_time", np.nan),
            }
        )
    elif activity in {"left_turn", "right_turn"}:
        out.update(
            {
                "start_time_s": summary.get("start", np.nan),
                "end_time_s": summary.get("end", np.nan),
                "duration_s": summary.get("turn_duration", np.nan),
                "step_count": summary.get("turn_step_count", np.nan),
                "peak_angular_velocity": summary.get("peak_angular_velocity", np.nan),
                "mean_angular_velocity": summary.get("mean_angular_velocity", np.nan),
            }
        )
    else:
        out.update(
            {
                "start_time_s": summary.get("start", np.nan),
                "end_time_s": summary.get("end", np.nan),
                "duration_s": summary.get("duration", np.nan),
                "flexion_peak": summary.get("flexion_peak", np.nan),
                "extension_peak": summary.get("extension", np.nan),
                "time_to_peak_acc_s": summary.get("time_to_peak_acc", np.nan),
                "peak_acc": summary.get("peak_acc", np.nan),
                "peak_gyro": summary.get("peak_gyro", np.nan),
            }
        )
    return out


def summarize_sensor(activity_dir: Path, sensor: str, activity: str, config):
    loaded = load_sensor(activity_dir, sensor, config)
    if loaded is None:
        return None, ""
    df, meta = loaded
    if activity == "walk":
        return walking_summary(df, meta, sensor), ""
    if activity == "left_turn":
        return turn_summary(df, meta, config, "left_turn"), ""
    if activity == "right_turn":
        return turn_summary(df, meta, config, "right_turn"), ""
    return sts_summary(df, meta, config), ""


def build_activity_rows(root: Path, activity: str, config) -> pd.DataFrame:
    rows = []
    spec = ACTIVITY_SPECS[activity]
    subject_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("SRS")])
    for subject_dir in subject_dirs:
        visit_dirs = sorted([p for p in subject_dir.iterdir() if p.is_dir() and p.name.lower().startswith("visit")])
        for visit_dir in visit_dirs:
            activity_dirs = find_activity_dirs(visit_dir, spec["prefixes"])
            for activity_dir in activity_dirs:
                trial_id = natural_trial_number(activity_dir)
                row = {
                    "subject_id": subject_dir.name,
                    "visit_id": visit_id_from_folder(visit_dir.name),
                    "activity": activity,
                    "trial_id": trial_id,
                    "trial_name": activity_dir.name,
                }
                for sensor in SENSORS:
                    try:
                        summary, error = summarize_sensor(activity_dir, sensor, activity, config)
                    except Exception as exc:
                        summary, error = None, str(exc)
                    values = normalize_summary(summary, activity)
                    for key, value in values.items():
                        row[f"{sensor}_{key}"] = value

                if activity == "walk":
                    qc_row = {
                        "sacrum_step_count": row.get("sacrum_step_count", np.nan),
                        "trunk_step_count": row.get("trunk_step_count", np.nan),
                        "sacrum_duration": row.get("sacrum_duration_s", np.nan),
                        "trunk_duration": row.get("trunk_duration_s", np.nan),
                    }
                    add_walking_sacrum_qc(qc_row)
                    row["sacrum_qc_step_count"] = qc_row.get("sacrum_qc_step_count", np.nan)
                    row["sacrum_qc_cadence"] = qc_row.get("sacrum_qc_cadence", np.nan)
                    row["sacrum_qc_mean_step_time"] = qc_row.get("sacrum_qc_mean_step_time", np.nan)
                    row["sacrum_qc_source"] = qc_row.get("sacrum_qc_source", "")

                rows.append(row)
    return pd.DataFrame(rows)


def ordered_columns(df: pd.DataFrame) -> list[str]:
    front = ["subject_id", "visit_id", "activity", "trial_id", "trial_name"]
    sensor_cols = []
    for sensor in SENSORS:
        sensor_cols.extend([f"{sensor}_{feature}" for feature in GENERIC_FEATURES if f"{sensor}_{feature}" in df])
    qc_cols = [col for col in ("sacrum_qc_step_count", "sacrum_qc_cadence", "sacrum_qc_mean_step_time", "sacrum_qc_source") if col in df]
    tail = []
    rest = [col for col in df.columns if col not in front + sensor_cols + qc_cols + tail]
    return [col for col in front + sensor_cols + qc_cols + rest + tail if col in df]


def write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        df = df[ordered_columns(df)].copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].round(3)
    df.to_csv(path, index=False, float_format="%.3f")
    print(f"Wrote {path} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Rebuild trial-level XSENS results from Data_Sensors/XSENS_synchronized.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    config = load_config()
    config.filtering.enabled = True

    all_rows = []
    for activity, spec in ACTIVITY_SPECS.items():
        df = build_activity_rows(args.root, activity, config)
        write_csv(df, args.output_dir / spec["output"])
        all_rows.append(df)
    combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    write_csv(combined, args.output_dir / "xsens_results_long.csv")


if __name__ == "__main__":
    main()
