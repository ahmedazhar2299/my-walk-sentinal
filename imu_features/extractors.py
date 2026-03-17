from pathlib import Path

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .transition_features import (
    extract_transition_features,
    transition_feature_names,
)
from .turn_features import extract_turn_features, turn_feature_names
from .utils import (
    ACTIVITY_ORDER,
    build_nan_feature_dict,
    read_and_preprocess_csv,
    resolve_activity_files,
)
from .walk_features import WALK_FEATURE_NAMES, extract_walk_features

ASYMMETRY_FEATURES = [
    "turn_duration_difference",
    "turn_velocity_difference",
    "turn_pause_difference",
    "abs_turn_duration_difference",
    "abs_turn_velocity_difference",
    "abs_turn_pause_difference",
]


def _safe_diff(a, b):
    if np.isfinite(a) and np.isfinite(b):
        return float(a - b)
    return np.nan


def _turn_asymmetry(row):
    duration_diff = _safe_diff(
        row.get("left_turn_duration", np.nan),
        row.get("right_turn_duration", np.nan),
    )
    velocity_diff = _safe_diff(
        row.get("left_turn_mean_angular_velocity", np.nan),
        row.get("right_turn_mean_angular_velocity", np.nan),
    )
    pause_diff = _safe_diff(
        row.get("left_turn_pause_time", np.nan),
        row.get("right_turn_pause_time", np.nan),
    )
    return {
        "turn_duration_difference": duration_diff,
        "turn_velocity_difference": velocity_diff,
        "turn_pause_difference": pause_diff,
        "abs_turn_duration_difference": abs(duration_diff) if np.isfinite(duration_diff) else np.nan,
        "abs_turn_velocity_difference": abs(velocity_diff) if np.isfinite(velocity_diff) else np.nan,
        "abs_turn_pause_difference": abs(pause_diff) if np.isfinite(pause_diff) else np.nan,
    }


def _empty_features_for_activity(activity):
    if activity == "walk":
        return build_nan_feature_dict(WALK_FEATURE_NAMES)
    if activity == "left_turn":
        return build_nan_feature_dict(turn_feature_names("left_turn"))
    if activity == "right_turn":
        return build_nan_feature_dict(turn_feature_names("right_turn"))
    if activity == "sit_to_stand":
        return build_nan_feature_dict(transition_feature_names("sit_to_stand"))
    if activity == "stand_to_sit":
        return build_nan_feature_dict(transition_feature_names("stand_to_sit"))
    return {}


def _extract_activity_features(activity, csv_path, config, verbose):
    if csv_path is None:
        return _empty_features_for_activity(activity)

    try:
        df, meta = read_and_preprocess_csv(csv_path, config)
    except Exception as exc:
        if verbose:
            print(f"[WARN] Failed to process {csv_path}: {exc}")
        return _empty_features_for_activity(activity)

    if activity == "walk":
        return extract_walk_features(df, meta, config)
    if activity == "left_turn":
        return extract_turn_features(df, meta, config, prefix="left_turn")
    if activity == "right_turn":
        return extract_turn_features(df, meta, config, prefix="right_turn")
    if activity == "sit_to_stand":
        return extract_transition_features(df, meta, config, prefix="sit_to_stand")
    if activity == "stand_to_sit":
        return extract_transition_features(df, meta, config, prefix="stand_to_sit")
    return {}


def _iter_patient_date_dirs(dataset_root):
    folders = []
    for patient_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
        date_dirs = sorted([d for d in patient_dir.iterdir() if d.is_dir()])
        for date_dir in date_dirs:
            folders.append((patient_dir.name, date_dir.name, date_dir))
    return folders


def run_validation_checks(df):
    """Run lightweight validation checks on the final feature table."""
    checks = []
    if df.empty:
        checks.append("Dataset is empty.")
        return checks

    if {"patient_id", "date"}.issubset(df.columns):
        duplicated = df.duplicated(subset=["patient_id", "date"]).sum()
        if duplicated:
            checks.append(f"Found {duplicated} duplicate patient/date rows.")

    feature_cols = [c for c in df.columns if c not in {"patient_id", "date"}]
    all_nan_rows = int(df[feature_cols].isna().all(axis=1).sum()) if feature_cols else 0
    if all_nan_rows:
        checks.append(f"{all_nan_rows} rows have all features as NaN.")

    for col in ["walk_duration", "left_turn_duration", "right_turn_duration"]:
        if col in df.columns:
            missing_pct = 100.0 * float(df[col].isna().mean())
            if missing_pct > 20.0:
                checks.append(f"{col} missing in {missing_pct:.1f}% of rows.")
    return checks


def extract_dataset_features(
    dataset_root,
    config,
    output_csv=None,
    save_csv=True,
    verbose=True,
):
    """
    Extract one feature row per patient/date from a patient/date/activity folder tree.
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    folders = _iter_patient_date_dirs(root)
    if verbose:
        print(f"[INFO] Found {len(folders)} patient-date folders in {root}")

    rows = []
    for patient_id, date_str, date_dir in folders:
        activity_files = resolve_activity_files(date_dir)

        row = {
            "patient_id": patient_id,
            "date": date_str,
        }
        for activity in ACTIVITY_ORDER:
            features = _extract_activity_features(
                activity=activity,
                csv_path=activity_files.get(activity),
                config=config,
                verbose=verbose,
            )
            row.update(features)

        row.update(_turn_asymmetry(row))
        rows.append(row)

        if verbose:
            found_count = sum(path is not None for path in activity_files.values())
            print(f"[INFO] {patient_id}/{date_str}: found {found_count}/5 activity files")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["patient_id", "date"]).reset_index(drop=True)

    output_path = Path(output_csv or config.output_csv)
    if save_csv:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        if verbose:
            print(f"[INFO] Saved feature dataset to {output_path} ({len(df)} rows)")

    return df


def merge_with_labels(features_df, labels_df, keys=("patient_id", "date")):
    """
    Left-join labels or metadata onto extracted features.
    Useful for adding TUG_score, fall_risk, diagnosis, age, sex, etc.
    """
    return features_df.merge(labels_df, on=list(keys), how="left")
