from pathlib import Path

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .transition_features import (
    extract_transition_features,
    transition_flexion_extension_peaks,
    transition_feature_names,
)
from .turn_features import extract_turn_features, turn_feature_names
from .utils import (
    ACTIVITY_ORDER,
    build_nan_feature_dict,
    detect_turn_window,
    read_and_preprocess_csv,
    resolve_activity_files,
    robust_p2p_threshold,
    select_turn_angular_signal,
)
from .walk_features import (
    WALK_FEATURE_NAMES,
    WALK_NONLINEAR_FEATURE_NAMES,
    extract_walk_features,
    extract_walk_nonlinear_features,
)

WALK_DATASET_FEATURE_NAMES = WALK_FEATURE_NAMES + WALK_NONLINEAR_FEATURE_NAMES

TURN_XCORR_FEATURES = [
    "turn_left_right_xcorr_symmetry_score",
    "turn_left_right_xcorr_peak_correlation",
]

TRANSITION_XCORR_FEATURES = [
    "sitstand_standsit_xcorr_symmetry_score",
    "sitstand_standsit_xcorr_peak_correlation",
]

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
        return build_nan_feature_dict(WALK_DATASET_FEATURE_NAMES)
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
        features = extract_walk_features(df, meta, config)
        features.update(extract_walk_nonlinear_features(df, meta, config))
        return features
    if activity == "left_turn":
        return extract_turn_features(df, meta, config, prefix="left_turn")
    if activity == "right_turn":
        return extract_turn_features(df, meta, config, prefix="right_turn")
    if activity == "sit_to_stand":
        return extract_transition_features(df, meta, config, prefix="sit_to_stand")
    if activity == "stand_to_sit":
        return extract_transition_features(df, meta, config, prefix="stand_to_sit")
    return {}


def _window_signal(time_s, signal, start_t, end_t):
    time_s = np.asarray(time_s, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if len(time_s) != len(signal) or len(time_s) < 3:
        return np.array([]), np.array([])
    mask = (time_s >= start_t) & (time_s <= end_t) & np.isfinite(time_s) & np.isfinite(signal)
    return time_s[mask], signal[mask]


def _resample_signal_to_n(time_s, signal, n_points=101):
    time_s = np.asarray(time_s, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if len(time_s) < 3 or len(signal) < 3 or n_points < 3:
        return np.array([])

    order = np.argsort(time_s)
    time_s = time_s[order]
    signal = signal[order]
    keep = np.isfinite(time_s) & np.isfinite(signal)
    time_s = time_s[keep]
    signal = signal[keep]
    if len(time_s) < 3:
        return np.array([])

    unique_t, unique_idx = np.unique(time_s, return_index=True)
    time_s = unique_t
    signal = signal[unique_idx]
    if len(time_s) < 3 or time_s[-1] <= time_s[0]:
        return np.array([])

    resampled_t = np.linspace(time_s[0], time_s[-1], n_points)
    return np.interp(resampled_t, time_s, signal)


def _normalize_signal(signal):
    signal = np.asarray(signal, dtype=float)
    signal = signal[np.isfinite(signal)]
    if len(signal) < 3:
        return np.array([])
    signal = signal - float(np.nanmean(signal))
    std = float(np.nanstd(signal))
    if not np.isfinite(std) or std == 0:
        return np.array([])
    return signal / std


def _cycle_xcorr_features(data_a, data_b, prefix):
    empty = {
        f"{prefix}_xcorr_symmetry_score": np.nan,
        f"{prefix}_xcorr_peak_correlation": np.nan,
    }
    if data_a is None or data_b is None:
        return empty

    ta, xa = _window_signal(
        data_a["time_s"],
        data_a["signal"],
        data_a["start_time_s"],
        data_a["end_time_s"],
    )
    tb, xb = _window_signal(
        data_b["time_s"],
        data_b["signal"],
        data_b["start_time_s"],
        data_b["end_time_s"],
    )
    xa = _normalize_signal(_resample_signal_to_n(ta, xa, n_points=101))
    xb = _normalize_signal(_resample_signal_to_n(tb, xb, n_points=101))
    if len(xa) < 3 or len(xb) < 3 or len(xa) != len(xb):
        return empty

    denom = np.linalg.norm(xa) * np.linalg.norm(xb)
    if not np.isfinite(denom) or denom == 0:
        return empty

    corr = np.correlate(xa, xb, mode="full") / denom
    if len(corr) == 0:
        return empty

    peak = float(corr[int(np.argmax(np.abs(corr)))])
    return {
        f"{prefix}_xcorr_symmetry_score": float(abs(peak)),
        f"{prefix}_xcorr_peak_correlation": peak,
    }


def _load_turn_xcorr_data(csv_path, config, verbose):
    if csv_path is None:
        return None
    try:
        df, meta = read_and_preprocess_csv(csv_path, config)
        time_s = df["time_s"].to_numpy(dtype=float)
        angular_signal, _ = select_turn_angular_signal(df, config, meta.fs_hz)
        angular_abs = np.abs(angular_signal)
        threshold, _ = robust_p2p_threshold(
            time_s=time_s,
            signal=angular_abs,
            window_sec=config.window_gate.window_sec,
            k=config.window_gate.turn_threshold_k,
            fallback=config.window_gate.turn_min_amp_threshold,
        )
        start_t, end_t, _, _ = detect_turn_window(
            angular_signal_abs=angular_abs,
            time_s=time_s,
            threshold=threshold,
            window_sec=config.window_gate.window_sec,
        )
        return {
            "time_s": time_s,
            "signal": angular_signal,
            "start_time_s": start_t,
            "end_time_s": end_t,
        }
    except Exception as exc:
        if verbose:
            print(f"[WARN] Failed to process turn xcorr data {csv_path}: {exc}")
        return None


def _load_transition_xcorr_data(csv_path, config, verbose):
    if csv_path is None:
        return None
    try:
        df, meta = read_and_preprocess_csv(csv_path, config)
        flex_ext = transition_flexion_extension_peaks(df, meta, config)
        return {
            "time_s": flex_ext["time_s"],
            "signal": flex_ext["gyro_signal"],
            "start_time_s": flex_ext["start_time_s"],
            "end_time_s": flex_ext["end_time_s"],
        }
    except Exception as exc:
        if verbose:
            print(f"[WARN] Failed to process transition xcorr data {csv_path}: {exc}")
        return None


def _dataset_columns():
    columns = ["patient_id", "date"]
    columns.extend(WALK_DATASET_FEATURE_NAMES)
    columns.extend(turn_feature_names("left_turn"))
    columns.extend(turn_feature_names("right_turn"))
    columns.extend(ASYMMETRY_FEATURES)
    columns.extend(TURN_XCORR_FEATURES)
    columns.extend(transition_feature_names("sit_to_stand"))
    columns.extend(transition_feature_names("stand_to_sit"))
    columns.extend(TRANSITION_XCORR_FEATURES)
    return columns


def _iter_patient_date_dirs(dataset_root):
    folders = []
    for patient_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
        if patient_dir.name.startswith(".") or patient_dir.name.lower() == "template":
            continue
        date_dirs = sorted([d for d in patient_dir.iterdir() if d.is_dir()])
        for date_dir in date_dirs:
            if date_dir.name.startswith(".") or date_dir.name.lower() == "template":
                continue
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
        row.update(_cycle_xcorr_features(
            _load_turn_xcorr_data(activity_files.get("left_turn"), config, verbose),
            _load_turn_xcorr_data(activity_files.get("right_turn"), config, verbose),
            prefix="turn_left_right",
        ))
        row.update(_cycle_xcorr_features(
            _load_transition_xcorr_data(activity_files.get("sit_to_stand"), config, verbose),
            _load_transition_xcorr_data(activity_files.get("stand_to_sit"), config, verbose),
            prefix="sitstand_standsit",
        ))
        rows.append(row)

        if verbose:
            found_count = sum(path is not None for path in activity_files.values())
            print(f"[INFO] {patient_id}/{date_str}: found {found_count}/5 activity files")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["patient_id", "date"]).reset_index(drop=True)
        ordered_columns = _dataset_columns()
        extra_columns = [col for col in df.columns if col not in ordered_columns]
        df = df.reindex(columns=ordered_columns + extra_columns)

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
