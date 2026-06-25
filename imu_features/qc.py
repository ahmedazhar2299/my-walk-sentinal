import json
from pathlib import Path

import numpy as np


QC_SCORE_COLUMNS = [
    "walk_qc_score",
    "left_turn_qc_score",
    "right_turn_qc_score",
    "sit_to_stand_qc_score",
    "stand_to_sit_qc_score",
]


DEFAULT_QC_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "qc_thresholds.json"


def load_qc_thresholds(path=None):
    """Load QC thresholds from JSON."""
    config_path = Path(path) if path is not None else DEFAULT_QC_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _finite(value):
    try:
        return float(value) if np.isfinite(float(value)) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _between(value, min_value, max_value):
    value = _finite(value)
    return np.isfinite(value) and float(min_value) <= value <= float(max_value)


def _relative_error(observed, expected):
    observed = _finite(observed)
    expected = _finite(expected)
    if not np.isfinite(observed) or not np.isfinite(expected) or expected == 0:
        return np.nan
    return abs(observed - expected) / abs(expected)


def score_walk_qc(row, thresholds):
    score = 0
    duration = _finite(row.get("walk_duration"))
    step_count = _finite(row.get("step_count"))
    cadence = _finite(row.get("cadence"))
    mean_step_time = _finite(row.get("mean_step_time"))

    if _between(duration, thresholds["duration_min_s"], thresholds["duration_max_s"]):
        score += 1
    if np.isfinite(step_count) and step_count >= float(thresholds["step_count_min"]):
        score += 1
    if _between(cadence, thresholds["cadence_min"], thresholds["cadence_max"]):
        score += 1
    if np.isfinite(step_count) and np.isfinite(duration) and duration > 0:
        expected_cadence = step_count / duration * 60.0
        err = _relative_error(cadence, expected_cadence)
        if np.isfinite(err) and err < float(thresholds["cadence_consistency_error_max"]):
            score += 1
    if np.isfinite(cadence) and cadence > 0:
        expected_step_time = 60.0 / cadence
        err = _relative_error(mean_step_time, expected_step_time)
        if np.isfinite(err) and err < float(thresholds["mean_step_time_consistency_error_max"]):
            score += 1
    return int(score)


def score_turn_qc(row, prefix, thresholds):
    score = 0
    duration = _finite(row.get(f"{prefix}_duration"))
    step_count = _finite(row.get(f"{prefix}_step_count"))
    peak_velocity = _finite(row.get(f"{prefix}_peak_angular_velocity"))
    mean_velocity = _finite(row.get(f"{prefix}_mean_angular_velocity"))
    pause_time = _finite(row.get(f"{prefix}_pause_time"))

    if _between(duration, thresholds["duration_min_s"], thresholds["duration_max_s"]):
        score += 1
    if _between(step_count, thresholds["step_count_min"], thresholds["step_count_max"]):
        score += 1
    if np.isfinite(peak_velocity) and peak_velocity > float(thresholds["peak_angular_velocity_min"]):
        score += 1
    if np.isfinite(peak_velocity) and np.isfinite(mean_velocity) and peak_velocity > mean_velocity:
        score += 1
    if np.isfinite(pause_time) and np.isfinite(duration) and pause_time < duration:
        score += 1
    return int(score)


def score_transition_qc(row, prefix, thresholds):
    score = 0
    duration = _finite(row.get(f"{prefix}_duration"))
    time_to_peak_acc = _finite(row.get(f"{prefix}_time_to_peak_acc"))
    peak_acc = _finite(row.get(f"{prefix}_peak_acc"))
    acc_rms = _finite(row.get(f"{prefix}_acc_rms"))
    peak_gyro = _finite(row.get(f"{prefix}_peak_gyro"))

    if _between(duration, thresholds["duration_min_s"], thresholds["duration_max_s"]):
        score += 1
    if np.isfinite(time_to_peak_acc) and np.isfinite(duration) and time_to_peak_acc < duration:
        score += 1
    if np.isfinite(peak_acc) and np.isfinite(acc_rms) and peak_acc > acc_rms:
        score += 1
    if np.isfinite(peak_gyro) and peak_gyro > float(thresholds["peak_gyro_min"]):
        score += 1
    return int(score)


def add_qc_scores(row, thresholds):
    """Add aggregate QC score columns to a completed feature row."""
    row["walk_qc_score"] = score_walk_qc(row, thresholds["walk"])
    row["left_turn_qc_score"] = score_turn_qc(row, "left_turn", thresholds["turn"])
    row["right_turn_qc_score"] = score_turn_qc(row, "right_turn", thresholds["turn"])
    row["sit_to_stand_qc_score"] = score_transition_qc(row, "sit_to_stand", thresholds["transition"])
    row["stand_to_sit_qc_score"] = score_transition_qc(row, "stand_to_sit", thresholds["transition"])
    return row
