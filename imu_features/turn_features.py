import numpy as np
import pandas as pd

from .config import PipelineConfig
from .utils import (
    SignalMeta,
    adaptive_amplitude_threshold,
    adaptive_pause_min_duration,
    build_nan_feature_dict,
    count_pauses,
    detect_turn_window,
    safe_gradient,
    select_turn_angular_signal,
    spectral_entropy,
)

TURN_SUFFIXES = [
    "duration",
    "mean_angular_velocity",
    "peak_angular_velocity",
    "ang_vel_std",
    "pause_count",
    "pause_time",
    "jerk_std",
    "entropy",
]


def turn_feature_names(prefix):
    return [f"{prefix}_{suffix}" for suffix in TURN_SUFFIXES]


def extract_turn_features(df, meta, config, prefix):
    """Extract turn features for left/right activities."""
    names = turn_feature_names(prefix)
    if df is None or meta is None or len(df) < config.min_rows_per_activity:
        return build_nan_feature_dict(names)

    out = build_nan_feature_dict(names)
    time_s = df["time_s"].to_numpy(dtype=float)
    angular_signal, _ = select_turn_angular_signal(df, config, meta.fs_hz)
    angular_abs = np.abs(angular_signal)

    turn_threshold, _, _ = adaptive_amplitude_threshold(
        signal=angular_abs,
        fs_hz=meta.fs_hz,
        config=config.adaptive_thresholds,
        k_value=config.adaptive_thresholds.turn_threshold_k,
        min_value=config.adaptive_thresholds.turn_threshold_min,
    )
    pause_threshold, _, _ = adaptive_amplitude_threshold(
        signal=angular_abs,
        fs_hz=meta.fs_hz,
        config=config.adaptive_thresholds,
        k_value=config.adaptive_thresholds.pause_threshold_k,
        min_value=config.adaptive_thresholds.pause_threshold_min,
    )

    start_t, end_t, duration, turn_mask = detect_turn_window(
        angular_signal_abs=angular_abs,
        time_s=time_s,
        threshold=turn_threshold,
    )
    if np.any(turn_mask):
        ang_for_stats = angular_signal[turn_mask]
        ang_abs_for_stats = angular_abs[turn_mask]
        t_for_stats = time_s[turn_mask]
    else:
        ang_for_stats = angular_signal
        ang_abs_for_stats = angular_abs
        t_for_stats = time_s

    out[f"{prefix}_duration"] = duration
    out[f"{prefix}_mean_angular_velocity"] = float(np.nanmean(ang_abs_for_stats))
    out[f"{prefix}_peak_angular_velocity"] = float(np.nanmax(ang_abs_for_stats))
    out[f"{prefix}_ang_vel_std"] = float(np.nanstd(ang_for_stats))

    pause_min_duration_s = adaptive_pause_min_duration(
        angular_velocity=ang_for_stats,
        time_s=t_for_stats,
        threshold=pause_threshold,
        beta=config.turn_pause.adaptive_beta,
    )

    pause_count, pause_time = count_pauses(
        angular_velocity=ang_for_stats,
        time_s=t_for_stats,
        threshold=pause_threshold,
        min_duration_s=pause_min_duration_s,
    )
    out[f"{prefix}_pause_count"] = pause_count
    out[f"{prefix}_pause_time"] = pause_time

    angular_jerk = safe_gradient(ang_abs_for_stats, t_for_stats)
    out[f"{prefix}_jerk_std"] = float(np.nanstd(angular_jerk))
    out[f"{prefix}_entropy"] = spectral_entropy(ang_abs_for_stats, meta.fs_hz)
    return out
