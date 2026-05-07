import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .config import PipelineConfig
from .utils import (
    SignalMeta,
    adaptive_pause_min_duration,
    build_nan_feature_dict,
    count_pauses,
    detect_threshold_turn_window,
    estimate_sampling_interval_s,
    robust_p2p_threshold,
    safe_gradient,
    select_motion_acc_signal,
    select_turn_angular_signal,
    spectral_entropy,
    truncate_activity_dataframe,
)

TURN_SUFFIXES = [
    "duration",
    "mean_angular_velocity",
    "peak_angular_velocity",
    "ang_vel_std",
    "step_count",
    "pause_time",
    "jerk_std",
    "entropy",
]


def turn_feature_names(prefix):
    return [f"{prefix}_{suffix}" for suffix in TURN_SUFFIXES]


def _turn_step_count_from_acc(df, time_s, turn_mask, config):
    """Count turn steps from acceleration peaks inside the detected turn window."""
    if df is None or not np.any(turn_mask):
        return np.nan

    acc_signal, _ = select_motion_acc_signal(df, config.prefer_useracc_for_motion)
    t_turn = time_s[turn_mask]
    acc_turn = acc_signal[turn_mask]
    if len(acc_turn) < 3:
        return np.nan

    fs_turn = 1.0 / estimate_sampling_interval_s(t_turn) if len(t_turn) > 2 else np.nan
    if not np.isfinite(fs_turn) or fs_turn <= 0:
        return np.nan

    acc_turn_smooth = gaussian_filter1d(acc_turn, sigma=1)
    if not np.isfinite(acc_turn_smooth).any():
        return np.nan

    threshold = float(np.nanmean(acc_turn_smooth))
    if not np.isfinite(threshold):
        return np.nan

    min_distance = max(1, int(0.2 * fs_turn))
    peaks, _ = find_peaks(acc_turn_smooth, height=threshold, distance=min_distance)
    return float(len(peaks))


def extract_turn_features(df, meta, config, prefix):
    """Extract turn features for left/right activities."""
    names = turn_feature_names(prefix)
    if df is None or meta is None or len(df) < config.min_rows_per_activity:
        return build_nan_feature_dict(names)

    activity = "left_turn" if str(prefix).startswith("left") else "right_turn"
    df, meta = truncate_activity_dataframe(df, meta, activity)
    out = build_nan_feature_dict(names)
    time_s = df["time_s"].to_numpy(dtype=float)
    angular_signal, _ = select_turn_angular_signal(df, config, meta.fs_hz)
    angular_abs = np.abs(angular_signal)
    turn_window_sec = getattr(config.window_gate, "turn_window_sec", config.window_gate.window_sec)
    turn_threshold, _ = robust_p2p_threshold(
        time_s=time_s,
        signal=angular_abs,
        window_sec=turn_window_sec,
        k=config.window_gate.turn_threshold_k,
        fallback=config.window_gate.turn_min_amp_threshold,
    )
    pause_threshold = turn_threshold

    start_t, end_t, duration, turn_mask = detect_threshold_turn_window(
        angular_signal_abs=angular_abs,
        time_s=time_s,
        threshold=turn_threshold,
        window_sec=turn_window_sec,
        min_duration_s=config.window_gate.turn_min_duration_s,
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
    out[f"{prefix}_step_count"] = _turn_step_count_from_acc(df, time_s, turn_mask, config)

    pause_min_duration_s = adaptive_pause_min_duration(
        angular_velocity=ang_for_stats,
        time_s=t_for_stats,
        threshold=pause_threshold,
        beta=config.turn_pause.adaptive_beta,
    )

    _, pause_time = count_pauses(
        angular_velocity=ang_for_stats,
        time_s=t_for_stats,
        threshold=pause_threshold,
        min_duration_s=pause_min_duration_s,
    )
    out[f"{prefix}_pause_time"] = pause_time

    angular_jerk = safe_gradient(ang_abs_for_stats, t_for_stats)
    out[f"{prefix}_jerk_std"] = float(np.nanstd(angular_jerk))
    out[f"{prefix}_entropy"] = spectral_entropy(ang_abs_for_stats, meta.fs_hz)
    return out
