import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .config import PipelineConfig
from .utils import (
    SignalMeta,
    adaptive_pause_min_duration,
    apply_lowpass_filter,
    build_nan_feature_dict,
    count_pauses,
    estimate_sampling_interval_s,
    mask_close_gaps,
    safe_gradient,
    select_motion_acc_signal,
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

    if "acc_mag_gravity_removed" in df:
        acc_signal = np.abs(df["acc_mag_gravity_removed"].to_numpy(dtype=float))
    else:
        acc_signal = np.array([], dtype=float)
    if len(acc_signal) == 0 or not np.isfinite(acc_signal).any() or np.nanstd(acc_signal) <= 1e-8:
        acc_signal, _ = select_motion_acc_signal(df, config.prefer_useracc_for_motion)
        acc_signal = np.abs(acc_signal)
    t_turn = time_s[turn_mask]
    acc_turn = acc_signal[turn_mask]
    if len(acc_turn) < 3:
        return np.nan

    fs_turn = 1.0 / estimate_sampling_interval_s(t_turn) if len(t_turn) > 2 else np.nan
    if not np.isfinite(fs_turn) or fs_turn <= 0:
        return np.nan

    acc_turn_smooth = gaussian_filter1d(acc_turn, sigma=2)
    if not np.isfinite(acc_turn_smooth).any():
        return np.nan

    peak_percentile = float(getattr(config.window_gate, "turn_step_peak_percentile", 55.0))
    threshold = float(np.nanpercentile(acc_turn_smooth, peak_percentile))
    if not np.isfinite(threshold):
        return np.nan

    min_interval_s = float(getattr(config.window_gate, "turn_step_min_interval_s", 0.50))
    min_distance = max(1, int(min_interval_s * fs_turn))
    peaks, _ = find_peaks(acc_turn_smooth, height=threshold, distance=min_distance)
    return float(len(peaks))


def _turn_window_from_gyro_magnitude(df, meta, config):
    time_s = df["time_s"].to_numpy(dtype=float)
    if "gyro_mag" in df.columns:
        angular_signal = df["gyro_mag"].to_numpy(dtype=float)
    else:
        angular_signal = np.sqrt(
            df["gyro_x"].to_numpy(dtype=float) ** 2
            + df["gyro_y"].to_numpy(dtype=float) ** 2
            + df["gyro_z"].to_numpy(dtype=float) ** 2
        )
    if config.filtering.enabled and np.isfinite(meta.fs_hz):
        angular_signal = apply_lowpass_filter(
            angular_signal,
            fs_hz=meta.fs_hz,
            cutoff_hz=config.filtering.cutoff_hz,
            order=config.filtering.order,
        )

    threshold = 0.25 * float(np.nanmax(angular_signal)) if np.isfinite(angular_signal).any() else np.nan
    active = np.isfinite(angular_signal) & np.isfinite(threshold) & (angular_signal >= threshold)
    if np.isfinite(meta.fs_hz) and meta.fs_hz > 0:
        active = mask_close_gaps(active, max_gap_samples=max(1, int(round(0.20 * meta.fs_hz))))

    turn_mask = np.zeros(len(time_s), dtype=bool)
    start_t = np.nan
    end_t = np.nan
    duration = np.nan
    if np.any(active):
        idx = np.where(active)[0]
        pad = max(1, int(round(0.10 * meta.fs_hz))) if np.isfinite(meta.fs_hz) and meta.fs_hz > 0 else 1
        i0 = max(0, int(idx[0]) - pad)
        i1 = min(len(time_s) - 1, int(idx[-1]) + pad)
        turn_mask[i0 : i1 + 1] = True
        start_t = float(time_s[i0])
        end_t = float(time_s[i1])
        duration = float(end_t - start_t)
    return angular_signal, start_t, end_t, duration, turn_mask, threshold


def extract_turn_features(df, meta, config, prefix):
    """Extract turn features for left/right activities."""
    names = turn_feature_names(prefix)
    if df is None or meta is None or len(df) < config.min_rows_per_activity:
        return build_nan_feature_dict(names)

    activity = "left_turn" if str(prefix).startswith("left") else "right_turn"
    df, meta = truncate_activity_dataframe(df, meta, activity)
    out = build_nan_feature_dict(names)
    time_s = df["time_s"].to_numpy(dtype=float)
    angular_abs, start_t, end_t, duration, turn_mask, turn_threshold = _turn_window_from_gyro_magnitude(
        df,
        meta,
        config,
    )
    stats_angular_abs = angular_abs
    if "gyro_x" in df.columns:
        stats_angular_abs = np.abs(df["gyro_x"].to_numpy(dtype=float))
        if config.filtering.enabled and np.isfinite(meta.fs_hz):
            stats_angular_abs = apply_lowpass_filter(
                stats_angular_abs,
                fs_hz=meta.fs_hz,
                cutoff_hz=config.filtering.cutoff_hz,
                order=config.filtering.order,
            )
    pause_threshold = turn_threshold
    if np.any(turn_mask):
        ang_abs_for_stats = stats_angular_abs[turn_mask]
        ang_abs_for_pause = angular_abs[turn_mask]
        t_for_stats = time_s[turn_mask]
    else:
        ang_abs_for_stats = stats_angular_abs
        ang_abs_for_pause = angular_abs
        t_for_stats = time_s

    out[f"{prefix}_duration"] = duration
    out[f"{prefix}_mean_angular_velocity"] = float(np.nanmean(ang_abs_for_stats))
    out[f"{prefix}_peak_angular_velocity"] = float(np.nanmax(ang_abs_for_stats))
    out[f"{prefix}_ang_vel_std"] = float(np.nanstd(ang_abs_for_stats))
    out[f"{prefix}_step_count"] = _turn_step_count_from_acc(df, time_s, turn_mask, config)

    pause_min_duration_s = adaptive_pause_min_duration(
        angular_velocity=ang_abs_for_pause,
        time_s=t_for_stats,
        threshold=pause_threshold,
        beta=config.turn_pause.adaptive_beta,
    )

    _, pause_time = count_pauses(
        angular_velocity=ang_abs_for_pause,
        time_s=t_for_stats,
        threshold=pause_threshold,
        min_duration_s=pause_min_duration_s,
    )
    out[f"{prefix}_pause_time"] = pause_time

    angular_jerk = safe_gradient(ang_abs_for_stats, t_for_stats)
    out[f"{prefix}_jerk_std"] = float(np.nanstd(angular_jerk))
    out[f"{prefix}_entropy"] = spectral_entropy(ang_abs_for_stats, meta.fs_hz)
    return out
