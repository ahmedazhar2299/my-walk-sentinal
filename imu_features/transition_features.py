import numpy as np
import pandas as pd

from .config import PipelineConfig
from .utils import (
    SignalMeta,
    build_nan_feature_dict,
    detect_active_window,
    robust_p2p_threshold,
    detect_peaks,
    rms,
    safe_gradient,
    select_motion_acc_signal,
    spectral_entropy,
    truncate_activity_dataframe,
)

TRANSITION_SUFFIXES = [
    "duration",
    "time_to_peak_acc",
    "peak_acc",
    "flexion_peak",
    "extension_peak",
    "peak_gyro",
    "acc_rms",
    "jerk_mean",
    "jerk_std",
    "peak_count",
    "entropy",
]

DEFAULT_TRANSITION_MIN_DISTANCE_S = 0.15
GYRO_AXIS_COLUMNS = ("gyro_x", "gyro_y", "gyro_z")


def transition_feature_names(prefix):
    return [f"{prefix}_{suffix}" for suffix in TRANSITION_SUFFIXES]


def _transition_window(time_s, signal, config):
    transition_threshold, _ = robust_p2p_threshold(
        time_s=time_s,
        signal=signal,
        window_sec=config.window_gate.window_sec,
        k=config.window_gate.transition_threshold_k,
        fallback=config.window_gate.transition_min_amp_threshold,
    )
    return detect_active_window(
        signal=signal,
        time_s=time_s,
        min_duration_s=config.window_gate.transition_min_duration_s,
        threshold=transition_threshold,
        window_sec=config.window_gate.window_sec,
    )


def _median_mad_threshold(values, k, fallback=np.nan):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float(fallback) if np.isfinite(fallback) else np.nan
    median = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - median)))
    threshold = median + float(k) * mad
    max_value = float(np.nanmax(values))
    if np.isfinite(max_value) and max_value > 0:
        threshold = min(threshold, 0.3 * max_value)
    if not np.isfinite(threshold):
        threshold = float(fallback) if np.isfinite(fallback) else np.nan
    return float(threshold)


def _refine_window_from_abs_signal(time_s, signal, broad_mask, config):
    time_s = np.asarray(time_s, dtype=float)
    signal = np.asarray(signal, dtype=float)
    broad_mask = np.asarray(broad_mask, dtype=bool)
    if len(time_s) == 0 or len(signal) != len(time_s):
        return np.nan, np.nan, np.nan, np.zeros(len(time_s), dtype=bool), np.nan

    if not np.any(broad_mask):
        broad_mask = np.isfinite(signal) & np.isfinite(time_s)
    abs_signal = np.abs(signal)
    threshold = _median_mad_threshold(
        abs_signal[broad_mask],
        k=config.window_gate.transition_threshold_k,
        fallback=config.window_gate.transition_min_amp_threshold,
    )
    active_mask = broad_mask & np.isfinite(abs_signal) & np.isfinite(threshold) & (abs_signal >= threshold)
    if not np.any(active_mask):
        active_mask = broad_mask & np.isfinite(signal)

    active_idx = np.where(active_mask)[0]
    if len(active_idx) == 0:
        return np.nan, np.nan, np.nan, np.zeros(len(time_s), dtype=bool), threshold

    start_t = float(time_s[int(active_idx[0])])
    end_t = float(time_s[int(active_idx[-1])])
    duration = float(end_t - start_t)
    if not np.isfinite(duration) or duration < config.window_gate.transition_min_duration_s:
        broad_idx = np.where(broad_mask & np.isfinite(signal))[0]
        if len(broad_idx) == 0:
            return np.nan, np.nan, np.nan, np.zeros(len(time_s), dtype=bool), threshold
        start_t = float(time_s[int(broad_idx[0])])
        end_t = float(time_s[int(broad_idx[-1])])
        duration = float(end_t - start_t)

    refined_mask = (time_s >= start_t) & (time_s <= end_t) & np.isfinite(signal)
    return start_t, end_t, duration, refined_mask, threshold


def _dominant_gyro_axis(df, mask):
    best_axis = None
    best_range = -np.inf
    for axis in GYRO_AXIS_COLUMNS:
        if axis not in df.columns:
            continue
        values = df[axis].to_numpy(dtype=float)
        values = values[mask] if np.any(mask) else values
        if not np.isfinite(values).any():
            continue
        axis_range = float(np.nanmax(values) - np.nanmin(values))
        if axis_range > best_range:
            best_axis = axis
            best_range = axis_range
    return best_axis


def transition_flexion_extension_peaks(df, meta, config):
    """Return flexion/extension peak details from the dominant signed gyro axis."""
    empty = {
        "start_time_s": np.nan,
        "end_time_s": np.nan,
        "duration_s": np.nan,
        "threshold": np.nan,
        "gyro_axis": None,
        "time_s": np.array([], dtype=float),
        "gyro_signal": np.array([], dtype=float),
        "transition_mask": np.array([], dtype=bool),
        "flexion_peak": np.nan,
        "flexion_peak_time_s": np.nan,
        "extension_peak": np.nan,
        "extension_peak_time_s": np.nan,
    }
    if df is None or meta is None or len(df) < config.min_rows_per_activity:
        return empty

    activity = "sit_to_stand"
    df, meta = truncate_activity_dataframe(df, meta, activity)
    time_s = df["time_s"].to_numpy(dtype=float)
    gyro_mag = df["gyro_mag"].to_numpy(dtype=float)
    prelim_start_t, prelim_end_t, prelim_duration, prelim_mask, prelim_threshold = _transition_window(
        time_s,
        gyro_mag,
        config,
    )
    axis = _dominant_gyro_axis(df, prelim_mask)
    if axis is None:
        empty.update({
            "start_time_s": prelim_start_t,
            "end_time_s": prelim_end_t,
            "duration_s": prelim_duration,
            "threshold": prelim_threshold,
            "transition_mask": prelim_mask,
        })
        return empty

    gyro_signal = df[axis].to_numpy(dtype=float)
    start_t, end_t, duration, transition_mask, threshold = _refine_window_from_abs_signal(
        time_s,
        gyro_signal,
        prelim_mask,
        config,
    )
    if not np.isfinite(start_t) or not np.isfinite(end_t):
        start_t = prelim_start_t
        end_t = prelim_end_t
        duration = prelim_duration
        transition_mask = prelim_mask
        threshold = prelim_threshold

    peak_search_mask = transition_mask & np.isfinite(gyro_signal)
    if np.isfinite(start_t):
        peak_search_mask &= time_s >= start_t
    if not np.any(peak_search_mask):
        peak_search_mask = prelim_mask & np.isfinite(gyro_signal)
    if not np.any(peak_search_mask):
        peak_search_mask = np.isfinite(gyro_signal)

    peak_idx = np.where(peak_search_mask)[0]
    flex_global_idx = None
    ext_global_idx = None
    if len(peak_idx):
        gyro_for_peaks = gyro_signal[peak_idx]
        if np.isfinite(gyro_for_peaks).any():
            flex_global_idx = int(peak_idx[int(np.nanargmax(gyro_for_peaks))])
            ext_global_idx = int(peak_idx[int(np.nanargmin(gyro_for_peaks))])

    if flex_global_idx is not None and ext_global_idx is not None and np.isfinite(threshold):
        last_motion_peak_idx = max(flex_global_idx, ext_global_idx)
        search_idx = peak_idx[peak_idx > last_motion_peak_idx]
        below_idx = search_idx[
            np.isfinite(gyro_signal[search_idx]) & (np.abs(gyro_signal[search_idx]) < threshold)
        ]
        if len(below_idx):
            end_idx = int(below_idx[0])
            end_t = float(time_s[end_idx])
            duration = float(end_t - start_t) if np.isfinite(start_t) else np.nan
            transition_mask = (time_s >= start_t) & (time_s <= end_t) & np.isfinite(gyro_signal)

    details = {
        "start_time_s": start_t,
        "end_time_s": end_t,
        "duration_s": duration,
        "threshold": threshold,
        "gyro_axis": axis,
        "time_s": time_s,
        "gyro_signal": gyro_signal,
        "transition_mask": transition_mask,
    }
    if flex_global_idx is not None and ext_global_idx is not None:
        details.update({
            "flexion_peak": float(gyro_signal[flex_global_idx]),
            "flexion_peak_time_s": float(time_s[flex_global_idx]),
            "extension_peak": float(gyro_signal[ext_global_idx]),
            "extension_peak_time_s": float(time_s[ext_global_idx]),
        })
    else:
        details.update({
            "flexion_peak": np.nan,
            "flexion_peak_time_s": np.nan,
            "extension_peak": np.nan,
            "extension_peak_time_s": np.nan,
        })
    return details


def extract_transition_features(df, meta, config, prefix):
    """Extract sit-to-stand or stand-to-sit transition features."""
    names = transition_feature_names(prefix)
    if df is None or meta is None or len(df) < config.min_rows_per_activity:
        return build_nan_feature_dict(names)

    df, meta = truncate_activity_dataframe(df, meta, prefix)
    out = build_nan_feature_dict(names)
    time_s = df["time_s"].to_numpy(dtype=float)
    acc_signal, _ = select_motion_acc_signal(df, config.prefer_useracc_for_motion)
    gyro_signal = df["gyro_mag"].to_numpy(dtype=float)

    flex_ext = transition_flexion_extension_peaks(df, meta, config)
    duration = flex_ext.get("duration_s", np.nan)
    transition_mask = flex_ext.get("transition_mask", np.zeros(len(df), dtype=bool))
    out[f"{prefix}_duration"] = duration
    if np.any(transition_mask):
        acc_for_stats = acc_signal[transition_mask]
        gyro_for_stats = gyro_signal[transition_mask]
        time_for_stats = time_s[transition_mask]
    else:
        acc_for_stats = acc_signal
        gyro_for_stats = gyro_signal
        time_for_stats = time_s

    if np.isfinite(acc_for_stats).any():
        peak_idx = int(np.nanargmax(acc_for_stats))
        out[f"{prefix}_time_to_peak_acc"] = float(time_for_stats[peak_idx] - time_for_stats[0])
        out[f"{prefix}_peak_acc"] = float(acc_for_stats[peak_idx])

    out[f"{prefix}_flexion_peak"] = flex_ext.get("flexion_peak", np.nan)
    out[f"{prefix}_extension_peak"] = flex_ext.get("extension_peak", np.nan)
    out[f"{prefix}_peak_gyro"] = float(np.nanmax(gyro_for_stats))
    out[f"{prefix}_acc_rms"] = rms(acc_for_stats)

    acc_jerk = safe_gradient(acc_for_stats, time_for_stats)
    out[f"{prefix}_jerk_mean"] = float(np.nanmean(np.abs(acc_jerk)))
    out[f"{prefix}_jerk_std"] = float(np.nanstd(acc_jerk))

    peaks = detect_peaks(
        signal=acc_for_stats,
        fs_hz=meta.fs_hz,
        min_distance_s=DEFAULT_TRANSITION_MIN_DISTANCE_S,
        height=None,
    )
    out[f"{prefix}_peak_count"] = float(len(peaks))
    out[f"{prefix}_entropy"] = spectral_entropy(acc_for_stats, meta.fs_hz)
    return out
