import numpy as np
import pandas as pd

from .config import PipelineConfig
from .utils import (
    SignalMeta,
    adaptive_amplitude_threshold,
    build_nan_feature_dict,
    detect_active_window,
    detect_peaks,
    rms,
    safe_gradient,
    select_motion_acc_signal,
    spectral_entropy,
)

TRANSITION_SUFFIXES = [
    "duration",
    "time_to_peak_acc",
    "peak_acc",
    "peak_gyro",
    "acc_rms",
    "jerk_mean",
    "jerk_std",
    "peak_count",
    "entropy",
]

DEFAULT_TRANSITION_MIN_DISTANCE_S = 0.15


def transition_feature_names(prefix):
    return [f"{prefix}_{suffix}" for suffix in TRANSITION_SUFFIXES]


def extract_transition_features(df, meta, config, prefix):
    """Extract sit-to-stand or stand-to-sit transition features."""
    names = transition_feature_names(prefix)
    if df is None or meta is None or len(df) < config.min_rows_per_activity:
        return build_nan_feature_dict(names)

    out = build_nan_feature_dict(names)
    time_s = df["time_s"].to_numpy(dtype=float)
    acc_signal, _ = select_motion_acc_signal(df, config.prefer_useracc_for_motion)
    gyro_signal = df["gyro_mag"].to_numpy(dtype=float)

    start_t, end_t, duration, transition_mask, _ = detect_active_window(
        signal=acc_signal, time_s=time_s, min_duration_s=0.5
    )
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

    out[f"{prefix}_peak_gyro"] = float(np.nanmax(gyro_for_stats))
    out[f"{prefix}_acc_rms"] = rms(acc_for_stats)

    acc_jerk = safe_gradient(acc_for_stats, time_for_stats)
    out[f"{prefix}_jerk_mean"] = float(np.nanmean(np.abs(acc_jerk)))
    out[f"{prefix}_jerk_std"] = float(np.nanstd(acc_jerk))

    transition_threshold, _, _ = adaptive_amplitude_threshold(
        signal=acc_for_stats,
        fs_hz=meta.fs_hz,
        config=config.adaptive_thresholds,
        k_value=config.adaptive_thresholds.transition_threshold_k,
        min_value=config.adaptive_thresholds.transition_threshold_min,
    )

    peaks = detect_peaks(
        signal=acc_for_stats,
        fs_hz=meta.fs_hz,
        min_distance_s=DEFAULT_TRANSITION_MIN_DISTANCE_S,
        height=transition_threshold,
    )
    out[f"{prefix}_peak_count"] = float(len(peaks))
    out[f"{prefix}_entropy"] = spectral_entropy(acc_for_stats, meta.fs_hz)
    return out
