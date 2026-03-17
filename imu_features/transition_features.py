import numpy as np
import pandas as pd

from .config import PipelineConfig
from .utils import (
    SignalMeta,
    build_nan_feature_dict,
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

    out[f"{prefix}_duration"] = float(meta.duration_s)

    if np.isfinite(acc_signal).any():
        peak_idx = int(np.nanargmax(acc_signal))
        out[f"{prefix}_time_to_peak_acc"] = float(time_s[peak_idx] - time_s[0])
        out[f"{prefix}_peak_acc"] = float(acc_signal[peak_idx])

    out[f"{prefix}_peak_gyro"] = float(np.nanmax(gyro_signal))
    out[f"{prefix}_acc_rms"] = rms(acc_signal)

    acc_jerk = safe_gradient(acc_signal, time_s)
    out[f"{prefix}_jerk_mean"] = float(np.nanmean(np.abs(acc_jerk)))
    out[f"{prefix}_jerk_std"] = float(np.nanstd(acc_jerk))

    peaks = detect_peaks(
        signal=acc_signal,
        fs_hz=meta.fs_hz,
        min_distance_s=config.transition_peaks.min_distance_s,
        prominence=config.transition_peaks.prominence,
        height=None,
    )
    out[f"{prefix}_peak_count"] = float(len(peaks))
    out[f"{prefix}_entropy"] = spectral_entropy(acc_signal, meta.fs_hz)
    return out
