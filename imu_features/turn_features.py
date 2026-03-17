import numpy as np
import pandas as pd

from .config import PipelineConfig
from .utils import (
    SignalMeta,
    build_nan_feature_dict,
    count_pauses,
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

    out[f"{prefix}_duration"] = float(meta.duration_s)
    out[f"{prefix}_mean_angular_velocity"] = float(np.nanmean(np.abs(angular_signal)))
    out[f"{prefix}_peak_angular_velocity"] = float(np.nanmax(np.abs(angular_signal)))
    out[f"{prefix}_ang_vel_std"] = float(np.nanstd(angular_signal))

    pause_count, pause_time = count_pauses(
        angular_velocity=angular_signal,
        time_s=time_s,
        threshold=config.turn_pause.velocity_threshold,
        min_duration_s=config.turn_pause.min_pause_duration_s,
    )
    out[f"{prefix}_pause_count"] = pause_count
    out[f"{prefix}_pause_time"] = pause_time

    angular_jerk = safe_gradient(np.abs(angular_signal), time_s)
    out[f"{prefix}_jerk_std"] = float(np.nanstd(angular_jerk))
    out[f"{prefix}_entropy"] = spectral_entropy(np.abs(angular_signal), meta.fs_hz)
    return out
