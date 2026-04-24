import numpy as np
import pandas as pd

from .config import PipelineConfig
from .utils import (
    SignalMeta,
    build_nan_feature_dict,
    detect_active_window,
    robust_p2p_threshold,
    dominant_frequency,
    rms,
    select_motion_acc_signal,
    spectral_entropy,
    wavelet_step_summary,
)

WALK_FEATURE_NAMES = [
    "walk_duration",
    "step_count",
    "cadence",
    "walking_speed",
    "mean_step_time",
    "step_time_std",
    "step_time_cv",
    "step_regularity",
    "stride_regularity",
    "walk_acc_mag_mean",
    "walk_acc_mag_std",
    "walk_acc_mag_rms",
    "walk_gyro_mag_std",
    "walk_dominant_frequency",
    "walk_spectral_entropy",
    "walk_jerk_mean",
    "walk_jerk_std",
]


def _autocorrelation(signal):
    signal = np.asarray(signal, dtype=float)
    finite = np.isfinite(signal)
    if finite.sum() < 3:
        return np.array([], dtype=float)
    x = signal[finite] - np.nanmean(signal[finite])
    if np.allclose(x, 0.0):
        return np.array([], dtype=float)
    corr = np.correlate(x, x, mode="full")
    corr = corr[len(x) - 1 :]
    if corr[0] == 0:
        return np.array([], dtype=float)
    return corr / corr[0]


def _regularity_from_acf(acf, lag_samples):
    if len(acf) == 0 or lag_samples <= 0 or lag_samples >= len(acf):
        return np.nan
    return float(acf[lag_samples])


def extract_walk_features(df, meta, config):
    """Extract gait and signal features from walk activity."""
    if df is None or meta is None or len(df) < config.min_rows_per_activity:
        return build_nan_feature_dict(WALK_FEATURE_NAMES)

    out = build_nan_feature_dict(WALK_FEATURE_NAMES)
    time_s = df["time_s"].to_numpy(dtype=float)
    acc_signal, acc_source = select_motion_acc_signal(df, config.prefer_useracc_for_motion)
    wavelet = wavelet_step_summary(
        time_s=time_s,
        signal=acc_signal,
        wavelet_config=config.wavelet_steps,
        fixed_window=False,
    )

    start_t = float(wavelet.get("start_time_s", np.nan))
    end_t = float(wavelet.get("end_time_s", np.nan))
    duration = float(wavelet.get("duration_s", np.nan))
    step_count = float(wavelet.get("step_count", np.nan))
    cadence = float(wavelet.get("cadence", np.nan))

    if np.isfinite(start_t) and np.isfinite(end_t):
        walk_mask = (time_s >= start_t) & (time_s <= end_t)
    else:
        walk_threshold, _ = robust_p2p_threshold(
            time_s=time_s,
            signal=acc_signal,
            window_sec=config.window_gate.window_sec,
            k=config.window_gate.walk_threshold_k,
            fallback=config.window_gate.walk_min_amp_threshold,
        )
        start_t, end_t, duration, walk_mask, _ = detect_active_window(
            acc_signal,
            time_s,
            min_duration_s=config.window_gate.walk_min_duration_s,
            threshold=walk_threshold,
            window_sec=config.window_gate.window_sec,
        )

    out["walk_duration"] = duration
    out["step_count"] = step_count
    out["cadence"] = cadence if np.isfinite(cadence) else (
        60.0 * step_count / duration if np.isfinite(duration) and duration > 0 else np.nan
    )
    out["walking_speed"] = (
        config.walk_distance_m / duration if np.isfinite(duration) and duration > 0 else np.nan
    )

    step_time_mean = np.nan
    active_cad = np.asarray(wavelet.get("cad", []), dtype=float)
    active_cad = active_cad[np.isfinite(active_cad) & (active_cad > 0)]
    if len(active_cad):
        step_times = 1.0 / active_cad
        step_time_mean = float(np.nanmean(step_times))
        step_time_std = float(np.nanstd(step_times))
        out["mean_step_time"] = step_time_mean
        out["step_time_std"] = step_time_std
        out["step_time_cv"] = step_time_std / step_time_mean if step_time_mean > 0 else np.nan

    if np.any(walk_mask):
        acc_for_reg = acc_signal[walk_mask]
    else:
        acc_for_reg = acc_signal
    acf = _autocorrelation(acc_for_reg)
    lag_samples = np.nan
    if np.isfinite(step_time_mean) and step_time_mean > 0 and np.isfinite(meta.fs_hz):
        lag_samples = int(round(step_time_mean * meta.fs_hz))
    elif np.isfinite(meta.fs_hz):
        dom_freq = dominant_frequency(acc_signal, meta.fs_hz)
        if np.isfinite(dom_freq) and dom_freq > 0:
            lag_samples = int(round(meta.fs_hz / dom_freq))

    if np.isfinite(lag_samples):
        lag_samples = int(lag_samples)
        out["step_regularity"] = _regularity_from_acf(acf, lag_samples)
        out["stride_regularity"] = _regularity_from_acf(acf, 2 * lag_samples)

    if np.any(walk_mask):
        acc_stats = acc_signal[walk_mask]
        gyro_stats = df["gyro_mag"].to_numpy(dtype=float)[walk_mask]
    else:
        acc_stats = acc_signal
        gyro_stats = df["gyro_mag"].to_numpy(dtype=float)

    out["walk_acc_mag_mean"] = float(np.nanmean(acc_stats))
    out["walk_acc_mag_std"] = float(np.nanstd(acc_stats))
    out["walk_acc_mag_rms"] = rms(acc_stats)
    out["walk_gyro_mag_std"] = float(np.nanstd(gyro_stats))
    out["walk_dominant_frequency"] = dominant_frequency(acc_stats, meta.fs_hz)
    out["walk_spectral_entropy"] = spectral_entropy(acc_stats, meta.fs_hz)

    jerk_col = "useracc_jerk" if acc_source == "useracc_mag" else "acc_jerk"
    jerk = df[jerk_col].to_numpy(dtype=float)
    if np.any(walk_mask):
        jerk = jerk[walk_mask]
    out["walk_jerk_mean"] = float(np.nanmean(np.abs(jerk)))
    out["walk_jerk_std"] = float(np.nanstd(jerk))

    return out
