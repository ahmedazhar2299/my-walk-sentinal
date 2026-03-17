from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, welch

from .config import PipelineConfig

ACTIVITY_ORDER = (
    "walk",
    "left_turn",
    "right_turn",
    "sit_to_stand",
    "stand_to_sit",
)

CANONICAL_COLUMNS = (
    "timestamp",
    "accel_x",
    "accel_y",
    "accel_z",
    "useracc_x",
    "useracc_y",
    "useracc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
)


@dataclass
class SignalMeta:
    dt_s: float
    fs_hz: float
    duration_s: float
    n_samples: int


def build_nan_feature_dict(feature_names):
    return {name: np.nan for name in feature_names}


def resolve_column_name(columns, candidates):
    """Match candidate column names case-insensitively."""
    lookup = {str(col).strip().lower(): col for col in columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return lookup[key]
    return None


def standardize_sensor_columns(df, column_candidates):
    """Return dataframe with canonical sensor column names."""
    standardized = pd.DataFrame(index=df.index)
    for canonical in CANONICAL_COLUMNS:
        source_name = resolve_column_name(df.columns, column_candidates.get(canonical, []))
        if source_name is None:
            standardized[canonical] = np.nan
        else:
            standardized[canonical] = pd.to_numeric(df[source_name], errors="coerce")
    return standardized


def timestamp_to_seconds(series):
    """Convert numeric or datetime-like timestamps to seconds."""
    ts_numeric = pd.to_numeric(series, errors="coerce")
    numeric_ratio = float(ts_numeric.notna().mean()) if len(ts_numeric) else 0.0
    if numeric_ratio >= 0.8:
        diffs = np.diff(ts_numeric.to_numpy(dtype=float))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        median_diff = float(np.median(diffs)) if len(diffs) else np.nan
        scale = 1.0
        if np.isfinite(median_diff):
            if median_diff > 1e8:
                scale = 1e9  # nanoseconds
            elif median_diff > 1e5:
                scale = 1e6  # microseconds
            elif median_diff > 1.0:
                scale = 1e3  # milliseconds
        return ts_numeric / scale

    dt = pd.to_datetime(series, errors="coerce", utc=True)
    dt_ns = dt.astype("int64").astype(float)
    dt_ns[dt.isna()] = np.nan
    return dt_ns / 1e9


def estimate_sampling_interval_s(time_s):
    diffs = np.diff(time_s)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return np.nan
    return float(np.median(diffs))


def _fill_nans_linear(values, x_axis):
    if len(values) == 0:
        return values
    valid = np.isfinite(values)
    if valid.sum() == 0:
        return values.copy()
    if valid.sum() == len(values):
        return values.copy()
    filled = values.copy()
    filled[~valid] = np.interp(x_axis[~valid], x_axis[valid], values[valid])
    return filled


def apply_lowpass_filter(signal, fs_hz, cutoff_hz, order):
    """Apply Butterworth low-pass filter; returns original signal when unsafe."""
    values = np.asarray(signal, dtype=float)
    if len(values) < 5 or not np.isfinite(fs_hz) or fs_hz <= 0:
        return values
    if not np.isfinite(cutoff_hz) or cutoff_hz <= 0:
        return values

    nyquist = 0.5 * fs_hz
    if cutoff_hz >= nyquist * 0.95:
        return values

    x_axis = np.arange(len(values), dtype=float)
    filled = _fill_nans_linear(values, x_axis)
    if np.isnan(filled).all():
        return values

    b, a = butter(order, cutoff_hz / nyquist, btype="low", analog=False)
    padlen = 3 * (max(len(a), len(b)) - 1)
    if len(filled) <= padlen:
        return values

    try:
        filtered = filtfilt(b, a, filled)
    except ValueError:
        return values

    filtered[~np.isfinite(values)] = np.nan
    return filtered


def safe_gradient(values, time_s):
    """Compute derivative d(values)/dt for irregular sampling."""
    values = np.asarray(values, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    if len(values) != len(time_s) or len(values) < 2:
        return np.full(len(values), np.nan, dtype=float)
    if not np.isfinite(time_s).all():
        return np.full(len(values), np.nan, dtype=float)
    if np.all(np.diff(time_s) <= 0):
        return np.full(len(values), np.nan, dtype=float)

    valid = np.isfinite(values)
    if valid.sum() < 2:
        return np.full(len(values), np.nan, dtype=float)

    filled = _fill_nans_linear(values, time_s)
    try:
        grad = np.gradient(filled, time_s)
    except ValueError:
        dt = estimate_sampling_interval_s(time_s)
        if not np.isfinite(dt) or dt <= 0:
            return np.full(len(values), np.nan, dtype=float)
        grad = np.gradient(filled, dt)
    grad = np.asarray(grad, dtype=float)
    grad[~valid] = np.nan
    return grad


def rms(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan
    return float(np.sqrt(np.nanmean(values**2)))


def dominant_frequency(values, fs_hz):
    values = np.asarray(values, dtype=float)
    if len(values) < 8 or not np.isfinite(fs_hz) or fs_hz <= 0:
        return np.nan
    centered = values - np.nanmean(values)
    centered = centered[np.isfinite(centered)]
    if len(centered) < 8:
        return np.nan

    freqs, psd = welch(centered, fs=fs_hz, nperseg=min(256, len(centered)))
    valid = np.isfinite(freqs) & np.isfinite(psd) & (freqs > 0)
    if valid.sum() == 0:
        return np.nan
    idx = np.argmax(psd[valid])
    return float(freqs[valid][idx])


def spectral_entropy(values, fs_hz):
    values = np.asarray(values, dtype=float)
    if len(values) < 8 or not np.isfinite(fs_hz) or fs_hz <= 0:
        return np.nan
    centered = values - np.nanmean(values)
    centered = centered[np.isfinite(centered)]
    if len(centered) < 8:
        return np.nan

    _, psd = welch(centered, fs=fs_hz, nperseg=min(256, len(centered)))
    psd = psd[np.isfinite(psd) & (psd >= 0)]
    if len(psd) < 2:
        return np.nan
    total = float(np.sum(psd))
    if total <= 0:
        return np.nan

    prob = psd / total
    entropy = -np.sum(prob * np.log2(prob + 1e-12))
    return float(entropy / np.log2(len(prob)))


def detect_peaks(signal, fs_hz, min_distance_s, prominence, height=None):
    values = np.asarray(signal, dtype=float)
    if len(values) < 3:
        return np.array([], dtype=int)
    x_axis = np.arange(len(values), dtype=float)
    filled = _fill_nans_linear(values, x_axis)
    if np.isnan(filled).all():
        return np.array([], dtype=int)

    kwargs = {}
    if np.isfinite(fs_hz) and fs_hz > 0 and np.isfinite(min_distance_s):
        kwargs["distance"] = max(1, int(round(min_distance_s * fs_hz)))
    if prominence is not None:
        kwargs["prominence"] = prominence
    if height is not None:
        kwargs["height"] = height

    peaks, _ = find_peaks(filled, **kwargs)
    return peaks.astype(int)


def count_pauses(angular_velocity, time_s, threshold, min_duration_s):
    values = np.asarray(angular_velocity, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    if len(values) != len(time_s) or len(values) < 2:
        return np.nan, np.nan
    if not np.isfinite(values).any() or not np.isfinite(time_s).all():
        return np.nan, np.nan

    abs_values = np.abs(values)
    pause_mask = abs_values < threshold
    if not pause_mask.any():
        return 0.0, 0.0

    starts = np.where(~pause_mask[:-1] & pause_mask[1:])[0] + 1
    ends = np.where(pause_mask[:-1] & ~pause_mask[1:])[0] + 1
    if pause_mask[0]:
        starts = np.insert(starts, 0, 0)
    if pause_mask[-1]:
        ends = np.append(ends, len(pause_mask))

    pause_count = 0.0
    total_pause_time = 0.0
    for start, end in zip(starts, ends):
        if end - start < 2:
            continue
        duration = float(time_s[end - 1] - time_s[start])
        if duration >= min_duration_s:
            pause_count += 1.0
            total_pause_time += duration
    return pause_count, total_pause_time


def select_motion_acc_signal(df, prefer_useracc=True):
    if prefer_useracc and "useracc_mag" in df and not np.all(np.isnan(df["useracc_mag"].to_numpy())):
        return df["useracc_mag"].to_numpy(dtype=float), "useracc_mag"
    return df["acc_mag"].to_numpy(dtype=float), "acc_mag"


def select_turn_angular_signal(df, config, fs_hz):
    axis_candidates = ["gyro_x", "gyro_y", "gyro_z"]
    dominant_axis = "gyro_mag"
    best_score = -np.inf
    for axis in axis_candidates:
        if axis not in df:
            continue
        values = df[axis].to_numpy(dtype=float)
        score = np.nanmean(np.abs(values))
        if np.isfinite(score) and score > best_score:
            best_score = score
            dominant_axis = axis

    if dominant_axis == "gyro_mag" or dominant_axis not in df:
        signal = df["gyro_mag"].to_numpy(dtype=float)
    else:
        signal = np.abs(df[dominant_axis].to_numpy(dtype=float))

    if config.filtering.enabled:
        signal = apply_lowpass_filter(
            signal,
            fs_hz=fs_hz,
            cutoff_hz=config.filtering.cutoff_hz,
            order=config.filtering.order,
        )
    return signal, dominant_axis


def preprocess_activity_dataframe(df_raw, config):
    """Shared preprocessing used by all activities."""
    df = standardize_sensor_columns(df_raw, config.column_candidates)
    df["timestamp_s"] = timestamp_to_seconds(df["timestamp"])
    df = df.dropna(subset=["timestamp_s"]).copy()
    df = df.sort_values("timestamp_s", kind="mergesort")
    if df.empty:
        meta = SignalMeta(dt_s=np.nan, fs_hz=np.nan, duration_s=np.nan, n_samples=0)
        return df, meta

    df = df.groupby("timestamp_s", as_index=False).mean(numeric_only=True)
    df = df.sort_values("timestamp_s", kind="mergesort").reset_index(drop=True)

    for col in CANONICAL_COLUMNS[1:]:
        if col not in df:
            df[col] = np.nan
        if df[col].notna().any():
            df[col] = df[col].interpolate(method="linear", limit_direction="both")
            df[col] = df[col].ffill().bfill()

    df["time_s"] = df["timestamp_s"] - df["timestamp_s"].iloc[0]
    time_s = df["time_s"].to_numpy(dtype=float)
    dt_s = estimate_sampling_interval_s(time_s)
    fs_hz = float(1.0 / dt_s) if np.isfinite(dt_s) and dt_s > 0 else np.nan
    duration_s = float(time_s[-1] - time_s[0]) if len(time_s) > 1 else 0.0

    df["acc_mag"] = np.sqrt(df["accel_x"] ** 2 + df["accel_y"] ** 2 + df["accel_z"] ** 2)
    df["useracc_mag"] = np.sqrt(df["useracc_x"] ** 2 + df["useracc_y"] ** 2 + df["useracc_z"] ** 2)
    df["gyro_mag"] = np.sqrt(df["gyro_x"] ** 2 + df["gyro_y"] ** 2 + df["gyro_z"] ** 2)

    if config.filtering.enabled and np.isfinite(fs_hz):
        for mag_col in ("acc_mag", "useracc_mag", "gyro_mag"):
            df[mag_col] = apply_lowpass_filter(
                df[mag_col].to_numpy(dtype=float),
                fs_hz=fs_hz,
                cutoff_hz=config.filtering.cutoff_hz,
                order=config.filtering.order,
            )

    df["acc_jerk"] = safe_gradient(df["acc_mag"].to_numpy(dtype=float), time_s)
    df["useracc_jerk"] = safe_gradient(df["useracc_mag"].to_numpy(dtype=float), time_s)
    df["gyro_jerk"] = safe_gradient(df["gyro_mag"].to_numpy(dtype=float), time_s)

    meta = SignalMeta(dt_s=dt_s, fs_hz=fs_hz, duration_s=duration_s, n_samples=len(df))
    return df, meta


def read_and_preprocess_csv(csv_path, config):
    df_raw = pd.read_csv(csv_path)
    return preprocess_activity_dataframe(df_raw, config)


def resolve_activity_files(date_dir):
    """Resolve activity files using strict exact filenames only."""
    resolved = {}
    for activity in ACTIVITY_ORDER:
        expected_path = date_dir / f"{activity}.csv"
        resolved[activity] = expected_path if expected_path.is_file() else None
    return resolved


def plot_debug_signal(
    csv_path,
    config,
    signal_name="motion_acc",
    peak_mode=None,
    output_path=None,
):
    """
    Plot a debug signal and optionally annotate detected peaks.

    peak_mode options:
    - "steps": uses step detection settings
    - "transition": uses transition peak settings
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for debug plotting.") from exc

    df, meta = read_and_preprocess_csv(Path(csv_path), config)
    if df.empty:
        raise ValueError(f"No valid rows after preprocessing: {csv_path}")

    if signal_name == "motion_acc":
        values, chosen = select_motion_acc_signal(df, config.prefer_useracc_for_motion)
        title_signal = f"motion_acc ({chosen})"
    else:
        if signal_name not in df.columns:
            raise ValueError(f"Unknown signal '{signal_name}'. Available: {sorted(df.columns)}")
        values = df[signal_name].to_numpy(dtype=float)
        title_signal = signal_name

    time_s = df["time_s"].to_numpy(dtype=float)
    peaks = np.array([], dtype=int)
    if peak_mode == "steps":
        peaks = detect_peaks(
            values,
            fs_hz=meta.fs_hz,
            min_distance_s=config.step_detection.min_distance_s,
            prominence=config.step_detection.prominence,
            height=config.step_detection.height,
        )
    elif peak_mode == "transition":
        peaks = detect_peaks(
            values,
            fs_hz=meta.fs_hz,
            min_distance_s=config.transition_peaks.min_distance_s,
            prominence=config.transition_peaks.prominence,
            height=None,
        )

    plt.figure(figsize=(12, 4))
    plt.plot(time_s, values, linewidth=1.3, label=title_signal)
    if len(peaks):
        plt.scatter(time_s[peaks], values[peaks], c="red", s=20, label=f"peaks={len(peaks)}")
    plt.xlabel("Time (s)")
    plt.ylabel("Signal")
    plt.title(Path(csv_path).name)
    plt.legend(loc="best")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()

    return peaks
