from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.stats import entropy as scipy_entropy

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
    """Estimate dominant frequency from FFT power peak (excluding 0 Hz)."""
    values = np.asarray(values, dtype=float)
    if len(values) < 8 or not np.isfinite(fs_hz) or fs_hz <= 0:
        return np.nan
    centered = values - np.nanmean(values)
    centered = centered[np.isfinite(centered)]
    if len(centered) < 8:
        return np.nan

    # FFT-based dominant frequency (ignoring DC at 0 Hz).
    n = len(centered)
    window = np.hanning(n)
    spectrum = np.fft.rfft(centered * window)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    power = np.abs(spectrum) ** 2

    valid = np.isfinite(freqs) & np.isfinite(power) & (freqs > 0)
    if valid.sum() == 0:
        return np.nan
    idx = np.argmax(power[valid])
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
    # Normalized Shannon entropy in [0, 1] using SciPy.
    ent = scipy_entropy(prob, base=2)
    return float(ent / np.log2(len(prob)))


def detect_peaks(signal, fs_hz, min_distance_s, height=None):
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
    if height is not None:
        kwargs["height"] = height

    peaks, _ = find_peaks(filled, **kwargs)
    return peaks.astype(int)


def clamp_threshold(value, min_value):
    if not np.isfinite(value):
        return min_value
    return float(max(value, min_value))


def quiet_window_stats(signal, fs_hz, window_sec=1.0, min_samples=10):
    values = np.asarray(signal, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < max(3, min_samples):
        return np.nan, np.nan

    x_axis = np.arange(len(values), dtype=float)
    filled = _fill_nans_linear(values, x_axis)
    if np.isnan(filled).all():
        return np.nan, np.nan

    if np.isfinite(fs_hz) and fs_hz > 0 and np.isfinite(window_sec) and window_sec > 0:
        window_n = int(round(window_sec * fs_hz))
    else:
        window_n = min_samples
    window_n = max(min_samples, window_n)
    window_n = min(window_n, len(filled))
    if window_n < 3:
        return np.nan, np.nan

    best_spread = np.inf
    best_slice = None
    for start in range(0, len(filled) - window_n + 1):
        window = filled[start : start + window_n]
        window_median = float(np.nanmedian(window))
        window_mad = float(np.nanmedian(np.abs(window - window_median)))
        if np.isfinite(window_mad) and window_mad < best_spread:
            best_spread = window_mad
            best_slice = window

    if best_slice is None:
        return np.nan, np.nan
    baseline_mean = float(np.nanmean(best_slice))
    baseline_median = float(np.nanmedian(best_slice))
    spread_mad = float(np.nanmedian(np.abs(best_slice - baseline_median)))
    return baseline_mean, spread_mad


def adaptive_amplitude_threshold(signal, fs_hz, config, k_value, min_value):
    baseline_mean, spread_mad = quiet_window_stats(
        signal=signal,
        fs_hz=fs_hz,
        window_sec=config.quiet_window_sec,
        min_samples=config.min_window_samples,
    )
    if not np.isfinite(baseline_mean) or not np.isfinite(spread_mad) or spread_mad <= 0:
        return min_value, baseline_mean, spread_mad
    return (
        clamp_threshold(baseline_mean + k_value * spread_mad, min_value),
        baseline_mean,
        spread_mad,
    )


def adaptive_step_min_distance(time_s, candidate_peaks, alpha):
    time_s = np.asarray(time_s, dtype=float)
    candidate_peaks = np.asarray(candidate_peaks, dtype=int)
    if len(candidate_peaks) < 2:
        return np.nan

    dt = np.diff(time_s[candidate_peaks])
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return np.nan

    return float(alpha * np.nanmedian(dt))


def pause_segment_durations(angular_velocity, time_s, threshold):
    values = np.asarray(angular_velocity, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    if len(values) != len(time_s) or len(values) < 2:
        return np.array([], dtype=float)
    if not np.isfinite(values).any() or not np.isfinite(time_s).all():
        return np.array([], dtype=float)

    pause_mask = np.abs(values) < threshold
    if not pause_mask.any():
        return np.array([], dtype=float)

    starts = np.where(~pause_mask[:-1] & pause_mask[1:])[0] + 1
    ends = np.where(pause_mask[:-1] & ~pause_mask[1:])[0] + 1
    if pause_mask[0]:
        starts = np.insert(starts, 0, 0)
    if pause_mask[-1]:
        ends = np.append(ends, len(pause_mask))

    durations = []
    for start, end in zip(starts, ends):
        if end - start < 2:
            continue
        duration = float(time_s[end - 1] - time_s[start])
        if np.isfinite(duration) and duration > 0:
            durations.append(duration)
    return np.asarray(durations, dtype=float)


def adaptive_pause_min_duration(angular_velocity, time_s, threshold, beta):
    durations = pause_segment_durations(angular_velocity, time_s, threshold)
    if len(durations) == 0:
        return 0.0
    return float(beta * np.nanmedian(durations))


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


def mask_close_gaps(mask, max_gap_samples):
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0 or max_gap_samples <= 0:
        return mask

    out = mask.copy()
    i = 0
    n = len(out)
    while i < n:
        if out[i]:
            i += 1
            continue
        j = i
        while j < n and not out[j]:
            j += 1
        gap_len = j - i
        left_on = i > 0 and out[i - 1]
        right_on = j < n and out[j]
        if left_on and right_on and gap_len <= max_gap_samples:
            out[i:j] = True
        i = j
    return out


def largest_true_segment(mask):
    mask = np.asarray(mask, dtype=bool)
    best = None
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if best is None or (j - i) > (best[1] - best[0]):
            best = (i, j - 1)
        i = j
    return best


def detect_active_window(signal, time_s, min_duration_s=0.8, threshold=None):
    signal = np.asarray(signal, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    if len(signal) == 0 or len(time_s) == 0:
        return np.nan, np.nan, np.nan, np.zeros(0, dtype=bool), np.nan

    fs_hz = 1.0 / estimate_sampling_interval_s(time_s) if len(time_s) > 2 else np.nan
    baseline = np.nanmedian(signal)
    motion = np.abs(signal - baseline)

    if threshold is None:
        if np.isfinite(motion).any():
            q70 = np.nanpercentile(motion, 70)
            q90 = np.nanpercentile(motion, 90)
            threshold = q70 + 0.15 * (q90 - q70) if np.isfinite(q70) and np.isfinite(q90) else np.nan
        else:
            threshold = np.nan

    if not np.isfinite(threshold):
        return np.nan, np.nan, np.nan, np.zeros(len(time_s), dtype=bool), np.nan

    active = np.isfinite(motion) & (motion >= threshold)
    if np.isfinite(fs_hz) and fs_hz > 0:
        active = mask_close_gaps(active, max_gap_samples=max(1, int(0.25 * fs_hz)))

    seg = largest_true_segment(active)
    if seg is None:
        return np.nan, np.nan, np.nan, np.zeros(len(time_s), dtype=bool), threshold

    i0, i1 = seg
    start_t = float(time_s[i0])
    end_t = float(time_s[i1])
    duration = float(end_t - start_t)
    if duration < min_duration_s:
        return np.nan, np.nan, np.nan, np.zeros(len(time_s), dtype=bool), threshold

    mask = np.zeros(len(time_s), dtype=bool)
    mask[i0 : i1 + 1] = True
    return start_t, end_t, duration, mask, threshold


def detect_walk_window_from_peaks(acc_signal, time_s, peaks):
    acc_signal = np.asarray(acc_signal, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    peaks = np.asarray(peaks, dtype=int)
    if len(time_s) == 0:
        return np.nan, np.nan, np.nan, np.zeros(0, dtype=bool)

    motion_start, motion_end, _, _, _ = detect_active_window(acc_signal, time_s, min_duration_s=1.0)

    if len(peaks) >= 2:
        step_dt = np.diff(time_s[peaks])
        pad = float(np.nanmedian(step_dt) * 0.20) if len(step_dt) else 0.12
        pad = pad if np.isfinite(pad) and pad > 0 else 0.12
        start_t = max(float(time_s[0]), float(time_s[peaks[0]] - pad))
        end_t = min(float(time_s[-1]), float(time_s[peaks[-1]] + pad))
        if np.isfinite(motion_start):
            start_t = min(start_t, float(motion_start))
        if np.isfinite(motion_end):
            end_t = max(end_t, float(motion_end))
        start_t = max(float(time_s[0]), start_t)
        end_t = min(float(time_s[-1]), end_t)
        if end_t > start_t:
            mask = (time_s >= start_t) & (time_s <= end_t)
            return start_t, end_t, float(end_t - start_t), mask

    start_t, end_t, duration, mask, _ = detect_active_window(acc_signal, time_s, min_duration_s=1.0)
    return start_t, end_t, duration, mask


def detect_turn_window(angular_signal_abs, time_s, threshold):
    angular_signal_abs = np.asarray(angular_signal_abs, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    if len(time_s) == 0:
        return np.nan, np.nan, np.nan, np.zeros(0, dtype=bool)

    fs_hz = 1.0 / estimate_sampling_interval_s(time_s) if len(time_s) > 2 else np.nan
    active = np.isfinite(angular_signal_abs) & (angular_signal_abs >= threshold)
    if np.isfinite(fs_hz) and fs_hz > 0:
        active = mask_close_gaps(active, max_gap_samples=max(1, int(0.60 * fs_hz)))

    min_run = max(1, int(0.25 * fs_hz)) if np.isfinite(fs_hz) and fs_hz > 0 else 1
    cleaned = np.zeros(len(active), dtype=bool)
    i = 0
    while i < len(active):
        if not active[i]:
            i += 1
            continue
        j = i
        while j < len(active) and active[j]:
            j += 1
        if (j - i) >= min_run:
            cleaned[i:j] = True
        i = j
    active = cleaned

    if not np.any(active):
        start_t, end_t, duration, mask, _ = detect_active_window(
            angular_signal_abs, time_s, min_duration_s=0.8
        )
        return start_t, end_t, duration, mask

    idx = np.where(active)[0]
    i0 = int(idx[0])
    i1 = int(idx[-1])
    pad = int(0.15 * fs_hz) if np.isfinite(fs_hz) and fs_hz > 0 else 1
    i0 = max(0, i0 - pad)
    i1 = min(len(time_s) - 1, i1 + pad)

    start_t = float(time_s[i0])
    end_t = float(time_s[i1])
    mask = np.zeros(len(time_s), dtype=bool)
    mask[i0 : i1 + 1] = True
    return start_t, end_t, float(end_t - start_t), mask


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
