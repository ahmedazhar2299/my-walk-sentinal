from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.signal.windows import tukey
from scipy.stats import entropy as scipy_entropy

from .config import PipelineConfig

ACTIVITY_ORDER = (
    "walk",
    "left_turn",
    "right_turn",
    "sit_to_stand",
    "stand_to_sit",
)

ACTIVITY_MAX_DURATION_S = {
    "walk": 31.0,
    "left_turn": 11.0,
    "right_turn": 11.0,
    "sit_to_stand": 17.0,
    "stand_to_sit": 17.0,
}

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


def window_peak_to_peak(time_s, signal, window_sec=1.0):
    time_s = np.asarray(time_s, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if len(time_s) != len(signal) or len(time_s) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    if not np.isfinite(window_sec) or window_sec <= 0:
        window_sec = 1.0

    valid = np.isfinite(time_s) & np.isfinite(signal)
    time_s = time_s[valid]
    signal = signal[valid]
    if len(time_s) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    t0 = float(time_s[0])
    t1 = float(time_s[-1])
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 < t0:
        return np.array([], dtype=float), np.array([], dtype=float)

    n_windows = int(np.floor((t1 - t0) / window_sec)) + 1
    if n_windows <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    starts = []
    pp = []
    for i in range(n_windows):
        start_t = t0 + i * window_sec
        end_t = start_t + window_sec
        if i == n_windows - 1:
            mask = (time_s >= start_t) & (time_s <= end_t)
        else:
            mask = (time_s >= start_t) & (time_s < end_t)
        window = signal[mask]
        starts.append(start_t)
        if len(window) == 0:
            pp.append(np.nan)
        else:
            pp.append(float(np.nanmax(window) - np.nanmin(window)))

    return np.asarray(starts, dtype=float), np.asarray(pp, dtype=float)


def p2p_distribution_threshold(pp, k=1.0, fallback=np.nan):
    pp = np.asarray(pp, dtype=float)
    valid_pp = pp[np.isfinite(pp)]
    if len(valid_pp) == 0:
        return float(fallback) if np.isfinite(fallback) else np.nan

    median_pp = float(np.nanmedian(valid_pp))
    mad_pp = float(np.nanmedian(np.abs(valid_pp - median_pp)))
    threshold = median_pp + float(k) * mad_pp
    threshold = min(threshold, 0.3 * float(np.nanmax(valid_pp)))
    if not np.isfinite(threshold):
        threshold = float(fallback) if np.isfinite(fallback) else np.nan
    return float(threshold)


def robust_p2p_threshold(time_s, signal, window_sec=1.0, k=1.0, fallback=np.nan):
    _, pp = window_peak_to_peak(time_s, signal, window_sec=window_sec)
    threshold = p2p_distribution_threshold(pp, k=k, fallback=fallback)
    return float(threshold), pp


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


def _wavelet_adjust_bout(values, fs_hz):
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or not np.isfinite(fs_hz) or fs_hz <= 0:
        return np.array([])

    rem = len(values) % int(fs_hz)
    if rem >= int(np.ceil(0.7 * fs_hz)):
        pad_n = int(fs_hz) - rem
        if pad_n < fs_hz:
            values = np.append(values, np.repeat(values[-1], pad_n))
    elif rem != 0:
        values = values[: (len(values) // int(fs_hz)) * int(fs_hz)]
    return values


def _wavelet_preprocess_bout(time_s, signal, fs_hz):
    time_s = np.asarray(time_s, dtype=float)
    signal = np.asarray(signal, dtype=float)
    keep = np.isfinite(time_s) & np.isfinite(signal)
    time_s = time_s[keep]
    signal = signal[keep]
    if len(time_s) < 2 or len(signal) < 2 or not np.isfinite(fs_hz) or fs_hz <= 0:
        return np.array([]), np.array([])

    t_rel = time_s - time_s[0]
    t_interp = np.arange(t_rel[0], t_rel[-1], 1.0 / fs_hz)
    if len(t_interp) < 2:
        return np.array([]), np.array([])

    t_interp = t_interp + time_s[0]
    interp_fn = interpolate.interp1d(time_s, signal, bounds_error=False, fill_value="extrapolate")
    signal_interp = interp_fn(t_interp)
    signal_interp = _wavelet_adjust_bout(signal_interp, fs_hz)
    t_interp = t_interp[: len(signal_interp)]
    if len(signal_interp) < int(fs_hz):
        return np.array([]), np.array([])
    return t_interp, signal_interp


def _wavelet_compute_cwt(signal, fs_hz):
    signal = np.asarray(signal, dtype=float)
    if len(signal) < int(fs_hz):
        return np.array([]), np.array([[]])

    try:
        from ssqueezepy import ssq_cwt
    except Exception:
        return np.array([]), np.array([[]])

    window = tukey(len(signal), alpha=0.02, sym=True)
    padded = np.concatenate((np.zeros(5 * int(fs_hz)), signal * window, np.zeros(5 * int(fs_hz))))
    _, Wx, ssq_freqs, *_ = ssq_cwt(
        padded[:-1],
        wavelet=("gmw", {"beta": 90, "gamma": 3}),
        fs=fs_hz,
    )
    coefs = np.abs(Wx.astype("complex128") ** 2)
    coefs = np.append(coefs, coefs[:, -1:], axis=1)

    freqs_interp = np.arange(0.5, 4.5, 0.05)
    coefs_interp = np.vstack(
        [
            np.interp(freqs_interp, ssq_freqs[::-1], coefs[:, j][::-1])
            for j in range(coefs.shape[1])
        ]
    ).T
    coefs_interp = coefs_interp[:, 5 * int(fs_hz) : -5 * int(fs_hz)]
    return freqs_interp, coefs_interp


def _wavelet_identify_peaks(freqs_interp, coefs_interp, fs_hz, step_freq, alpha, beta):
    if coefs_interp.size == 0:
        return np.array([[]])

    num_rows, num_cols = coefs_interp.shape
    num_cols_sec = int(num_cols / int(fs_hz))
    dominant = np.zeros((num_rows, num_cols_sec))
    loc_min = np.argmin(np.abs(freqs_interp - step_freq[0]))
    loc_max = np.argmin(np.abs(freqs_interp - step_freq[1]))

    for i in range(num_cols_sec):
        x_start = i * int(fs_hz)
        x_end = (i + 1) * int(fs_hz)
        window = np.sum(coefs_interp[:, np.arange(x_start, x_end)], axis=1)
        locs = detect_peaks(window, fs_hz=np.nan, min_distance_s=np.nan, height=None)
        locs = np.asarray(locs, dtype=int)
        if len(locs) == 0:
            continue

        pks = window[locs]
        order = np.argsort(-pks)
        locs = locs[order]
        pks = pks[order]

        in_range_idx = None
        for j, loc_j in enumerate(locs):
            if loc_min <= loc_j <= loc_max:
                in_range_idx = j
                break
        if in_range_idx is None:
            continue

        peak_vec = np.zeros(num_rows)
        if locs[0] > loc_max:
            if pks[0] / pks[in_range_idx] < beta:
                peak_vec[locs[in_range_idx]] = 1
        elif locs[0] < loc_min:
            if pks[0] / pks[in_range_idx] < alpha:
                peak_vec[locs[in_range_idx]] = 1
        else:
            peak_vec[locs[in_range_idx]] = 1
        dominant[:, i] = peak_vec
    return dominant


def _wavelet_find_continuous_peaks(valid_peaks, min_t, delta):
    valid_peaks = np.asarray(valid_peaks, dtype=float)
    if valid_peaks.size == 0:
        return valid_peaks

    num_rows, num_cols = valid_peaks.shape
    extended = np.zeros((num_rows, num_cols + 1), dtype=valid_peaks.dtype)
    extended[:, :num_cols] = valid_peaks
    cont_peaks = np.zeros_like(extended)

    for slice_ind in range(num_cols + 1 - min_t):
        slice_mat = extended[:, slice_ind : slice_ind + min_t].copy()
        windows = list(range(min_t)) + list(range(min_t - 2, -1, -1))
        stop = True
        for win_ind in windows:
            present_rows = np.where(slice_mat[:, win_ind] != 0)[0]
            stop = True
            for row_idx in present_rows:
                index = np.arange(max(0, row_idx - delta), min(row_idx + delta + 1, num_rows))
                peaks1 = slice_mat[row_idx, win_ind]
                peaks2 = peaks1
                if win_ind == 0:
                    peaks1 += slice_mat[index, win_ind + 1]
                elif win_ind == min_t - 1:
                    peaks1 += slice_mat[index, win_ind - 1]
                else:
                    peaks1 += slice_mat[index, win_ind - 1]
                    peaks2 += slice_mat[index, win_ind + 1]

                if win_ind == 0 or win_ind == min_t - 1:
                    if np.any(peaks1 > 1):
                        stop = False
                    else:
                        slice_mat[row_idx, win_ind] = 0
                else:
                    if np.any(peaks1 > 1) and np.any(peaks2 > 1):
                        stop = False
                    else:
                        slice_mat[row_idx, win_ind] = 0
            if stop:
                break
        if not stop:
            cont_peaks[:, slice_ind : slice_ind + min_t] += slice_mat

    cont_peaks = np.where(cont_peaks > 0, 1, 0)
    return cont_peaks[:, :num_cols]


def wavelet_step_summary(time_s, signal, wavelet_config, fixed_window=False):
    compare_fs = int(wavelet_config.resample_fs_hz)
    fallback_min_amp = float(
        getattr(
            wavelet_config,
            "walk_min_amp_threshold",
            getattr(wavelet_config, "min_amp_threshold", 0.3),
        )
    )
    threshold_k = float(getattr(wavelet_config, "walk_threshold_k", 1.0))
    min_t = int(wavelet_config.min_active_windows)
    step_freq = (
        float(wavelet_config.step_freq_min_hz),
        float(wavelet_config.step_freq_max_hz),
    )

    t_res, signal_bout = _wavelet_preprocess_bout(time_s, signal, compare_fs)
    if len(signal_bout) < compare_fs:
        return {
            "t_res": np.array([]),
            "signal_bout": np.array([]),
            "cad": np.array([]),
            "dominant_freq_hz": np.array([]),
            "pp": np.array([]),
            "min_amp": fallback_min_amp,
            "active_mask": np.array([], dtype=bool),
            "start_time_s": np.nan,
            "end_time_s": np.nan,
            "duration_s": np.nan,
            "step_count": np.nan,
            "cadence": np.nan,
        }

    pp = np.ptp(signal_bout.reshape((compare_fs, -1), order="F"), axis=0)
    min_amp = p2p_distribution_threshold(pp, k=threshold_k, fallback=fallback_min_amp)
    valid = np.ones(len(pp), dtype=bool)
    valid[pp < min_amp] = False
    cad = np.zeros(len(pp), dtype=float)
    dominant_freq_hz = np.full(len(pp), np.nan, dtype=float)

    if np.sum(valid) >= min_t:
        tapered_bout = signal_bout[np.repeat(valid, compare_fs)]
        freqs_interp, coefs_interp = _wavelet_compute_cwt(tapered_bout, compare_fs)
        if coefs_interp.size:
            dp = _wavelet_identify_peaks(
                freqs_interp,
                coefs_interp,
                compare_fs,
                step_freq,
                wavelet_config.alpha,
                wavelet_config.beta,
            )
            valid_peaks = np.zeros((dp.shape[0], len(valid)))
            valid_peaks[:, valid] = dp
            cont_peaks = _wavelet_find_continuous_peaks(
                valid_peaks,
                min_t=min_t,
                delta=int(wavelet_config.delta),
            )
            for i in range(len(cad)):
                ind_freqs = np.where(cont_peaks[:, i] > 0)[0]
                if len(ind_freqs) > 0:
                    dominant_freq_hz[i] = freqs_interp[ind_freqs[0]]
                    cad[i] = freqs_interp[ind_freqs[0]]

    active_mask = cad > 0
    if fixed_window:
        start_time_s = float(time_s[0]) if len(time_s) else np.nan
        end_time_s = float(time_s[-1]) if len(time_s) else np.nan
        duration_s = float(end_time_s - start_time_s) if np.isfinite(start_time_s) and np.isfinite(end_time_s) else np.nan
        step_count = float(np.nansum(cad[active_mask])) if np.any(active_mask) else 0.0
        cadence = float((step_count / duration_s) * 60.0) if np.isfinite(duration_s) and duration_s > 0 else np.nan
    elif np.any(active_mask):
        first_sec = np.where(active_mask)[0][0]
        last_sec = np.where(active_mask)[0][-1]
        start_time_s = float(t_res[first_sec * compare_fs])
        end_idx = min(len(t_res) - 1, (last_sec + 1) * compare_fs - 1)
        end_time_s = float(t_res[end_idx])
        duration_s = float(end_time_s - start_time_s)
        step_count = float(np.nansum(cad[active_mask]))
        cadence = float(np.nanmean(cad[active_mask]) * 60.0)
    else:
        start_time_s = np.nan
        end_time_s = np.nan
        duration_s = np.nan
        step_count = 0.0
        cadence = np.nan

    return {
        "t_res": t_res,
        "signal_bout": signal_bout,
        "cad": cad,
        "dominant_freq_hz": dominant_freq_hz,
        "pp": pp,
        "min_amp": min_amp,
        "active_mask": active_mask,
        "start_time_s": start_time_s,
        "end_time_s": end_time_s,
        "duration_s": duration_s,
        "step_count": step_count,
        "cadence": cadence,
    }


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


def detect_active_window(signal, time_s, min_duration_s=0.8, threshold=None, window_sec=1.0):
    signal = np.asarray(signal, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    if len(signal) == 0 or len(time_s) == 0:
        return np.nan, np.nan, np.nan, np.zeros(0, dtype=bool), np.nan

    if not np.isfinite(window_sec) or window_sec <= 0:
        window_sec = 1.0

    try:
        min_amp = float(threshold)
    except (TypeError, ValueError):
        min_amp = np.nan
    if not np.isfinite(min_amp):
        min_amp = 0.3
    starts, pp = window_peak_to_peak(time_s, signal, window_sec=window_sec)
    if len(starts) == 0:
        return np.nan, np.nan, np.nan, np.zeros(len(time_s), dtype=bool), min_amp

    active_windows = np.isfinite(pp) & (pp >= min_amp)
    active_idx = np.where(active_windows)[0]
    if len(active_idx) == 0:
        return np.nan, np.nan, np.nan, np.zeros(len(time_s), dtype=bool), min_amp

    w0 = int(active_idx[0])
    w1 = int(active_idx[-1])
    start_t = float(starts[w0])
    end_t = float(min(time_s[-1], starts[w1] + window_sec))
    duration = float(end_t - start_t)
    if duration < min_duration_s:
        return np.nan, np.nan, np.nan, np.zeros(len(time_s), dtype=bool), min_amp

    mask = (time_s >= start_t) & (time_s <= end_t) & np.isfinite(signal)
    return start_t, end_t, duration, mask, min_amp


def detect_turn_window(angular_signal_abs, time_s, threshold, window_sec=1.0):
    start_t, end_t, duration, mask, _ = detect_active_window(
        angular_signal_abs,
        time_s,
        min_duration_s=0.8,
        threshold=threshold,
        window_sec=window_sec,
    )
    return start_t, end_t, duration, mask


def detect_threshold_turn_window(
    angular_signal_abs,
    time_s,
    threshold,
    window_sec=1.0,
    min_duration_s=0.8,
):
    """Detect turn start/end from the same threshold-crossing logic used in notebooks."""
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
            angular_signal_abs,
            time_s,
            min_duration_s=min_duration_s,
            threshold=None,
            window_sec=window_sec,
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
    mask[i0:i1 + 1] = True
    return start_t, end_t, float(end_t - start_t), mask


def _has_usable_motion_signal(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return False
    return bool(np.nanmax(np.abs(finite)) > 1e-8 and np.nanstd(finite) > 1e-8)


def select_motion_acc_signal(df, prefer_useracc=True):
    if prefer_useracc and "useracc_mag" in df and _has_usable_motion_signal(df["useracc_mag"].to_numpy()):
        return df["useracc_mag"].to_numpy(dtype=float), "useracc_mag"
    if "acc_mag_gravity_removed" in df:
        return df["acc_mag_gravity_removed"].to_numpy(dtype=float), "acc_mag_gravity_removed"
    return (df["acc_mag"].to_numpy(dtype=float) - np.nanmean(df["acc_mag"].to_numpy(dtype=float))), "acc_mag_gravity_removed"


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

    df["acc_mag_gravity_removed"] = df["acc_mag"] - float(np.nanmean(df["acc_mag"]))
    df["acc_jerk"] = safe_gradient(df["acc_mag"].to_numpy(dtype=float), time_s)
    df["acc_gravity_removed_jerk"] = safe_gradient(
        df["acc_mag_gravity_removed"].to_numpy(dtype=float),
        time_s,
    )
    df["useracc_jerk"] = safe_gradient(df["useracc_mag"].to_numpy(dtype=float), time_s)
    df["gyro_jerk"] = safe_gradient(df["gyro_mag"].to_numpy(dtype=float), time_s)

    meta = SignalMeta(dt_s=dt_s, fs_hz=fs_hz, duration_s=duration_s, n_samples=len(df))
    return df, meta


def read_and_preprocess_csv(csv_path, config):
    df_raw = pd.read_csv(csv_path)
    return preprocess_activity_dataframe(df_raw, config)


def read_preprocess_activity_csv(csv_path, config, activity):
    """Read, preprocess, and select the fixed activity segment for one CSV."""
    df, meta = read_and_preprocess_csv(csv_path, config)
    df, meta = truncate_activity_dataframe(df, meta, activity)
    df.attrs["activity"] = activity
    df.attrs["source_csv"] = str(csv_path)
    df.attrs["selected_segment_rule"] = f"last_{ACTIVITY_MAX_DURATION_S.get(activity, 'full')}_seconds_if_longer"
    return df, meta


def _robust_scale(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.nan, np.nan
    center = float(np.nanmedian(finite))
    mad = float(np.nanmedian(np.abs(finite - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale == 0:
        scale = float(np.nanstd(finite))
    return center, scale


def _activity_score(df):
    score_parts = []
    for col in ("useracc_mag", "acc_mag", "gyro_mag"):
        if col not in df:
            continue
        values = df[col].to_numpy(dtype=float)
        center, scale = _robust_scale(values)
        if not np.isfinite(scale) or scale == 0:
            continue
        score_parts.append(np.abs(values - center) / scale)
    if not score_parts:
        return np.full(len(df), np.nan, dtype=float)
    return np.nanmax(np.vstack(score_parts), axis=0)


def _recompute_meta_and_derived(df):
    df = df.reset_index(drop=True).copy()
    if df.empty:
        return df, SignalMeta(dt_s=np.nan, fs_hz=np.nan, duration_s=np.nan, n_samples=0)

    if "timestamp_s" in df:
        df["timestamp_s"] = df["timestamp_s"] - df["timestamp_s"].iloc[0]
    df["time_s"] = df["time_s"] - df["time_s"].iloc[0]
    time_s = df["time_s"].to_numpy(dtype=float)
    dt_s = estimate_sampling_interval_s(time_s)
    fs_hz = float(1.0 / dt_s) if np.isfinite(dt_s) and dt_s > 0 else np.nan
    duration_s = float(time_s[-1] - time_s[0]) if len(time_s) > 1 else 0.0

    if "acc_mag" in df:
        df["acc_mag_gravity_removed"] = df["acc_mag"] - float(np.nanmean(df["acc_mag"]))
        df["acc_jerk"] = safe_gradient(df["acc_mag"].to_numpy(dtype=float), time_s)
    if "acc_mag_gravity_removed" in df:
        df["acc_gravity_removed_jerk"] = safe_gradient(
            df["acc_mag_gravity_removed"].to_numpy(dtype=float),
            time_s,
        )
    if "useracc_mag" in df:
        df["useracc_jerk"] = safe_gradient(df["useracc_mag"].to_numpy(dtype=float), time_s)
    if "gyro_mag" in df:
        df["gyro_jerk"] = safe_gradient(df["gyro_mag"].to_numpy(dtype=float), time_s)

    meta = SignalMeta(dt_s=dt_s, fs_hz=fs_hz, duration_s=duration_s, n_samples=len(df))
    return df, meta


def truncate_to_last_activity_window(df, meta, max_duration_s, truncate_after_s=None):
    """Keep the last fixed-duration slice when recordings run too long."""
    if df is None or meta is None or df.empty:
        return df, meta
    if not np.isfinite(max_duration_s) or max_duration_s <= 0:
        return df, meta
    if "time_s" not in df:
        return df, meta

    time_s = df["time_s"].to_numpy(dtype=float)
    if len(time_s) < 2 or not np.isfinite(time_s).any():
        return df, meta

    total_duration = float(np.nanmax(time_s) - np.nanmin(time_s))
    truncate_after_s = float(truncate_after_s) if truncate_after_s is not None else float(max_duration_s)
    if total_duration <= truncate_after_s:
        return df, meta

    end_t = float(np.nanmax(time_s))
    start_t = max(float(np.nanmin(time_s)), end_t - float(max_duration_s))
    keep = (time_s >= start_t) & (time_s <= end_t)
    if int(np.sum(keep)) < 2:
        keep = time_s >= (float(np.nanmax(time_s)) - float(max_duration_s))

    truncated = df.loc[keep].copy()
    return _recompute_meta_and_derived(truncated)


def truncate_activity_dataframe(df, meta, activity):
    max_duration_s = ACTIVITY_MAX_DURATION_S.get(activity)
    if max_duration_s is None:
        return df, meta
    return truncate_to_last_activity_window(df, meta, max_duration_s=max_duration_s)


def resolve_activity_files(date_dir):
    """Resolve activity files from flat or nested patient/date folders."""
    aliases = {
        "walk": ("walk", "10_mw"),
        "left_turn": ("left_turn", "360_leftturn", "turnl", "turn_l"),
        "right_turn": ("right_turn", "360_rightturn", "turnr", "turn_r"),
        "sit_to_stand": ("sit_to_stand", "sit_stand", "sittostand", "sitstand"),
        "stand_to_sit": ("stand_to_sit", "stand_sit", "standtosit", "standsit"),
    }

    def normalize(value):
        text = str(value).lower()
        chars = [ch if ch.isalnum() else "_" for ch in text]
        normalized = "_".join("".join(chars).split("_"))
        return f"_{normalized}_"

    def alias_matches(normalized_value, alias):
        alias = str(alias).lower()
        return f"_{alias}_" in normalized_value or f"_{alias}" in normalized_value

    csv_files = sorted(Path(date_dir).rglob("*.csv"))
    resolved = {}
    for activity in ACTIVITY_ORDER:
        expected_path = Path(date_dir) / f"{activity}.csv"
        if expected_path.is_file():
            resolved[activity] = expected_path
            continue

        best = None
        best_score = -1
        for path in csv_files:
            normalized_name = normalize(path.stem)
            normalized_rel = normalize(path.relative_to(date_dir))
            if not any(alias_matches(normalized_rel, alias) for alias in aliases.get(activity, (activity,))):
                continue

            score = 0
            if "synchronized" in normalized_rel:
                score += 20
            if any(alias_matches(normalized_name, alias) for alias in aliases.get(activity, (activity,))):
                score += 10
            if activity in normalized_name:
                score += 5
            if score > best_score:
                best = path
                best_score = score
        resolved[activity] = best
    return resolved
