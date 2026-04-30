from pathlib import Path
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.signal.windows import tukey
from ssqueezepy import ssq_cwt

from imu_features.config import PipelineConfig
from imu_features.transition_features import transition_flexion_extension_peaks
from imu_features.utils import (
    count_pauses,
    detect_active_window as core_detect_active_window,
    detect_peaks,
    estimate_sampling_interval_s,
    robust_p2p_threshold,
    p2p_distribution_threshold,
    window_peak_to_peak,
    preprocess_activity_dataframe,
    safe_gradient,
    select_motion_acc_signal,
    select_turn_angular_signal,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


def default_plot_params():
    return {
        "step_min_distance_alpha": 0.5,
        "pause_min_duration_beta": 0.6,
        "plot_walk_histogram": True,
        "walk_variance_window_sec": 0.50,
        "walk_compare_fs_hz": 10,
        "walk_compare_min_amp": 0.30,
        "walk_compare_threshold_k": 1.0,
        "walk_compare_min_t_sec": 2.0,
        "walk_compare_step_freq_hz": (0.8, 2.3),
        "walk_compare_window_sec": 1.0,
        "turn_compare_fs_hz": 10,
        "turn_compare_min_amp": 0.20,
        "turn_compare_threshold_k": 1.0,
        "turn_compare_min_t_sec": 3.0,
        "turn_compare_step_freq_hz": (0.8, 2.3),
        "turn_compare_window_sec": 1.0,
        "window_gate": {
            "window_sec": 1.0,
            "walk_min_amp_threshold": 0.30,
            "walk_threshold_k": 1.0,
            "turn_min_amp_threshold": 0.10,
            "transition_min_amp_threshold": 0.30,
            "transition_threshold_k": 1.0,
        },
    }


PIPELINE_CONFIG = PipelineConfig()
PLOT_PARAMS = default_plot_params()


def configure(pipeline_config=None, plot_params=None):
    global PIPELINE_CONFIG, PLOT_PARAMS
    if pipeline_config is not None:
        PIPELINE_CONFIG = pipeline_config
    if plot_params is not None:
        PLOT_PARAMS = plot_params
    return PIPELINE_CONFIG, PLOT_PARAMS


ACTIVITY_ALIASES = {
    "walk": ["walk", "10_mw", "10mw", "1m", "1 meter"],
    "left_turn": ["left_turn", "left turn", "l_360", "360 left", "left"],
    "right_turn": ["right_turn", "right turn", "r_360", "360 right", "right"],
    "sit_to_stand": ["sit_to_stand", "sit to stand", "sit_stand", "sitstand"],
    "stand_to_sit": ["stand_to_sit", "stand to sit", "stand_sit", "standsit"],
}


def _norm_text(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def resolve_date_dir(data_root, patient_id, date):
    patient_dir = Path(data_root) / patient_id
    if not patient_dir.exists():
        raise FileNotFoundError(f"Patient folder not found: {patient_dir}")

    if date is None:
        date_dirs = sorted([d for d in patient_dir.iterdir() if d.is_dir()])
        if not date_dirs:
            raise FileNotFoundError(f"No date folders under: {patient_dir}")
        return date_dirs[-1]

    direct = patient_dir / str(date)
    if direct.exists():
        return direct

    candidates = [d for d in patient_dir.rglob("*") if d.is_dir() and d.name == str(date)]
    if candidates:
        return sorted(candidates, key=lambda p: len(p.parts))[0]

    raise FileNotFoundError(f"Date folder not found: {date}")


def discover_activity_files(base_dir, activity_filter=None):
    # Recursive discovery with synchronized-file preference
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return {}

    csv_paths = sorted([p for p in base_dir.rglob("*.csv") if p.is_file()])
    discovered = {}

    for activity, aliases in ACTIVITY_ALIASES.items():
        if activity_filter and activity not in set(activity_filter):
            continue

        candidates = []
        for p in csv_paths:
            probe = _norm_text(p.stem + " " + p.parent.name + " " + str(p))
            alias_rank = None
            for rank, alias in enumerate(aliases):
                if _norm_text(alias) in probe:
                    alias_rank = rank
                    break
            if alias_rank is None:
                continue

            sync_rank = 0 if "synchronized" in p.name.lower() else 1
            score = (alias_rank, sync_rank, len(p.parts), len(p.name), p.name.lower())
            candidates.append((score, p))

        if candidates:
            discovered[activity] = sorted(candidates, key=lambda x: x[0])[0][1]

    return discovered


def load_activity_file(path):
    raw = pd.read_csv(path)
    return preprocess_activity_dataframe(raw, PIPELINE_CONFIG)


def standardize_columns(df):
    return standardize_sensor_columns(df, PIPELINE_CONFIG.column_candidates)


def compute_time_seconds(df):
    ts = pd.Series(timestamp_to_seconds(df["timestamp"]), dtype=float)
    return ts - ts.dropna().iloc[0] if ts.notna().any() else ts


def compute_acc_magnitude(df):
    if "useracc_mag" in df and np.isfinite(df["useracc_mag"]).any():
        return df["useracc_mag"].to_numpy(dtype=float), "useracc_mag"
    if all(c in df for c in ["accel_x", "accel_y", "accel_z"]):
        m = np.sqrt(df["accel_x"]**2 + df["accel_y"]**2 + df["accel_z"]**2)
        return m.to_numpy(dtype=float), "acc_mag"
    return np.full(len(df), np.nan), "acc_mag"


def compute_gyro_magnitude(df):
    if "gyro_mag" in df:
        return df["gyro_mag"].to_numpy(dtype=float)
    if all(c in df for c in ["gyro_x", "gyro_y", "gyro_z"]):
        m = np.sqrt(df["gyro_x"]**2 + df["gyro_y"]**2 + df["gyro_z"]**2)
        return m.to_numpy(dtype=float)
    return np.full(len(df), np.nan)


def _acf_norm(sig):
    x = np.asarray(sig, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.array([])
    x = x - np.mean(x)
    if np.allclose(x, 0):
        return np.array([])
    acf = np.correlate(x, x, mode="full")
    acf = acf[len(x)-1:]
    return acf / acf[0] if acf[0] != 0 else np.array([])


def _window_signal(time_s, signal, start_t, end_t):
    time_s = np.asarray(time_s, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if len(time_s) != len(signal) or len(time_s) < 3:
        return np.array([]), np.array([])
    if not np.isfinite(start_t) or not np.isfinite(end_t) or end_t <= start_t:
        return np.array([]), np.array([])

    mask = (time_s >= start_t) & (time_s <= end_t) & np.isfinite(signal) & np.isfinite(time_s)
    if mask.sum() < 3:
        return np.array([]), np.array([])
    return time_s[mask], signal[mask]


def _window_peak_to_peak_from_signal(t, signal, window_sec=1.0):
    t = np.asarray(t, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if len(t) == 0 or len(signal) == 0 or len(t) != len(signal):
        return np.array([], dtype=float), np.array([], dtype=float)
    if not np.isfinite(window_sec) or window_sec <= 0:
        window_sec = 1.0
    t0 = float(t[0])
    t1 = float(t[-1])
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 < t0:
        return np.array([], dtype=float), np.array([], dtype=float)
    n_windows = int(np.floor((t1 - t0) / window_sec)) + 1
    if n_windows <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    starts = []
    pp = []
    for i in range(n_windows):
        start = t0 + i * window_sec
        end = start + window_sec
        if i == n_windows - 1:
            mask = (t >= start) & (t <= end)
        else:
            mask = (t >= start) & (t < end)
        window = signal[mask]
        window = window[np.isfinite(window)]
        starts.append(start)
        if len(window) == 0:
            pp.append(np.nan)
        else:
            pp.append(float(np.nanmax(window) - np.nanmin(window)))
    return np.asarray(starts, dtype=float), np.asarray(pp, dtype=float)


def _window_signal_envelope(t, signal, window_sec=1.0):
    t = np.asarray(t, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if len(t) == 0 or len(signal) == 0 or len(t) != len(signal):
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)
    if not np.isfinite(window_sec) or window_sec <= 0:
        window_sec = 1.0
    t0 = float(t[0])
    t1 = float(t[-1])
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 < t0:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)
    n_windows = int(np.floor((t1 - t0) / window_sec)) + 1
    if n_windows <= 0:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)
    starts = []
    mins = []
    maxs = []
    mids = []
    for i in range(n_windows):
        start = t0 + i * window_sec
        end = start + window_sec
        if i == n_windows - 1:
            mask = (t >= start) & (t <= end)
        else:
            mask = (t >= start) & (t < end)
        window = signal[mask]
        window = window[np.isfinite(window)]
        starts.append(start)
        if len(window) == 0:
            mins.append(np.nan)
            maxs.append(np.nan)
            mids.append(np.nan)
        else:
            wmin = float(np.nanmin(window))
            wmax = float(np.nanmax(window))
            mins.append(wmin)
            maxs.append(wmax)
            mids.append(0.5 * (wmin + wmax))
    return np.asarray(starts, dtype=float), np.asarray(mins, dtype=float), np.asarray(maxs, dtype=float), np.asarray(mids, dtype=float)


def _resample_signal_to_n(time_s, signal, n_points=101):
    time_s = np.asarray(time_s, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if len(time_s) < 3 or len(signal) < 3 or n_points < 3:
        return np.array([]), np.array([])

    order = np.argsort(time_s)
    time_s = time_s[order]
    signal = signal[order]
    keep = np.isfinite(time_s) & np.isfinite(signal)
    time_s = time_s[keep]
    signal = signal[keep]
    if len(time_s) < 3:
        return np.array([]), np.array([])

    unique_t, unique_idx = np.unique(time_s, return_index=True)
    signal = signal[unique_idx]
    time_s = unique_t
    if len(time_s) < 3 or time_s[-1] <= time_s[0]:
        return np.array([]), np.array([])

    resampled_t = np.linspace(time_s[0], time_s[-1], n_points)
    resampled_x = np.interp(resampled_t, time_s, signal)
    return resampled_t, resampled_x


def _resample_signal_to_cycle_percent(time_s, signal, n_cycle_points=101):
    _, resampled_x = _resample_signal_to_n(time_s, signal, n_points=n_cycle_points)
    if len(resampled_x) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    cycle_percent = np.linspace(0.0, 100.0, len(resampled_x))
    return cycle_percent, resampled_x


def _normalize_signal(signal):
    signal = np.asarray(signal, dtype=float)
    if len(signal) < 3:
        return np.array([])
    signal = signal[np.isfinite(signal)]
    if len(signal) < 3:
        return np.array([])
    signal = signal - np.mean(signal)
    std = np.std(signal)
    if not np.isfinite(std) or std == 0:
        return np.array([])
    return signal / std


def _cycle_normalized_pair(data_a, data_b, n_cycle_points=101):
    ta, xa = _window_signal(
        data_a['time'], data_a['signal'],
        data_a['summary']['start_time_s'], data_a['summary']['end_time_s']
    )
    tb, xb = _window_signal(
        data_b['time'], data_b['signal'],
        data_b['summary']['start_time_s'], data_b['summary']['end_time_s']
    )
    if len(ta) < 3 or len(tb) < 3:
        return np.array([]), np.array([]), np.array([]), np.nan

    cycle_percent, xa_r = _resample_signal_to_cycle_percent(ta, xa, n_cycle_points=n_cycle_points)
    _, xb_r = _resample_signal_to_cycle_percent(tb, xb, n_cycle_points=n_cycle_points)
    xa_n = _normalize_signal(xa_r)
    xb_n = _normalize_signal(xb_r)
    if len(xa_n) < 3 or len(xb_n) < 3:
        return np.array([]), np.array([]), np.array([]), np.nan

    cycle_step_percent = 100.0 / (n_cycle_points - 1) if n_cycle_points > 1 else np.nan
    return cycle_percent, xa_n, xb_n, cycle_step_percent


def _windowed_resampled_pair(data_a, data_b, n_points=101):
    _, xa_n, xb_n, cycle_step_percent = _cycle_normalized_pair(
        data_a,
        data_b,
        n_cycle_points=n_points,
    )
    return xa_n, xb_n, cycle_step_percent


def _normalized_xcorr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or len(y) < 3 or len(x) != len(y):
        return np.array([]), np.array([])

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or len(x) != len(y):
        return np.array([]), np.array([])

    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return np.array([]), np.array([])

    corr = np.correlate(x, y, mode='full') / denom
    lags = np.arange(-len(x) + 1, len(x))
    return lags, corr


def _best_xcorr_summary(data_a, data_b, n_points=101):
    x, y, lag_step_percent = _windowed_resampled_pair(data_a, data_b, n_points=n_points)
    lags, corr = _normalized_xcorr(x, y)
    if len(corr) == 0:
        return lags, corr, np.nan, np.nan, np.nan

    i = int(np.argmax(np.abs(corr)))
    peak = float(corr[i])
    symmetry_score = float(abs(peak))
    lag_percent = float(lags[i] * lag_step_percent) if np.isfinite(lag_step_percent) else np.nan
    return lags, corr, peak, symmetry_score, lag_percent


def _mask_close_gaps(mask, max_gap_samples):
    if max_gap_samples <= 0 or len(mask) == 0:
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


def _largest_segment(mask):
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


def _detect_active_window(signal, t, min_duration_s=0.8, threshold=None):
    gate_cfg = _apply_notebook_window_gate_settings()
    return core_detect_active_window(
        signal=np.asarray(signal, dtype=float),
        time_s=np.asarray(t, dtype=float),
        min_duration_s=min_duration_s,
        threshold=threshold,
        window_sec=gate_cfg.window_sec,
    )




def _apply_notebook_window_gate_settings():
    gate_cfg = PIPELINE_CONFIG.window_gate
    settings = PLOT_PARAMS.get('window_gate', {})
    for key, value in settings.items():
        if hasattr(gate_cfg, key):
            setattr(gate_cfg, key, value)
    return gate_cfg


def _adaptive_step_min_distance(t, candidate_peaks):
    t = np.asarray(t, dtype=float)
    candidate_peaks = np.asarray(candidate_peaks, dtype=int)
    if len(candidate_peaks) < 2:
        return np.nan

    dt = np.diff(t[candidate_peaks])
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return np.nan

    return float(PLOT_PARAMS.get('step_min_distance_alpha', 0.5) * np.nanmedian(dt))


def _pause_segment_durations(values, t, threshold):
    values = np.asarray(values, dtype=float)
    t = np.asarray(t, dtype=float)
    if len(values) != len(t) or len(values) < 2:
        return np.array([], dtype=float)

    pause_mask = np.isfinite(values) & (np.abs(values) < threshold)
    if not np.any(pause_mask):
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
        duration = float(t[end - 1] - t[start])
        if np.isfinite(duration) and duration > 0:
            durations.append(duration)
    return np.asarray(durations, dtype=float)


def _adaptive_pause_min_duration(values, t, threshold):
    durations = _pause_segment_durations(values, t, threshold)
    if len(durations) == 0:
        return 0.0
    return float(PLOT_PARAMS.get('pause_min_duration_beta', 0.5) * np.nanmedian(durations))






def _compare_adjust_bout(inarray, fs=10):
    inarray = np.asarray(inarray, dtype=float)
    if len(inarray) == 0:
        return np.array([])
    rem = len(inarray) % fs
    if rem >= int(np.ceil(0.7 * fs)):
        pad_n = fs - rem
        if pad_n < fs:
            inarray = np.append(inarray, np.repeat(inarray[-1], pad_n))
    elif rem != 0:
        inarray = inarray[: (len(inarray) // fs) * fs]
    return inarray


def _compare_preprocess_bout(t_bout, vm_bout, fs=10):
    t_bout = np.asarray(t_bout, dtype=float)
    vm_bout = np.asarray(vm_bout, dtype=float)
    keep = np.isfinite(t_bout) & np.isfinite(vm_bout)
    t_bout = t_bout[keep]
    vm_bout = vm_bout[keep]
    if len(t_bout) < 2 or len(vm_bout) < 2 or not np.isfinite(fs) or fs <= 0:
        return np.array([]), np.array([])
    t_rel = t_bout - t_bout[0]
    t_interp = np.arange(t_rel[0], t_rel[-1], 1.0 / fs)
    if len(t_interp) < 2:
        return np.array([]), np.array([])
    t_interp = t_interp + t_bout[0]
    f = interpolate.interp1d(t_bout, vm_bout, bounds_error=False, fill_value='extrapolate')
    vm_interp = f(t_interp)
    vm_interp = _compare_adjust_bout(vm_interp, fs)
    t_interp = t_interp[:len(vm_interp)]
    if len(vm_interp) < fs:
        return np.array([]), np.array([])
    return t_interp, vm_interp


def _compare_get_pp(vm_bout, fs=10, window_sec=1.0):
    vm_bout = np.asarray(vm_bout, dtype=float)
    if fs <= 0 or not np.isfinite(window_sec) or window_sec <= 0:
        return np.array([], dtype=float)
    window_n = max(1, int(round(fs * window_sec)))
    if len(vm_bout) < window_n:
        return np.array([], dtype=float)
    vm_bout = vm_bout[: (len(vm_bout) // window_n) * window_n]
    if len(vm_bout) == 0:
        return np.array([], dtype=float)
    vm_res = vm_bout.reshape((window_n, -1), order='F')
    return np.ptp(vm_res, axis=0)


def _compare_compute_interpolate_cwt(tapered_bout, fs=10):
    tapered_bout = np.asarray(tapered_bout, dtype=float)
    if len(tapered_bout) < fs:
        return np.array([]), np.array([[]])
    window = tukey(len(tapered_bout), alpha=0.02, sym=True)
    padded = np.concatenate((np.zeros(5 * fs), tapered_bout * window, np.zeros(5 * fs)))
    Tx, Wx, ssq_freqs, scales, *_ = ssq_cwt(padded[:-1], wavelet=('gmw', {'beta': 90, 'gamma': 3}), fs=fs)
    coefs = np.abs(Wx.astype('complex128') ** 2)
    coefs = np.append(coefs, coefs[:, -1:], axis=1)
    freqs_interp = np.arange(0.5, 4.5, 0.05)
    # scipy interp2d is gone in recent versions; use row-wise np.interp
    time_idx = np.arange(coefs.shape[1], dtype=float)
    coefs_interp = np.vstack([
        np.interp(freqs_interp, ssq_freqs[::-1], coefs[:, j][::-1])
        for j in range(coefs.shape[1])
    ]).T
    coefs_interp = coefs_interp[:, 5 * fs : -5 * fs]
    return freqs_interp, coefs_interp


def _compare_identify_peaks_in_cwt(freqs_interp, coefs_interp, fs=10, window_sec=1.0, step_freq=(1.4, 2.3), alpha=0.6, beta=2.5):
    if coefs_interp.size == 0:
        return np.array([[]])
    num_rows, num_cols = coefs_interp.shape
    window_n = max(1, int(round(fs * window_sec)))
    num_cols2 = int(num_cols / window_n)
    dp = np.zeros((num_rows, num_cols2))
    loc_min = np.argmin(np.abs(freqs_interp - step_freq[0]))
    loc_max = np.argmin(np.abs(freqs_interp - step_freq[1]))
    for i in range(num_cols2):
        x_start = i * window_n
        x_end = (i + 1) * window_n
        window = np.sum(coefs_interp[:, np.arange(x_start, x_end)], axis=1)
        locs = detect_peaks(window, fs_hz=np.nan, min_distance_s=np.nan, height=None)
        locs = np.asarray(locs, dtype=int)
        if len(locs) == 0:
            continue
        pks = window[locs]
        ind = np.argsort(-pks)
        locs = locs[ind]
        pks = pks[ind]
        index_in_range = None
        for j, locs_j in enumerate(locs):
            if loc_min <= locs_j <= loc_max:
                index_in_range = j
                break
        peak_vec = np.zeros(num_rows)
        if index_in_range is not None:
            if locs[0] > loc_max:
                if pks[0] / pks[index_in_range] < beta:
                    peak_vec[locs[index_in_range]] = 1
            elif locs[0] < loc_min:
                if pks[0] / pks[index_in_range] < alpha:
                    peak_vec[locs[index_in_range]] = 1
            else:
                peak_vec[locs[index_in_range]] = 1
        dp[:, i] = peak_vec
    return dp


def _compare_find_continuous_dominant_peaks(valid_peaks, min_t=3, delta=20):
    valid_peaks = np.asarray(valid_peaks, dtype=float)
    if valid_peaks.size == 0:
        return valid_peaks

    num_rows, num_cols = valid_peaks.shape
    min_t = max(1, int(min_t))

    # If the user allows a single accepted window, or the bout is too short to
    # evaluate continuity, keep the currently valid peaks as-is.
    if num_cols <= 1 or min_t <= 1:
        return np.where(valid_peaks > 0, 1, 0)

    min_t = min(min_t, num_cols)
    extended_peaks = np.zeros((num_rows, num_cols + 1), dtype=valid_peaks.dtype)
    extended_peaks[:, :num_cols] = valid_peaks
    cont_peaks = np.zeros_like(extended_peaks)
    for slice_ind in range(num_cols + 1 - min_t):
        slice_mat = extended_peaks[:, slice_ind:slice_ind + min_t].copy()
        windows = list(range(min_t)) + list(range(min_t - 2, -1, -1))
        stop = True
        for win_ind in windows:
            pr = np.where(slice_mat[:, win_ind] != 0)[0]
            stop = True
            for p in pr:
                index = np.arange(max(0, p - delta), min(p + delta + 1, num_rows))
                peaks1 = slice_mat[p, win_ind]
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
                        slice_mat[p, win_ind] = 0
                else:
                    if np.any(peaks1 > 1) and np.any(peaks2 > 1):
                        stop = False
                    else:
                        slice_mat[p, win_ind] = 0
            if stop:
                break
        if not stop:
            cont_peaks[:, slice_ind:slice_ind + min_t] += slice_mat
    cont_peaks = np.where(cont_peaks > 0, 1, 0)
    return cont_peaks[:, :num_cols]


def _compare_params(prefix):
    return {
        "compare_fs": int(PLOT_PARAMS.get(f"{prefix}_fs_hz", 10)),
        "fallback_min_amp": float(PLOT_PARAMS.get(f"{prefix}_min_amp", 0.30)),
        "threshold_k": float(PLOT_PARAMS.get(f"{prefix}_threshold_k", 1.0)),
        "min_t_sec": float(PLOT_PARAMS.get(f"{prefix}_min_t_sec", 2.0)),
        "step_freq": PLOT_PARAMS.get(f"{prefix}_step_freq_hz", (0.8, 2.3)),
        "window_sec": float(PLOT_PARAMS.get(f"{prefix}_window_sec", 1.0)),
    }


def exact_compare_walk_summary(t, acc, fs_hz, prefix='walk_compare'):
    params = _compare_params(prefix)
    compare_fs = params["compare_fs"]
    fallback_min_amp = params["fallback_min_amp"]
    threshold_k = params["threshold_k"]
    min_t_sec = params["min_t_sec"]
    step_freq = params["step_freq"]
    window_sec = params["window_sec"]
    min_t = max(1, int(round(min_t_sec / max(window_sec, 1e-9))))
    alpha = 0.6
    beta = 2.5
    delta = 20

    t_res, vm_bout = _compare_preprocess_bout(t, acc, fs=compare_fs)
    if len(vm_bout) < max(1, int(round(compare_fs * window_sec))):
        return {'t_res': np.array([]), 'vm_bout': np.array([]), 'cad': np.array([]), 'dominant_freq_hz': np.array([]), 'pp': np.array([]), 'min_amp': fallback_min_amp, 'walk_mask': np.array([], dtype=bool), 'start_time_s': np.nan, 'end_time_s': np.nan, 'duration_s': np.nan, 'step_count': np.nan, 'cadence': np.nan, 'window_sec': window_sec}

    pp = _compare_get_pp(vm_bout, compare_fs, window_sec=window_sec)
    min_amp = p2p_distribution_threshold(pp, k=threshold_k, fallback=fallback_min_amp)
    valid = np.ones(len(pp), dtype=bool)
    valid[pp < min_amp] = False
    cad = np.zeros(len(pp), dtype=float)
    dominant_freq_hz = np.full(len(pp), np.nan, dtype=float)
    if np.sum(valid) >= min_t:
        tapered_bout = vm_bout[np.repeat(valid, max(1, int(round(compare_fs * window_sec))))]
        freqs_interp, coefs_interp = _compare_compute_interpolate_cwt(tapered_bout, fs=compare_fs)
        if coefs_interp.size:
            dp = _compare_identify_peaks_in_cwt(freqs_interp, coefs_interp, compare_fs, window_sec=window_sec, step_freq=step_freq, alpha=alpha, beta=beta)
            valid_peaks = np.zeros((dp.shape[0], len(valid)))
            valid_peaks[:, valid] = dp
            cont_peaks = _compare_find_continuous_dominant_peaks(valid_peaks, min_t=min_t, delta=delta)
            for i in range(len(cad)):
                ind_freqs = np.where(cont_peaks[:, i] > 0)[0]
                if len(ind_freqs) > 0:
                    dominant_freq_hz[i] = freqs_interp[ind_freqs[0]]
                    cad[i] = freqs_interp[ind_freqs[0]]
    walk_sec = cad > 0
    if not np.any(walk_sec):
        return {'t_res': t_res, 'vm_bout': vm_bout, 'cad': cad, 'dominant_freq_hz': dominant_freq_hz, 'pp': pp, 'min_amp': min_amp, 'walk_mask': walk_sec, 'start_time_s': np.nan, 'end_time_s': np.nan, 'duration_s': np.nan, 'step_count': 0.0, 'cadence': np.nan, 'window_sec': window_sec}
    window_n = max(1, int(round(compare_fs * window_sec)))
    first_sec = np.where(walk_sec)[0][0]
    last_sec = np.where(walk_sec)[0][-1]
    start_time = float(t_res[first_sec * window_n])
    end_idx = min(len(t_res) - 1, (last_sec + 1) * window_n - 1)
    end_time = float(t_res[end_idx])
    duration = float(end_time - start_time)
    step_count = float(np.nansum(cad[walk_sec]) * window_sec)
    cadence = float(np.nanmean(cad[walk_sec]) * 60.0) if np.any(walk_sec) else np.nan
    return {'t_res': t_res, 'vm_bout': vm_bout, 'cad': cad, 'dominant_freq_hz': dominant_freq_hz, 'pp': pp, 'min_amp': min_amp, 'walk_mask': walk_sec, 'start_time_s': start_time, 'end_time_s': end_time, 'duration_s': duration, 'step_count': step_count, 'cadence': cadence, 'window_sec': window_sec}


def exact_compare_fixed_window_summary(t, acc, fs_hz, prefix='turn_compare'):
    params = _compare_params(prefix)
    compare_fs = params["compare_fs"]
    fallback_min_amp = params["fallback_min_amp"]
    threshold_k = params["threshold_k"]
    min_t_sec = params["min_t_sec"]
    step_freq = params["step_freq"]
    window_sec = params["window_sec"]
    min_t = max(1, int(round(min_t_sec / max(window_sec, 1e-9))))
    alpha = 0.6
    beta = 2.5
    delta = 20

    t_res, vm_bout = _compare_preprocess_bout(t, acc, fs=compare_fs)
    if len(vm_bout) < max(1, int(round(compare_fs * window_sec))):
        return {'t_res': np.array([]), 'vm_bout': np.array([]), 'cad': np.array([]), 'dominant_freq_hz': np.array([]), 'pp': np.array([]), 'min_amp': fallback_min_amp, 'walk_mask': np.array([], dtype=bool), 'start_time_s': np.nan, 'end_time_s': np.nan, 'duration_s': np.nan, 'step_count': np.nan, 'cadence': np.nan, 'window_sec': window_sec}

    pp = _compare_get_pp(vm_bout, compare_fs, window_sec=window_sec)
    min_amp = p2p_distribution_threshold(pp, k=threshold_k, fallback=fallback_min_amp)
    valid = np.ones(len(pp), dtype=bool)
    valid[pp < min_amp] = False
    cad = np.zeros(len(pp), dtype=float)
    dominant_freq_hz = np.full(len(pp), np.nan, dtype=float)
    if np.sum(valid) >= min_t:
        tapered_bout = vm_bout[np.repeat(valid, max(1, int(round(compare_fs * window_sec))))]
        freqs_interp, coefs_interp = _compare_compute_interpolate_cwt(tapered_bout, fs=compare_fs)
        if coefs_interp.size:
            dp = _compare_identify_peaks_in_cwt(freqs_interp, coefs_interp, compare_fs, window_sec=window_sec, step_freq=step_freq, alpha=alpha, beta=beta)
            valid_peaks = np.zeros((dp.shape[0], len(valid)))
            valid_peaks[:, valid] = dp
            cont_peaks = _compare_find_continuous_dominant_peaks(valid_peaks, min_t=min_t, delta=delta)
            for i in range(len(cad)):
                ind_freqs = np.where(cont_peaks[:, i] > 0)[0]
                if len(ind_freqs) > 0:
                    dominant_freq_hz[i] = freqs_interp[ind_freqs[0]]
                    cad[i] = freqs_interp[ind_freqs[0]]

    active_sec = cad > 0
    start_time = float(t[0]) if len(t) else np.nan
    end_time = float(t[-1]) if len(t) else np.nan
    duration = float(end_time - start_time) if np.isfinite(start_time) and np.isfinite(end_time) else np.nan
    step_count = float(np.nansum(cad[active_sec]) * window_sec) if np.any(active_sec) else 0.0
    cadence = float((step_count / duration) * 60.0) if np.isfinite(duration) and duration > 0 else np.nan
    return {'t_res': t_res, 'vm_bout': vm_bout, 'cad': cad, 'dominant_freq_hz': dominant_freq_hz, 'pp': pp, 'min_amp': min_amp, 'walk_mask': active_sec, 'start_time_s': start_time, 'end_time_s': end_time, 'duration_s': duration, 'step_count': step_count, 'cadence': cadence, 'window_sec': window_sec}


def _turn_thresholds(ang, t, meta):
    gate_cfg = _apply_notebook_window_gate_settings()
    pp_starts, pp = window_peak_to_peak(t, ang, window_sec=gate_cfg.window_sec)
    turn_threshold, _ = robust_p2p_threshold(
        time_s=t,
        signal=ang,
        window_sec=gate_cfg.window_sec,
        k=getattr(gate_cfg, 'turn_threshold_k', 1.0),
        fallback=gate_cfg.turn_min_amp_threshold,
    )
    return float(turn_threshold), float(turn_threshold), pp_starts, pp







def plot_walk_validation(activity_name, df, meta):
    t = df['time_s'].to_numpy(dtype=float)
    acc, src = select_motion_acc_signal(df, PIPELINE_CONFIG.prefer_useracc_for_motion)

    compare_walk = exact_compare_walk_summary(t, acc, meta.fs_hz)
    start_t = float(compare_walk.get('start_time_s', np.nan))
    end_t = float(compare_walk.get('end_time_s', np.nan))
    duration_t = float(compare_walk.get('duration_s', np.nan))
    step_count = float(compare_walk.get('step_count', np.nan))
    cadence = float(compare_walk.get('cadence', np.nan))

    if len(compare_walk['t_res']):
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(t, acc, color='steelblue', linewidth=1.5, label=src)
        if np.isfinite(start_t):
            ax1.axvline(start_t, color='green', ls='--', linewidth=1.5, label='start')
        if np.isfinite(end_t):
            ax1.axvline(end_t, color='gray', ls='--', linewidth=1.5, label='end')
        ax1.set_title(f"{activity_name}: Wavelet Walking Comparison")
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Acceleration Magnitude')
        ax1.grid(alpha=0.3)

        env_times, env_min, env_max, env_mid = _window_signal_envelope(t, acc, window_sec=1.0)
        min_amp = float(compare_walk.get('min_amp', np.nan))
        if len(env_times) == len(env_mid):
            valid_env = np.isfinite(env_min) & np.isfinite(env_max)
            if np.any(valid_env):
                ax1.fill_between(env_times[valid_env], env_min[valid_env], env_max[valid_env], step='post', color='crimson', alpha=0.10, label='min-max')
                ax1.step(env_times, env_mid, where='post', color='crimson', linewidth=2.0, alpha=0.9, label='peak to peak')
        if np.isfinite(min_amp):
            ax1.axhline(min_amp, color='firebrick', linestyle='--', linewidth=1.6, alpha=0.9, label=f'threshold={min_amp:.2f}')

        ax2 = ax1.twinx()
        if len(compare_walk['cad']):
            walk_window_n = max(1, int(round(int(PLOT_PARAMS.get('walk_compare_fs_hz', 10)) * float(PLOT_PARAMS.get('walk_compare_window_sec', 1.0)))))
            sec_times = compare_walk['t_res'][::walk_window_n][:len(compare_walk['cad'])]
            cad = np.asarray(compare_walk['cad'], dtype=float)
            if len(sec_times) == len(cad):
                ax2.step(sec_times, cad, where='post', color='darkorange', linewidth=2.0, label='cadence (steps/s)')
        ax2.set_ylabel('Cadence (steps/s)')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(0.995, 0.995), borderaxespad=0.0, ncol=1, fontsize=7, framealpha=0.80, borderpad=0.25, labelspacing=0.25, columnspacing=0.8, handlelength=1.6)

        ax1.text(
            0.01,
            0.98,
            f"start={start_t:.3f}s | end={end_t:.3f}s | duration={duration_t:.2f}s\nsteps≈{step_count:.1f} | cadence={cadence:.2f}",
            transform=ax1.transAxes,
            va='top',
            ha='left',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.25'),
        )
        plt.show()

        if len(compare_walk['cad']):
            walk_window_n = max(1, int(round(int(PLOT_PARAMS.get('walk_compare_fs_hz', 10)) * float(PLOT_PARAMS.get('walk_compare_window_sec', 1.0)))))
            sec_times = compare_walk['t_res'][::walk_window_n][:len(compare_walk['cad'])]
            freq_hz = np.asarray(compare_walk.get('dominant_freq_hz', []), dtype=float)
            plt.figure(figsize=(10, 3.8))
            if len(sec_times) == len(freq_hz):
                freq_plot = np.where(np.isfinite(freq_hz), freq_hz, 0.0)
                plt.step(sec_times, freq_plot, where='post', color='darkorange', linewidth=2.2, label='dominant wavelet frequency (Hz)')
                active = np.isfinite(freq_hz) & (freq_hz > 0)
                if np.any(active):
                    plt.scatter(sec_times[active], freq_hz[active], color='crimson', s=36, zorder=3, label='active frequency windows')
            freq_band = PLOT_PARAMS.get('walk_compare_step_freq_hz', (0.8, 2.3))
            if len(freq_band) == 2:
                plt.axhline(float(freq_band[0]), color='firebrick', ls='--', linewidth=1.4, alpha=0.8, label=f'freq min={float(freq_band[0]):.2f} Hz')
                plt.axhline(float(freq_band[1]), color='brown', ls=':', linewidth=1.4, alpha=0.8, label=f'freq max={float(freq_band[1]):.2f} Hz')
            if np.isfinite(start_t):
                plt.axvline(start_t, color='green', ls='--', linewidth=1.5, label='start')
            if np.isfinite(end_t):
                plt.axvline(end_t, color='gray', ls='--', linewidth=1.5, label='end')
            if len(t):
                plt.xlim(float(t[0]), float(t[-1]))
            plt.title(f"{activity_name}: Wavelet Dominant Frequency by Window")
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.grid(alpha=0.3)
            plt.legend(loc='upper right', fontsize=7, framealpha=0.80, borderpad=0.25, labelspacing=0.25, handlelength=1.6)
            plt.show()

    plt.figure(figsize=(9, 4))
    if np.isfinite(start_t) and np.isfinite(end_t):
        walk_mask = (t >= start_t) & (t <= end_t)
        acc_for_fft = acc[walk_mask]
    else:
        acc_for_fft = acc
    x = acc_for_fft[np.isfinite(acc_for_fft)]
    if len(x) >= 8 and np.isfinite(meta.fs_hz) and meta.fs_hz > 0:
        x = x - np.mean(x)
        n = len(x)
        f = np.fft.rfftfreq(n, d=1.0 / meta.fs_hz)
        mag = np.abs(np.fft.rfft(x * np.hanning(n)))
        min_plot_hz = 0.05
        mask = f >= min_plot_hz
        plt.plot(f[mask], mag[mask], label='FFT magnitude')
        if np.any(mask):
            i = np.argmax(mag[mask])
            plt.scatter(f[mask][i], mag[mask][i], c='red', s=35, label=f'dominant={f[mask][i]:.3f}Hz')
            plt.xlim(left=min_plot_hz)
    plt.title(f"{activity_name}: Frequency Spectrum (FFT, Walk Window)")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(alpha=0.3)
    plt.legend(loc='upper right', fontsize=7, framealpha=0.80, borderpad=0.25, labelspacing=0.25, handlelength=1.6)
    plt.show()

    return {
        'activity': activity_name,
        'start_time_s': start_t,
        'end_time_s': end_t,
        'duration_s': duration_t,
        'step_count': step_count,
        'cadence': cadence,
    }

def _turn_window_from_threshold(ang, t, threshold):
    if len(t) == 0:
        return np.nan, np.nan, np.nan, np.zeros(0, dtype=bool)

    ang = np.asarray(ang, dtype=float)
    t = np.asarray(t, dtype=float)
    fs_hz = 1.0 / estimate_sampling_interval_s(t) if len(t) > 2 else np.nan

    active = np.isfinite(ang) & (ang >= threshold)
    if np.isfinite(fs_hz) and fs_hz > 0:
        active = _mask_close_gaps(active, max_gap_samples=max(1, int(0.60 * fs_hz)))

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
        start_t, end_t, duration, mask, _ = _detect_active_window(ang, t, min_duration_s=0.8)
        return start_t, end_t, duration, mask

    idx = np.where(active)[0]
    i0 = int(idx[0])
    i1 = int(idx[-1])
    pad = int(0.15 * fs_hz) if np.isfinite(fs_hz) and fs_hz > 0 else 1
    i0 = max(0, i0 - pad)
    i1 = min(len(t) - 1, i1 + pad)

    start_t = float(t[i0])
    end_t = float(t[i1])
    mask = np.zeros(len(t), dtype=bool)
    mask[i0:i1 + 1] = True
    return start_t, end_t, float(end_t - start_t), mask


def plot_turn_validation(activity_name, df, meta):
    t = df['time_s'].to_numpy(dtype=float)

    ang, src = select_turn_angular_signal(df, PIPELINE_CONFIG, meta.fs_hz)

    turn_threshold, pause_threshold, turn_pp_starts, turn_pp = _turn_thresholds(ang, t, meta)
    start_t, end_t, duration_t, turn_mask = _turn_window_from_threshold(ang, t, turn_threshold)

    if 'useracc_mag' in df and np.isfinite(df['useracc_mag'].to_numpy(dtype=float)).any():
        acc_src = 'useracc_mag'
        acc = df[acc_src].to_numpy(dtype=float)
    elif 'acc_mag' in df and np.isfinite(df['acc_mag'].to_numpy(dtype=float)).any():
        acc_src = 'acc_mag'
        acc = df[acc_src].to_numpy(dtype=float)
    else:
        acc_src = 'useracc_mag'
        acc = np.full(len(df), np.nan, dtype=float)
    acc_turn = np.array([], dtype=float)
    t_turn = np.array([], dtype=float)
    acc_turn_smooth = np.array([], dtype=float)
    turn_peak_indices = np.array([], dtype=int)
    turn_peak_threshold = np.nan
    turn_peak_distance = np.nan
    fs_turn = meta.fs_hz
    if np.any(turn_mask):
        idx = np.where(turn_mask)[0]
        acc_turn = acc[idx]
        t_turn = t[idx]
        fs_turn = (1.0 / estimate_sampling_interval_s(t_turn)) if len(t_turn) > 2 else meta.fs_hz
        if len(acc_turn) >= 3 and np.isfinite(fs_turn) and fs_turn > 0:
            acc_turn_smooth = gaussian_filter1d(acc_turn, sigma=1)
            turn_compare_k = float(PLOT_PARAMS.get('turn_compare_threshold_k', 1.0))
            turn_peak_threshold = turn_compare_k * float(np.nanmean(acc_turn_smooth))
            turn_peak_distance = max(1, int(0.2 * fs_turn))
            turn_peak_indices, _ = find_peaks(
                acc_turn_smooth,
                height=turn_peak_threshold,
                distance=turn_peak_distance,
            )

    step_count_turn = float(len(turn_peak_indices)) if len(t_turn) else np.nan

    if len(t_turn) and len(acc_turn_smooth):
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(t_turn, acc_turn, color='lightsteelblue', linewidth=1.0, alpha=0.8, label=acc_src)
        ax1.plot(t_turn, acc_turn_smooth, color='steelblue', linewidth=1.8, label=f'smoothed {acc_src}')
        if np.isfinite(start_t):
            ax1.axvline(start_t, color='green', ls='--', linewidth=1.5, label='turn start')
        if np.isfinite(end_t):
            ax1.axvline(end_t, color='gray', ls='--', linewidth=1.5, label='turn end')
        if np.isfinite(turn_peak_threshold):
            ax1.axhline(
                turn_peak_threshold,
                color='firebrick',
                linestyle='--',
                linewidth=1.6,
                alpha=0.9,
                label=f'K*mean threshold={turn_peak_threshold:.3f}',
            )
        if len(turn_peak_indices):
            ax1.scatter(
                t_turn[turn_peak_indices],
                acc_turn_smooth[turn_peak_indices],
                color='crimson',
                s=45,
                zorder=5,
                label=f'peaks={len(turn_peak_indices)}',
            )
        ax1.set_title(f"{activity_name}: find_peaks Turn Step Estimate")
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Acceleration Magnitude')
        ax1.grid(alpha=0.3)
        turn_leg = ax1.legend(
            loc='upper right',
            bbox_to_anchor=(0.995, 0.995),
            bbox_transform=ax1.transAxes,
            borderaxespad=0.0,
            ncol=1,
            fontsize=7,
            framealpha=0.80,
            borderpad=0.25,
            labelspacing=0.25,
            columnspacing=0.8,
            handlelength=1.6,
        )
        turn_leg.set_bbox_to_anchor((0.995, 0.995), transform=ax1.transAxes)
        turn_leg._loc = 1
        ax1.text(
            0.01,
            0.90,
            f"turn start={start_t:.3f}s | turn end={end_t:.2f}s | duration={duration_t:.2f}s\n"
            f"find_peaks steps={len(turn_peak_indices)} | K={turn_compare_k:.2f} | distance={turn_peak_distance / fs_turn:.2f}s",
            transform=ax1.transAxes,
            va='top',
            ha='left',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.25'),
        )
        plt.show()

    mean_v = float(np.nanmean(ang[turn_mask])) if np.any(turn_mask) else np.nan
    peak_v = float(np.nanmax(ang[turn_mask])) if np.any(turn_mask) else np.nan
    pause_min_duration_s = _adaptive_pause_min_duration(ang[turn_mask], t[turn_mask], pause_threshold) if np.any(turn_mask) else np.nan
    pause_count, pause_time = count_pauses(ang[turn_mask], t[turn_mask], pause_threshold, pause_min_duration_s) if np.any(turn_mask) else (np.nan, np.nan)

    plt.figure(figsize=(10, 4))
    plt.plot(t, ang, label=f'angular_velocity ({src})')

    env_times, env_min, env_max, env_mid = _window_signal_envelope(
        t,
        ang,
        window_sec=PIPELINE_CONFIG.window_gate.window_sec,
    )
    if len(turn_pp_starts) == len(env_mid):
        env_times = turn_pp_starts
    if len(env_times) == len(env_mid):
        valid_env = np.isfinite(env_min) & np.isfinite(env_max)
        if np.any(valid_env):
            plt.fill_between(
                env_times[valid_env],
                env_min[valid_env],
                env_max[valid_env],
                step='post',
                color='crimson',
                alpha=0.10,
                label='min-max',
            )

    if len(turn_pp):
        pp_times = np.asarray(turn_pp_starts, dtype=float)
        valid_pp = np.isfinite(turn_pp)
        if np.any(valid_pp):
            plt.step(
                pp_times[valid_pp],
                turn_pp[valid_pp],
                where='post',
                color='crimson',
                linewidth=2.0,
                alpha=0.9,
                label='window p2p',
            )
    if np.isfinite(turn_threshold):
        plt.axhline(
            turn_threshold,
            color='firebrick',
            ls='--',
            linewidth=2.2,
            alpha=1.0,
            zorder=6,
            label=f'threshold={turn_threshold:.3f}',
        )
    if np.isfinite(start_t):
        plt.axvline(start_t, color='green', ls='--', label='turn start')
    if np.isfinite(end_t):
        plt.axvline(end_t, color='gray', ls='--', label='turn end')
    plt.title(f"{activity_name}: Angular Velocity Time Series")
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (abs)')
    plt.grid(alpha=0.3)
    plt.legend(loc='upper right', fontsize=7, framealpha=0.80, borderpad=0.25, labelspacing=0.25, handlelength=1.6)
    plt.text(0.01, 0.98, f"start={start_t:.3f} | end={end_t:.3f} | duration={duration_t:.2f}s | steps={step_count_turn:.1f} | turn_thr={turn_threshold:.3f}", transform=plt.gca().transAxes, va='top')
    plt.show()

    plt.figure(figsize=(10, 4))
    thr = turn_threshold
    plt.plot(t, ang, label='angular_velocity')
    plt.axhline(thr, color='orange', ls='--', label=f'turn/pause threshold={thr:.3f}')
    turn_only = np.isfinite(ang) & (ang >= thr)
    pause_only = np.isfinite(ang) & (ang < thr)
    if np.any(turn_only):
        plt.fill_between(t, 0, ang, where=turn_only, alpha=0.22, color='green', label='turn region', interpolate=True)
    if np.any(pause_only):
        plt.fill_between(t, 0, ang, where=pause_only, alpha=0.18, color='purple', label='pause region', interpolate=True)
    if np.isfinite(start_t):
        plt.axvline(start_t, color='green', ls='--', linewidth=1.3, label='start')
    if np.isfinite(end_t):
        plt.axvline(end_t, color='gray', ls='--', linewidth=1.3, label='end')
    plt.title(f"{activity_name}: Turn and Pause Regions")
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (abs)')
    plt.grid(alpha=0.3)
    plt.legend(loc='upper right', fontsize=7, framealpha=0.80, borderpad=0.25, labelspacing=0.25, handlelength=1.6)
    plt.text(0.01, 0.98, f'pause_time={pause_time:.3f}s', transform=plt.gca().transAxes, va='top')
    plt.show()

    summary = {
        'activity': activity_name,
        'start_time_s': start_t,
        'end_time_s': end_t,
        'duration_s': duration_t,
        'step_count': float(step_count_turn),
        'pause_duration': float(pause_time),
    }
    return summary, {'time': t, 'signal': ang, 'summary': summary, 'fs_hz': meta.fs_hz}


def plot_turn_symmetry_comparison(left_data, right_data):
    lt, lsig = _window_signal(
        left_data['time'], left_data['signal'],
        left_data['summary']['start_time_s'], left_data['summary']['end_time_s']
    )
    rt, rsig = _window_signal(
        right_data['time'], right_data['signal'],
        right_data['summary']['start_time_s'], right_data['summary']['end_time_s']
    )

    plt.figure(figsize=(10, 4))
    if len(lt):
        plt.plot(lt - lt[0], lsig, label='left_turn')
    if len(rt):
        plt.plot(rt - rt[0], rsig, label='right_turn')
    max_duration = np.nanmax([left_data['summary']['duration_s'], right_data['summary']['duration_s']])
    if np.isfinite(max_duration) and max_duration > 0:
        plt.xlim(0, max_duration)
    plt.title('Turn Symmetry Comparison')
    plt.xlabel('Time from Start (s)')
    plt.ylabel('Angular Velocity (abs)')
    plt.grid(alpha=0.3)
    plt.legend(loc='upper right', fontsize=7, framealpha=0.80, borderpad=0.25, labelspacing=0.25, handlelength=1.6)

    d = left_data['summary']['duration_s'] - right_data['summary']['duration_s']
    ls = left_data['summary']['start_time_s']
    le = left_data['summary']['end_time_s']
    rs = right_data['summary']['start_time_s']
    re = right_data['summary']['end_time_s']
    plt.text(0.01, 0.98, f'left: {ls:.3f}->{le:.3f}s | right: {rs:.3f}->{re:.3f}s | duration_diff={d:.3f}s', transform=plt.gca().transAxes, va='top')
    plt.show()


def plot_turn_pair_xcorr(left_data, right_data):
    lags, corr, peak, symmetry_score, lag_percent = _best_xcorr_summary(
        left_data,
        right_data,
        n_points=101,
    )

    plt.figure(figsize=(10, 4))
    if len(corr):
        lag_step_percent = 100.0 / 100.0
        lags_x = lags * lag_step_percent
        plt.plot(lags_x, corr, label='normalized cross-correlation')
        plt.xlabel('Cycle shift (% of movement cycle)')
    else:
        plt.xlabel('Cycle shift (% of movement cycle)')
    plt.title('Left vs Right Turn: Cycle-Normalized Cross-Correlation')
    plt.ylabel('Correlation')
    plt.grid(alpha=0.3)
    plt.legend(loc='upper right', fontsize=7, framealpha=0.80, borderpad=0.25, labelspacing=0.25, handlelength=1.6)
    plt.text(
        0.01,
        0.98,
        f'symmetry score={symmetry_score:.3f} | peak correlation={peak:.3f}',
        transform=plt.gca().transAxes,
        va='top',
    )
    plt.show()

    return {
        'activity': 'turn_left_right_xcorr',
        'xcorr_symmetry_score': symmetry_score,
        'xcorr_peak_correlation': peak,
    }


def plot_transition_validation(activity_name, df, meta):
    t = df['time_s'].to_numpy(dtype=float)
    gate_cfg = _apply_notebook_window_gate_settings()
    flex_ext = transition_flexion_extension_peaks(df, meta, PIPELINE_CONFIG)
    gyro = flex_ext["gyro_signal"]
    src = flex_ext["gyro_axis"] or "selected gyro axis"
    acc, _ = select_motion_acc_signal(df, PIPELINE_CONFIG.prefer_useracc_for_motion)

    start_t = flex_ext["start_time_s"]
    end_t = flex_ext["end_time_s"]
    duration_t = flex_ext["duration_s"]
    trans_mask = flex_ext["transition_mask"]
    thr = flex_ext["threshold"]

    jerk = safe_gradient(acc, t)
    jerk_mean = float(np.nanmean(np.abs(jerk[trans_mask]))) if np.any(trans_mask) else np.nan
    jerk_std = float(np.nanstd(jerk[trans_mask])) if np.any(trans_mask) else np.nan

    plt.figure(figsize=(10, 4))
    if len(gyro):
        plt.plot(t, gyro, color='steelblue', linewidth=1.5, label=f'{src} angular velocity')
    if np.isfinite(start_t):
        plt.axvline(start_t, color='green', ls='--', label='start')
    if np.isfinite(end_t):
        plt.axvline(end_t, color='gray', ls='--', label='end')
    plt.axhline(0, color='black', linewidth=1.0, alpha=0.45)
    env_times, env_min, env_max, env_mid = _window_signal_envelope(
        t,
        gyro,
        window_sec=gate_cfg.window_sec,
    )
    if len(env_times) == len(env_mid):
        valid_env = np.isfinite(env_min) & np.isfinite(env_max)
        if np.any(valid_env):
            plt.fill_between(
                env_times[valid_env],
                env_min[valid_env],
                env_max[valid_env],
                step='post',
                color='crimson',
                alpha=0.10,
                label='min-max',
            )
    if np.isfinite(thr):
        plt.axhline(thr, color='firebrick', ls='--', linewidth=1.6, alpha=0.9, label=f'+abs threshold={thr:.3f}')
        plt.axhline(-thr, color='firebrick', ls='--', linewidth=1.6, alpha=0.9, label=f'-abs threshold={-thr:.3f}')
    if np.isfinite(flex_ext["flexion_peak"]) and np.isfinite(flex_ext["flexion_peak_time_s"]):
        plt.scatter(
            flex_ext["flexion_peak_time_s"],
            flex_ext["flexion_peak"],
            color='crimson',
            s=70,
            zorder=6,
            label='flexion peak',
        )
        plt.annotate(
            f'flexion={flex_ext["flexion_peak"]:.3f}',
            (flex_ext["flexion_peak_time_s"], flex_ext["flexion_peak"]),
            textcoords='offset points',
            xytext=(8, 8),
        )
    if np.isfinite(flex_ext["extension_peak"]) and np.isfinite(flex_ext["extension_peak_time_s"]):
        plt.scatter(
            flex_ext["extension_peak_time_s"],
            flex_ext["extension_peak"],
            color='royalblue',
            s=70,
            zorder=6,
            label='extension peak',
        )
        plt.annotate(
            f'extension={flex_ext["extension_peak"]:.3f}',
            (flex_ext["extension_peak_time_s"], flex_ext["extension_peak"]),
            textcoords='offset points',
            xytext=(8, -14),
        )
    plt.title(f"{activity_name}: Duration, Flexion, and Extension")
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity')
    plt.grid(alpha=0.3)
    plt.legend(loc='upper right', fontsize=7, framealpha=0.80, borderpad=0.25, labelspacing=0.25, handlelength=1.6)
    plt.text(
        0.01,
        0.98,
        f'start={start_t:.3f} | end={end_t:.3f} | duration={duration_t:.2f}s\n'
        f'flexion={flex_ext["flexion_peak"]:.3f} | extension={flex_ext["extension_peak"]:.3f}',
        transform=plt.gca().transAxes,
        va='top',
    )
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(t, jerk, label='jerk')
    if np.isfinite(start_t):
        plt.axvline(start_t, color='green', ls='--', label='start')
    if np.isfinite(end_t):
        plt.axvline(end_t, color='gray', ls='--', label='end')
    plt.title(f"{activity_name}: Jerk Signal")
    plt.xlabel('Time (s)')
    plt.ylabel('Jerk')
    plt.grid(alpha=0.3)
    plt.legend(loc='upper right', fontsize=7, framealpha=0.80, borderpad=0.25, labelspacing=0.25, handlelength=1.6)
    plt.text(0.01, 0.98, f'jerk_mean={jerk_mean:.3f} | jerk_std={jerk_std:.3f}', transform=plt.gca().transAxes, va='top')
    plt.show()

    summary = {
        'activity': activity_name,
        'start_time_s': start_t,
        'end_time_s': end_t,
        'duration_s': duration_t,
        'flexion_peak': flex_ext.get('flexion_peak', np.nan),
        'extension_peak': flex_ext.get('extension_peak', np.nan),
    }
    return summary, {'time': t, 'signal': gyro, 'summary': summary, 'fs_hz': meta.fs_hz}


def plot_transition_pair_xcorr(sit_stand_data, stand_sit_data):
    lags, corr, peak, symmetry_score, lag_percent = _best_xcorr_summary(
        sit_stand_data,
        stand_sit_data,
        n_points=101,
    )

    plt.figure(figsize=(10, 4))
    if len(corr):
        lag_step_percent = 100.0 / 100.0
        lags_x = lags * lag_step_percent
        plt.plot(lags_x, corr, label='normalized cross-correlation')
        plt.xlabel('Cycle shift (% of movement cycle)')
    else:
        plt.xlabel('Cycle shift (% of movement cycle)')
    plt.title('Sit-to-Stand vs Stand-to-Sit: Cycle-Normalized Cross-Correlation')
    plt.ylabel('Correlation')
    plt.grid(alpha=0.3)
    plt.legend(loc='upper right', fontsize=7, framealpha=0.80, borderpad=0.25, labelspacing=0.25, handlelength=1.6)
    plt.text(
        0.01,
        0.98,
        f'symmetry score={symmetry_score:.3f} | peak correlation={peak:.3f}',
        transform=plt.gca().transAxes,
        va='top',
    )
    plt.show()

    return {
        'activity': 'sitstand_standsit_xcorr',
        'xcorr_symmetry_score': symmetry_score,
        'xcorr_peak_correlation': peak,
    }
