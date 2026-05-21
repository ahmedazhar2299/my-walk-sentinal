import math

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .utils import (
    SignalMeta,
    build_nan_feature_dict,
    detect_active_window,
    robust_p2p_threshold,
    dominant_frequency,
    peak_step_summary,
    rms,
    select_motion_acc_signal,
    spectral_entropy,
    truncate_activity_dataframe,
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

WALK_NONLINEAR_FEATURE_NAMES = [
    "approximate_entropy",
    "sample_entropy",
    "symbolic_entropy",
    "permutation_entropy",
    "rosenstein_lyapunov_exponent",
    "wolf_lyapunov_exponent",
    "rqa_REC",
    "rqa_DET",
    "rqa_LAM",
    "rqa_MeanL",
    "rqa_MaxL",
    "rqa_EntrL",
    "rqa_EntrV",
    "rqa_EntrW",
]

NONLINEAR_MAX_SAMPLES = 900
EMBED_DIM = 3
ENTROPY_M = 2
PERMUTATION_ORDER = 3
TAU = 1


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


def _finite_centered_signal(values, max_samples=NONLINEAR_MAX_SAMPLES):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) > max_samples:
        idx = np.linspace(0, len(x) - 1, max_samples).astype(int)
        x = x[idx]
    if len(x) == 0:
        return x
    std = np.nanstd(x)
    if np.isfinite(std) and std > 0:
        return (x - np.nanmean(x)) / std
    return x - np.nanmean(x)


def _embed_signal(x, dim=EMBED_DIM, tau=TAU):
    x = np.asarray(x, dtype=float)
    n_vectors = len(x) - (dim - 1) * tau
    if n_vectors <= 1:
        return np.empty((0, dim), dtype=float)
    return np.column_stack([x[i * tau : i * tau + n_vectors] for i in range(dim)])


def _maxdist_count(x, m, r):
    emb = _embed_signal(x, dim=m, tau=1)
    n = len(emb)
    if n < 2 or not np.isfinite(r) or r <= 0:
        return np.nan, 0.0, 0.0
    count = 0.0
    total = 0.0
    for i in range(n):
        dist = np.max(np.abs(emb - emb[i]), axis=1)
        count += np.sum(dist <= r) - 1
        total += n - 1
    return count / total if total else np.nan, count, total


def _approximate_entropy(x, m=ENTROPY_M, r=None):
    x = np.asarray(x, dtype=float)
    if r is None:
        r = 0.2 * np.nanstd(x)
    if len(x) < m + 2 or not np.isfinite(r) or r <= 0:
        return np.nan

    def phi(order):
        emb = _embed_signal(x, dim=order, tau=1)
        n = len(emb)
        if n < 2:
            return np.nan
        c_vals = []
        for i in range(n):
            dist = np.max(np.abs(emb - emb[i]), axis=1)
            c_vals.append(np.mean(dist <= r))
        c_vals = np.asarray(c_vals, dtype=float)
        c_vals = c_vals[c_vals > 0]
        return np.mean(np.log(c_vals)) if len(c_vals) else np.nan

    phi_m = phi(m)
    phi_m1 = phi(m + 1)
    if not np.isfinite(phi_m) or not np.isfinite(phi_m1):
        return np.nan
    return float(phi_m - phi_m1)


def _sample_entropy_counts(x, m=ENTROPY_M, r=None):
    x = np.asarray(x, dtype=float)
    if r is None:
        r = 0.2 * np.nanstd(x)
    if len(x) < m + 2 or not np.isfinite(r) or r <= 0:
        return np.nan, 0.0, 0.0
    _, b_count, _ = _maxdist_count(x, m, r)
    _, a_count, _ = _maxdist_count(x, m + 1, r)
    if b_count <= 0 or a_count <= 0:
        return np.nan, a_count, b_count
    return float(-np.log(a_count / b_count)), a_count, b_count


def _sample_entropy(x, m=ENTROPY_M, r=None):
    value, _, _ = _sample_entropy_counts(x, m=m, r=r)
    return value


def _symbolic_entropy(x, word_length=2, n_bins=4):
    x = np.asarray(x, dtype=float)
    if len(x) < word_length:
        return np.nan
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.quantile(x, quantiles)
    symbols = np.digitize(x, edges)
    words = [tuple(symbols[i : i + word_length]) for i in range(len(symbols) - word_length + 1)]
    if not words:
        return np.nan
    _, counts = np.unique(words, axis=0, return_counts=True)
    prob = counts / counts.sum()
    return float(-np.sum(prob * np.log2(prob)))


def _permutation_entropy(x, order=PERMUTATION_ORDER, tau=TAU):
    x = np.asarray(x, dtype=float)
    emb = _embed_signal(x, dim=order, tau=tau)
    if len(emb) < 2:
        return np.nan
    patterns = np.argsort(emb, axis=1)
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    prob = counts / counts.sum()
    entropy = -np.sum(prob * np.log2(prob))
    return float(entropy / np.log2(math.factorial(order)))


def _nearest_neighbors(embedded, theiler=10):
    n = len(embedded)
    if n < 3:
        return np.full(n, -1, dtype=int), np.full(n, np.nan, dtype=float)
    neighbors = np.full(n, -1, dtype=int)
    distances = np.full(n, np.nan, dtype=float)
    for i in range(n):
        dist = np.linalg.norm(embedded - embedded[i], axis=1)
        lo = max(0, i - theiler)
        hi = min(n, i + theiler + 1)
        dist[lo:hi] = np.inf
        j = int(np.argmin(dist))
        if np.isfinite(dist[j]) and dist[j] > 0:
            neighbors[i] = j
            distances[i] = dist[j]
    return neighbors, distances


def _rosenstein_lyapunov(x, fs_hz, dim=EMBED_DIM, tau=TAU, max_k=20):
    embedded = _embed_signal(x, dim=dim, tau=tau)
    n = len(embedded)
    if n < max_k + 5 or not np.isfinite(fs_hz) or fs_hz <= 0:
        return np.nan
    neighbors, _ = _nearest_neighbors(embedded, theiler=max(10, dim * tau))
    divergence = []
    k_values = []
    for k in range(1, min(max_k, n - 2)):
        vals = []
        for i, j in enumerate(neighbors):
            if j < 0 or i + k >= n or j + k >= n:
                continue
            d = np.linalg.norm(embedded[i + k] - embedded[j + k])
            if np.isfinite(d) and d > 0:
                vals.append(np.log(d))
        if vals:
            divergence.append(np.mean(vals))
            k_values.append(k / fs_hz)
    if len(divergence) < 3:
        return np.nan
    fit_n = max(3, min(8, len(divergence)))
    slope, _ = np.polyfit(k_values[:fit_n], divergence[:fit_n], 1)
    return float(slope)


def _wolf_lyapunov(x, fs_hz, dim=EMBED_DIM, tau=TAU):
    embedded = _embed_signal(x, dim=dim, tau=tau)
    n = len(embedded)
    if n < 5 or not np.isfinite(fs_hz) or fs_hz <= 0:
        return np.nan
    neighbors, dist_now = _nearest_neighbors(embedded, theiler=max(10, dim * tau))
    rates = []
    for i, j in enumerate(neighbors):
        if j < 0 or i + 1 >= n or j + 1 >= n or not np.isfinite(dist_now[i]) or dist_now[i] <= 0:
            continue
        dist_next = np.linalg.norm(embedded[i + 1] - embedded[j + 1])
        if np.isfinite(dist_next) and dist_next > 0:
            rates.append(np.log(dist_next / dist_now[i]) * fs_hz)
    return float(np.nanmean(rates)) if rates else np.nan


def _line_lengths(mask):
    lengths = []
    n = mask.shape[0]
    for offset in range(-n + 1, n):
        diag = np.diagonal(mask, offset=offset)
        run = 0
        for value in diag:
            if value:
                run += 1
            elif run:
                lengths.append(run)
                run = 0
        if run:
            lengths.append(run)
    return np.asarray(lengths, dtype=int)


def _vertical_lengths(mask, target=True):
    lengths = []
    for col in range(mask.shape[1]):
        run = 0
        for value in mask[:, col] == target:
            if value:
                run += 1
            elif run:
                lengths.append(run)
                run = 0
        if run:
            lengths.append(run)
    return np.asarray(lengths, dtype=int)


def _length_entropy(lengths):
    lengths = np.asarray(lengths, dtype=int)
    lengths = lengths[lengths > 0]
    if len(lengths) == 0:
        return np.nan
    _, counts = np.unique(lengths, return_counts=True)
    prob = counts / counts.sum()
    return float(-np.sum(prob * np.log2(prob)))


def _rqa_features(x, dim=EMBED_DIM, tau=TAU, radius_percentile=10.0, min_line=2):
    empty = {name: np.nan for name in ["REC", "DET", "LAM", "MeanL", "MaxL", "EntrL", "EntrV", "EntrW"]}
    embedded = _embed_signal(x, dim=dim, tau=tau)
    n = len(embedded)
    if n < 5:
        return empty
    diff = embedded[:, None, :] - embedded[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    nonzero_dist = dist[np.triu_indices(n, k=1)]
    nonzero_dist = nonzero_dist[np.isfinite(nonzero_dist) & (nonzero_dist > 0)]
    if len(nonzero_dist) == 0:
        return empty
    radius = np.percentile(nonzero_dist, radius_percentile)
    recurrence = dist <= radius
    np.fill_diagonal(recurrence, False)

    rec_points = float(np.sum(recurrence))
    rec_rate = rec_points / float(n * (n - 1)) if n > 1 else np.nan

    diag_lengths = _line_lengths(recurrence)
    diag_valid = diag_lengths[diag_lengths >= min_line]
    vert_lengths = _vertical_lengths(recurrence, target=True)
    vert_valid = vert_lengths[vert_lengths >= min_line]
    white_lengths = _vertical_lengths(recurrence, target=False)
    white_valid = white_lengths[white_lengths >= min_line]

    det = float(np.sum(diag_valid) / rec_points) if rec_points > 0 and len(diag_valid) else np.nan
    lam = float(np.sum(vert_valid) / rec_points) if rec_points > 0 and len(vert_valid) else np.nan

    return {
        "REC": rec_rate,
        "DET": det,
        "LAM": lam,
        "MeanL": float(np.mean(diag_valid)) if len(diag_valid) else np.nan,
        "MaxL": float(np.max(diag_valid)) if len(diag_valid) else np.nan,
        "EntrL": _length_entropy(diag_valid),
        "EntrV": _length_entropy(vert_valid),
        "EntrW": _length_entropy(white_valid),
    }


def extract_walk_nonlinear_features(df, meta, config, max_samples=NONLINEAR_MAX_SAMPLES):
    """Extract nonlinear walking features from the detected walking window."""
    if df is None or meta is None or len(df) < config.min_rows_per_activity:
        return build_nan_feature_dict(WALK_NONLINEAR_FEATURE_NAMES)

    df, meta = truncate_activity_dataframe(df, meta, "walk")
    out = build_nan_feature_dict(WALK_NONLINEAR_FEATURE_NAMES)
    time_s = df["time_s"].to_numpy(dtype=float)
    acc_signal, _ = select_motion_acc_signal(df, config.prefer_useracc_for_motion)
    walk_window = _walk_summary_for_features(time_s, acc_signal, meta, config)

    start_t = float(walk_window.get("start_time_s", np.nan))
    end_t = float(walk_window.get("end_time_s", np.nan))
    if np.isfinite(start_t) and np.isfinite(end_t):
        mask = (time_s >= start_t) & (time_s <= end_t)
        nonlinear_signal = acc_signal[mask]
    else:
        nonlinear_signal = acc_signal

    x = _finite_centered_signal(nonlinear_signal, max_samples=max_samples)
    if len(x) < 10:
        return out

    fs_hz = float(meta.fs_hz)
    r = 0.2 * np.nanstd(x)
    rqa = _rqa_features(x)

    out["approximate_entropy"] = _approximate_entropy(x, m=ENTROPY_M, r=r)
    out["sample_entropy"] = _sample_entropy(x, m=ENTROPY_M, r=r)
    out["symbolic_entropy"] = _symbolic_entropy(x, word_length=2, n_bins=4)
    out["permutation_entropy"] = _permutation_entropy(x, order=PERMUTATION_ORDER, tau=TAU)
    out["rosenstein_lyapunov_exponent"] = _rosenstein_lyapunov(x, fs_hz)
    out["wolf_lyapunov_exponent"] = _wolf_lyapunov(x, fs_hz)
    for feature in ["REC", "DET", "LAM", "MeanL", "MaxL", "EntrL", "EntrV", "EntrW"]:
        out[f"rqa_{feature}"] = rqa.get(feature, np.nan)

    return out


def _walk_summary_for_features(time_s, acc_signal, meta, config):
    """Use the same walk summary logic as the validation notebook."""
    try:
        from validation_plots import shared_validation as validation

        if validation.PIPELINE_CONFIG is not config:
            validation.configure(config, validation.plot_params_from_config(config))
        summary = validation.exact_compare_walk_summary(time_s, acc_signal, meta.fs_hz)
        if np.isfinite(float(summary.get("start_time_s", np.nan))) and np.isfinite(
            float(summary.get("end_time_s", np.nan))
        ):
            return summary
    except Exception:
        pass

    return peak_step_summary(
        time_s=time_s,
        signal=acc_signal,
        window_sec=config.window_gate.window_sec,
        threshold_k=config.window_gate.walk_threshold_k,
        fallback_min_amp=config.window_gate.walk_min_amp_threshold,
        min_duration_s=config.window_gate.walk_min_duration_s,
        smooth_sigma=1.0,
        min_peak_distance_s=0.3,
    )


def extract_walk_features(df, meta, config):
    """Extract gait and signal features from walk activity."""
    if df is None or meta is None or len(df) < config.min_rows_per_activity:
        return build_nan_feature_dict(WALK_FEATURE_NAMES)

    df, meta = truncate_activity_dataframe(df, meta, "walk")
    out = build_nan_feature_dict(WALK_FEATURE_NAMES)
    time_s = df["time_s"].to_numpy(dtype=float)
    acc_signal, acc_source = select_motion_acc_signal(df, config.prefer_useracc_for_motion)
    wavelet = _walk_summary_for_features(time_s, acc_signal, meta, config)

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
    peak_times = np.asarray(wavelet.get("peak_times", []), dtype=float)
    peak_times = peak_times[np.isfinite(peak_times)]
    if len(peak_times) >= 2:
        step_times = np.diff(peak_times)
    else:
        active_cad = np.asarray(wavelet.get("cad", []), dtype=float)
        active_cad = active_cad[np.isfinite(active_cad) & (active_cad > 0)]
        step_times = 1.0 / active_cad if len(active_cad) else np.array([], dtype=float)
    if len(step_times):
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

    jerk_col = "useracc_jerk" if acc_source == "useracc_mag" else "acc_gravity_removed_jerk"
    if jerk_col not in df:
        jerk_col = "acc_jerk"
    jerk = df[jerk_col].to_numpy(dtype=float)
    if np.any(walk_mask):
        jerk = jerk[walk_mask]
    out["walk_jerk_mean"] = float(np.nanmean(np.abs(jerk)))
    out["walk_jerk_std"] = float(np.nanstd(jerk))

    return out
