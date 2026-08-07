from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


PELVIS_MARKERS = ("LASIS", "RASIS", "LPSIS", "RPSIS")
SHOULDER_MARKERS = ("LSHO", "RSHO")
FOOT_MARKERS = ("RHEE", "RMT2", "LHEE", "LMT2")


def _to_float(value):
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _clean_marker_name(raw_name):
    name = str(raw_name).strip()
    if ":" in name:
        name = name.split(":", 1)[1]
    return name


def load_vicon_marker_trajectories(csv_path):
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        rows = list(csv.reader(f))

    traj_idx = None
    for i, row in enumerate(rows):
        if row and row[0].strip().lower() == "trajectories":
            traj_idx = i
            break
    if traj_idx is None:
        raise ValueError(f"No Trajectories block found in {csv_path}")

    fs_hz = _to_float(rows[traj_idx + 1][0] if rows[traj_idx + 1] else np.nan)
    marker_row = rows[traj_idx + 2]
    unit_row = rows[traj_idx + 4]
    data_rows = rows[traj_idx + 5 :]

    marker_cols = []
    for col_idx, raw_name in enumerate(marker_row):
        if str(raw_name).strip():
            marker_cols.append((col_idx, _clean_marker_name(raw_name)))

    records = []
    for row in data_rows:
        if len(row) < 2:
            continue
        frame = _to_float(row[0])
        sub_frame = _to_float(row[1])
        if not np.isfinite(frame):
            continue
        record = {"frame": int(frame), "sub_frame": int(sub_frame) if np.isfinite(sub_frame) else 0}
        for start_col, marker in marker_cols:
            values = []
            for offset in range(3):
                idx = start_col + offset
                value = row[idx] if idx < len(row) else ""
                values.append(_to_float(value))
            record[f"{marker}_X"] = values[0]
            record[f"{marker}_Y"] = values[1]
            record[f"{marker}_Z"] = values[2]
        records.append(record)

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No trajectory rows found in {csv_path}")

    if pd.notna(fs_hz) and fs_hz > 0:
        fs_hz = float(fs_hz)
        df["time_s"] = (df["frame"] - df["frame"].iloc[0]) / fs_hz
    else:
        fs_hz = np.nan
        df["time_s"] = np.arange(len(df), dtype=float)

    units = "mm" if any("mm" in str(x).lower() for x in unit_row) else "unknown"
    return df, {
        "csv_path": csv_path,
        "fs_hz": fs_hz,
        "markers": [marker for _, marker in marker_cols],
        "units": units,
    }


def existing_markers(df, markers):
    return [m for m in markers if all(f"{m}_{axis}" in df for axis in ("X", "Y", "Z"))]


def marker_center(df, markers):
    markers = existing_markers(df, markers)
    if not markers:
        return pd.DataFrame(index=df.index, columns=["X", "Y", "Z"], dtype=float)
    out = {}
    for axis in ("X", "Y", "Z"):
        cols = [f"{m}_{axis}" for m in markers]
        out[axis] = df[cols].mean(axis=1, skipna=True)
    return pd.DataFrame(out)


def robust_mad(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    med = np.nanmedian(arr)
    return 1.4826 * np.nanmedian(np.abs(arr - med))


def _contiguous_regions(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(changes[::2], changes[1::2]))


def motion_window(time_s, signal, fs_hz, threshold_k=3.0, min_duration_s=0.25, pad_s=0.0):
    time_s = np.asarray(time_s, dtype=float)
    sig = np.asarray(signal, dtype=float)
    finite = np.isfinite(sig)
    if finite.sum() < 3:
        return 0, len(time_s) - 1

    smooth = gaussian_filter1d(np.nan_to_num(sig, nan=np.nanmedian(sig[finite])), sigma=max(fs_hz * 0.04, 1))
    baseline = np.nanmedian(smooth)
    amp = np.abs(smooth - baseline)
    thr = np.nanmedian(amp) + threshold_k * robust_mad(amp)
    min_len = max(int(round(min_duration_s * fs_hz)), 1) if np.isfinite(fs_hz) and fs_hz > 0 else 1
    regions = [(s, e) for s, e in _contiguous_regions(amp > thr) if e - s >= min_len]
    if not regions:
        idx = int(np.nanargmax(amp))
        return max(0, idx - min_len), min(len(time_s) - 1, idx + min_len)

    # Keep the largest sustained motion region, then extend to the nearest baseline crossing.
    start, end = max(regions, key=lambda r: np.nanmax(amp[r[0] : r[1]]) * (r[1] - r[0]))
    exit_thr = max(thr * 0.35, np.nanmedian(amp) + 0.5 * robust_mad(amp))
    while start > 0 and amp[start] > exit_thr:
        start -= 1
    while end < len(time_s) - 1 and amp[end] > exit_thr:
        end += 1
    pad = int(round(pad_s * fs_hz)) if np.isfinite(fs_hz) and fs_hz > 0 else 0
    return max(0, start - pad), min(len(time_s) - 1, end + pad)


def count_foot_steps(df, fs_hz, start_idx=0, end_idx=None, min_interval_s=0.30):
    if end_idx is None:
        end_idx = len(df) - 1
    step_counts = []
    peak_times = []
    for marker in ("RHEE", "LHEE", "RMT2", "LMT2"):
        if not all(f"{marker}_{axis}" in df for axis in ("X", "Y", "Z")):
            continue
        xyz = df.loc[start_idx:end_idx, [f"{marker}_X", f"{marker}_Y", f"{marker}_Z"]].interpolate(limit_direction="both")
        if len(xyz) < 5:
            continue
        # Foot speed is less axis-dependent than raw Z peaks and works for both walking and turning.
        arr = xyz.to_numpy(dtype=float)
        dt = 1.0 / fs_hz if np.isfinite(fs_hz) and fs_hz > 0 else np.nanmedian(np.diff(df["time_s"]))
        speed = np.linalg.norm(np.gradient(arr, dt, axis=0), axis=1)
        smooth = gaussian_filter1d(speed, sigma=max(fs_hz * 0.035, 1))
        thr = np.nanmedian(smooth) + 1.5 * robust_mad(smooth)
        prominence = max(robust_mad(smooth) * 1.0, np.nanstd(smooth) * 0.15)
        distance = max(int(round(min_interval_s * fs_hz)), 1)
        peaks, _ = find_peaks(smooth, height=thr, prominence=prominence, distance=distance)
        step_counts.append(len(peaks))
        if len(peaks):
            peak_times.extend(df["time_s"].iloc[start_idx:end_idx + 1].iloc[peaks].to_numpy(dtype=float))

    if not step_counts:
        return 0, []
    # Heel and toe markers on the same foot often duplicate each other; average across available foot markers.
    count = int(round(float(np.nanmean(step_counts)) * 2.0))
    return count, sorted(float(t) for t in peak_times)


def _merge_event_times(times, merge_gap_s=0.18):
    times = sorted(float(t) for t in times if np.isfinite(t))
    if not times:
        return []
    clusters = [[times[0]]]
    for value in times[1:]:
        if value - clusters[-1][-1] <= merge_gap_s:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [float(np.mean(cluster)) for cluster in clusters]


def ten_step_window_from_foot_markers(df, fs_hz, target_steps=10):
    _, peak_times = count_foot_steps(df, fs_hz, 0, len(df) - 1, min_interval_s=0.35)
    step_times = _merge_event_times(peak_times, merge_gap_s=0.18)
    if len(step_times) < target_steps:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    selected = step_times[:target_steps]
    start = float(selected[0])
    end = float(selected[-1])
    duration = float(end - start)
    cadence = float(target_steps / duration * 60.0) if duration > 0 else np.nan
    mean_step = float(duration / target_steps) if duration > 0 else np.nan
    return start, end, duration, cadence, mean_step


def summarize_nw(csv_path):
    df, meta = load_vicon_marker_trajectories(csv_path)
    fs = meta["fs_hz"]
    time_s = df["time_s"].to_numpy(dtype=float)
    ten_start, ten_end, ten_duration, ten_cadence, ten_mean_step = ten_step_window_from_foot_markers(df, fs)
    ten_count = 10.0 if np.isfinite(ten_duration) else np.nan
    return {
        "activity": "walk",
        "start_time_s": round(ten_start, 3) if np.isfinite(ten_start) else np.nan,
        "end_time_s": round(ten_end, 3) if np.isfinite(ten_end) else np.nan,
        "duration_s": round(ten_duration, 3) if np.isfinite(ten_duration) else np.nan,
        "step_count": ten_count,
        "cadence_steps_min": round(ten_cadence, 3) if np.isfinite(ten_cadence) else np.nan,
        "mean_step_time_s": round(ten_mean_step, 3) if np.isfinite(ten_mean_step) else np.nan,
        "ten_step_start_time_s": round(ten_start, 3) if np.isfinite(ten_start) else np.nan,
        "ten_step_end_time_s": round(ten_end, 3) if np.isfinite(ten_end) else np.nan,
        "ten_step_duration_s": round(ten_duration, 3) if np.isfinite(ten_duration) else np.nan,
        "ten_step_count": ten_count,
        "ten_step_cadence_steps_min": round(ten_cadence, 3) if np.isfinite(ten_cadence) else np.nan,
        "ten_step_mean_step_time_s": round(ten_mean_step, 3) if np.isfinite(ten_mean_step) else np.nan,
        "time_to_peak_s": np.nan,
    }


def summarize_turn(csv_path, side):
    df, meta = load_vicon_marker_trajectories(csv_path)
    fs = meta["fs_hz"]
    time_s = df["time_s"].to_numpy(dtype=float)
    pelvis = marker_center(df, PELVIS_MARKERS).interpolate(limit_direction="both")
    xy = pelvis[["X", "Y"]].to_numpy(dtype=float)
    dt = 1.0 / fs if np.isfinite(fs) and fs > 0 else np.nanmedian(np.diff(time_s))
    speed = np.linalg.norm(np.gradient(xy, dt, axis=0), axis=1)
    smooth_speed = gaussian_filter1d(speed, sigma=max(fs * 0.08, 1))
    amp = np.abs(smooth_speed - np.nanmedian(smooth_speed))
    thr = np.nanmedian(amp) + 0.5 * robust_mad(amp)
    min_len = max(int(round(0.20 * fs)), 1) if np.isfinite(fs) and fs > 0 else 1
    regions = [(s, e) for s, e in _contiguous_regions(amp > thr) if e - s >= min_len]
    if regions:
        start_idx, end_idx = regions[0][0], regions[-1][1]
        pad = max(int(round(0.15 * fs)), 1) if np.isfinite(fs) and fs > 0 else 0
        start_idx = max(0, start_idx - pad)
        end_idx = min(len(time_s) - 1, end_idx + pad)
    else:
        start_idx, end_idx = motion_window(time_s, speed, fs, threshold_k=2.0, min_duration_s=0.40, pad_s=0.15)
    duration = float(time_s[end_idx] - time_s[start_idx])
    window_speed = smooth_speed[start_idx : end_idx + 1]
    peak_rel = int(np.nanargmax(window_speed)) if len(window_speed) else 0
    time_to_peak = float(time_s[start_idx + peak_rel] - time_s[start_idx])
    step_count, _ = count_foot_steps(df, fs, start_idx, end_idx, min_interval_s=0.30)
    return {
        "activity": side,
        "start_time_s": round(float(time_s[start_idx]), 3),
        "end_time_s": round(float(time_s[end_idx]), 3),
        "duration_s": round(duration, 3),
        "step_count": float(step_count),
        "cadence_steps_min": np.nan,
        "mean_step_time_s": np.nan,
        "time_to_peak_s": round(time_to_peak, 3),
    }


def summarize_sts(csv_path):
    df, meta = load_vicon_marker_trajectories(csv_path)
    fs = meta["fs_hz"]
    time_s = df["time_s"].to_numpy(dtype=float)
    body = marker_center(df, PELVIS_MARKERS + SHOULDER_MARKERS).interpolate(limit_direction="both")
    z = body["Z"].to_numpy(dtype=float)
    smooth_z = gaussian_filter1d(z, sigma=max(fs * 0.06, 1))
    dt = 1.0 / fs if np.isfinite(fs) and fs > 0 else np.nanmedian(np.diff(time_s))
    velocity = np.gradient(smooth_z, dt)
    start_idx, end_idx = motion_window(time_s, velocity, fs, threshold_k=2.5, min_duration_s=0.30, pad_s=0.05)
    duration = float(time_s[end_idx] - time_s[start_idx])
    window_velocity = velocity[start_idx : end_idx + 1]
    peak_rel = int(np.nanargmax(np.abs(window_velocity))) if len(window_velocity) else 0
    time_to_peak = float(time_s[start_idx + peak_rel] - time_s[start_idx])
    return {
        "activity": "sit_to_stand",
        "start_time_s": round(float(time_s[start_idx]), 3),
        "end_time_s": round(float(time_s[end_idx]), 3),
        "duration_s": round(duration, 3),
        "step_count": np.nan,
        "cadence_steps_min": np.nan,
        "mean_step_time_s": np.nan,
        "time_to_peak_s": round(time_to_peak, 3),
    }


def activity_from_trial_name(name):
    key = name.lower()
    if key.startswith("nw"):
        return "walk"
    if key.startswith("sts"):
        return "sit_to_stand"
    if key.startswith("turnl"):
        return "left_turn"
    if key.startswith("turnr"):
        return "right_turn"
    return None


def trial_sort_key(path):
    name = path.parent.name
    match = re.search(r"(\d+)", name)
    number = int(match.group(1)) if match else 9999
    return (activity_from_trial_name(name) or "", number, name.lower())


def summarize_vicon_trial(csv_path):
    name = Path(csv_path).parent.name
    activity = activity_from_trial_name(name)
    if activity == "walk":
        return summarize_nw(csv_path)
    if activity == "left_turn":
        return summarize_turn(csv_path, "left_turn")
    if activity == "right_turn":
        return summarize_turn(csv_path, "right_turn")
    if activity == "sit_to_stand":
        return summarize_sts(csv_path)
    raise ValueError(f"Unsupported Vicon activity: {csv_path}")
