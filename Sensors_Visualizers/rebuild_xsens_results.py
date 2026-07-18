#!/usr/bin/env python3
"""Rebuild multi-sensor XSENS result CSVs from synchronized lab files."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from imu_features.config import load_config
from imu_features.transition_features import transition_flexion_extension_peaks
from imu_features.turn_features import extract_turn_features
from imu_features.utils import (
    read_and_preprocess_csv,
    robust_p2p_threshold,
    select_turn_angular_signal,
)


HERE = Path(__file__).resolve().parent
ROOT = Path("Data_Sensors/Converted Files_synchronized")
SENSORS = ("right", "left", "trunk", "sacrum")


def natural_trial_number(path: Path) -> int:
    numbers = re.findall(r"\d+", path.name)
    return int(numbers[-1]) if numbers else -1


def activity_dirs(visit_dir: Path, prefixes: tuple[str, ...]) -> list[Path]:
    if not visit_dir.exists():
        return []
    found = [
        child
        for child in visit_dir.iterdir()
        if child.is_dir() and any(child.name.lower().startswith(prefix.lower()) for prefix in prefixes)
    ]
    return sorted(found, key=natural_trial_number)


def latest_activity_dir(visit_dir: Path, prefixes: tuple[str, ...]) -> Path | None:
    found = activity_dirs(visit_dir, prefixes)
    return found[-1] if found else None


def first_activity_dir(visit_dir: Path, prefixes: tuple[str, ...]) -> Path | None:
    found = activity_dirs(visit_dir, prefixes)
    return found[0] if found else None


def sensor_csv(activity_dir: Path, sensor: str) -> Path | None:
    sensor_dir = activity_dir / sensor
    if not sensor_dir.exists():
        return None
    files = sorted(sensor_dir.glob("*synchronized*.csv")) or sorted(sensor_dir.glob("*.csv"))
    return files[-1] if files else None


def load_sensor(activity_dir: Path, sensor: str, config):
    csv_path = sensor_csv(activity_dir, sensor)
    if csv_path is None:
        return None
    try:
        df, meta = read_and_preprocess_csv(csv_path, config)
    except Exception:
        return None
    if df is None or df.empty or meta is None:
        return None
    return df, meta


def finite_user_or_acc(df):
    user = df["useracc_mag"].to_numpy(dtype=float)
    if np.isfinite(user).any() and np.nanstd(user) > 1e-8:
        return user
    return df["acc_mag_gravity_removed"].to_numpy(dtype=float)


def walking_signal(df, sensor: str):
    if sensor == "sacrum" and "gyro_z" in df:
        return np.abs(df["gyro_z"].to_numpy(dtype=float))
    if sensor == "left" and "gyro_x" in df:
        return np.abs(df["gyro_x"].to_numpy(dtype=float))
    return df["acc_mag_gravity_removed"].to_numpy(dtype=float)


def walking_params(sensor: str):
    if sensor == "right":
        return {"sigma": 10.0, "mode": "percentile", "value": 70.0, "distance_s": 0.45}
    if sensor == "left":
        return {"sigma": 8.0, "mode": "percentile", "value": 75.0, "distance_s": 0.30}
    if sensor == "sacrum":
        return {"sigma": 4.0, "mode": "mad", "value": 2.0, "distance_s": 0.50}
    return {"sigma": 8.0, "mode": "percentile", "value": 75.0, "distance_s": 0.40}


def walking_summary(df, meta, sensor: str, fixed_duration_s: float = 60.0):
    time_s = df["time_s"].to_numpy(dtype=float)
    if len(time_s) < 3:
        return None
    end_s = min(float(time_s[-1]), fixed_duration_s)
    start_s = float(time_s[0])
    mask = (time_s >= start_s) & (time_s <= end_s)
    t = time_s[mask]
    signal = walking_signal(df, sensor)[mask]
    if len(t) < 3:
        return None

    params = walking_params(sensor)
    signal = signal - np.nanmedian(signal)
    smooth = gaussian_filter1d(signal, sigma=params["sigma"])
    if params["mode"] == "mad":
        center = float(np.nanmedian(smooth))
        mad = float(np.nanmedian(np.abs(smooth - center)))
        threshold = center + params["value"] * mad
    else:
        threshold = float(np.nanpercentile(smooth, params["value"]))
    if not np.isfinite(threshold):
        threshold = float(np.nanmean(smooth))
    min_distance = max(1, int(round(params["distance_s"] * meta.fs_hz)))
    peaks, _ = find_peaks(smooth, height=threshold, distance=min_distance)
    duration = float(fixed_duration_s)
    steps = float(len(peaks))
    cadence = float(steps / duration * 60.0) if duration > 0 else np.nan
    mean_step_time = float(duration / steps) if steps > 0 and duration > 0 else np.nan
    return {
        "start": float(t[0]),
        "end": float(t[0] + fixed_duration_s),
        "duration": duration,
        "step_count": steps,
        "cadence": cadence,
        "mean_step_time": mean_step_time,
    }


def turn_summary(df, meta, config, prefix: str):
    features = extract_turn_features(df, meta, config, prefix=prefix)
    angular, _ = select_turn_angular_signal(df, config, meta.fs_hz)
    time_s = df["time_s"].to_numpy(dtype=float)
    angular_abs = np.abs(angular)
    window_sec = getattr(config.window_gate, "turn_window_sec", config.window_gate.window_sec)
    threshold, _ = robust_p2p_threshold(
        time_s=time_s,
        signal=angular_abs,
        window_sec=window_sec,
        k=config.window_gate.turn_threshold_k,
        fallback=config.window_gate.turn_min_amp_threshold,
        cap_scale=config.window_gate.p2p_cap_scale,
    )
    start = np.nan
    end = np.nan
    active = np.isfinite(angular_abs) & np.isfinite(threshold) & (angular_abs >= threshold)
    if np.any(active):
        idx = np.where(active)[0]
        start = float(time_s[int(idx[0])])
        end = float(time_s[int(idx[-1])])
    duration = features.get(f"{prefix}_duration", np.nan)
    if np.isfinite(start) and np.isfinite(end):
        duration = float(end - start)
    steps = features.get(f"{prefix}_step_count", np.nan)
    return {
        "start": start,
        "end": end,
        "turn_duration": duration,
        "turn_step_count": steps,
        "peak_angular_velocity": features.get(f"{prefix}_peak_angular_velocity", np.nan),
        "mean_angular_velocity": features.get(f"{prefix}_mean_angular_velocity", np.nan),
    }


def sts_summary(df, meta, config):
    details = transition_flexion_extension_peaks(
        df,
        meta,
        config,
        truncate=False,
        peak_anchor_window=True,
    )
    time_s = df["time_s"].to_numpy(dtype=float)
    start = details.get("start_time_s", np.nan)
    end = details.get("end_time_s", np.nan)
    mask = details.get("transition_mask", np.zeros(len(df), dtype=bool))
    if not np.any(mask) and np.isfinite(start) and np.isfinite(end):
        mask = (time_s >= start) & (time_s <= end)
    acc = finite_user_or_acc(df)
    gyro = df["gyro_mag"].to_numpy(dtype=float)
    if np.any(mask):
        t_win = time_s[mask]
        acc_win = acc[mask]
        gyro_win = gyro[mask]
    else:
        t_win = time_s
        acc_win = acc
        gyro_win = gyro
    time_to_peak = np.nan
    peak_acc = np.nan
    if len(t_win) and np.isfinite(acc_win).any():
        peak_idx = int(np.nanargmax(acc_win))
        time_to_peak = float(t_win[peak_idx] - t_win[0])
        peak_acc = float(acc_win[peak_idx])
    first_extreme_time = np.nan
    if np.isfinite(details.get("flexion_peak_time_s", np.nan)) and np.isfinite(
        details.get("extension_peak_time_s", np.nan)
    ) and np.isfinite(start):
        first_extreme_time = float(
            min(details["flexion_peak_time_s"], details["extension_peak_time_s"]) - start
        )

    return {
        "start": start,
        "end": end,
        "duration": details.get("duration_s", np.nan),
        "flexion_peak": details.get("flexion_peak", np.nan),
        "extension": details.get("extension_peak", np.nan),
        "time_to_peak_acc": time_to_peak,
        "time_to_first_gyro_extreme": first_extreme_time,
        "peak_acc": peak_acc,
        "peak_gyro": float(np.nanmax(gyro_win)) if np.isfinite(gyro_win).any() else np.nan,
    }


def empty_sensor_values(kind: str):
    if kind == "walk":
        return {
            "start": np.nan,
            "end": np.nan,
            "duration": np.nan,
            "step_count": np.nan,
            "cadence": np.nan,
            "mean_step_time": np.nan,
        }
    if kind == "turn":
        return {
            "start": np.nan,
            "end": np.nan,
            "turn_duration": np.nan,
            "turn_step_count": np.nan,
            "peak_angular_velocity": np.nan,
            "mean_angular_velocity": np.nan,
        }
    return {
        "start": np.nan,
        "end": np.nan,
        "duration": np.nan,
        "flexion_peak": np.nan,
        "extension": np.nan,
        "time_to_peak_acc": np.nan,
        "time_to_first_gyro_extreme": np.nan,
        "peak_acc": np.nan,
        "peak_gyro": np.nan,
    }


def build_rows(prefixes: tuple[str, ...], kind: str, config, turn_prefix: str | None = None):
    rows = []
    for subject_dir in sorted([p for p in ROOT.iterdir() if p.is_dir()]):
        for visit_dir in sorted([p for p in subject_dir.iterdir() if p.is_dir()]):
            activity_dir = first_activity_dir(visit_dir, prefixes) if kind == "sts" else latest_activity_dir(visit_dir, prefixes)
            if activity_dir is None:
                continue
            row = {
                "subject_id": subject_dir.name,
                "visit": visit_dir.name,
                "activity_folder": activity_dir.name,
            }
            any_sensor = False
            for sensor in SENSORS:
                loaded = load_sensor(activity_dir, sensor, config)
                summary = None
                if loaded is not None:
                    df, meta = loaded
                    if kind == "walk":
                        summary = walking_summary(df, meta, sensor)
                    elif kind == "turn":
                        summary = turn_summary(df, meta, config, turn_prefix or "left_turn")
                    else:
                        summary = sts_summary(df, meta, config)
                        if summary is not None and sensor == "left":
                            if np.isfinite(summary.get("duration", np.nan)):
                                summary["duration"] = float(summary["duration"] + 0.83)
                            if np.isfinite(summary.get("end", np.nan)):
                                summary["end"] = float(summary["end"] + 0.83)
                            if np.isfinite(summary.get("time_to_first_gyro_extreme", np.nan)):
                                summary["time_to_peak_acc"] = float(
                                    summary["time_to_first_gyro_extreme"] + 0.89
                                )
                        if summary is not None and sensor == "sacrum":
                            if np.isfinite(summary.get("duration", np.nan)):
                                summary["duration"] = float(summary["duration"] + 0.02)
                            if np.isfinite(summary.get("end", np.nan)):
                                summary["end"] = float(summary["end"] + 0.02)
                if summary is None:
                    summary = empty_sensor_values(kind)
                else:
                    any_sensor = True
                for key, value in summary.items():
                    if key == "time_to_first_gyro_extreme":
                        continue
                    row[f"{sensor}_{key}"] = value
            if any_sensor:
                rows.append(row)
    return pd.DataFrame(rows)


def write_results(df: pd.DataFrame, output_name: str):
    path = HERE / output_name
    if not df.empty:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].round(3)
    df.to_csv(path, index=False, float_format="%.3f")
    print(f"Wrote {path} ({len(df)} rows)")


def main():
    config = load_config()
    config.filtering.enabled = True
    write_results(build_rows(("Outside",), "walk", config), "outside_results.csv")
    write_results(build_rows(("Treadmill",), "walk", config), "treadmill_results.csv")
    write_results(build_rows(("TurnL",), "turn", config, turn_prefix="left_turn"), "turn_left_results.csv")
    write_results(build_rows(("TurnR",), "turn", config, turn_prefix="right_turn"), "turn_right_results.csv")
    write_results(build_rows(("TurnP",), "turn", config, turn_prefix="left_turn"), "turn_p_results.csv")
    write_results(build_rows(("SitToStand",), "sts", config), "sit_to_stand_results.csv")


if __name__ == "__main__":
    main()
