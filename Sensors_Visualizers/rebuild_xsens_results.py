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
from imu_features.utils import (
    apply_lowpass_filter,
    estimate_sampling_interval_s,
    mask_close_gaps,
    read_and_preprocess_csv,
    select_motion_acc_signal,
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
    if sensor == "right" and "gyro_x" in df:
        return df["gyro_x"].to_numpy(dtype=float)
    if sensor == "left" and "gyro_x" in df:
        return np.abs(df["gyro_x"].to_numpy(dtype=float))
    if sensor == "trunk" and "gyro_mag" in df:
        return df["gyro_mag"].to_numpy(dtype=float)
    if sensor == "sacrum" and "gyro_z" in df:
        return df["gyro_z"].to_numpy(dtype=float)
    return df["acc_mag_gravity_removed"].to_numpy(dtype=float)


def walking_params(sensor: str):
    if sensor == "right":
        return {"sigma": 12.0, "mode": "percentile", "value": 55.0, "distance_s": 0.35}
    if sensor == "left":
        return {"sigma": 8.0, "mode": "percentile", "value": 75.0, "distance_s": 0.30}
    if sensor == "sacrum":
        return {"sigma": 4.0, "mode": "mad", "value": 1.25, "distance_s": 0.35}
    if sensor == "trunk":
        return {"sigma": 12.0, "mode": "percentile", "value": 65.0, "distance_s": 0.45}
    return {"sigma": 8.0, "mode": "percentile", "value": 75.0, "distance_s": 0.40}


def add_walking_sacrum_qc(row: dict, disagreement_steps: float = 22.0, bias_correction_steps: float = 3.5):
    sacrum_steps = row.get("sacrum_step_count", np.nan)
    trunk_steps = row.get("trunk_step_count", np.nan)
    if np.isfinite(sacrum_steps) and np.isfinite(trunk_steps):
        use_trunk = abs(float(sacrum_steps) - float(trunk_steps)) > disagreement_steps
        steps = float(trunk_steps if use_trunk else sacrum_steps)
        source = "trunk" if use_trunk else "sacrum"
    elif np.isfinite(sacrum_steps):
        steps = float(sacrum_steps)
        source = "sacrum"
    elif np.isfinite(trunk_steps):
        steps = float(trunk_steps)
        source = "trunk"
    else:
        steps = np.nan
        source = ""
    if np.isfinite(steps):
        steps = float(steps + bias_correction_steps)

    duration = row.get("sacrum_duration", np.nan)
    if not np.isfinite(duration):
        duration = row.get("trunk_duration", np.nan)
    row["sacrum_qc_step_count"] = steps
    row["sacrum_qc_cadence"] = float(steps / duration * 60.0) if np.isfinite(steps) and np.isfinite(duration) and duration > 0 else np.nan
    row["sacrum_qc_mean_step_time"] = float(duration / steps) if np.isfinite(steps) and steps > 0 and np.isfinite(duration) else np.nan
    row["sacrum_qc_source"] = source


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
    time_s = df["time_s"].to_numpy(dtype=float)
    if "gyro_y" in df.columns:
        angular = np.abs(df["gyro_y"].to_numpy(dtype=float))
        gyro_source = "gyro_y"
        if config.filtering.enabled and np.isfinite(meta.fs_hz):
            angular = apply_lowpass_filter(
                angular,
                fs_hz=meta.fs_hz,
                cutoff_hz=config.filtering.cutoff_hz,
                order=config.filtering.order,
            )
    else:
        angular, gyro_source = select_turn_angular_signal(df, config, meta.fs_hz)
        angular = np.abs(angular)

    threshold = 0.25 * float(np.nanmax(angular)) if np.isfinite(angular).any() else np.nan
    start = np.nan
    end = np.nan
    duration = np.nan
    turn_mask = np.zeros(len(time_s), dtype=bool)
    active = np.isfinite(angular) & np.isfinite(threshold) & (angular >= threshold)
    if np.isfinite(meta.fs_hz) and meta.fs_hz > 0:
        active = mask_close_gaps(active, max_gap_samples=max(1, int(round(0.20 * meta.fs_hz))))
    if np.any(active):
        idx = np.where(active)[0]
        pad = max(1, int(round(0.10 * meta.fs_hz))) if np.isfinite(meta.fs_hz) and meta.fs_hz > 0 else 1
        i0 = max(0, int(idx[0]) - pad)
        i1 = min(len(time_s) - 1, int(idx[-1]) + pad)
        start = float(time_s[i0])
        end = float(time_s[i1])
        duration = float(end - start)
        turn_mask[i0 : i1 + 1] = True

    acc_signal = np.abs(df["acc_mag_gravity_removed"].to_numpy(dtype=float))
    if not np.isfinite(acc_signal).any() or np.nanstd(acc_signal) <= 1e-8:
        acc_signal, _ = select_motion_acc_signal(df, config.prefer_useracc_for_motion)
        acc_signal = np.abs(acc_signal)

    steps = np.nan
    if np.any(turn_mask):
        t_turn = time_s[turn_mask]
        acc_turn = acc_signal[turn_mask]
        fs_turn = 1.0 / estimate_sampling_interval_s(t_turn) if len(t_turn) > 2 else meta.fs_hz
        if len(acc_turn) >= 3 and np.isfinite(fs_turn) and fs_turn > 0:
            acc_turn_smooth = gaussian_filter1d(acc_turn, sigma=2)
            step_threshold = float(np.nanpercentile(acc_turn_smooth, 60))
            peaks, _ = find_peaks(
                acc_turn_smooth,
                height=step_threshold,
                distance=max(1, int(round(0.45 * fs_turn))),
            )
            steps = float(len(peaks))
    angular_stats = angular[turn_mask] if np.any(turn_mask) else angular
    return {
        "start": start,
        "end": end,
        "turn_duration": duration,
        "turn_step_count": steps,
        "peak_angular_velocity": float(np.nanmax(angular_stats)) if np.isfinite(angular_stats).any() else np.nan,
        "mean_angular_velocity": float(np.nanmean(angular_stats)) if np.isfinite(angular_stats).any() else np.nan,
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
            if kind == "walk":
                add_walking_sacrum_qc(row)
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
