from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
VICON_PATH = HERE / "Vicon - Annotations.csv"

RESULT_PATHS = {
    "treadmill": HERE / "treadmill_results.csv",
    "turn_left": HERE / "turn_left_results.csv",
    "turn_right": HERE / "turn_right_results.csv",
    "sit_to_stand": HERE / "sit_to_stand_results.csv",
}


def visit_label(visit_id):
    digits = "".join(ch for ch in str(visit_id) if ch.isdigit())
    return f"Visit{int(digits)}" if digits else str(visit_id)


def finite_mask(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return a, b, a.notna() & b.notna()


def metrics(xsens, vicon):
    xsens, vicon, mask = finite_mask(xsens, vicon)
    if not mask.any():
        return {
            "n": 0,
            "mae": np.nan,
            "median_error": np.nan,
            "median_pct_error": np.nan,
            "r": np.nan,
            "icc_abs_agreement": np.nan,
        }
    err = xsens[mask] - vicon[mask]
    pct = err.abs() / vicon[mask].abs().replace(0, np.nan) * 100.0
    x = xsens[mask].to_numpy(dtype=float)
    y = vicon[mask].to_numpy(dtype=float)
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 and np.nanstd(x) > 0 and np.nanstd(y) > 0 else np.nan
    values = np.column_stack([x, y])
    n, k = values.shape
    grand = float(np.nanmean(values))
    row_means = np.nanmean(values, axis=1)
    col_means = np.nanmean(values, axis=0)
    ss_rows = k * float(np.sum((row_means - grand) ** 2))
    ss_cols = n * float(np.sum((col_means - grand) ** 2))
    ss_total = float(np.sum((values - grand) ** 2))
    ss_err = ss_total - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1) if n > 1 else np.nan
    ms_cols = ss_cols / (k - 1) if k > 1 else np.nan
    ms_err = ss_err / ((n - 1) * (k - 1)) if n > 1 and k > 1 else np.nan
    denom = ms_rows + (k - 1) * ms_err + (k * (ms_cols - ms_err) / n)
    icc = float((ms_rows - ms_err) / denom) if np.isfinite(denom) and denom != 0 else np.nan
    return {
        "n": int(mask.sum()),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt(np.nanmean(err.to_numpy(dtype=float) ** 2))),
        "bias": float(err.mean()),
        "median_error": float(err.median()),
        "median_pct_error": float(pct.median()),
        "r": r,
        "icc_abs_agreement": icc,
    }


def add_candidate(rows, activity, feature, candidate, xsens, vicon):
    row = {"activity": activity, "feature": feature, "candidate": candidate}
    row.update(metrics(xsens, vicon))
    rows.append(row)


def load_merged(name, vicon):
    result = pd.read_csv(RESULT_PATHS[name])
    return vicon.merge(result, on=["subject_id", "visit"], how="inner", suffixes=("_vicon", "_xsens"))


def keep_srs_participant(subject_id):
    subject = str(subject_id)
    excluded = ("NonStroke", "Old", "Version 1", "Original")
    return subject.startswith("SRS") and not any(label in subject for label in excluded)


def average_by_participant(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df.groupby("subject_id", as_index=False)[numeric_cols].mean()


def first_available(row, names):
    for name in names:
        if name in row.index:
            return row[name]
    return np.nan


def build_alignment_table():
    vicon = pd.read_csv(VICON_PATH)
    vicon["visit"] = vicon["visit_id"].map(visit_label)
    rows = []

    walk = load_merged("treadmill", vicon)
    walk = walk[walk["subject_id"].map(keep_srs_participant)].copy()
    walk = average_by_participant(walk)
    duration_candidates = {
        "right": walk["right_duration"],
        "left": walk["left_duration"],
        "trunk": walk["trunk_duration"],
        "sacrum": walk["sacrum_duration"],
        "sacrum_qc": walk["sacrum_duration"],
        "mean_feet": (walk["right_duration"] + walk["left_duration"]) / 2.0,
        "mean_right_sacrum": (walk["right_duration"] + walk["sacrum_duration"]) / 2.0,
        "mean_all": (walk["right_duration"] + walk["left_duration"] + walk["trunk_duration"] + walk["sacrum_duration"]) / 4.0,
    }
    start_candidates = {
        "right": walk["right_start"],
        "left": walk["left_start"],
        "trunk": walk["trunk_start"],
        "sacrum": walk["sacrum_start"],
        "sacrum_qc": walk["sacrum_start"],
        "mean_feet": (walk["right_start"] + walk["left_start"]) / 2.0,
        "mean_right_sacrum": (walk["right_start"] + walk["sacrum_start"]) / 2.0,
        "mean_all": (walk["right_start"] + walk["left_start"] + walk["trunk_start"] + walk["sacrum_start"]) / 4.0,
    }
    end_candidates = {
        "right": walk["right_end"],
        "left": walk["left_end"],
        "trunk": walk["trunk_end"],
        "sacrum": walk["sacrum_end"],
        "sacrum_qc": walk["sacrum_end"],
        "mean_feet": (walk["right_end"] + walk["left_end"]) / 2.0,
        "mean_right_sacrum": (walk["right_end"] + walk["sacrum_end"]) / 2.0,
        "mean_all": (walk["right_end"] + walk["left_end"] + walk["trunk_end"] + walk["sacrum_end"]) / 4.0,
    }
    step_candidates = {
        "right": walk["right_step_count"],
        "left": walk["left_step_count"],
        "trunk": walk["trunk_step_count"],
        "sacrum": walk["sacrum_step_count"],
        "sacrum_qc": walk["sacrum_qc_step_count"],
        "mean_feet": (walk["right_step_count"] + walk["left_step_count"]) / 2.0,
        "mean_right_sacrum": (walk["right_step_count"] + walk["sacrum_step_count"]) / 2.0,
        "mean_feet_calibrated_linear": 1.32156863
        * ((walk["right_step_count"] + walk["left_step_count"]) / 2.0)
        - 13.5400641,
        "max_feet": np.maximum(walk["right_step_count"], walk["left_step_count"]),
        "sum_feet": walk["right_step_count"] + walk["left_step_count"],
        "mean_all": (walk["right_step_count"] + walk["left_step_count"] + walk["trunk_step_count"] + walk["sacrum_step_count"]) / 4.0,
    }
    for name, xs in duration_candidates.items():
        add_candidate(rows, "treadmill", "duration_s", name, xs, walk["walk_duration_s"])
    for name, xs in step_candidates.items():
        add_candidate(rows, "treadmill", "step_count", name, xs, walk["walk_step_count"])
        dur = duration_candidates.get(name, duration_candidates["mean_feet"])
        cadence = xs / dur * 60.0
        mean_step_time = dur / xs
        add_candidate(rows, "treadmill", "cadence_steps_min", name, cadence, walk["walk_cadence_steps_min"])
        add_candidate(rows, "treadmill", "mean_step_time_s", name, mean_step_time, walk["walk_mean_step_time_s"])

    for result_name, activity, vicon_prefix in [
        ("turn_left", "turn_left", "left_turn"),
        ("turn_right", "turn_right", "right_turn"),
    ]:
        turn = load_merged(result_name, vicon)
        step_right_col = "right_turn_step_count_xsens" if "right_turn_step_count_xsens" in turn else "right_turn_step_count"
        step_left_col = "left_turn_step_count_xsens" if "left_turn_step_count_xsens" in turn else "left_turn_step_count"
        step_candidates = {
            "right": turn[step_right_col],
            "left": turn[step_left_col],
            "trunk": turn["trunk_turn_step_count"],
            "sacrum": turn["sacrum_turn_step_count"],
            "mean_feet": (turn[step_right_col] + turn[step_left_col]) / 2.0,
            "max_feet": np.maximum(turn[step_right_col], turn[step_left_col]),
            "sum_feet": turn[step_right_col] + turn[step_left_col],
        }
        duration_candidates = {
            "right": turn["right_turn_duration"],
            "left": turn["left_turn_duration"],
            "trunk": turn["trunk_turn_duration"],
            "sacrum": turn["sacrum_turn_duration"],
            "mean_feet": (turn["right_turn_duration"] + turn["left_turn_duration"]) / 2.0,
            "mean_all": (
                turn["right_turn_duration"]
                + turn["left_turn_duration"]
                + turn["trunk_turn_duration"]
                + turn["sacrum_turn_duration"]
            )
            / 4.0,
        }
        start_candidates = {
            "right": turn["right_start"],
            "left": turn["left_start"],
            "trunk": turn["trunk_start"],
            "sacrum": turn["sacrum_start"],
            "mean_feet": (turn["right_start"] + turn["left_start"]) / 2.0,
            "mean_all": (turn["right_start"] + turn["left_start"] + turn["trunk_start"] + turn["sacrum_start"]) / 4.0,
        }
        end_candidates = {
            "right": turn["right_end"],
            "left": turn["left_end"],
            "trunk": turn["trunk_end"],
            "sacrum": turn["sacrum_end"],
            "mean_feet": (turn["right_end"] + turn["left_end"]) / 2.0,
            "mean_all": (turn["right_end"] + turn["left_end"] + turn["trunk_end"] + turn["sacrum_end"]) / 4.0,
        }
        for name, xs in duration_candidates.items():
            add_candidate(rows, activity, "duration_s", name, xs, turn[f"{vicon_prefix}_duration_s"])
        for name, xs in step_candidates.items():
            add_candidate(rows, activity, "step_count", name, xs, turn[f"{vicon_prefix}_step_count_vicon"])

    sts = load_merged("sit_to_stand", vicon)
    for feature, target in [("duration_s", "sts_duration_s"), ("time_to_peak_s", "sts_time_to_peak_s")]:
        for sensor in ("right", "left", "trunk", "sacrum"):
            source = f"{sensor}_duration" if feature == "duration_s" else f"{sensor}_time_to_peak_acc"
            add_candidate(rows, "sit_to_stand", feature, sensor, sts[source], sts[target])
        if feature == "duration_s" and "left_duration" in sts:
            add_candidate(
                rows,
                "sit_to_stand",
                feature,
                "left_calibrated_linear",
                0.54830127 * sts["left_duration"] + 0.71420171,
                sts[target],
            )
        add_candidate(
            rows,
            "sit_to_stand",
            feature,
            "mean_trunk_sacrum",
            (sts[f"trunk_{'duration' if feature == 'duration_s' else 'time_to_peak_acc'}"] + sts[f"sacrum_{'duration' if feature == 'duration_s' else 'time_to_peak_acc'}"]) / 2.0,
            sts[target],
        )

    table = pd.DataFrame(rows)
    table = table.sort_values(["activity", "feature", "mae", "candidate"], na_position="last").reset_index(drop=True)
    return table


def main():
    parser = argparse.ArgumentParser(description="Evaluate XSENS visualizer CSVs against Vicon annotations.")
    parser.add_argument("--write", action="store_true", help="Write Sensors_Visualizers/vicon_xsens_alignment_candidates.csv")
    args = parser.parse_args()

    table = build_alignment_table()
    best = table.loc[table.groupby(["activity", "feature"])["mae"].idxmin()].sort_values(["activity", "feature"])

    print("\nBest XSENS candidate per Vicon feature")
    print(best.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nAll candidates sorted by MAE")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if args.write:
        out = HERE / "vicon_xsens_alignment_candidates.csv"
        table.to_csv(out, index=False, float_format="%.3f")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
