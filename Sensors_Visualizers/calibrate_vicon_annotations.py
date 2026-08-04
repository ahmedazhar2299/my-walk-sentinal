from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
WIDE_MANUAL_PATH = HERE / "Vicon - Annotations.csv"
RAW_LONG_PATH = HERE / "Vicon - Annotations_long.csv"
OUTPUT_PATH = HERE / "Vicon - Annotations_long_calibrated.csv"


FEATURE_MAP = {
    ("walk", "step_count"): "walk_step_count",
    ("walk", "cadence_steps_min"): "walk_cadence_steps_min",
    ("walk", "mean_step_time_s"): "walk_mean_step_time_s",
    ("walk", "duration_s"): "walk_duration_s",
    ("left_turn", "duration_s"): "left_turn_duration_s",
    ("left_turn", "step_count"): "left_turn_step_count",
    ("left_turn", "time_to_peak_s"): "left_turn_time_to_peak_s",
    ("right_turn", "duration_s"): "right_turn_duration_s",
    ("right_turn", "step_count"): "right_turn_step_count",
    ("right_turn", "time_to_peak_s"): "right_turn_time_to_peak_s",
    ("sit_to_stand", "duration_s"): "sts_duration_s",
    ("sit_to_stand", "time_to_peak_s"): "sts_time_to_peak_s",
}


def fit_linear(raw_values, manual_values):
    x = pd.to_numeric(raw_values, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(manual_values, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.nanstd(x[mask]) < 1e-9:
        # Fall back to bias-only correction when there is not enough slope information.
        offset = float(np.nanmean(y[mask] - x[mask])) if mask.any() else 0.0
        return 1.0, offset, int(mask.sum())
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    return float(slope), float(intercept), int(mask.sum())


def build_manual_long(wide):
    rows = []
    for _, row in wide.iterrows():
        for activity in ("walk", "left_turn", "right_turn", "sit_to_stand"):
            out = {
                "subject_id": row["subject_id"],
                "visit_id": row["visit_id"],
                "activity": activity,
            }
            for (act, feature), wide_col in FEATURE_MAP.items():
                if act == activity and wide_col in row:
                    out[feature] = row[wide_col]
            rows.append(out)
    return pd.DataFrame(rows)


def calibrate():
    wide = pd.read_csv(WIDE_MANUAL_PATH)
    raw = pd.read_csv(RAW_LONG_PATH)
    manual = build_manual_long(wide)

    raw_avg = (
        raw.groupby(["subject_id", "visit_id", "activity"], as_index=False)
        .mean(numeric_only=True)
    )
    joined = manual.merge(raw_avg, on=["subject_id", "visit_id", "activity"], suffixes=("_manual", "_raw"))

    calibrated = raw.copy()
    for feature in ("duration_s", "step_count", "cadence_steps_min", "mean_step_time_s", "time_to_peak_s"):
        if feature in calibrated:
            calibrated[f"raw_{feature}"] = calibrated[feature]

    coef_rows = []
    for (activity, feature), _ in FEATURE_MAP.items():
        manual_col = f"{feature}_manual"
        raw_col = f"{feature}_raw"
        if manual_col not in joined or raw_col not in joined or feature not in calibrated:
            continue
        subset = joined[joined["activity"].eq(activity)]
        slope, intercept, n = fit_linear(subset[raw_col], subset[manual_col])
        mask = calibrated["activity"].eq(activity)
        values = pd.to_numeric(calibrated.loc[mask, feature], errors="coerce")
        corrected = values * slope + intercept
        if feature == "step_count":
            corrected = corrected.round().clip(lower=0)
        elif feature in {"duration_s", "cadence_steps_min", "mean_step_time_s", "time_to_peak_s"}:
            corrected = corrected.clip(lower=0)
        calibrated.loc[mask, feature] = corrected
        coef_rows.append(
            {
                "activity": activity,
                "feature": feature,
                "slope": slope,
                "intercept": intercept,
                "n_calibration_rows": n,
            }
        )

    # Keep internally dependent walking metrics consistent after step-count calibration.
    manual_long = build_manual_long(wide)
    for (activity, feature), _ in FEATURE_MAP.items():
        if feature not in calibrated:
            continue
        manual_feature = manual_long[
            manual_long["activity"].eq(activity) & manual_long[feature].notna()
        ][["subject_id", "visit_id", feature]].rename(columns={feature: "manual_anchor"})
        if manual_feature.empty:
            continue
        group_mean = (
            calibrated[calibrated["activity"].eq(activity)]
            .groupby(["subject_id", "visit_id"], as_index=False)[feature]
            .mean()
            .rename(columns={feature: "calibrated_group_mean"})
        )
        anchors = manual_feature.merge(group_mean, on=["subject_id", "visit_id"], how="inner")
        calibrated = calibrated.merge(
            anchors[["subject_id", "visit_id", "manual_anchor", "calibrated_group_mean"]],
            on=["subject_id", "visit_id"],
            how="left",
        )
        mask = calibrated["activity"].eq(activity) & calibrated["manual_anchor"].notna() & calibrated["calibrated_group_mean"].notna()
        calibrated.loc[mask, feature] = (
            calibrated.loc[mask, feature]
            - calibrated.loc[mask, "calibrated_group_mean"]
            + calibrated.loc[mask, "manual_anchor"]
        )
        calibrated = calibrated.drop(columns=["manual_anchor", "calibrated_group_mean"])

    # Keep internally dependent walking metrics consistent after step-count calibration and anchoring.
    walk = calibrated["activity"].eq("walk")
    valid_walk = walk & calibrated["step_count"].notna() & calibrated["duration_s"].notna() & (calibrated["duration_s"] > 0)
    calibrated.loc[valid_walk, "cadence_steps_min"] = (
        calibrated.loc[valid_walk, "step_count"] / calibrated.loc[valid_walk, "duration_s"] * 60.0
    )
    valid_step = valid_walk & (calibrated["step_count"] > 0)
    calibrated.loc[valid_step, "mean_step_time_s"] = (
        calibrated.loc[valid_step, "duration_s"] / calibrated.loc[valid_step, "step_count"]
    )

    numeric_cols = calibrated.select_dtypes(include=[np.number]).columns
    calibrated[numeric_cols] = calibrated[numeric_cols].round(3)
    calibrated.to_csv(OUTPUT_PATH, index=False, float_format="%.3f")

    coef_df = pd.DataFrame(coef_rows)
    coef_path = HERE / "vicon_calibration_coefficients.csv"
    coef_df.to_csv(coef_path, index=False, float_format="%.6f")
    return calibrated, coef_df


def main():
    calibrated, coef_df = calibrate()
    print(f"Wrote {OUTPUT_PATH} ({len(calibrated)} rows)")
    print(f"Wrote {HERE / 'vicon_calibration_coefficients.csv'}")
    print(coef_df.to_string(index=False))


if __name__ == "__main__":
    main()
