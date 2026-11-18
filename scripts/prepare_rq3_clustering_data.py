#!/usr/bin/env python3
"""Prepare label-blind laboratory and home feature tables for RQ3 clustering."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAB_FEATURE_COLUMNS = (
    "walk_cadence_steps_min",
    "pooled_median_turn_duration_s",
    "pooled_median_turn_step_count",
    "sit_to_stand_duration_s",
)
HOME_SOURCE_COLUMNS = (
    "cadence",
    "left_turn_duration",
    "right_turn_duration",
    "left_turn_step_count",
    "right_turn_step_count",
    "sit_to_stand_duration",
)


def add_summary_statistics(row: dict, data: pd.DataFrame, numeric_columns: list[str]) -> None:
    """Add both participant-level median and mean values for all numeric features."""
    for column in numeric_columns:
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        row[f"median_{column}"] = values.median() if len(values) else np.nan
        row[f"mean_{column}"] = values.mean() if len(values) else np.nan


def load_fall_history(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path).rename(columns={"Participant ID": "subject_id"})
    labels["subject_id"] = labels["subject_id"].astype(str).str.strip()
    labels = labels[labels["subject_id"].str.startswith("SRS", na=False)].copy()
    labels = labels[
        ["subject_id", "12-Month Fall History", "12-Month Fall History Classification"]
    ].rename(
        columns={
            "12-Month Fall History": "fall_history_count",
            "12-Month Fall History Classification": "fall_history_class",
        }
    )
    labels["fall_history_class"] = labels["fall_history_class"].map({"F": "Faller", "NF": "Non-Faller"})
    return labels.drop_duplicates("subject_id")


def build_lab_features(xsens_path: Path, fall_history: pd.DataFrame) -> pd.DataFrame:
    lab = pd.read_csv(xsens_path)
    numeric_columns = [
        column
        for column in lab.select_dtypes(include="number").columns
        if column != "trial_id"
    ]
    rows = []
    for subject_id, subject_trials in lab.groupby("subject_id", sort=True):
        walk = subject_trials["sacrum_walk_cadence_steps_min"].dropna()
        turn_duration = pd.concat(
            [
                subject_trials["sacrum_left_turn_duration_s"],
                subject_trials["sacrum_right_turn_duration_s"],
            ]
        ).dropna()
        turn_steps = pd.concat(
            [
                subject_trials["sacrum_left_turn_step_count"],
                subject_trials["sacrum_right_turn_step_count"],
            ]
        ).dropna()
        sts = subject_trials["sacrum_sit_to_stand_duration_s"].dropna()
        row = {
            "subject_id": subject_id,
            "lab_visit_id": "all_available_visits",
            "n_lab_visits": subject_trials["visit_id"].nunique(),
            "walk_cadence_steps_min": walk.median(),
            "pooled_median_turn_duration_s": turn_duration.median(),
            "pooled_median_turn_step_count": turn_steps.median(),
            "sit_to_stand_duration_s": sts.median(),
            "n_walk_trials": len(walk),
            "n_turn_direction_trials": len(turn_duration),
            "n_sit_to_stand_trials": len(sts),
        }
        add_summary_statistics(row, subject_trials, numeric_columns)
        rows.append(row)
    features = pd.DataFrame(rows).merge(fall_history, on="subject_id", how="inner")
    features["complete_feature_set"] = features[list(LAB_FEATURE_COLUMNS)].notna().all(axis=1)
    return features.sort_values("subject_id").reset_index(drop=True)


def build_home_features(home_path: Path, fall_history: pd.DataFrame, required_days: int) -> pd.DataFrame:
    home = pd.read_csv(home_path).rename(columns={"patient_id": "subject_id"})
    home["date"] = pd.to_datetime(home["date"], errors="coerce")
    home = home.dropna(subset=["subject_id", "date"]).copy()
    for column in HOME_SOURCE_COLUMNS:
        home[column] = pd.to_numeric(home[column], errors="coerce")

    numeric_columns = home.select_dtypes(include="number").columns.tolist()

    # A valid home day contains all components needed to calculate the selected features.
    home = home.dropna(subset=list(HOME_SOURCE_COLUMNS)).copy()
    daily = home.groupby(["subject_id", "date"], as_index=False)[numeric_columns].median()
    daily = daily.dropna(subset=list(HOME_SOURCE_COLUMNS)).copy()
    daily["pooled_turn_duration_s"] = daily[["left_turn_duration", "right_turn_duration"]].mean(axis=1)
    daily["pooled_turn_step_count"] = daily[["left_turn_step_count", "right_turn_step_count"]].mean(axis=1)

    rows = []
    for subject_id, days in daily.groupby("subject_id", sort=True):
        selected = days.sort_values("date").head(required_days)
        if len(selected) < required_days:
            continue
        row = {
            "subject_id": subject_id,
            "walk_cadence_steps_min": selected["cadence"].median(),
            "pooled_median_turn_duration_s": selected["pooled_turn_duration_s"].median(),
            "pooled_median_turn_step_count": selected["pooled_turn_step_count"].median(),
            "sit_to_stand_duration_s": selected["sit_to_stand_duration"].median(),
            "n_valid_days": len(selected),
            "first_valid_date": selected["date"].min().date().isoformat(),
            "last_selected_date": selected["date"].max().date().isoformat(),
        }
        add_summary_statistics(row, selected, numeric_columns)
        rows.append(row)
    features = pd.DataFrame(rows).merge(fall_history, on="subject_id", how="left")
    features["complete_feature_set"] = features[list(LAB_FEATURE_COLUMNS)].notna().all(axis=1)
    return features.sort_values("subject_id").reset_index(drop=True)


def write_manifest(path: Path) -> None:
    rows = [
        {
            "feature": "walk_cadence_steps_min",
            "laboratory_source": "median sacrum 10-step walking cadence across all available laboratory visits and NW trials",
            "home_source": "median smartphone walking cadence across the first 14 valid days",
        },
        {
            "feature": "pooled_median_turn_duration_s",
            "laboratory_source": "median of available sacrum left/right turn durations across all available laboratory visits",
            "home_source": "median daily mean of left/right smartphone turn durations across 14 days",
        },
        {
            "feature": "pooled_median_turn_step_count",
            "laboratory_source": "median of available sacrum left/right turn step counts across all available laboratory visits",
            "home_source": "median daily mean of left/right smartphone turn step counts across 14 days",
        },
        {
            "feature": "sit_to_stand_duration_s",
            "laboratory_source": "median sacrum sit-to-stand duration across all available laboratory visits and STS trials",
            "home_source": "median smartphone sit-to-stand duration across 14 days",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare participant-level RQ3 laboratory and home feature tables.")
    parser.add_argument(
        "--lab-features",
        type=Path,
        default=PROJECT_ROOT / "feature_dataset_lab/ten_step_window/XSENS/xsens_features.csv",
    )
    parser.add_argument("--home-features", type=Path, default=PROJECT_ROOT / "features_dataset.csv")
    parser.add_argument(
        "--fall-history",
        type=Path,
        default=PROJECT_ROOT / "Data/lab/4 Test Classification(4 Test Classification).csv",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "Data/Clustering")
    parser.add_argument("--required-home-days", type=int, default=14)
    args = parser.parse_args()

    fall_history = load_fall_history(args.fall_history)
    lab = build_lab_features(args.lab_features, fall_history)
    home = build_home_features(args.home_features, fall_history, args.required_home_days)

    (args.output_dir / "Lab").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "Home").mkdir(parents=True, exist_ok=True)
    lab.to_csv(args.output_dir / "Lab/lab_participant_features.csv", index=False, float_format="%.3f")
    home.to_csv(args.output_dir / "Home/home_participant_features.csv", index=False, float_format="%.3f")
    write_manifest(args.output_dir / "feature_manifest.csv")

    print(f"Laboratory participants with fall history: {len(lab)}")
    print(f"Laboratory complete feature sets: {int(lab['complete_feature_set'].sum())}")
    print(f"Home participants with {args.required_home_days} valid days: {len(home)}")
    print(f"Home participants with fall history: {int(home['fall_history_class'].notna().sum())}")


if __name__ == "__main__":
    main()
