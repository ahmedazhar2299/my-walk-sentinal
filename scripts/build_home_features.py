#!/usr/bin/env python3
"""Build stroke-only home smartphone feature tables."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from imu_features import extract_dataset_features, load_config


LEGACY_PATIENT_ID_TO_SRS = {
    72: "SRS07",
    73: "SRS07",
    74: "SRS02",
    78: "SRS01",
}

TASK_COLUMNS = {
    "Walk": ["walk_duration", "step_count", "cadence", "mean_step_time"],
    "Left Turn": ["left_turn_duration", "left_turn_step_count"],
    "Right Turn": ["right_turn_duration", "right_turn_step_count"],
    "Sit-to-Stand": ["sit_to_stand_duration", "sit_to_stand_peak_acc"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SRS-only home smartphone feature outputs.")
    parser.add_argument("--dataset-root", type=Path, default=Path("Home_Smartphone_Data"))
    parser.add_argument("--subjects-json", type=Path, default=Path("subjects.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("features_dataset.csv"))
    parser.add_argument("--per-subject-dir", type=Path, default=Path("feature_datasets_by_subject"))
    parser.add_argument(
        "--coverage-csv",
        type=Path,
        default=Path("Sensors_Visualizers/home_recording_coverage_30day.csv"),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def load_subject_map(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        srs_to_patient = json.load(f)
    patient_to_srs = {f"patient_{int(pid)}": srs for srs, pid in srs_to_patient.items()}
    for patient_id, srs in LEGACY_PATIENT_ID_TO_SRS.items():
        patient_to_srs.setdefault(f"patient_{patient_id}", srs)
    for srs in srs_to_patient:
        patient_to_srs[srs] = srs
    return patient_to_srs


def canonical_sort_key(subject_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", str(subject_id))
    if match:
        return int(match.group(1)), str(subject_id)
    return 9999, str(subject_id)


def source_rank(source_patient_id: str, subject_map: dict[str, str]) -> int:
    canonical = subject_map.get(str(source_patient_id), str(source_patient_id))
    if str(source_patient_id) == canonical:
        return 0
    if str(source_patient_id).startswith("patient_"):
        try:
            patient_num = int(str(source_patient_id).split("_", 1)[1])
        except ValueError:
            return 2
        if patient_num in LEGACY_PATIENT_ID_TO_SRS:
            return 2
        return 1
    return 1


def is_valid_feature_row(row: pd.Series) -> bool:
    ignored = {"patient_id", "date", "source_patient_id"}
    feature_cols = [col for col in row.index if col not in ignored]
    values = pd.to_numeric(row[feature_cols], errors="coerce")
    return bool(values.notna().any())


def canonicalize_features(raw: pd.DataFrame, subject_map: dict[str, str]) -> pd.DataFrame:
    if raw.empty:
        return raw

    df = raw.copy()
    df["source_patient_id"] = df["patient_id"].astype(str)
    df["patient_id"] = df["source_patient_id"].map(subject_map)
    df = df[df["patient_id"].notna()].copy()
    df = df[df["patient_id"].astype(str).str.match(r"^SRS\d+$", na=False)].copy()
    df["source_rank"] = df["source_patient_id"].map(lambda value: source_rank(value, subject_map))
    df = df[df.apply(is_valid_feature_row, axis=1)].copy()
    df = df.sort_values(["patient_id", "date", "source_rank", "source_patient_id"])
    df = df.drop_duplicates(subset=["patient_id", "date"], keep="first")
    df = df.drop(columns=["source_rank"])

    first_cols = ["patient_id", "date", "source_patient_id"]
    ordered = first_cols + [col for col in df.columns if col not in first_cols]
    df = df[ordered]
    df = df.sort_values(
        by=["patient_id", "date"],
        key=lambda s: s.map(canonical_sort_key) if s.name == "patient_id" else s,
    )
    return df.reset_index(drop=True)


def write_per_subject_files(df: pd.DataFrame, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for subject_id, group in df.groupby("patient_id", sort=False):
        path = output_dir / f"{subject_id}.csv"
        group.drop(columns=["source_patient_id"], errors="ignore").to_csv(path, index=False, float_format="%.3f")


def valid_task_dates(df: pd.DataFrame, columns: list[str]) -> list[pd.Timestamp]:
    available = [col for col in columns if col in df.columns]
    if not available:
        return []
    values = df[available].apply(pd.to_numeric, errors="coerce")
    valid = values.notna().any(axis=1)
    dates = pd.to_datetime(df.loc[valid, "date"], errors="coerce").dropna()
    return sorted(dates.dt.normalize().drop_duplicates().tolist())


def best_30day_window(dates: list[pd.Timestamp]) -> tuple[pd.Timestamp | pd.NaT, pd.Timestamp | pd.NaT, list[pd.Timestamp]]:
    if not dates:
        return pd.NaT, pd.NaT, []
    unique = sorted(pd.Series(dates).drop_duplicates().tolist())
    best_dates: list[pd.Timestamp] = []
    best_start = unique[0]
    for start in unique:
        end = start + pd.Timedelta(days=29)
        inside = [date for date in unique if start <= date <= end]
        if len(inside) > len(best_dates):
            best_dates = inside
            best_start = start
    return best_start, best_start + pd.Timedelta(days=29), best_dates


def build_coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject_id, subject_df in df.groupby("patient_id", sort=False):
        for task, columns in TASK_COLUMNS.items():
            dates = valid_task_dates(subject_df, columns)
            start, end, inside = best_30day_window(dates)
            rows.append(
                {
                    "participant": subject_id,
                    "task": task,
                    "first_valid_recording_date": dates[0].strftime("%Y-%m-%d") if dates else "",
                    "last_valid_recording_date_all_data": dates[-1].strftime("%Y-%m-%d") if dates else "",
                    "total_valid_recording_days_all_data": len(dates),
                    "best_30day_window_start": start.strftime("%Y-%m-%d") if pd.notna(start) else "",
                    "best_30day_window_end": end.strftime("%Y-%m-%d") if pd.notna(end) else "",
                    "last_recording_date_within_30_days": inside[-1].strftime("%Y-%m-%d") if inside else "",
                    "valid_recording_days_within_30_days": len(inside),
                    "recording_dates_within_30_days": ", ".join(date.strftime("%Y-%m-%d") for date in inside),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config = load_config(None)
    raw = extract_dataset_features(
        dataset_root=args.dataset_root,
        config=config,
        output_csv=None,
        save_csv=False,
        verbose=not args.quiet,
    )
    subject_map = load_subject_map(args.subjects_json)
    features = canonicalize_features(raw, subject_map)
    features = features.drop(columns=[col for col in features.columns if col.endswith("_qc_score")], errors="ignore")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    features.drop(columns=["source_patient_id"], errors="ignore").to_csv(
        args.output_csv, index=False, float_format="%.3f"
    )
    write_per_subject_files(features, args.per_subject_dir)

    coverage = build_coverage_table(features)
    args.coverage_csv.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(args.coverage_csv, index=False)

    if not args.quiet:
        print(f"Wrote {args.output_csv} ({len(features)} rows)")
        print(f"Wrote per-participant CSVs to {args.per_subject_dir}")
        print(f"Wrote {args.coverage_csv}")
        print(f"Participants: {features['patient_id'].nunique() if not features.empty else 0}")


if __name__ == "__main__":
    main()
