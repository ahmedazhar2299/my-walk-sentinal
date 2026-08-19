#!/usr/bin/env python3
"""Merge combined SRS smartphone folders into the Scrapper patient layout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

ACTIVITY_MAP = {
    "Walk": "walk",
    "360_LeftTurn": "left_turn",
    "360_RightTurn": "right_turn",
    "Sit_To_Stand": "sit_to_stand",
    "Stand_To_Sit": "stand_to_sit",
}


@dataclass(frozen=True)
class SourceDateFolder:
    subject_id: str
    patient_id: str
    date: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy Data_Stroke/Combined_Smartphone_Data_JSON into Scrapper/patient_<id> folders."
    )
    parser.add_argument("--source", default="Data_Stroke/Combined_Smartphone_Data_JSON", type=Path)
    parser.add_argument("--scrapper", default="Scrapper", type=Path)
    parser.add_argument("--subjects", default="subjects.json", type=Path)
    parser.add_argument("--manifest", default="Scrapper/_combined_smartphone_merge_manifest.csv", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_subject_map(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping = {str(key): str(value) for key, value in data.items()}

    # Data_Stroke uses SRS07, while the website export used SRS007.
    if "SRS07" not in mapping and "SRS007" in mapping:
        mapping["SRS07"] = mapping["SRS007"]

    return mapping


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_source_date_folders(source_root: Path, subject_map: dict[str, str]) -> tuple[list[SourceDateFolder], list[str]]:
    folders = []
    unmapped = []
    for subject_dir in sorted(source_root.iterdir()):
        if not subject_dir.is_dir() or subject_dir.name.startswith("_") or subject_dir.name.startswith("."):
            continue
        subject_id = subject_dir.name
        patient_id = subject_map.get(subject_id)
        if not patient_id:
            unmapped.append(subject_id)
            continue
        for date_dir in sorted(subject_dir.iterdir()):
            if not date_dir.is_dir() or date_dir.name.startswith("_") or date_dir.name.startswith("."):
                continue
            match = DATE_RE.search(date_dir.name)
            if not match:
                continue
            folders.append(
                SourceDateFolder(
                    subject_id=subject_id,
                    patient_id=patient_id,
                    date=match.group(1),
                    path=date_dir,
                )
            )
    return folders, sorted(set(unmapped))


def safe_copy(src: Path, dest: Path, subject_id: str) -> tuple[Path, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(src, dest)
        return dest, "copied"

    try:
        if md5_file(src) == md5_file(dest):
            return dest, "duplicate_same_content_skipped"
    except Exception:
        pass

    stem = dest.stem
    suffix = dest.suffix
    candidate = dest.parent / f"{stem}_{subject_id}{suffix}"
    idx = 1
    while candidate.exists():
        try:
            if md5_file(src) == md5_file(candidate):
                return candidate, "duplicate_same_content_skipped"
        except Exception:
            pass
        candidate = dest.parent / f"{stem}_{subject_id}_{idx}{suffix}"
        idx += 1

    shutil.copy2(src, candidate)
    return candidate, "copied_with_renamed_collision"


def copy_date_folder(folder: SourceDateFolder, scrapper_root: Path, dry_run: bool) -> list[dict[str, str]]:
    rows = []
    target_date_dir = scrapper_root / f"patient_{folder.patient_id}" / folder.date
    if not dry_run:
        target_date_dir.mkdir(parents=True, exist_ok=True)

    for activity_dir in sorted(item for item in folder.path.iterdir() if item.is_dir()):
        target_activity = ACTIVITY_MAP.get(activity_dir.name)
        if not target_activity:
            rows.append(
                {
                    "status": "unknown_activity_skipped",
                    "subject_id": folder.subject_id,
                    "patient_id": folder.patient_id,
                    "date": folder.date,
                    "activity": activity_dir.name,
                    "source_path": str(activity_dir),
                    "destination_path": "",
                }
            )
            continue

        target_activity_dir = target_date_dir / target_activity
        if not dry_run:
            target_activity_dir.mkdir(parents=True, exist_ok=True)
            for source_subdir in sorted(item for item in activity_dir.rglob("*") if item.is_dir()):
                rel_dir = source_subdir.relative_to(activity_dir)
                (target_activity_dir / rel_dir).mkdir(parents=True, exist_ok=True)

        for src_file in sorted(item for item in activity_dir.rglob("*") if item.is_file()):
            rel_file = src_file.relative_to(activity_dir)
            dest_file = target_activity_dir / rel_file
            final_dest = dest_file
            status = "dry_run_copy"
            if not dry_run:
                final_dest, status = safe_copy(src_file, dest_file, folder.subject_id)

            rows.append(
                {
                    "status": status,
                    "subject_id": folder.subject_id,
                    "patient_id": folder.patient_id,
                    "date": folder.date,
                    "activity": target_activity,
                    "source_path": str(src_file),
                    "destination_path": str(final_dest),
                }
            )

    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "subject_id",
        "patient_id",
        "date",
        "activity",
        "source_path",
        "destination_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    subject_map = load_subject_map(args.subjects)
    source_folders, unmapped = iter_source_date_folders(args.source, subject_map)

    rows = []
    for folder in sorted(source_folders, key=lambda item: (item.patient_id, item.date, item.subject_id)):
        rows.extend(copy_date_folder(folder, args.scrapper, args.dry_run))

    rows.sort(key=lambda row: (row["patient_id"], row["date"], row["activity"], row["source_path"]))
    if not args.dry_run:
        write_manifest(args.manifest, rows)

    status_counts = Counter(row["status"] for row in rows)
    patients = defaultdict(set)
    for folder in source_folders:
        patients[folder.patient_id].add(folder.date)

    print(f"Source date folders processed: {len(source_folders)}")
    print(f"Unmapped subjects: {', '.join(unmapped) if unmapped else 'none'}")
    print("Status counts:")
    for key, value in sorted(status_counts.items()):
        print(f"  {key}: {value}")
    print("Patient date ranges:")
    for patient_id in sorted(patients, key=lambda value: int(value)):
        dates = sorted(patients[patient_id])
        print(f"  patient_{patient_id}: {len(dates)} dates, {dates[0]} to {dates[-1]}")


if __name__ == "__main__":
    main()
