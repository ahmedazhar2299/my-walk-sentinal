#!/usr/bin/env python3
"""Combine organized smartphone datasets into one chronological folder tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


@dataclass(frozen=True)
class DateFolder:
    source_name: str
    subject_id: str
    date: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Data_Stroke/Organized_Data and Data_Stroke/Smartphone_database."
    )
    parser.add_argument("--organized", default="Data_Stroke/Organized_Data", type=Path)
    parser.add_argument("--smartphone", default="Data_Stroke/Smartphone_database", type=Path)
    parser.add_argument("--target", default="Data_Stroke/Combined_Smartphone_Data", type=Path)
    parser.add_argument("--allow-existing-target", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_date_folders(root: Path, source_name: str) -> list[DateFolder]:
    folders = []
    for subject_dir in sorted(root.iterdir()):
        if not subject_dir.is_dir() or subject_dir.name.startswith("_") or subject_dir.name.startswith("."):
            continue
        if not re.fullmatch(r"SRS\d{1,3}", subject_dir.name):
            continue
        for date_dir in sorted(subject_dir.iterdir()):
            if not date_dir.is_dir() or date_dir.name.startswith("_") or date_dir.name.startswith("."):
                continue
            match = DATE_RE.search(date_dir.name)
            if not match:
                continue
            folders.append(
                DateFolder(
                    source_name=source_name,
                    subject_id=subject_dir.name,
                    date=match.group(1),
                    path=date_dir,
                )
            )
    return folders


def build_day_lookup(date_folders: list[DateFolder]) -> dict[tuple[str, str], str]:
    by_subject = defaultdict(set)
    for folder in date_folders:
        by_subject[folder.subject_id].add(folder.date)

    lookup = {}
    for subject_id, dates in by_subject.items():
        for idx, date in enumerate(sorted(dates), start=1):
            lookup[(subject_id, date)] = f"D{idx}_{date}"
    return lookup


def copy_file_with_collision_handling(src: Path, dest: Path, source_name: str) -> tuple[Path, str]:
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
    candidate = dest.parent / f"{stem}_{source_name}{suffix}"
    idx = 1
    while candidate.exists():
        try:
            if md5_file(src) == md5_file(candidate):
                return candidate, "duplicate_same_content_skipped"
        except Exception:
            pass
        candidate = dest.parent / f"{stem}_{source_name}_{idx}{suffix}"
        idx += 1

    shutil.copy2(src, candidate)
    return candidate, "copied_with_renamed_collision"


def iter_activity_dirs(date_dir: Path) -> list[Path]:
    excluded = {"csv", "json", "synchronized"}
    return [
        child
        for child in sorted(date_dir.iterdir())
        if child.is_dir() and not child.name.startswith(".") and child.name.lower() not in excluded
    ]


def copy_activity_tree(
    folder: DateFolder,
    target_root: Path,
    target_day_folder: str,
    dry_run: bool,
) -> list[dict[str, str]]:
    rows = []
    target_date_dir = target_root / folder.subject_id / target_day_folder
    if not dry_run:
        target_date_dir.mkdir(parents=True, exist_ok=True)

    for activity_dir in iter_activity_dirs(folder.path):
        target_activity_dir = target_date_dir / activity_dir.name
        if not dry_run:
            target_activity_dir.mkdir(parents=True, exist_ok=True)
            for src_dir in sorted(item for item in activity_dir.rglob("*") if item.is_dir()):
                rel_dir = src_dir.relative_to(activity_dir)
                (target_activity_dir / rel_dir).mkdir(parents=True, exist_ok=True)

        for src_file in sorted(activity_dir.rglob("*")):
            if not src_file.is_file() or src_file.name.startswith("."):
                continue

            rel = src_file.relative_to(activity_dir)
            dest_file = target_activity_dir / rel
            status = "dry_run_copy"
            final_dest = dest_file

            if not dry_run:
                final_dest, status = copy_file_with_collision_handling(
                    src_file, dest_file, folder.source_name
                )

            rows.append(
                {
                    "status": status,
                    "source_dataset": folder.source_name,
                    "subject_id": folder.subject_id,
                    "date": folder.date,
                    "combined_day_folder": target_day_folder,
                    "activity": activity_dir.name,
                    "source_path": str(src_file),
                    "destination_path": str(final_dest),
                }
            )
    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "source_dataset",
        "subject_id",
        "date",
        "combined_day_folder",
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
    organized_root = args.organized.resolve()
    smartphone_root = args.smartphone.resolve()
    target_root = args.target.resolve()

    if not organized_root.exists():
        raise SystemExit(f"Missing organized root: {organized_root}")
    if not smartphone_root.exists():
        raise SystemExit(f"Missing smartphone root: {smartphone_root}")
    if target_root.exists() and not args.allow_existing_target and not args.dry_run:
        raise SystemExit(
            f"Target already exists: {target_root}. Use --allow-existing-target intentionally."
        )

    date_folders = []
    date_folders.extend(iter_date_folders(organized_root, "organized"))
    date_folders.extend(iter_date_folders(smartphone_root, "smartphone"))
    day_lookup = build_day_lookup(date_folders)

    manifest_rows = []
    for folder in sorted(date_folders, key=lambda item: (item.subject_id, item.date, item.source_name)):
        combined_day = day_lookup[(folder.subject_id, folder.date)]
        manifest_rows.extend(copy_activity_tree(folder, target_root, combined_day, args.dry_run))

    manifest_rows.sort(
        key=lambda row: (
            row["subject_id"],
            row["date"],
            row["activity"],
            row["source_dataset"],
            row["source_path"],
        )
    )

    if not args.dry_run:
        write_manifest(target_root / "_combined_manifest.csv", manifest_rows)

    status_counts = Counter(row["status"] for row in manifest_rows)
    source_counts = Counter(row["source_dataset"] for row in manifest_rows if row["status"].startswith("copied"))
    subject_dates = defaultdict(set)
    for folder in date_folders:
        subject_dates[folder.subject_id].add(folder.date)

    print(f"Date folders read: {len(date_folders)}")
    print(f"Combined subjects: {len(subject_dates)}")
    print("Status counts:")
    for key, value in sorted(status_counts.items()):
        print(f"  {key}: {value}")
    print("Copied files by source:")
    for key, value in sorted(source_counts.items()):
        print(f"  {key}: {value}")
    print("Days by subject:")
    for subject_id in sorted(subject_dates):
        dates = sorted(subject_dates[subject_id])
        print(f"  {subject_id}: {len(dates)} days, {dates[0]} to {dates[-1]}")


if __name__ == "__main__":
    main()
