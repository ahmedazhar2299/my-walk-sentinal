#!/usr/bin/env python3
"""Organize raw smartphone JSON exports into the stroke dataset layout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ACTIVITY_FROM_FILENAME = {
    "1 meter walk": "Walk",
    "360 left turn": "360_LeftTurn",
    "360 right turn": "360_RightTurn",
    "sit to stand": "Sit_To_Stand",
    "stand to sit": "Stand_To_Sit",
}

ACTIVITY_FROM_EXERCISE_TYPE = {
    1: "Sit_To_Stand",
    2: "Stand_To_Sit",
    3: "360_LeftTurn",
    5: "360_RightTurn",
    6: "Walk",
}

FILENAME_RE = re.compile(
    r"^(?P<activity>.+?)-(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})$"
)


@dataclass(frozen=True)
class JsonRecord:
    source_path: Path
    subject_id: str
    filename_activity: str
    date: str
    time: str
    content_hash: str
    exercise_type: str
    exercise_activity: str
    timestamp_date: str
    status: str
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Organize Data_Smartphone_Database JSON files into SRS/date/activity/json folders."
    )
    parser.add_argument("--source", default="Data_Smartphone_Database", type=Path)
    parser.add_argument("--target", default="Data_Stroke/Smartphone_database", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-existing-target",
        action="store_true",
        help="Allow writing into an existing target folder.",
    )
    return parser.parse_args()


def normalize_activity_name(raw: str) -> str:
    text = raw.lower().replace("°", "").strip()
    text = re.sub(r"\s+", " ", text)
    return ACTIVITY_FROM_FILENAME.get(text, "")


def infer_subject_id(path: Path, source_root: Path) -> str:
    rel_parts = path.relative_to(source_root).parts
    for part in rel_parts[:-1]:
        match = re.search(r"\b(SRS\d{1,3})\b", part, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return ""


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_json(path: Path) -> tuple[str, str, str, str]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        return "", "", "", f"invalid_json:{type(exc).__name__}"

    exercise_type = data.get("exercise_type") if isinstance(data, dict) else None
    exercise_activity = ""
    try:
        exercise_activity = ACTIVITY_FROM_EXERCISE_TYPE.get(int(exercise_type), "")
    except Exception:
        pass

    timestamp_date = ""
    samples = data.get("exercise_data") if isinstance(data, dict) else None
    if isinstance(samples, list) and samples:
        timestamp = samples[0].get("timestamp") if isinstance(samples[0], dict) else None
        try:
            ts = float(timestamp)
            if ts > 10_000_000_000:
                ts /= 1000.0
            timestamp_date = datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
        except Exception:
            timestamp_date = ""

    return str(exercise_type), exercise_activity, timestamp_date, ""


def build_records(source_root: Path) -> list[JsonRecord]:
    records = []
    for path in sorted(source_root.rglob("*.json")):
        subject_id = infer_subject_id(path, source_root)
        match = FILENAME_RE.match(path.stem)
        filename_activity = ""
        date = ""
        time = ""
        notes = []

        if match:
            filename_activity = normalize_activity_name(match.group("activity"))
            date = match.group("date")
            time = match.group("time")
            if not filename_activity:
                notes.append(f"unknown_filename_activity:{match.group('activity')}")
        else:
            notes.append("filename_date_parse_failed")

        if not subject_id:
            notes.append("subject_parse_failed")

        content_hash = md5_file(path)
        exercise_type, exercise_activity, timestamp_date, json_note = inspect_json(path)
        if json_note:
            notes.append(json_note)

        if filename_activity and exercise_activity and filename_activity != exercise_activity:
            notes.append(f"activity_mismatch:{filename_activity}!={exercise_activity}")

        if date and timestamp_date and date != timestamp_date:
            notes.append(f"date_mismatch_filename_vs_utc_timestamp:{date}!={timestamp_date}")

        status = "candidate"
        if json_note:
            status = "invalid_json"
        elif not subject_id or not filename_activity or not date:
            status = "unroutable"

        records.append(
            JsonRecord(
                source_path=path,
                subject_id=subject_id,
                filename_activity=filename_activity,
                date=date,
                time=time,
                content_hash=content_hash,
                exercise_type=exercise_type,
                exercise_activity=exercise_activity,
                timestamp_date=timestamp_date,
                status=status,
                notes=";".join(notes),
            )
        )
    return records


def choose_unique_records(records: list[JsonRecord]) -> tuple[list[JsonRecord], list[JsonRecord]]:
    by_hash = defaultdict(list)
    for record in records:
        by_hash[record.content_hash].append(record)

    chosen = []
    duplicates = []
    for _, group in by_hash.items():
        valid_group = [item for item in group if item.status == "candidate"]
        if valid_group:
            sorted_group = sorted(
                valid_group,
                key=lambda item: (
                    "backup" in [part.lower() for part in item.source_path.parts],
                    len(item.source_path.parts),
                    str(item.source_path),
                ),
            )
            chosen.append(sorted_group[0])
            duplicates.extend([item for item in group if item != sorted_group[0]])
        else:
            sorted_group = sorted(group, key=lambda item: str(item.source_path))
            chosen.append(sorted_group[0])
            duplicates.extend(sorted_group[1:])

    return sorted(chosen, key=lambda item: str(item.source_path)), sorted(
        duplicates, key=lambda item: str(item.source_path)
    )


def build_day_lookup(records: list[JsonRecord]) -> dict[tuple[str, str], str]:
    by_subject = defaultdict(set)
    for record in records:
        if record.status == "candidate":
            by_subject[record.subject_id].add(record.date)

    lookup = {}
    for subject_id, dates in by_subject.items():
        for idx, date in enumerate(sorted(dates), start=1):
            lookup[(subject_id, date)] = f"D{idx}_{date}"
    return lookup


def safe_copy(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(src, dest)
        return dest

    stem = dest.stem
    suffix = dest.suffix
    idx = 1
    while True:
        candidate = dest.parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            shutil.copy2(src, candidate)
            return candidate
        idx += 1


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "subject_id",
        "date",
        "day_folder",
        "activity",
        "time",
        "exercise_type",
        "exercise_activity",
        "timestamp_date",
        "content_hash",
        "source_path",
        "destination_path",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    source_root = args.source.resolve()
    target_root = args.target.resolve()

    if not source_root.exists():
        raise SystemExit(f"Source folder does not exist: {source_root}")
    if target_root.exists() and not args.allow_existing_target and not args.dry_run:
        raise SystemExit(
            f"Target folder already exists: {target_root}. "
            "Use --allow-existing-target to add to it intentionally."
        )

    records = build_records(source_root)
    unique_records, duplicate_records = choose_unique_records(records)
    day_lookup = build_day_lookup(unique_records)

    manifest_rows = []
    summary = Counter()
    copied_by_subject = Counter()

    duplicate_hashes = {record.content_hash for record in duplicate_records}
    duplicate_paths = {record.source_path for record in duplicate_records}

    for record in unique_records:
        destination = ""
        day_folder = ""
        status = record.status
        notes = record.notes

        if record.status == "candidate":
            day_folder = day_lookup[(record.subject_id, record.date)]
            destination_path = (
                target_root
                / record.subject_id
                / day_folder
                / record.filename_activity
                / "json"
                / record.source_path.name
            )
            status = "copied" if not args.dry_run else "dry_run_copy"
            if not args.dry_run:
                destination_path = safe_copy(record.source_path, destination_path)
            destination = str(destination_path)
            copied_by_subject[record.subject_id] += 1
        elif record.status == "invalid_json":
            destination_path = target_root / "_quarantine" / "invalid_json" / record.source_path.name
            status = "quarantined_invalid_json" if not args.dry_run else "dry_run_quarantine"
            if not args.dry_run:
                destination_path = safe_copy(record.source_path, destination_path)
            destination = str(destination_path)
        else:
            destination_path = target_root / "_quarantine" / "unroutable" / record.source_path.name
            status = "quarantined_unroutable" if not args.dry_run else "dry_run_quarantine"
            if not args.dry_run:
                destination_path = safe_copy(record.source_path, destination_path)
            destination = str(destination_path)

        if record.content_hash in duplicate_hashes:
            notes = ";".join(filter(None, [notes, "has_duplicate_copies"]))

        summary[status] += 1
        manifest_rows.append(
            {
                "status": status,
                "subject_id": record.subject_id,
                "date": record.date,
                "day_folder": day_folder,
                "activity": record.filename_activity,
                "time": record.time,
                "exercise_type": record.exercise_type,
                "exercise_activity": record.exercise_activity,
                "timestamp_date": record.timestamp_date,
                "content_hash": record.content_hash,
                "source_path": str(record.source_path),
                "destination_path": destination,
                "notes": notes,
            }
        )

    for record in duplicate_records:
        summary["duplicate_skipped"] += 1
        day_folder = day_lookup.get((record.subject_id, record.date), "")
        status = "duplicate_skipped"
        if record.status != "candidate":
            status = f"duplicate_{record.status}"
            summary[status] += 1
        manifest_rows.append(
            {
                "status": status,
                "subject_id": record.subject_id,
                "date": record.date,
                "day_folder": day_folder,
                "activity": record.filename_activity,
                "time": record.time,
                "exercise_type": record.exercise_type,
                "exercise_activity": record.exercise_activity,
                "timestamp_date": record.timestamp_date,
                "content_hash": record.content_hash,
                "source_path": str(record.source_path),
                "destination_path": "",
                "notes": ";".join(filter(None, [record.notes, "duplicate_content_hash"])),
            }
        )

    manifest_rows.sort(key=lambda row: (row["subject_id"], row["date"], row["activity"], row["source_path"]))

    if not args.dry_run:
        write_manifest(target_root / "_manifest.csv", manifest_rows)
        summary_payload = {
            "source": str(source_root),
            "target": str(target_root),
            "total_json_seen": len(records),
            "unique_content_hashes": len({record.content_hash for record in records}),
            "unique_records_considered": len(unique_records),
            "duplicate_records": len(duplicate_records),
            "status_counts": dict(summary),
            "copied_by_subject": dict(sorted(copied_by_subject.items())),
            "days_by_subject": {
                subject: len({record.date for record in unique_records if record.subject_id == subject and record.status == "candidate"})
                for subject in sorted(copied_by_subject)
            },
        }
        (target_root / "_summary.json").write_text(
            json.dumps(summary_payload, indent=2), encoding="utf-8"
        )

    print(f"Source JSON files seen: {len(records)}")
    print(f"Unique content hashes: {len({record.content_hash for record in records})}")
    print(f"Duplicate records skipped: {len(duplicate_records)}")
    for key, value in sorted(summary.items()):
        print(f"{key}: {value}")
    print("Copied by subject:")
    for subject, count in sorted(copied_by_subject.items()):
        print(f"  {subject}: {count}")


if __name__ == "__main__":
    main()
