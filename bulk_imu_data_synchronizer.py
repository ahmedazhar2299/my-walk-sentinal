import csv
import json
import re
import shutil
import traceback
from pathlib import Path

import pandas as pd

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception:
    tk = None
    filedialog = None
    messagebox = None


# ---------- CONFIG ----------
OVERWRITE_EXISTING = True
ARCHIVE_SOURCE_FILES = False
CSV_READ_CHUNK_SIZE = 50_000
JSON_READ_CHUNK_SIZE = 64 * 1024
JSON_KEY_SCAN_BYTES = 1_000_000
LAST_DIR_FILE = Path(__file__).resolve().parent / ".bulk_imu_last_dir"


# ---------- JSON -> CSV ----------
def detect_activity_key_from_object(data):
    """Fallback detector using parsed JSON object."""
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return key
    return None


def detect_activity_key_from_file(json_path):
    """
    Detect the first list key from JSON text without loading full file into memory.
    For IMU exports this is commonly "exercise_data" or "data".
    """
    with open(json_path, "r", encoding="utf-8") as f:
        head = f.read(JSON_KEY_SCAN_BYTES)

    matches = list(re.finditer(r'"([^"]+)"\s*:\s*\[', head))
    if not matches:
        return None
    return matches[0].group(1)


def iter_json_array_objects(json_path, array_key):
    """
    Stream objects from one JSON array key using incremental decoding.
    This avoids loading large arrays fully in memory.
    """
    key_token = f'"{array_key}"'
    decoder = json.JSONDecoder()
    buffer = ""
    array_started = False
    idx = 0
    eof = False

    with open(json_path, "r", encoding="utf-8") as f:
        while True:
            if not eof:
                chunk = f.read(JSON_READ_CHUNK_SIZE)
                if chunk == "":
                    eof = True
                else:
                    buffer += chunk

            if not array_started:
                key_pos = buffer.find(key_token)
                if key_pos == -1:
                    if eof:
                        raise ValueError(f'Key "{array_key}" not found in JSON.')
                    keep_chars = max(len(key_token) * 2, 2048)
                    if len(buffer) > keep_chars:
                        buffer = buffer[-keep_chars:]
                    continue

                bracket_pos = buffer.find("[", key_pos + len(key_token))
                if bracket_pos == -1:
                    if eof:
                        raise ValueError(f'Array start for key "{array_key}" not found.')
                    continue

                array_started = True
                idx = bracket_pos + 1

            while True:
                while idx < len(buffer) and buffer[idx] in " \r\n\t,":
                    idx += 1

                if idx >= len(buffer):
                    break

                if buffer[idx] == "]":
                    return

                try:
                    obj, next_idx = decoder.raw_decode(buffer, idx)
                except json.JSONDecodeError:
                    break

                if not isinstance(obj, dict):
                    raise ValueError(f'Array "{array_key}" contains non-object rows.')

                yield obj
                idx = next_idx

            if eof:
                while idx < len(buffer) and buffer[idx] in " \r\n\t,":
                    idx += 1
                if idx < len(buffer) and buffer[idx] == "]":
                    return
                raise ValueError(f'Unexpected end of file while parsing key "{array_key}".')

            if idx > 0:
                buffer = buffer[idx:]
                idx = 0


def normalize_row(item):
    """Normalize JSON row keys to CSV schema expected by synchronization."""
    return {
        "Timestamp": item.get("timestamp", item.get("Timestamp", "")),
        "Sensor Type": item.get("sensorType", item.get("Sensor Type", "")),
        "X": item.get("x", item.get("X", "")),
        "Y": item.get("y", item.get("Y", "")),
        "Z": item.get("z", item.get("Z", "")),
    }


def convert_json_file_to_csv(json_path, overwrite=True, output_csv_path=None):
    """
    Convert one JSON file to CSV in the same directory, row-by-row (streaming).
    Output naming preserves original base filename:
    "abc.json" -> "abc.csv"
    """
    if json_path.stat().st_size == 0:
        raise ValueError("JSON file is empty (0 bytes).")

    activity_key = detect_activity_key_from_file(json_path)
    if not activity_key:
        # Fallback for valid JSON files where key pattern detection fails.
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                parsed = json.load(f)
            activity_key = detect_activity_key_from_object(parsed)
        except Exception as exc:
            raise ValueError(f"Unable to parse JSON structure: {exc}") from exc

    if not activity_key:
        raise ValueError("No list-of-rows activity key found in JSON.")

    out_path = output_csv_path if output_csv_path is not None else json_path.with_suffix(".csv")
    if out_path.exists() and not overwrite:
        return out_path, 0, False

    fieldnames = ["Timestamp", "Sensor Type", "X", "Y", "Z"]
    row_count = 0

    with open(out_path, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for item in iter_json_array_objects(json_path, activity_key):
            writer.writerow(normalize_row(item))
            row_count += 1

    if row_count == 0:
        raise ValueError("No valid rows found in JSON array.")

    return out_path, row_count, True


# ---------- CSV STANDARDIZATION ----------
def build_canonical_rename_map(columns):
    """
    Normalize supported column variants into:
    Timestamp, Sensor Type, X, Y, Z
    """
    alias_map = {
        "timestamp": "Timestamp",
        "sensortype": "Sensor Type",
        "x": "X",
        "y": "Y",
        "z": "Z",
    }
    rename_map = {}
    for col in columns:
        normalized = re.sub(r"[\s_\-]+", "", str(col)).strip().lower()
        if normalized in alias_map:
            rename_map[col] = alias_map[normalized]
    return rename_map


def load_sensor_frames_chunked(csv_path):
    """Read source CSV in chunks and split into sensor-specific DataFrames."""
    required_cols = ["Timestamp", "Sensor Type", "X", "Y", "Z"]
    header = pd.read_csv(csv_path, nrows=0)
    rename_map = build_canonical_rename_map(header.columns.tolist())

    canonical_to_source = {}
    for source_col in header.columns:
        canonical_col = rename_map.get(source_col)
        if canonical_col and canonical_col not in canonical_to_source:
            canonical_to_source[canonical_col] = source_col

    missing = [c for c in required_cols if c not in canonical_to_source]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {header.columns.tolist()}"
        )

    usecols = [canonical_to_source[c] for c in required_cols]
    inverse_map = {v: k for k, v in canonical_to_source.items()}

    accel_chunks = []
    user_accel_chunks = []
    gyro_chunks = []

    total_rows = 0
    invalid_timestamp_rows = 0
    unknown_sensor_rows = 0
    max_timestamp = None

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=CSV_READ_CHUNK_SIZE):
        chunk = chunk.rename(columns=inverse_map)
        chunk = chunk[required_cols].copy()
        total_rows += len(chunk)

        chunk["Sensor Type"] = chunk["Sensor Type"].astype(str).str.strip().str.lower()
        chunk["Timestamp"] = pd.to_numeric(chunk["Timestamp"], errors="coerce")
        invalid_timestamp_rows += int(chunk["Timestamp"].isna().sum())
        chunk = chunk.dropna(subset=["Timestamp"])

        if chunk.empty:
            continue

        chunk["Timestamp"] = chunk["Timestamp"].astype(float)
        chunk["X"] = pd.to_numeric(chunk["X"], errors="coerce")
        chunk["Y"] = pd.to_numeric(chunk["Y"], errors="coerce")
        chunk["Z"] = pd.to_numeric(chunk["Z"], errors="coerce")

        chunk_max = chunk["Timestamp"].max()
        if max_timestamp is None or chunk_max > max_timestamp:
            max_timestamp = chunk_max

        known = chunk["Sensor Type"].isin(["accelerometer", "useraccelerometer", "gyroscope"])
        unknown_sensor_rows += int((~known).sum())

        accel_chunk = chunk[chunk["Sensor Type"] == "accelerometer"][["Timestamp", "X", "Y", "Z"]]
        user_accel_chunk = chunk[chunk["Sensor Type"] == "useraccelerometer"][["Timestamp", "X", "Y", "Z"]]
        gyro_chunk = chunk[chunk["Sensor Type"] == "gyroscope"][["Timestamp", "X", "Y", "Z"]]

        if not accel_chunk.empty:
            accel_chunks.append(accel_chunk)
        if not user_accel_chunk.empty:
            user_accel_chunks.append(user_accel_chunk)
        if not gyro_chunk.empty:
            gyro_chunks.append(gyro_chunk)

    accel_df = (
        pd.concat(accel_chunks, ignore_index=True) if accel_chunks else pd.DataFrame(columns=["Timestamp", "X", "Y", "Z"])
    )
    user_accel_df = (
        pd.concat(user_accel_chunks, ignore_index=True)
        if user_accel_chunks
        else pd.DataFrame(columns=["Timestamp", "X", "Y", "Z"])
    )
    gyro_df = (
        pd.concat(gyro_chunks, ignore_index=True) if gyro_chunks else pd.DataFrame(columns=["Timestamp", "X", "Y", "Z"])
    )

    accel_df = accel_df.drop_duplicates(subset=["Timestamp", "X", "Y", "Z"]).sort_values("Timestamp").reset_index(drop=True)
    user_accel_df = (
        user_accel_df.drop_duplicates(subset=["Timestamp", "X", "Y", "Z"]).sort_values("Timestamp").reset_index(drop=True)
    )
    gyro_df = gyro_df.drop_duplicates(subset=["Timestamp", "X", "Y", "Z"]).sort_values("Timestamp").reset_index(drop=True)

    accel_df = accel_df.rename(columns={"X": "Accel_X", "Y": "Accel_Y", "Z": "Accel_Z"})[
        ["Timestamp", "Accel_X", "Accel_Y", "Accel_Z"]
    ]
    user_accel_df = user_accel_df.rename(columns={"X": "UserAccel_X", "Y": "UserAccel_Y", "Z": "UserAccel_Z"})[
        ["Timestamp", "UserAccel_X", "UserAccel_Y", "UserAccel_Z"]
    ]
    gyro_df = gyro_df.rename(columns={"X": "Gyro_X", "Y": "Gyro_Y", "Z": "Gyro_Z"})[
        ["Timestamp", "Gyro_X", "Gyro_Y", "Gyro_Z"]
    ]

    stats = {
        "total_rows": total_rows,
        "invalid_timestamp_rows": invalid_timestamp_rows,
        "unknown_sensor_rows": unknown_sensor_rows,
    }
    return accel_df, user_accel_df, gyro_df, stats, max_timestamp


# ---------- DATA SYNCHRONIZATION ----------
def synchronize_sensor_data(csv_path, output_path):
    """
    Synchronize sensor data so each timestamp has all three sensor types.
    Long format -> wide format.
    CSV input is read in chunks for memory safety.
    """
    print(f"[INFO] Synchronizing sensor data from: {csv_path}")
    accel_df, user_accel_df, gyro_df, stats, max_timestamp = load_sensor_frames_chunked(csv_path)

    print(f"[INFO] Total rows read: {stats['total_rows']}")
    if stats["invalid_timestamp_rows"] > 0:
        print(f"[WARNING] Dropped rows with invalid timestamps: {stats['invalid_timestamp_rows']}")
    if stats["unknown_sensor_rows"] > 0:
        print(f"[WARNING] Ignored rows with unknown sensor types: {stats['unknown_sensor_rows']}")

    print(f"[INFO] Accelerometer rows: {len(accel_df)}")
    print(f"[INFO] UserAccelerometer rows: {len(user_accel_df)}")
    print(f"[INFO] Gyroscope rows: {len(gyro_df)}")

    if max_timestamp is None:
        raise ValueError("No valid timestamps found in file.")

    tolerance = 100 if max_timestamp > 1e12 else 0.1
    scale = "ms" if max_timestamp > 1e12 else "s"
    print(f"[INFO] Using tolerance: {tolerance} (timestamp scale: {scale})")

    ts_frames = []
    if not accel_df.empty:
        ts_frames.append(accel_df[["Timestamp"]])
    if not user_accel_df.empty:
        ts_frames.append(user_accel_df[["Timestamp"]])
    if not gyro_df.empty:
        ts_frames.append(gyro_df[["Timestamp"]])

    if not ts_frames:
        raise ValueError("No recognized sensor rows found (accelerometer/userAccelerometer/gyroscope).")

    base_ts = (
        pd.concat(ts_frames, ignore_index=True)
        .drop_duplicates()
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )
    synchronized_df = base_ts

    if not accel_df.empty:
        synchronized_df = pd.merge_asof(
            synchronized_df,
            accel_df,
            on="Timestamp",
            direction="nearest",
            tolerance=tolerance,
        )
    else:
        synchronized_df[["Accel_X", "Accel_Y", "Accel_Z"]] = 0

    if not user_accel_df.empty:
        synchronized_df = pd.merge_asof(
            synchronized_df,
            user_accel_df,
            on="Timestamp",
            direction="nearest",
            tolerance=tolerance,
        )
    else:
        synchronized_df[["UserAccel_X", "UserAccel_Y", "UserAccel_Z"]] = 0

    if not gyro_df.empty:
        synchronized_df = pd.merge_asof(
            synchronized_df,
            gyro_df,
            on="Timestamp",
            direction="nearest",
            tolerance=tolerance,
        )
    else:
        synchronized_df[["Gyro_X", "Gyro_Y", "Gyro_Z"]] = 0

    sensor_cols = [c for c in synchronized_df.columns if c != "Timestamp"]
    synchronized_df[sensor_cols] = synchronized_df[sensor_cols].fillna(0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    synchronized_df.to_csv(output_path, index=False)
    print(f"[INFO] Synchronized data saved to: {output_path}")
    return len(synchronized_df)


# ---------- BULK PROCESS ----------
def should_skip_path(path):
    """Skip hidden files and generated/archive folders."""
    lower_parts = [p.lower() for p in path.parts]
    for part in lower_parts:
        if part == "synchronized":
            return True
        if re.fullmatch(r"(csv|json)(_\d+)?", part):
            return True
    if path.name.startswith("."):
        return True
    return False


def is_date_folder_name(name):
    name = str(name).strip()
    # Accept common variants:
    # D1_2025-03-18, 2025-03-18, Session_2025-03-18, D1_2025_03_18
    if re.search(r"\d{4}[-_]\d{2}[-_]\d{2}", name):
        return True
    return False


def is_processable_patient_folder(path):
    """Return True for real patient export folders and False for backups/internal folders."""
    if not path.is_dir() or path.name.startswith(".") or path.name.startswith("_"):
        return False
    if "backup" in path.name.lower():
        return False
    return True


def detect_root_mode(root_dir):
    """
    root mode:
    - scrapper_root: root/patient_id/date/.../files
    - patient_root: root/date/.../files
    - date_root: root/.../files where root itself is a date folder
    - invalid: anything else
    """
    child_dirs = [p for p in root_dir.iterdir() if is_processable_patient_folder(p)]

    if is_date_folder_name(root_dir.name):
        return "date_root"

    if any(is_date_folder_name(d.name) for d in child_dirs):
        return "patient_root"

    if any(
        any(is_date_folder_name(grandchild.name) for grandchild in child.iterdir() if grandchild.is_dir())
        for child in child_dirs
    ):
        return "scrapper_root"

    return "invalid"


def file_matches_strict_layout(file_path, root_dir, root_mode):
    """
    Strict layouts accepted:
    - scrapper_root: root/patient/date/.../file
    - patient_root: root/date/.../file
    - date_root: root/.../file (root is date folder)
    """
    rel = file_path.resolve().relative_to(root_dir.resolve())
    parts = rel.parts

    if root_mode == "scrapper_root":
        if len(parts) < 3:
            return False
        date_part = parts[1]
        return is_date_folder_name(date_part)

    if root_mode == "patient_root":
        if len(parts) < 2:
            return False
        date_part = parts[0]
        return is_date_folder_name(date_part)

    if root_mode == "date_root":
        return len(parts) >= 1

    return False


def find_files_recursively(root_dir, suffix, root_mode):
    paths = []
    for path in root_dir.rglob(f"*{suffix}"):
        if path.is_file() and not should_skip_path(path):
            if not file_matches_strict_layout(path, root_dir, root_mode):
                continue
            paths.append(path)
    return sorted(paths)


def safe_move(src, dest):
    if dest.exists():
        stem = src.stem
        suffix = src.suffix
        idx = 1
        while True:
            candidate = dest.parent / f"{stem}_{idx}{suffix}"
            if not candidate.exists():
                dest = candidate
                break
            idx += 1
    shutil.move(str(src), str(dest))
    return dest


def route_outside_files(root_dir, root_mode):
    """
    For files outside strict layout, route by type into sibling folders:
    - *.json -> json/
    - *.csv -> csv/
    - *_synchronized.csv -> synchronized/
    """
    moved = 0
    for path in sorted(root_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue

        if path.suffix.lower() not in {".csv", ".json"}:
            continue

        if file_matches_strict_layout(path, root_dir, root_mode):
            continue

        if path.name.lower().endswith("_synchronized.csv"):
            folder_name = "synchronized"
        elif path.suffix.lower() == ".json":
            folder_name = "json"
        else:
            folder_name = "csv"

        if path.parent.name.lower() == folder_name:
            continue

        target_dir = path.parent / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)
        dest = safe_move(path, target_dir / path.name)
        moved += 1
        print(f"[INFO] Routed outside file: {path} -> {dest}")
    return moved


def make_run_archive_dir(parent_dir, base_name):
    """
    Return archive folder for this parent.
    If it does not exist, create it.
    """
    candidate = parent_dir / base_name
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def move_files_to_archive(files, base_name):
    """
    Move files into per-parent archive folders (`csv` or `json`).
    Returns number of moved files.
    """
    by_parent = {}
    for fpath in files:
        by_parent.setdefault(fpath.parent, []).append(fpath)

    moved = 0
    for parent, parent_files in by_parent.items():
        archive_dir = make_run_archive_dir(parent, base_name)
        for src in sorted(parent_files):
            if not src.exists():
                continue
            dest = archive_dir / src.name
            if dest.exists():
                stem = src.stem
                suffix = src.suffix
                k = 1
                while True:
                    candidate = archive_dir / f"{stem}_{k}{suffix}"
                    if not candidate.exists():
                        dest = candidate
                        break
                    k += 1
            shutil.move(str(src), str(dest))
            moved += 1
            print(f"[INFO] Archived: {src} -> {dest}")
    return moved


def process_folder(root_dir, overwrite=True):
    print(f"[INFO] Starting bulk processing for: {root_dir}")

    root_mode = detect_root_mode(root_dir)
    if root_mode == "invalid":
        print("[WARNING] Invalid folder format. Conversion blocked.")
        print("[WARNING] Expected: root/patient/date/.../*.csv, root/date/.../*.csv, or *.json")
        print("[WARNING] Or, select a single date folder directly.")
        summary = {
            "root": str(root_dir),
            "mode": root_mode,
            "outside_files_routed": 0,
            "json_files_found": 0,
            "json_success": 0,
            "json_failed": 0,
            "csv_files_found": 0,
            "sync_success": 0,
            "sync_failed": 0,
            "json_archived": 0,
            "csv_archived": 0,
        }
        return summary

    def get_date_dirs(root_path, mode):
        if mode == "date_root":
            return [root_path]
        if mode == "scrapper_root":
            date_dirs = []
            patient_dirs = [d for d in root_path.iterdir() if is_processable_patient_folder(d)]
            for patient_dir in sorted(patient_dirs):
                date_dirs.extend(
                    sorted([d for d in patient_dir.iterdir() if d.is_dir() and is_date_folder_name(d.name)])
                )
            return date_dirs
        return sorted([d for d in root_path.iterdir() if d.is_dir() and is_date_folder_name(d.name)])

    def get_activity_dirs(date_dir):
        """
        Return leaf activity folders for both Scrapper layouts:
        - date/activity_name/file.csv
        - date/activity_1/activity_name/file.csv
        """
        excluded = {"csv", "json", "synchronized"}
        activity_dirs = []

        for child in sorted(date_dir.iterdir()):
            if not child.is_dir() or child.name.startswith(".") or child.name.lower() in excluded:
                continue

            if re.fullmatch(r"activity_\d+", child.name.lower()):
                for nested_activity in sorted(child.iterdir()):
                    if (
                        nested_activity.is_dir()
                        and not nested_activity.name.startswith(".")
                        and nested_activity.name.lower() not in excluded
                    ):
                        activity_dirs.append(nested_activity)
            else:
                activity_dirs.append(child)

        return activity_dirs

    def organize_activity_inputs(activity_dir):
        csv_dir = activity_dir / "csv"
        json_dir = activity_dir / "json"
        sync_dir = activity_dir / "synchronized"

        moved_csv_local = 0
        moved_json_local = 0
        moved_sync_local = 0
        for item in sorted(activity_dir.iterdir()):
            if not item.is_file() or item.name.startswith("."):
                continue

            if item.suffix.lower() == ".json":
                json_dir.mkdir(parents=True, exist_ok=True)
                safe_move(item, json_dir / item.name)
                moved_json_local += 1
            elif item.suffix.lower() == ".csv":
                if item.name.lower().endswith("_synchronized.csv"):
                    sync_dir.mkdir(parents=True, exist_ok=True)
                    safe_move(item, sync_dir / item.name)
                    moved_sync_local += 1
                else:
                    csv_dir.mkdir(parents=True, exist_ok=True)
                    safe_move(item, csv_dir / item.name)
                    moved_csv_local += 1

        return csv_dir, json_dir, sync_dir, moved_csv_local, moved_json_local, moved_sync_local

    date_dirs = get_date_dirs(root_dir, root_mode)
    activity_dirs = []
    for date_dir in date_dirs:
        activity_dirs.extend(get_activity_dirs(date_dir))

    if not activity_dirs:
        print("[WARNING] No activity folders found under date folders.")
        summary = {
            "root": str(root_dir),
            "mode": root_mode,
            "outside_files_routed": 0,
            "json_files_found": 0,
            "json_success": 0,
            "json_failed": 0,
            "csv_files_found": 0,
            "sync_success": 0,
            "sync_failed": 0,
            "json_archived": 0,
            "csv_archived": 0,
        }
        return summary

    json_found = 0
    csv_found = 0
    json_success = 0
    json_fail = 0
    sync_success = 0
    sync_fail = 0
    moved_csv = 0
    moved_json = 0
    moved_sync = 0

    for activity_dir in activity_dirs:
        csv_dir, json_dir, sync_dir, m_csv, m_json, m_sync = organize_activity_inputs(activity_dir)
        moved_csv += m_csv
        moved_json += m_json
        moved_sync += m_sync

        # Convert all JSON in activity/json -> activity/csv
        json_files = sorted([p for p in json_dir.glob("*.json") if p.is_file()]) if json_dir.exists() else []
        json_found += len(json_files)
        for json_path in json_files:
            try:
                csv_dir.mkdir(parents=True, exist_ok=True)
                output_csv = csv_dir / f"{json_path.stem}.csv"
                out_csv, row_count, wrote_file = convert_json_file_to_csv(
                    json_path,
                    overwrite=overwrite,
                    output_csv_path=output_csv,
                )
                json_success += 1
                action = "converted" if wrote_file else "skipped(existing)"
                print(f"[OK] JSON -> CSV ({action}): {json_path} -> {out_csv} | rows={row_count}")
            except Exception as exc:
                json_fail += 1
                print(f"[ERROR] JSON conversion failed: {json_path}")
                print(f"[ERROR] {exc}")

        # Rebuild synchronized outputs from activity/csv each run.
        if sync_dir.exists():
            for old_sync in sync_dir.glob("*_synchronized.csv"):
                try:
                    old_sync.unlink()
                except Exception:
                    pass

        csv_files = sorted([p for p in csv_dir.glob("*.csv") if p.is_file()]) if csv_dir.exists() else []
        csv_found += len(csv_files)
        for csv_path in csv_files:
            try:
                sync_dir.mkdir(parents=True, exist_ok=True)
                sync_name = f"{csv_path.stem}_synchronized.csv"
                sync_path = sync_dir / sync_name
                rows_synced = synchronize_sensor_data(csv_path, sync_path)
                sync_success += 1
                print(f"[OK] CSV synchronized: {csv_path} -> {sync_path} | rows={rows_synced}")
            except Exception as exc:
                sync_fail += 1
                print(f"[ERROR] Synchronization failed: {csv_path}")
                print(f"[ERROR] {exc}")
                print(traceback.format_exc())

    summary = {
        "root": str(root_dir),
        "mode": root_mode,
        "outside_files_routed": moved_sync,
        "json_files_found": json_found,
        "json_success": json_success,
        "json_failed": json_fail,
        "csv_files_found": csv_found,
        "sync_success": sync_success,
        "sync_failed": sync_fail,
        "json_archived": moved_json,
        "csv_archived": moved_csv,
    }

    print("[INFO] Processing summary:")
    for key, value in summary.items():
        print(f"  - {key}: {value}")

    return summary


# ---------- UI ----------
def choose_directory():
    if tk is None or filedialog is None:
        raise RuntimeError(
            "tkinter is not available in this Python environment. "
            "Install python-tk to use folder picker UI."
        )

    def center_window(win, width=420, height=120):
        win.update_idletasks()
        screen_w = win.winfo_screenwidth()
        screen_h = win.winfo_screenheight()
        x = max((screen_w - width) // 2, 0)
        y = max((screen_h - height) // 2, 0)
        win.geometry(f"{width}x{height}+{x}+{y}")

    root = tk.Tk()
    root.title("Select Folder")
    center_window(root, width=420, height=120)
    root.update_idletasks()
    root.lift()
    root.attributes("-topmost", True)
    root.after_idle(root.attributes, "-topmost", False)
    root.update()

    initial_dir = str(Path.home())
    try:
        if LAST_DIR_FILE.exists():
            remembered = Path(LAST_DIR_FILE.read_text(encoding="utf-8").strip())
            if remembered.exists() and remembered.is_dir():
                initial_dir = str(remembered)
    except Exception:
        pass

    selected_dir = filedialog.askdirectory(
        parent=root,
        title="Select root folder to bulk process",
        mustexist=True,
        initialdir=initial_dir,
    )

    try:
        root.destroy()
    except Exception:
        pass

    if not selected_dir:
        return None

    try:
        LAST_DIR_FILE.write_text(str(Path(selected_dir).resolve()), encoding="utf-8")
    except Exception:
        pass

    return Path(selected_dir)


def main():
    selected = choose_directory()
    if not selected:
        print("[INFO] No directory selected. Exiting.")
        return

    summary = process_folder(selected, overwrite=OVERWRITE_EXISTING)
    message = (
        f"Root: {summary['root']}\n\n"
        f"Detected mode: {summary['mode']}\n"
        f"Loose synchronized moved: {summary['outside_files_routed']}\n\n"
        f"JSON files found: {summary['json_files_found']}\n"
        f"JSON converted: {summary['json_success']}\n"
        f"JSON failed: {summary['json_failed']}\n"
        f"JSON moved to /json: {summary['json_archived']}\n\n"
        f"CSV files found: {summary['csv_files_found']}\n"
        f"Synchronized: {summary['sync_success']}\n"
        f"Sync failed: {summary['sync_failed']}\n"
        f"CSV moved to /csv: {summary['csv_archived']}\n\n"
        f"Overwrite mode: {OVERWRITE_EXISTING}"
    )
    try:
        popup = tk.Tk()
        popup.update_idletasks()
        screen_w = popup.winfo_screenwidth()
        screen_h = popup.winfo_screenheight()
        width, height = 420, 120
        x = max((screen_w - width) // 2, 0)
        y = max((screen_h - height) // 2, 0)
        popup.geometry(f"{width}x{height}+{x}+{y}")
        popup.withdraw()
        messagebox.showinfo("Bulk Processing Complete", message, parent=popup)
        popup.destroy()
    except Exception:
        print(message)


if __name__ == "__main__":
    main()
