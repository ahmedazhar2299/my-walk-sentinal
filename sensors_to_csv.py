from __future__ import annotations

import argparse
import os
import subprocess
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


DEFAULT_DURATION_SECONDS = 60.0
LAST_DIR_FILE = Path(__file__).resolve().parent / ".data_sensors_last_dir"
OUTPUT_COLUMNS = [
    "Timestamp",
    "Accel_X",
    "Accel_Y",
    "Accel_Z",
    "UserAccel_X",
    "UserAccel_Y",
    "UserAccel_Z",
    "Gyro_X",
    "Gyro_Y",
    "Gyro_Z",
]
SOURCE_COLUMNS = ["PacketCounter", "Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"]
SOURCE_COLUMN_ALIASES = {
    "PacketCounter": ["PacketCounter", "Packet Counter"],
    "Acc_X": ["Acc_X", "Acc_X(m/s^2)", "Accel_X", "Accelerometer_X"],
    "Acc_Y": ["Acc_Y", "Acc_Y(m/s^2)", "Accel_Y", "Accelerometer_Y"],
    "Acc_Z": ["Acc_Z", "Acc_Z(m/s^2)", "Accel_Z", "Accelerometer_Z"],
    "Gyr_X": ["Gyr_X", "Gyr_X(rad/s)", "Gyro_X", "Gyroscope_X"],
    "Gyr_Y": ["Gyr_Y", "Gyr_Y(rad/s)", "Gyro_Y", "Gyroscope_Y"],
    "Gyr_Z": ["Gyr_Z", "Gyr_Z(rad/s)", "Gyro_Z", "Gyroscope_Z"],
}
SENSOR_PLACEMENT_BY_DEVICE_ID = {
    "42760": "right",
    "426CC": "left",
    "4276D": "trunk",
    "42660": "sacrum",
}


def choose_directory() -> Path | None:
    if tk is None or filedialog is None:
        raise RuntimeError("tkinter is not available in this Python environment.")

    selected_path: Path | None = None

    root = tk.Tk()
    root.title("Select Data_Sensors Folder")
    root.geometry("680x190")
    root.resizable(False, False)
    root.update_idletasks()

    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    x = max((screen_w - 680) // 2, 0)
    y = max((screen_h - 190) // 2, 0)
    root.geometry(f"680x190+{x}+{y}")

    root.attributes("-topmost", True)
    root.lift()
    root.focus_force()

    try:
        subprocess.run(
            ["osascript", "-e", 'tell application "System Events" to set frontmost of first process whose unix id is %d to true' % os.getpid()],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass

    initial_dir = Path.home()
    try:
        remembered = Path(LAST_DIR_FILE.read_text(encoding="utf-8").strip())
        if remembered.exists() and remembered.is_dir():
            initial_dir = remembered
    except Exception:
        pass

    path_var = tk.StringVar(value=str(initial_dir))
    status_var = tk.StringVar(value="Choose the folder that contains your Data_Sensors TXT or XSENS CSV files.")

    def remember_and_close(path: Path) -> None:
        nonlocal selected_path
        selected_path = path.resolve()
        try:
            LAST_DIR_FILE.write_text(str(selected_path), encoding="utf-8")
        except Exception:
            pass
        root.quit()

    def browse() -> None:
        root.attributes("-topmost", True)
        root.lift()
        root.focus_force()
        selected = filedialog.askdirectory(
            parent=root,
            title="Select folder containing Data_Sensors TXT or XSENS CSV files",
            mustexist=True,
            initialdir=path_var.get() or str(initial_dir),
        )
        root.lift()
        root.focus_force()
        if selected:
            path_var.set(selected)
            remember_and_close(Path(selected))

    def use_typed_path() -> None:
        typed = path_var.get().strip().strip('"').strip("'")
        if not typed:
            status_var.set("Please enter a folder path or click Browse.")
            return
        path = Path(typed).expanduser()
        if not path.exists() or not path.is_dir():
            status_var.set(f"Not a valid folder: {path}")
            return
        remember_and_close(path)

    def cancel() -> None:
        root.quit()

    frame = tk.Frame(root, padx=18, pady=16)
    frame.pack(fill="both", expand=True)

    tk.Label(frame, textvariable=status_var, anchor="w").pack(fill="x")
    entry = tk.Entry(frame, textvariable=path_var)
    entry.pack(fill="x", pady=(14, 12))
    entry.focus_set()

    buttons = tk.Frame(frame)
    buttons.pack(fill="x")
    tk.Button(buttons, text="Browse...", width=14, command=browse).pack(side="left")
    tk.Button(buttons, text="Use This Folder", width=16, command=use_typed_path).pack(side="left", padx=(10, 0))
    tk.Button(buttons, text="Cancel", width=10, command=cancel).pack(side="right")

    root.bind("<Return>", lambda _event: use_typed_path())
    root.bind("<Escape>", lambda _event: cancel())
    root.after(2500, lambda: root.attributes("-topmost", False))

    try:
        root.mainloop()
    finally:
        root.destroy()

    return selected_path


def choose_directory_with_fallback() -> Path | None:
    try:
        return choose_directory()
    except Exception as exc:
        print(f"[WARNING] Tkinter folder picker failed: {exc}")
        print("[INFO] Paste the folder path here, or press Enter to exit.")
        typed = input("Folder path: ").strip().strip('"').strip("'")
        if not typed:
            return None
        path = Path(typed).expanduser().resolve()
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Selected path is not a valid directory: {path}")
        return path


def normalize_column_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def build_source_rename_map(columns: list[str]) -> dict[str, str]:
    normalized_to_original = {normalize_column_name(col): col for col in columns}
    rename_map = {}
    missing = []

    for canonical_col, aliases in SOURCE_COLUMN_ALIASES.items():
        source_col = None
        for alias in aliases:
            source_col = normalized_to_original.get(normalize_column_name(alias))
            if source_col is not None:
                break
        if source_col is None:
            missing.append(canonical_col)
        else:
            rename_map[source_col] = canonical_col

    if missing:
        raise ValueError(f"Missing required columns {missing}. Available columns: {columns}")
    return rename_map


def read_sensor_file(sensor_path: Path) -> pd.DataFrame:
    """Read an MT Manager TXT or XSENS CSV export and return packet, accel, and gyro columns."""
    if sensor_path.suffix.lower() == ".txt":
        df = pd.read_csv(sensor_path, comment="/")
    elif sensor_path.suffix.lower() == ".csv":
        df = pd.read_csv(sensor_path)
    else:
        raise ValueError(f"Unsupported sensor file type: {sensor_path.suffix}")

    rename_map = build_source_rename_map(df.columns.tolist())
    df = df.rename(columns=rename_map)[SOURCE_COLUMNS].copy()
    for col in SOURCE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=SOURCE_COLUMNS).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid sensor rows found after numeric cleanup.")

    return df


def build_synchronized_frame(df: pd.DataFrame, duration_seconds: float) -> tuple[pd.DataFrame, float]:
    """
    Generate synchronized-style rows.

    The exports represent roughly one minute of data, so frequency is estimated as N / 60.
    PacketCounter is used as the reference so packet gaps are reflected in generated time.
    """
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than zero.")

    row_count = len(df)
    frequency_hz = row_count / duration_seconds
    first_packet = df["PacketCounter"].iloc[0]

    out = pd.DataFrame(
        {
            "Timestamp": (df["PacketCounter"] - first_packet) / frequency_hz,
            "Accel_X": df["Acc_X"],
            "Accel_Y": df["Acc_Y"],
            "Accel_Z": df["Acc_Z"],
            "UserAccel_X": pd.NA,
            "UserAccel_Y": pd.NA,
            "UserAccel_Z": pd.NA,
            "Gyro_X": df["Gyr_X"],
            "Gyro_Y": df["Gyr_Y"],
            "Gyro_Z": df["Gyr_Z"],
        }
    )
    return out[OUTPUT_COLUMNS], frequency_hz


def output_root_for(input_dir: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir

    return input_dir.parent / f"{input_dir.name}_synchronized"


def placement_for_file(txt_path: Path) -> str:
    upper_name = txt_path.stem.upper()
    for device_suffix, placement in SENSOR_PLACEMENT_BY_DEVICE_ID.items():
        if upper_name.endswith(device_suffix.upper()) or f"_{device_suffix.upper()}" in upper_name:
            return placement
    return "unknown_sensor"


def output_path_for(txt_path: Path, input_dir: Path, output_dir: Path | None) -> Path:
    output_root = output_root_for(input_dir, output_dir)
    placement = placement_for_file(txt_path)

    rel_parent = txt_path.parent.relative_to(input_dir)
    target_dir = output_root / rel_parent / placement
    target_dir.mkdir(parents=True, exist_ok=True)

    target = target_dir / f"{txt_path.stem}_synchronized.csv"
    if not target.exists():
        return target

    suffix = 1
    while True:
        candidate = target_dir / f"{txt_path.stem}_synchronized_{suffix}.csv"
        if not candidate.exists():
            return candidate
        suffix += 1


def convert_file(txt_path: Path, input_dir: Path, output_dir: Path | None, duration_seconds: float) -> dict[str, object]:
    df = read_sensor_file(txt_path)
    synced, frequency_hz = build_synchronized_frame(df, duration_seconds)
    out_path = output_path_for(txt_path, input_dir, output_dir)
    synced.to_csv(out_path, index=False)

    packet_start = int(df["PacketCounter"].iloc[0])
    packet_end = int(df["PacketCounter"].iloc[-1])
    return {
        "source": str(txt_path),
        "output": str(out_path),
        "placement": placement_for_file(txt_path),
        "rows": len(synced),
        "frequency_hz": frequency_hz,
        "packet_start": packet_start,
        "packet_end": packet_end,
    }


def find_sensor_files(input_dir: Path) -> list[Path]:
    ignored_dirs = {"synchronized_csv", "__pycache__"}
    paths = []
    for path in input_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".csv"}:
            continue
        if any(part.startswith(".") or part in ignored_dirs for part in path.relative_to(input_dir).parts):
            continue
        if any(part.endswith("_synchronized") for part in path.relative_to(input_dir).parts):
            continue
        if path.name.lower().endswith("_synchronized.csv"):
            continue
        paths.append(path)
    return sorted(paths)


def convert_directory(input_dir: Path, output_dir: Path | None = None, duration_seconds: float = DEFAULT_DURATION_SECONDS) -> dict[str, object]:
    sensor_files = find_sensor_files(input_dir)
    summary: dict[str, object] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_root_for(input_dir, output_dir)),
        "sensor_files_found": len(sensor_files),
        "converted": 0,
        "failed": 0,
        "outputs": [],
        "errors": [],
    }

    if not sensor_files:
        print(f"[WARNING] No .txt or XSENS .csv files found under: {input_dir}")
        return summary

    for txt_path in sensor_files:
        try:
            result = convert_file(txt_path, input_dir, output_dir, duration_seconds)
            summary["converted"] = int(summary["converted"]) + 1
            summary["outputs"].append(result)  # type: ignore[union-attr]
            print(
                "[OK] Converted "
                f"{txt_path.name} -> {result['output']} | "
                f"placement={result['placement']} | "
                f"rows={result['rows']} | freq={float(result['frequency_hz']):.3f} Hz | "
                f"PacketCounter={result['packet_start']}..{result['packet_end']}"
            )
        except Exception as exc:
            summary["failed"] = int(summary["failed"]) + 1
            error = {"source": str(txt_path), "error": str(exc)}
            summary["errors"].append(error)  # type: ignore[union-attr]
            print(f"[ERROR] Failed: {txt_path}")
            print(f"[ERROR] {exc}")
            print(traceback.format_exc())

    print("[INFO] Conversion summary:")
    for key in ["input_dir", "output_dir", "sensor_files_found", "converted", "failed"]:
        print(f"  - {key}: {summary[key]}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Data_Sensors TXT or XSENS CSV files to synchronized-style CSV files."
    )
    parser.add_argument("--input-dir", type=Path, default=None, help="Folder containing TXT or XSENS CSV sensor files.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output folder.")
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=DEFAULT_DURATION_SECONDS,
        help="Recording duration used to estimate frequency. Defaults to 60.",
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Do not wait for Enter before closing when using the folder picker.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    launched_with_picker = args.input_dir is None
    input_dir = args.input_dir.resolve() if args.input_dir else choose_directory_with_fallback()
    if input_dir is None:
        print("[INFO] No directory selected. Exiting.")
        if launched_with_picker and not args.no_pause:
            input("Press Enter to close...")
        return

    try:
        output_dir = args.output_dir.resolve() if args.output_dir else None
        summary = convert_directory(input_dir, output_dir, args.duration_seconds)
    except Exception:
        print("[ERROR] Conversion failed before completion.")
        print(traceback.format_exc())
        if launched_with_picker and not args.no_pause:
            input("Press Enter to close...")
        raise

    message = (
        f"Input: {summary['input_dir']}\n"
        f"Output: {summary['output_dir']}\n\n"
        f"Sensor files found: {summary['sensor_files_found']}\n"
        f"Converted: {summary['converted']}\n"
        f"Failed: {summary['failed']}"
    )
    if tk is not None and messagebox is not None and launched_with_picker:
        try:
            popup = tk.Tk()
            popup.withdraw()
            messagebox.showinfo("Data Sensors Conversion Complete", message, parent=popup)
            popup.destroy()
        except Exception:
            print(message)
    else:
        print(message)

    if launched_with_picker and not args.no_pause:
        input("Press Enter to close...")


if __name__ == "__main__":
    main()
