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
    status_var = tk.StringVar(value="Choose the folder that contains your Data_Sensors .txt files.")

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
            title="Select folder containing Data_Sensors .txt files",
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


def read_sensor_txt(txt_path: Path) -> pd.DataFrame:
    """Read an MT Manager text export and return only packet, accel, and gyro columns."""
    df = pd.read_csv(txt_path, comment="/")
    missing = [col for col in SOURCE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}. Available columns: {df.columns.tolist()}")

    df = df[SOURCE_COLUMNS].copy()
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


def output_path_for(txt_path: Path, input_dir: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{txt_path.stem}_synchronized.csv"

    # Keep nested source structure if .txt files live in subfolders.
    rel_parent = txt_path.parent.relative_to(input_dir)
    target_dir = input_dir / "synchronized_csv" / rel_parent
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{txt_path.stem}_synchronized.csv"


def convert_file(txt_path: Path, input_dir: Path, output_dir: Path | None, duration_seconds: float) -> dict[str, object]:
    df = read_sensor_txt(txt_path)
    synced, frequency_hz = build_synchronized_frame(df, duration_seconds)
    out_path = output_path_for(txt_path, input_dir, output_dir)
    synced.to_csv(out_path, index=False)

    packet_start = int(df["PacketCounter"].iloc[0])
    packet_end = int(df["PacketCounter"].iloc[-1])
    return {
        "source": str(txt_path),
        "output": str(out_path),
        "rows": len(synced),
        "frequency_hz": frequency_hz,
        "packet_start": packet_start,
        "packet_end": packet_end,
    }


def find_txt_files(input_dir: Path) -> list[Path]:
    ignored_dirs = {"synchronized_csv", "__pycache__"}
    paths = []
    for path in input_dir.rglob("*.txt"):
        if any(part.startswith(".") or part in ignored_dirs for part in path.relative_to(input_dir).parts):
            continue
        paths.append(path)
    return sorted(paths)


def convert_directory(input_dir: Path, output_dir: Path | None = None, duration_seconds: float = DEFAULT_DURATION_SECONDS) -> dict[str, object]:
    txt_files = find_txt_files(input_dir)
    summary: dict[str, object] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir) if output_dir else str(input_dir / "synchronized_csv"),
        "txt_files_found": len(txt_files),
        "converted": 0,
        "failed": 0,
        "outputs": [],
        "errors": [],
    }

    if not txt_files:
        print(f"[WARNING] No .txt files found under: {input_dir}")
        return summary

    for txt_path in txt_files:
        try:
            result = convert_file(txt_path, input_dir, output_dir, duration_seconds)
            summary["converted"] = int(summary["converted"]) + 1
            summary["outputs"].append(result)  # type: ignore[union-attr]
            print(
                "[OK] Converted "
                f"{txt_path.name} -> {result['output']} | "
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
    for key in ["input_dir", "output_dir", "txt_files_found", "converted", "failed"]:
        print(f"  - {key}: {summary[key]}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Data_Sensors .txt files to synchronized-style CSV files."
    )
    parser.add_argument("--input-dir", type=Path, default=None, help="Folder containing .txt sensor files.")
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
        f"TXT files found: {summary['txt_files_found']}\n"
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
