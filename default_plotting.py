import json
import csv
import os
import re
import socket
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from datetime import datetime

# --- Dash and Data Handling Imports ---
import dash
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go
import webbrowser

# === Exercise ID → description (your mapping) ===
EXERCISE_NAME_BY_ID = {
    1: "Sit to Stand",
    2: "Stand to Sit",
    3: "360 Left Turn",
    4: "10m",
    5: "360 Right Turn",
    6: "1m",
}

# ---------- UTIL ----------
def find_free_port(start=8050, limit=20):
    """Find a free local port starting at 8050."""
    for p in range(start, start + limit):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return start  # fallback

# ---------- DATA SYNCHRONIZATION ----------
def synchronize_sensor_data(csv_path):
    """
    Synchronize sensor data so each timestamp has all three sensor types.
    Converts from long format (multiple rows per timestamp) to wide format
    (one row per timestamp with all sensor data).
    """
    print(f"[INFO] Synchronizing sensor data from: {csv_path}")
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    print(f"[INFO] Original data shape: {df.shape}")
    print(f"[INFO] Sensor types found: {df['Sensor Type'].unique()}")
    
    # Normalize sensor type names
    df['Sensor Type'] = df['Sensor Type'].str.strip().str.lower()
    
    # Convert timestamp to numeric for proper sorting and merging
    df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
    
    # Remove duplicate rows (same timestamp, sensor type, and values)
    df = df.drop_duplicates(subset=['Timestamp', 'Sensor Type', 'X', 'Y', 'Z'])
    print(f"[INFO] After removing duplicates: {df.shape}")
    
    # Separate by sensor type
    accel_df = df[df['Sensor Type'] == 'accelerometer'].copy()
    user_accel_df = df[df['Sensor Type'] == 'useraccelerometer'].copy()
    gyro_df = df[df['Sensor Type'] == 'gyroscope'].copy()
    
    print(f"[INFO] Accelerometer rows: {len(accel_df)}")
    print(f"[INFO] UserAccelerometer rows: {len(user_accel_df)}")
    print(f"[INFO] Gyroscope rows: {len(gyro_df)}")
    
    # Check if we have data for all sensor types
    if len(accel_df) == 0 or len(user_accel_df) == 0 or len(gyro_df) == 0:
        print("[WARNING] Missing data for one or more sensor types!")
        print(f"[WARNING] Available sensor types: {df['Sensor Type'].unique()}")
    
    # Sort by timestamp
    accel_df = accel_df.sort_values('Timestamp').reset_index(drop=True)
    user_accel_df = user_accel_df.sort_values('Timestamp').reset_index(drop=True)
    gyro_df = gyro_df.sort_values('Timestamp').reset_index(drop=True)
    
    # Rename columns to include sensor type prefix
    accel_df = accel_df.rename(columns={
        'X': 'Accel_X',
        'Y': 'Accel_Y',
        'Z': 'Accel_Z'
    })[['Timestamp', 'Accel_X', 'Accel_Y', 'Accel_Z']]
    
    user_accel_df = user_accel_df.rename(columns={
        'X': 'UserAccel_X',
        'Y': 'UserAccel_Y',
        'Z': 'UserAccel_Z'
    })[['Timestamp', 'UserAccel_X', 'UserAccel_Y', 'UserAccel_Z']]
    
    gyro_df = gyro_df.rename(columns={
        'X': 'Gyro_X',
        'Y': 'Gyro_Y',
        'Z': 'Gyro_Z'
    })[['Timestamp', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']]
    
    # Determine appropriate tolerance based on timestamp scale
    # If timestamps are in milliseconds (>1e12), use millisecond tolerance
    # Otherwise use fractional second tolerance
    max_timestamp = df['Timestamp'].max()
    if max_timestamp > 1e12:
        # Timestamps in milliseconds - use larger tolerance
        tolerance = 100  # 100 milliseconds tolerance (was too strict at 10)
    else:
        # Timestamps in seconds (epoch)
        tolerance = 0.1  # 0.1 second tolerance
    
    print(f"[INFO] Using tolerance: {tolerance} (timestamp scale: {'ms' if max_timestamp > 1e12 else 's'})")

    # Build a base timeline from whatever sensors exist
    ts_frames = []
    if not accel_df.empty:
        ts_frames.append(accel_df[['Timestamp']])
    if not user_accel_df.empty:
        ts_frames.append(user_accel_df[['Timestamp']])
    if not gyro_df.empty:
        ts_frames.append(gyro_df[['Timestamp']])

    if ts_frames:
        base_ts = (
            pd.concat(ts_frames, ignore_index=True)
            .drop_duplicates()
            .sort_values('Timestamp')
            .reset_index(drop=True)
        )
    else:
        base_ts = pd.DataFrame(columns=['Timestamp'])
    
    # Merge on nearest timestamp using merge_asof
    # Start with accelerometer as base
    
    synchronized_df = base_ts

    if not accel_df.empty:
        synchronized_df = pd.merge_asof(
            synchronized_df, accel_df,
            on='Timestamp', direction='nearest', tolerance=tolerance
        )
    else:
        synchronized_df[['Accel_X', 'Accel_Y', 'Accel_Z']] = 0

    print(f"[INFO] After accel merge: {len(synchronized_df)} rows")

    if not user_accel_df.empty:
        synchronized_df = pd.merge_asof(
            synchronized_df, user_accel_df,
            on='Timestamp', direction='nearest', tolerance=tolerance
        )
    else:
        synchronized_df[['UserAccel_X', 'UserAccel_Y', 'UserAccel_Z']] = 0

    print(f"[INFO] After user_accel merge: {len(synchronized_df)} rows")

    if not gyro_df.empty:
        synchronized_df = pd.merge_asof(
            synchronized_df, gyro_df,
            on='Timestamp', direction='nearest', tolerance=tolerance
        )
    else:
        synchronized_df[['Gyro_X', 'Gyro_Y', 'Gyro_Z']] = 0

    print(f"[INFO] After gyro merge: {len(synchronized_df)} rows")
    
    # Remove any rows with missing data
    rows_before = len(synchronized_df)
    # Replace NaNs (no match within tolerance) with 0s for sensor columns
    sensor_cols = [c for c in synchronized_df.columns if c != 'Timestamp']
    synchronized_df[sensor_cols] = synchronized_df[sensor_cols].fillna(0)
    rows_after = len(synchronized_df)
    
    print(f"[INFO] Rows dropped due to missing data: {rows_before - rows_after}")
    print(f"[INFO] Final synchronized rows: {rows_after}")
    print(f"[INFO] Columns: {synchronized_df.columns.tolist()}")
    
    if len(synchronized_df) == 0:
        print("[ERROR] No synchronized data! This might mean:")
        print("  - Timestamps don't align within tolerance")
        print("  - Missing sensor data")
        print("  - Data format issues")
    
    # Save synchronized data
    sync_path = csv_path.replace('.csv', '_synchronized.csv')
    synchronized_df.to_csv(sync_path, index=False)
    print(f"[INFO] Synchronized data saved to: {sync_path}")
    
    return sync_path, synchronized_df

# ---------- JSON → CSV HELPERS ----------
def sanitize_timestamp_for_filename(ts):
    """Convert timestamp strings into safe, filename-friendly text."""
    if not ts:
        return "unknown_timestamp"

    # numeric epoch seconds or ms
    try:
        ts_float = float(ts)
        dt = datetime.fromtimestamp(ts_float / 1000 if ts_float > 1e12 else ts_float)
        return dt.strftime("%Y-%m-%d_%H-%M-%S")
    except Exception:
        pass

    # several common string formats
    known_formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S.%f",
    ]
    for fmt in known_formats:
        try:
            dt = datetime.strptime(ts, fmt)
            return dt.strftime("%Y-%m-%d_%H-%M-%S")
        except ValueError:
            continue

    # last resort: strip illegal filename chars
    return re.sub(r'[\\/:"*?<>|]+', "_", str(ts))

def detect_activity_from_json(data):
    """Detect the activity key that holds a list of dicts."""
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return key
    return "Unknown"

def convert_json_to_csv(json_text, save_dir, username="Unknown"):
    """
    Convert JSON content to CSV and save automatically in the provided folder.
    """
    data = json.loads(json_text)

    # activity key (e.g., "exercise_data")
    activity = detect_activity_from_json(data)
    entries = data.get(activity)
    
    if not entries or not isinstance(entries, list):
        raise ValueError("No valid activity data found in the JSON file.")

    # normalize rows
    def normalize_row(item):
        return {
            "Timestamp": item.get("timestamp", item.get("Timestamp", "")),
            "Sensor Type": item.get("sensorType", item.get("Sensor Type", "")),
            "X": item.get("x", item.get("X", "")),
            "Y": item.get("y", item.get("Y", "")),
            "Z": item.get("z", item.get("Z", "")),
        }

    rows = [normalize_row(item) for item in entries]
    fieldnames = ["Timestamp", "Sensor Type", "X", "Y", "Z"]

    if not rows or not rows[0].get("Timestamp"):
        raise ValueError("No timestamp found in data for naming the file.")

    # ===== filename pieces =====
    safe_ts = sanitize_timestamp_for_filename(rows[0]["Timestamp"])

    # pull exercise_type from the JSON and map to description
    exercise_type = data.get("exercise_type", None)
    try:
        etype_int = int(exercise_type) if exercise_type is not None else None
    except Exception:
        etype_int = None

    exercise_name = EXERCISE_NAME_BY_ID.get(
        etype_int,
        activity
    )

    filename = f"{username} - {exercise_name} - {safe_ts}.csv"

    # sanitize and join path
    filename = re.sub(r'[\\/:"*?<>|]+', "_", filename)
    out_path = os.path.join(save_dir, filename)

    # write CSV
    with open(out_path, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return out_path

# ---------- FILE PICKER ----------
def ask_file_and_load(root):
    """Let the user choose a CSV or JSON file; convert JSON to CSV if needed."""
    try:
        root.update()
    except Exception:
        pass

    path = filedialog.askopenfilename(
        parent=root,
        title="Open CSV or JSON",
        filetypes=[
            ("CSV or JSON", "*.csv *.json"),
            ("CSV files", "*.csv"),
            ("JSON files", "*.json"),
            ("All files", "*.*"),
        ],
    )
    if not path:
        messagebox.showinfo("Cancelled", "No file selected.", parent=root)
        return None

    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return path

    if ext == ".json":
        username = simpledialog.askstring("Username", "Enter username (Patient ID):", parent=root)
        try:
            with open(path, "r", encoding="utf-8") as f:
                json_content = f.read()
            out_csv = convert_json_to_csv(json_content, os.path.dirname(path), username or "Unknown")
            messagebox.showinfo("Converted", f"CSV saved at:\n{out_csv}", parent=root)
            return out_csv
        except Exception as e:
            messagebox.showerror("Error", f"Failed to convert JSON:\n{e}", parent=root)
            return None

    messagebox.showwarning("Unsupported", f"Unsupported file type: {ext}", parent=root)
    return None

# ---------- DASH APP ----------
def launch_dash(csv_path):
    print(f"[INFO] Launching Dash with: {csv_path}")
    
    # Synchronize the data first
    sync_path, sync_df = synchronize_sensor_data(csv_path)
    
    # Use the synchronized dataframe
    _df = sync_df

    def process_sensor_data(df, sensor_prefix, window_size, baseline_count, threshold_multiplier, axis):
        """
        Process sensor data from synchronized wide-format DataFrame.
        sensor_prefix: 'Accel', 'UserAccel', or 'Gyro'
        """
        d = df.copy()
        
        # Get the appropriate columns based on sensor prefix
        x_col = f"{sensor_prefix}_X"
        y_col = f"{sensor_prefix}_Y"
        z_col = f"{sensor_prefix}_Z"
        
        if x_col not in d.columns or y_col not in d.columns or z_col not in d.columns:
            empty_df = pd.DataFrame({"Datetime": [], "Value": []})
            return empty_df, "s", 0.0, 0.0
        
        sdf = d[['Timestamp', x_col, y_col, z_col]].copy()
        sdf = sdf.dropna()
        sdf = sdf.sort_values('Timestamp').reset_index(drop=True)
        
        if sdf.empty:
            empty_df = pd.DataFrame({"Datetime": [], "Value": []})
            return empty_df, "s", 0.0, 0.0

        unit = "ms" if sdf["Timestamp"].max() > 1e12 else "s"
        sdf["Datetime"] = pd.to_datetime(sdf["Timestamp"], unit=unit)

        if axis == "Resultant":
            sdf["Value"] = np.sqrt(
                pd.to_numeric(sdf[x_col], errors="coerce") ** 2
                + pd.to_numeric(sdf[y_col], errors="coerce") ** 2
                + pd.to_numeric(sdf[z_col], errors="coerce") ** 2
            )
        else:
            col_name = f"{sensor_prefix}_{axis}"
            sdf["Value"] = pd.to_numeric(sdf[col_name], errors="coerce")

        window_size = max(int(window_size), 1)
        baseline_count = max(int(baseline_count), 1)
        threshold_multiplier = float(threshold_multiplier)

        sdf["Variance"] = sdf["Value"].rolling(window=window_size, min_periods=1).var()
        baseline_var = float(sdf["Variance"].iloc[:baseline_count].mean()) if len(sdf) else 0.0
        threshold_var = baseline_var * threshold_multiplier
        return sdf, unit, baseline_var, threshold_var

    app = Dash(__name__, external_stylesheets=[
        "https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"
    ])
    app.title = "Motion Detection Tool"

    sensor_list = [
        {"label": "Gyroscope", "value": "Gyro"},
        {"label": "Accelerometer", "value": "Accel"},
        {"label": "User Accelerometer", "value": "UserAccel"}
    ]

    app.layout = html.Div(className="container py-4", children=[
        html.H2("Motion Detection Tool", className="text-center mb-4"),
        html.P(f"Synchronized Data ({len(_df)} samples)", className="text-muted text-center mb-4"),
        html.Div(className="form-group row justify-content-center", children=[
            html.Label("Detection Sensor:", className="col-sm-2 col-form-label text-right"),
            html.Div(className="col-sm-4", children=[
                dcc.Dropdown(
                    id="detect-sensor",
                    options=sensor_list,
                    value="Gyro",
                    clearable=False
                )
            ])
        ]),
        html.Div(className="row justify-content-center mb-4", children=[
            html.Div(className="col-auto form-group", children=[
                html.Label("Axis:"),
                dcc.Dropdown(
                    id="axis-dropdown",
                    options=[{"label": a, "value": a} for a in ["Resultant", "X", "Y", "Z"]],
                    value="Resultant",
                    clearable=False
                )
            ]),
            html.Div(className="col-auto form-group", children=[
                html.Label("Window:"),
                dcc.Input(id="window-input", type="number", value=50, debounce=True)
            ]),
            html.Div(className="col-auto form-group", children=[
                html.Label("Baseline:"),
                dcc.Input(id="baseline-input", type="number", value=15, debounce=True)
            ]),
            html.Div(className="col-auto form-group", children=[
                html.Label("Thresh Mult:"),
                dcc.Input(id="thresh-input", type="number", value=2, debounce=True)
            ])
        ]),
        html.Hr(),
        dcc.Graph(id="graph-master")
    ])

    @app.callback(
        Output("graph-master", "figure"),
        Input("detect-sensor", "value"),
        Input("axis-dropdown", "value"),
        Input("window-input", "value"),
        Input("baseline-input", "value"),
        Input("thresh-input", "value"),
    )
    def update_master(sensor_prefix, axis, win, base_cnt, thr_mult):
        sdf, unit, base_var, thr_var = process_sensor_data(_df, sensor_prefix, win, base_cnt, thr_mult, axis)
        fig = go.Figure()
        if not sdf.empty:
            fig.add_trace(go.Scatter(x=sdf["Datetime"], y=sdf["Value"], mode="lines", name=sensor_prefix))
        
        sensor_name = next((s['label'] for s in sensor_list if s['value'] == sensor_prefix), sensor_prefix)
        fig.update_layout(
            title=f"{sensor_name} Signal - {axis}",
            xaxis_title="Time",
            yaxis_title=axis
        )
        return fig

    port = find_free_port(8050)
    url = f"http://127.0.0.1:{port}"
    print(f"[INFO] Dash running on {url}")
    try:
        webbrowser.open(url, new=2)
    except Exception:
        pass

    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)

# ---------- MAIN ----------
if __name__ == "__main__":
    print("[INFO] Starting application.")
    root = tk.Tk()
    root.update_idletasks()
    root.lift()
    root.attributes("-topmost", True)
    root.after_idle(root.attributes, "-topmost", False)

    csv_path = ask_file_and_load(root)

    try:
        root.destroy()
    except Exception:
        pass

    if csv_path:
        print(f"[INFO] Launching analysis for: {csv_path}")
        launch_dash(csv_path)
    else:
        print("[INFO] No file selected. Exiting.")