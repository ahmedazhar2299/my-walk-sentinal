# IMU Feature Extraction Pipeline

This project builds one feature row per `patient/date` from activity-specific IMU CSV files.

## Project Structure

- `extract_features.py`: CLI entry point for full extraction and validation checks.
- `imu_features/config.py`: Config dataclasses and optional JSON override loader.
- `imu_features/utils.py`: Shared preprocessing, filtering, peak detection, spectral metrics, and file resolution.
- `imu_features/walk_features.py`: Walk/gait feature extraction.
- `imu_features/turn_features.py`: Left/right turn feature extraction.
- `imu_features/transition_features.py`: Sit-to-stand and stand-to-sit feature extraction.
- `imu_features/extractors.py`: Patient/date loop, missing-file handling, asymmetry features, CSV writing.

## Preprocessing Pipeline

For each activity CSV:

1. Map columns to canonical names (configurable).
2. Convert timestamps to seconds and sort by time.
3. Collapse duplicate timestamps by averaging.
4. Interpolate/ffill/bfill missing sensor samples.
5. Compute magnitudes:
   - `acc_mag = sqrt(accel_x^2 + accel_y^2 + accel_z^2)`
   - `useracc_mag = sqrt(useracc_x^2 + useracc_y^2 + useracc_z^2)`
   - `gyro_mag = sqrt(gyro_x^2 + gyro_y^2 + gyro_z^2)`
6. Estimate sampling interval from median positive time difference.
7. Optional Butterworth low-pass filter on magnitude signals.
8. Compute jerk with irregular-time derivative:
   - `jerk = d(signal)/dt` using `np.gradient(signal, time_s)`

Required exact filenames per patient/date folder:
- `walk.csv`
- `left_turn.csv`
- `right_turn.csv`
- `sit_to_stand.csv`
- `stand_to_sit.csv`

## Feature Formulas

### Walk

- `walk_duration = detected_walk_end - detected_walk_start` (movement window inside file)
- `step_count = number_of_detected_peaks(acc_signal)`
- `cadence = 60 * step_count / walk_duration`
- `walking_speed = walk_distance_m / walk_duration` (default distance = 10m)
- `mean_step_time = mean(diff(step_times))`
- `step_time_std = std(diff(step_times))`
- `step_time_cv = step_time_std / mean_step_time`
- `step_regularity`, `stride_regularity`: normalized autocorrelation at step/stride lags.
- `walk_dominant_frequency`: FFT peak frequency of the walking acceleration signal (excluding 0 Hz).
- Signal stats from acceleration/user acceleration magnitude and gyroscope magnitude:
  mean, std, RMS, dominant frequency, spectral entropy, jerk mean/std.

### Left/Right Turn

- `*_duration = detected_turn_end - detected_turn_start` (turning window inside file)
- `*_mean_angular_velocity = mean(abs(turn_angular_signal))`
- `*_peak_angular_velocity = max(abs(turn_angular_signal))`
- `*_ang_vel_std = std(turn_angular_signal)`
- `*_pause_count`, `*_pause_time`: count and total time where `|angular_velocity|` is below threshold for at least minimum pause duration.
- `*_jerk_std = std(d(|angular_velocity|)/dt)`
- `*_entropy = spectral entropy of angular velocity`

### Sit-to-Stand / Stand-to-Sit

- `*_duration = detected_transition_end - detected_transition_start` (active transition window)
- `*_time_to_peak_acc = time_of_peak_acc - t_start`
- `*_peak_acc = max(acc_signal)`
- `*_peak_gyro = max(gyro_mag)`
- `*_acc_rms = sqrt(mean(acc_signal^2))`
- `*_jerk_mean = mean(abs(d(acc_signal)/dt))`
- `*_jerk_std = std(d(acc_signal)/dt)`
- `*_peak_count = number_of_detected_peaks(acc_signal)`
- `*_entropy = spectral entropy(acc_signal)`

### Turn Asymmetry

- `turn_duration_difference = left_turn_duration - right_turn_duration`
- `turn_velocity_difference = left_turn_mean_angular_velocity - right_turn_mean_angular_velocity`
- `turn_pause_difference = left_turn_pause_time - right_turn_pause_time`
- Absolute versions are `abs(...)`.

## Running the Pipeline

```bash
python3 extract_features.py \
  --dataset-root Data \
  --output-csv features_dataset.csv
```

## Optional Config Override (JSON)

Create `config_override.json`:

```json
{
  "walk_distance_m": 10.0,
  "filtering": {
    "enabled": true,
    "cutoff_hz": 5.0,
    "order": 4
  },
  "step_detection": {
    "min_distance_s": 0.45
  },
  "turn_pause": {
    "min_pause_duration_s": 0.2
  },
  "transition_peaks": {
    "min_distance_s": 0.15
  },
  "adaptive_thresholds": {
    "quiet_window_sec": 1.0,
    "step_threshold_k": 4.0,
    "turn_threshold_k": 3.0,
    "pause_threshold_k": 1.5,
    "transition_threshold_k": 2.0
  }
}
```

The pipeline estimates a quiet baseline window in each file and uses:

- threshold: `mean_quiet + k * std_quiet`

This single adaptive threshold is applied to walk step peaks, turn detection, pause detection, and transition peak counting.

Run:

```bash
python3 extract_features.py --config-json config_example.json
```

## Validation Checks

`extract_features.py` runs basic checks:

- duplicate `patient_id/date` rows
- rows where all features are `NaN`
- high missingness in key duration features
