# IMU Feature Extraction Pipeline

This project extracts one feature row per `patient_id/date` from synchronized IMU activity CSV files. The CSV output is aligned with the feature tables used in:

- `walk_validation.ipynb`
- `turn_validation.ipynb`
- `transition_validation.ipynb`

The default output is `features_dataset.csv`.

## Setup

From the repo root:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements-feature-pipeline.txt
```

Run extraction:

```bash
env/bin/python extract_features.py --dataset-root Data --output-csv features_dataset.csv
```

Quiet version:

```bash
env/bin/python extract_features.py --dataset-root Data --output-csv features_dataset.csv --quiet
```

Optional config override:

```bash
env/bin/python extract_features.py --dataset-root Data --output-csv features_dataset.csv --config-json config_example.json
```

## Project Structure

- `extract_features.py`: command-line entry point for full extraction and validation checks.
- `imu_features/config.py`: config dataclasses and optional JSON override loader.
- `imu_features/utils.py`: preprocessing, filtering, file discovery, wavelet helpers, and shared metrics.
- `imu_features/walk_features.py`: walking linear and nonlinear feature extraction.
- `imu_features/turn_features.py`: left/right turn feature extraction.
- `imu_features/transition_features.py`: sit-to-stand and stand-to-sit feature extraction.
- `imu_features/extractors.py`: patient/date loop, feature merging, symmetry features, and CSV writing.
- `validation_plots/`: plotting helpers used by the notebooks.

## Data Layout

The extractor scans `Data/<patient_id>/<date>/...` and writes one row per patient/date.

It supports both flat files and nested synchronized files. Examples:

```text
Data/patient_103/2026-03-08/walk.csv
Data/patient_103/2026-03-08/left_turn.csv
Data/patient_111/2026-04-28/Activity_2/sit_stand/synchronized/sit_to_stand_synchronized.csv
```

Recognized activity aliases:

- Walk: `walk`, `10_mw`
- Left turn: `left_turn`, `360_LeftTurn`
- Right turn: `right_turn`, `360_RightTurn`
- Sit-to-stand: `sit_to_stand`, `sit_stand`
- Stand-to-sit: `stand_to_sit`, `stand_sit`

Synchronized CSVs are preferred when both raw and synchronized files exist.

## Long Recording Truncation

Some sensor files can keep recording after the activity is over. Before plotting or extracting features, the code uses a simple fixed-range rule from the end of the file:

- Walk: files up to `31s` are unchanged; longer files keep the last `31s`.
- Left/right turn: files up to `11s` are unchanged; longer files keep the last `11s`.
- Sit-to-stand / stand-to-sit: files up to `17s` are unchanged; longer files keep the last `17s`.

After truncation, `time_s` is reset so plots start at `0s`.

This selected segment is used everywhere: raw accelerometer/gyroscope/user-accelerometer plots, wavelet/FFT/jerk plots, cross-correlation comparisons, and all feature tables.

## Preprocessing

For each activity CSV:

1. Map sensor columns to canonical names.
2. Convert timestamps to seconds.
3. Sort by time and average duplicate timestamps.
4. Interpolate/forward-fill/back-fill missing sensor samples.
5. Compute magnitudes:
   - `acc_mag = sqrt(accel_x^2 + accel_y^2 + accel_z^2)`
   - `useracc_mag = sqrt(useracc_x^2 + useracc_y^2 + useracc_z^2)`
   - `gyro_mag = sqrt(gyro_x^2 + gyro_y^2 + gyro_z^2)`
6. Estimate sampling frequency from median timestamp spacing.
7. Optionally apply a Butterworth low-pass filter.
8. Compute jerk using `np.gradient(signal, time_s)`.

## Output Format

`features_dataset.csv` is a wide table:

```text
patient_id, date, [walk features], [turn features], [sit/stand features]
```

The current extraction includes:

### Walking

Linear features:

- `walk_duration`
- `step_count`
- `cadence`
- `walking_speed`
- `mean_step_time`
- `step_time_std`
- `step_time_cv`
- `step_regularity`
- `stride_regularity`
- `walk_acc_mag_mean`
- `walk_acc_mag_std`
- `walk_acc_mag_rms`
- `walk_gyro_mag_std`
- `walk_dominant_frequency`
- `walk_jerk_mean`
- `walk_jerk_std`

Nonlinear features:

- `walk_spectral_entropy`
- `approximate_entropy`
- `sample_entropy`
- `symbolic_entropy`
- `permutation_entropy`
- `rosenstein_lyapunov_exponent`
- `wolf_lyapunov_exponent`
- `rqa_REC`
- `rqa_DET`
- `rqa_LAM`
- `rqa_MeanL`
- `rqa_MaxL`
- `rqa_EntrL`
- `rqa_EntrV`
- `rqa_EntrW`

### Turns

For both `left_turn_*` and `right_turn_*`:

- `duration`
- `mean_angular_velocity`
- `peak_angular_velocity`
- `ang_vel_std`
- `step_count`
- `pause_time`
- `jerk_std`
- `entropy`

Turn step count is estimated from acceleration peaks inside the gyro-detected turn start/end window.

Turn comparison features:

- `turn_duration_difference`
- `turn_velocity_difference`
- `turn_pause_difference`
- `abs_turn_duration_difference`
- `abs_turn_velocity_difference`
- `abs_turn_pause_difference`
- `turn_left_right_xcorr_symmetry_score`
- `turn_left_right_xcorr_peak_correlation`

The cross-correlation features resample each detected turn to a 0 to 100 percent movement cycle before computing normalized cross-correlation.

### Sit-To-Stand / Stand-To-Sit

For both `sit_to_stand_*` and `stand_to_sit_*`:

- `duration`
- `time_to_peak_acc`
- `peak_acc`
- `flexion_peak`
- `extension_peak`
- `peak_gyro`
- `acc_rms`
- `jerk_mean`
- `jerk_std`
- `peak_count`
- `entropy`

Flexion is the maximum value of the selected dominant signed gyro axis inside the detected transition region. Extension is the minimum value of that same signal. End time is refined after the later of the flexion/extension peaks.

Transition comparison features:

- `sitstand_standsit_xcorr_symmetry_score`
- `sitstand_standsit_xcorr_peak_correlation`

These are also computed after resampling each transition to a 0 to 100 percent movement cycle.

## Validation Notebooks

Use these notebooks to inspect individual signals, windows, and plots:

- `walk_validation.ipynb`
- `turn_validation.ipynb`
- `transition_validation.ipynb`

The CSV extractor uses the same core feature functions as the notebook feature tables.

## Validation Checks

By default, `extract_features.py` checks for:

- duplicate `patient_id/date` rows
- rows where all features are `NaN`
- high missingness in key duration features

Skip checks with:

```bash
env/bin/python extract_features.py --dataset-root Data --output-csv features_dataset.csv --skip-checks
```

## Notes

If Matplotlib prints a cache warning, extraction can still complete. It means the default Matplotlib cache directory is not writable. To avoid the warning:

```bash
mkdir -p .matplotlib-cache
MPLCONFIGDIR=.matplotlib-cache env/bin/python extract_features.py --dataset-root Data --output-csv features_dataset.csv
```
