# MyWalkSentinel IMU Feature Extraction

This repository provides the feature extraction and validation code used for the MyWalkSentinel stroke mobility study. The code supports two analysis workflows:

- smartphone-derived home mobility features from repeated unsupervised recordings
- laboratory validation using XSENS IMU recordings and Vicon motion-capture reference data

The repository does not include participant data or generated result tables. Study data are stored separately because they contain controlled research data. Access to the data files can be requested from the authors. After access is granted, place the downloaded folders in the structure shown below and rerun the commands in this README.

## 1. Install Packages

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-feature-pipeline.txt
```

For notebook-based visualization:

```bash
pip install notebook jupyterlab
jupyter lab
```

## 2. Data Folder Structure

Place the downloaded data in these folders:

```text
Home_Smartphone_Data/                    smartphone recordings arranged by participant and date
subjects.json                            maps SRS participant IDs to smartphone source IDs
Data_Sensors/XSENS_synchronized/         synchronized laboratory XSENS IMU files
Data_Sensors/Vicon_CSV_Files/            Vicon marker trajectory CSV files
```

Reference Vicon annotation data used to evaluate the XSENS features can be requested from the authors.

Expected smartphone layout:

```text
Home_Smartphone_Data/patient_<id>/<recording_date>/...
```

or, if the smartphone files have already been renamed by study ID:

```text
Home_Smartphone_Data/SRS01/<recording_date>/...
Home_Smartphone_Data/SRS02/<recording_date>/...
```

The extraction script accepts either layout.

## 3. Build Smartphone Feature Dataset

Run:

```bash
python scripts/build_home_features.py \
  --dataset-root Home_Smartphone_Data \
  --subjects-json subjects.json \
  --output-csv features_dataset.csv \
  --per-subject-dir feature_datasets_by_subject
```

This creates:

```text
features_dataset.csv
feature_datasets_by_subject/SRS01.csv
feature_datasets_by_subject/SRS02.csv
...
```

`features_dataset.csv` contains one row per participant recording day.

## 4. Build Laboratory Validation Files

Rebuild XSENS laboratory feature tables:

```bash
python Sensors_Visualizers/rebuild_xsens_trial_results.py \
  --root Data_Sensors/XSENS_synchronized \
  --output-dir Sensors_Visualizers/XSENS_results
```

The laboratory agreement notebook uses the XSENS result tables together with the reference Vicon annotation data provided by the authors:

```text
Sensors_Visualizers/XSENS_results/
Sensors_Visualizers/imu_vicon_features.csv
```

## 5. Run Agreement and Repeatability Analyses

Open:

```text
Sensors_Visualizers/participant_agreement.ipynb
```

This notebook compares XSENS-derived IMU features with Vicon motion-capture reference measures for the laboratory tasks.

Open:

```text
Sensors_Visualizers/rq2_home_repeatability.ipynb
```

This notebook evaluates repeatability of smartphone-derived home mobility features across repeated recordings.

## 6. Visualize Individual Recordings

Smartphone examples:

```text
walk_validation.ipynb
turn_validation.ipynb
transition_validation.ipynb
```

Laboratory XSENS examples:

```text
Sensors_Visualizers/walking_multi_sensor_visualizer.ipynb
Sensors_Visualizers/turn_multi_sensor_visualizer.ipynb
Sensors_Visualizers/sit_to_stand_multi_sensor_visualizer.ipynb
```

Vicon marker trajectory examples:

```text
Sensors_Visualizers/vicon_visualizer.ipynb
```
