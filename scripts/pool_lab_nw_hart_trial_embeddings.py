#!/usr/bin/env python3
"""Mean-pool HART embeddings within each participant/visit/NW trial.

HART processes each sensor axis independently. This script pools windows
separately for each axis, then concatenates the six 48-value axis summaries
into one 288-value vector per participant, visit, and NW trial.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "Data/embeddings/lab_nw"
OUTPUT_ROOT = INPUT_ROOT
GROUP_COLUMNS = ["participant_id", "visit", "trial"]
CHANNELS = ["Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"]


def pool_file(version: str) -> None:
    source = INPUT_ROOT / f"hart_{version}.csv"
    frame = pd.read_csv(source)
    embedding_columns = [column for column in frame.columns if column.startswith("embedding_")]
    if len(embedding_columns) != 48:
        raise ValueError(f"Expected 48 HART dimensions in {source}, found {len(embedding_columns)}")
    if set(frame["channel"].unique()) != set(CHANNELS):
        raise ValueError(f"Unexpected HART channel set in {source}: {sorted(frame['channel'].unique())}")

    records = []
    for key, group in frame.groupby(GROUP_COLUMNS, sort=True, dropna=False):
        record = dict(zip(GROUP_COLUMNS, key))
        record["n_temporal_windows"] = group[["source_file", "window_start_s"]].drop_duplicates().shape[0]
        record["n_axis_window_embeddings"] = len(group)
        record["n_source_files"] = group["source_file"].nunique()
        record["source_files"] = ";".join(sorted(group["source_file"].unique()))
        for channel in CHANNELS:
            channel_rows = group.loc[group["channel"].eq(channel)]
            means = channel_rows[embedding_columns].mean(axis=0)
            record.update({f"{channel}_{name}": value for name, value in means.items()})
        records.append(record)

    result = pd.DataFrame(records)
    output = OUTPUT_ROOT / f"hart_{version}_trial_mean.csv"
    result.to_csv(output, index=False)
    print(
        f"{version}: saved {len(result)} participant/visit/NW-trial vectors "
        f"(6 axes x 48 dimensions = 288 dimensions) to {output}"
    )


def main() -> None:
    for version in ("generic", "stroke_adapted"):
        pool_file(version)


if __name__ == "__main__":
    main()
