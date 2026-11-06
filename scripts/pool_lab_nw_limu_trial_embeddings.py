#!/usr/bin/env python3
"""Mean-pool LIMU window embeddings within each participant/visit/NW trial."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "Data/embeddings/lab_nw"
OUTPUT_ROOT = INPUT_ROOT
GROUP_COLUMNS = ["participant_id", "visit", "trial"]
META_COLUMNS = ["source_file", "recording_file"]


def pool_file(version: str) -> None:
    source = INPUT_ROOT / f"limu_{version}.csv"
    frame = pd.read_csv(source)
    embedding_columns = [column for column in frame.columns if column.startswith("embedding_")]
    if len(embedding_columns) != 72:
        raise ValueError(f"Expected 72 LIMU dimensions in {source}, found {len(embedding_columns)}")

    records = []
    for key, group in frame.groupby(GROUP_COLUMNS, sort=True, dropna=False):
        record = dict(zip(GROUP_COLUMNS, key))
        record["n_window_embeddings"] = len(group)
        record["n_source_files"] = group["source_file"].nunique()
        record["source_files"] = ";".join(sorted(group["source_file"].unique()))
        record["recording_files"] = ";".join(sorted(group["recording_file"].unique()))
        record.update(group[embedding_columns].mean(axis=0).to_dict())
        records.append(record)

    result = pd.DataFrame(records)
    output = OUTPUT_ROOT / f"limu_{version}_trial_mean.csv"
    result.to_csv(output, index=False)
    print(
        f"{version}: saved {len(result)} participant/visit/NW-trial vectors "
        f"({len(embedding_columns)} dimensions each) to {output}"
    )


def main() -> None:
    for version in ("generic", "stroke_adapted"):
        pool_file(version)


if __name__ == "__main__":
    main()
