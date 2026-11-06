#!/usr/bin/env python3
"""Mean-pool joint six-channel HART window embeddings within each NW trial."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "Data/embeddings/lab_nw"
OUTPUT_ROOT = INPUT_ROOT
GROUP_COLUMNS = ["participant_id", "visit", "trial"]


def pool_file(version: str) -> None:
    source = INPUT_ROOT / f"hart_{version}.csv"
    frame = pd.read_csv(source)
    embedding_columns = [column for column in frame.columns if column.startswith("embedding_")]
    if len(embedding_columns) != 288:
        raise ValueError(f"Expected 288 joint HART dimensions in {source}, found {len(embedding_columns)}")

    records = []
    for key, group in frame.groupby(GROUP_COLUMNS, sort=True, dropna=False):
        record = dict(zip(GROUP_COLUMNS, key))
        record["n_temporal_windows"] = len(group)
        record["n_source_files"] = group["source_file"].nunique()
        record["source_files"] = ";".join(sorted(group["source_file"].unique()))
        record.update(group[embedding_columns].mean(axis=0).to_dict())
        records.append(record)

    result = pd.DataFrame(records)
    output = OUTPUT_ROOT / f"hart_{version}_trial_mean.csv"
    result.to_csv(output, index=False)
    print(
        f"{version}: saved {len(result)} participant/visit/NW-trial vectors "
        f"(288 dimensions) to {output}"
    )


def main() -> None:
    for version in ("generic", "stroke_adapted"):
        pool_file(version)


if __name__ == "__main__":
    main()
