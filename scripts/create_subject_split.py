#!/usr/bin/env python3
"""Create a reproducible participant-level public-stroke train/validation split.

The split is stratified by the archive cohort folder, keeps all windows from a
participant together, and applies the three ID corrections in the released VAE
preprocessing code (Functions/Proces_data.py). It does not create a test set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# Matches the participant ID corrections in the downloaded VAE preprocessing code.
ID_ALIASES = {
    "S5223P": "S6669P",
    "S2706H": "S9317H",
    "S5649H": "S6567H",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-root", type=Path, default=Path("Data/windowed/public_stroke"))
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.train_fraction < 1:
        parser.error("--train-fraction must be between 0 and 1")

    limu_manifest_path = args.window_root / "LIMU-BERT-X/manifest.csv"
    participants = pd.read_csv(limu_manifest_path)[["subject_id", "cohort"]].drop_duplicates()
    participants["canonical_subject_id"] = participants["subject_id"].replace(ID_ALIASES)

    # A canonical participant must belong to one source cohort only.
    cohort_counts = participants.groupby("canonical_subject_id")["cohort"].nunique()
    if (cohort_counts > 1).any():
        raise ValueError("A canonical participant ID appears in multiple cohorts")

    canonical_participants = participants[["canonical_subject_id", "cohort"]].drop_duplicates()
    if canonical_participants["canonical_subject_id"].duplicated().any():
        raise ValueError("Participant aliases produced duplicate cohort assignments")

    assignments = []
    for cohort_index, (cohort, group) in enumerate(canonical_participants.groupby("cohort", sort=True)):
        ids = sorted(group["canonical_subject_id"].tolist())
        rng = pd.Series(ids).sample(frac=1, random_state=args.seed + cohort_index).tolist()
        n_train = round(len(ids) * args.train_fraction)
        train_ids = set(rng[:n_train])
        for canonical_id in ids:
            assignments.append({
                "canonical_subject_id": canonical_id,
                "cohort": cohort,
                "split": "train" if canonical_id in train_ids else "validation",
            })

    split_by_canonical = pd.DataFrame(assignments)
    participant_split = participants.merge(
        split_by_canonical, on=["canonical_subject_id", "cohort"], validate="many_to_one"
    )
    out = args.window_root
    out.mkdir(parents=True, exist_ok=True)
    participant_split.sort_values(["cohort", "canonical_subject_id", "subject_id"]).to_csv(
        out / "participant_split.csv", index=False
    )

    for model in ("LIMU-BERT-X", "HART"):
        manifest_path = out / model / "manifest.csv"
        manifest = pd.read_csv(manifest_path)
        manifest = manifest.drop(columns=["canonical_subject_id", "split"], errors="ignore")
        manifest = manifest.merge(
            participant_split[["subject_id", "cohort", "canonical_subject_id", "split"]],
            on=["subject_id", "cohort"],
            how="left",
            validate="many_to_one",
        )
        if manifest[["canonical_subject_id", "split"]].isna().any().any():
            raise ValueError(f"Some {model} windows did not receive a participant split")
        manifest.to_csv(manifest_path, index=False)

    print(f"Participants: {len(canonical_participants)} canonical IDs from {len(participants)} filename IDs")
    print("Split by cohort:")
    print(split_by_canonical.groupby(["cohort", "split"]).size().to_string())
    print("Saved participant_split.csv and added split columns to both window manifests.")


if __name__ == "__main__":
    main()
