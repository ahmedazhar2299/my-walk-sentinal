from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULT_DIR = HERE / "XSENS_results"
MODEL_PATH = HERE / "xsens_ensemble_models.csv"

ACTIVITY_FILES = {
    "walk": RESULT_DIR / "walking_results.csv",
    "left_turn": RESULT_DIR / "left_turn_results.csv",
    "right_turn": RESULT_DIR / "right_turn_results.csv",
    "sit_to_stand": RESULT_DIR / "sit_to_stand_results.csv",
}


def parse_vector(value):
    if pd.isna(value) or str(value).strip() == "":
        return np.array([], dtype=float)
    return np.array([float(x) for x in str(value).split("|")], dtype=float)


def apply_model(df, model):
    cols = str(model["cols"]).split("|")
    coef = parse_vector(model["coef"])
    mu = parse_vector(model["mu"])
    sd = parse_vector(model["sd"])
    if len(coef) != len(cols) + 1 or len(mu) != len(cols) or len(sd) != len(cols):
        raise ValueError(f"Bad model vector lengths for {model['model_col']}")
    X = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    # Fill missing sensor values with training means. This avoids dropping a trial because one sensor is absent.
    inds = np.where(~np.isfinite(X))
    if len(inds[0]):
        X[inds] = np.take(mu, inds[1])
    sd_safe = np.where(sd == 0, 1.0, sd)
    Z = (X - mu) / sd_safe
    pred = coef[0] + Z @ coef[1:]
    pred = np.clip(pred, 0, None)
    if "step_count" in str(model["model_col"]):
        pred = np.round(pred)
    return pred


def main():
    models = pd.read_csv(MODEL_PATH)
    combined_rows = []
    for activity, path in ACTIVITY_FILES.items():
        df = pd.read_csv(path)
        df = df.drop(columns=["visit"], errors="ignore")
        for _, model in models[models["activity"].eq(activity)].iterrows():
            df[model["model_col"]] = apply_model(df, model)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].round(3)
        df.to_csv(path, index=False, float_format="%.3f")
        combined_rows.append(df)
        print(f"Updated {path} ({len(df)} rows)")

    combined = pd.concat(combined_rows, ignore_index=True)
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    combined[numeric_cols] = combined[numeric_cols].round(3)
    out = RESULT_DIR / "xsens_results_long.csv"
    combined.to_csv(out, index=False, float_format="%.3f")
    print(f"Updated {out} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
