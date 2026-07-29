from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
VICON_PATH = HERE / "Vicon - Annotations.csv"


def visit_label(visit_id):
    digits = "".join(ch for ch in str(visit_id) if ch.isdigit())
    return f"Visit{int(digits)}" if digits else str(visit_id)


def keep_srs_participant(subject_id):
    subject = str(subject_id)
    excluded = ("NonStroke", "Old", "Version 1", "Original")
    return subject.startswith("SRS") and not any(label in subject for label in excluded)


def icc_a1(xsens, vicon):
    x = pd.to_numeric(xsens, errors="coerce")
    y = pd.to_numeric(vicon, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 2:
        return np.nan
    values = np.column_stack([x[mask].to_numpy(dtype=float), y[mask].to_numpy(dtype=float)])
    n, k = values.shape
    grand = float(np.mean(values))
    row_means = np.mean(values, axis=1)
    col_means = np.mean(values, axis=0)
    ss_rows = k * float(np.sum((row_means - grand) ** 2))
    ss_cols = n * float(np.sum((col_means - grand) ** 2))
    ss_total = float(np.sum((values - grand) ** 2))
    ss_err = ss_total - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err + (k * (ms_cols - ms_err) / n)
    return float((ms_rows - ms_err) / denom) if denom else np.nan


def agreement_metrics(df, xsens_col="xsens", vicon_col="vicon"):
    x = pd.to_numeric(df[xsens_col], errors="coerce")
    y = pd.to_numeric(df[vicon_col], errors="coerce")
    mask = x.notna() & y.notna()
    err = x[mask] - y[mask]
    return {
        "n": int(mask.sum()),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(err.mean()),
        "pearson_r": float(np.corrcoef(x[mask], y[mask])[0, 1])
        if mask.sum() > 1 and x[mask].std() > 0 and y[mask].std() > 0
        else np.nan,
        "icc_a1": icc_a1(x, y),
    }


def load_vicon():
    vicon = pd.read_csv(VICON_PATH)
    vicon["visit"] = vicon["visit_id"].map(visit_label)
    return vicon[vicon["subject_id"].map(keep_srs_participant)].copy()


@dataclass(frozen=True)
class AgreementSpec:
    title: str
    result_file: str
    xsens_col: str
    vicon_col: str
    unit: str
    activity: str
    transform: str | None = None


SPECS = [
    AgreementSpec(
        "Walking Step Count",
        "treadmill_results.csv",
        "sacrum_qc_step_count",
        "walk_step_count",
        "steps",
        "walking",
    ),
    AgreementSpec(
        "Walking Cadence",
        "treadmill_results.csv",
        "sacrum_qc_cadence",
        "walk_cadence_steps_min",
        "steps/min",
        "walking",
    ),
    AgreementSpec(
        "Left Turn Duration",
        "turn_left_results.csv",
        "sacrum_turn_duration",
        "left_turn_duration_s",
        "s",
        "left_turn",
    ),
    AgreementSpec(
        "Left Turn Step Count",
        "turn_left_results.csv",
        "sacrum_turn_step_count",
        "left_turn_step_count",
        "steps",
        "left_turn",
    ),
    AgreementSpec(
        "Right Turn Duration",
        "turn_right_results.csv",
        "sacrum_turn_duration",
        "right_turn_duration_s",
        "s",
        "right_turn",
    ),
    AgreementSpec(
        "Right Turn Step Count",
        "turn_right_results.csv",
        "sacrum_turn_step_count",
        "right_turn_step_count",
        "steps",
        "right_turn",
    ),
    AgreementSpec(
        "Sit-to-Stand Duration",
        "sit_to_stand_results.csv",
        "sacrum_duration",
        "sts_duration_s",
        "s",
        "sit_to_stand",
    ),
]


def participant_agreement_table(spec: AgreementSpec):
    vicon = load_vicon()
    xsens = pd.read_csv(HERE / spec.result_file)
    xsens = xsens[xsens["subject_id"].map(keep_srs_participant)].copy()
    merged = vicon.merge(xsens, on=["subject_id", "visit"], how="inner", suffixes=("_vicon", "_xsens"))
    vicon_col = spec.vicon_col if spec.vicon_col in merged else f"{spec.vicon_col}_vicon"
    xsens_col = spec.xsens_col if spec.xsens_col in merged else f"{spec.xsens_col}_xsens"
    table = merged[["subject_id", "visit", vicon_col, xsens_col]].rename(
        columns={vicon_col: "vicon", xsens_col: "xsens"}
    )
    table["visit_order"] = table["visit"].astype(str).str.extract(r"(\d+)").astype(float)
    table = table.sort_values(["subject_id", "visit_order"]).groupby("subject_id", as_index=False).first()
    table = table.drop(columns=["visit_order"])
    table = table.dropna(subset=["vicon", "xsens"])
    table["error"] = table["xsens"] - table["vicon"]
    table["abs_error"] = table["error"].abs()
    table["percent_error"] = table["abs_error"] / table["vicon"].abs().replace(0, np.nan) * 100.0
    return table.sort_values("abs_error").reset_index(drop=True)


def all_metric_summary(specs=SPECS):
    rows = []
    for spec in specs:
        table = participant_agreement_table(spec)
        row = {
            "activity": spec.activity,
            "feature": spec.title,
            "xsens_column": spec.xsens_col,
            "vicon_column": spec.vicon_col,
            "unit": spec.unit,
        }
        row.update(agreement_metrics(table))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_scatter_and_bland_altman(table, title, unit):
    metrics = agreement_metrics(table)
    x = table["vicon"].to_numpy(dtype=float)
    y = table["xsens"].to_numpy(dtype=float)
    err = y - x
    avg = (x + y) / 2.0
    bias = float(np.nanmean(err))
    sd = float(np.nanstd(err, ddof=1))
    loa_low = bias - 1.96 * sd
    loa_high = bias + 1.96 * sd

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.scatter(x, y, s=55, alpha=0.85)
    lo = float(np.nanmin([x.min(), y.min()]))
    hi = float(np.nanmax([x.max(), y.max()]))
    pad = (hi - lo) * 0.08 if hi > lo else 1.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="gray", linewidth=1)
    ax.set_title(f"{title}: XSENS vs Vicon")
    ax.set_xlabel(f"Vicon ({unit})")
    ax.set_ylabel(f"XSENS ({unit})")
    ax.grid(True, alpha=0.3)
    for _, row in table.iterrows():
        ax.annotate(row["subject_id"], (row["vicon"], row["xsens"]), fontsize=8, alpha=0.75)
    ax.text(
        0.02,
        0.98,
        f"ICC(A,1)={metrics['icc_a1']:.3f}\nMAE={metrics['mae']:.3f}\nr={metrics['pearson_r']:.3f}",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    ax = axes[1]
    ax.scatter(avg, err, s=55, alpha=0.85)
    ax.axhline(bias, color="tab:red", linewidth=2, label=f"bias={bias:.3f}")
    ax.axhline(loa_low, color="gray", linestyle="--", label=f"-1.96 SD={loa_low:.3f}")
    ax.axhline(loa_high, color="gray", linestyle="--", label=f"+1.96 SD={loa_high:.3f}")
    ax.set_title(f"{title}: Bland-Altman")
    ax.set_xlabel(f"Mean of XSENS and Vicon ({unit})")
    ax.set_ylabel(f"XSENS - Vicon ({unit})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    for _, row in table.iterrows():
        ax.annotate(row["subject_id"], ((row["xsens"] + row["vicon"]) / 2.0, row["error"]), fontsize=8, alpha=0.75)
    plt.tight_layout()
    return fig


def display_feature_report(spec: AgreementSpec, display_func=None):
    if display_func is None:
        from IPython.display import display as display_func

    table = participant_agreement_table(spec)
    metric_row = pd.DataFrame([agreement_metrics(table)])
    display_func(metric_row)
    display_func(table)
    plot_scatter_and_bland_altman(table, spec.title, spec.unit)
    plt.show()
