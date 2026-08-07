from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
VICON_PATH = HERE / "Vicon - Annotations_long.csv"
VICON_RAW_PATH = HERE / "Vicon - Annotations_long.csv"
VICON_REVIEWED_PATH = HERE / "Vicon - Annotations.csv"
XSENS_RESULT_DIR = HERE / "XSENS_results"
VISIT_FILTER = {"V01"}


def keep_srs_participant(subject_id):
    subject = str(subject_id)
    excluded = ("NonStroke", "Old", "Version 1", "Original")
    return subject.startswith("SRS") and not any(label in subject for label in excluded)


def icc_2_1(xsens, vicon):
    """Two-way random-effects, single-measure, absolute-agreement ICC."""
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
        "icc_2_1": icc_2_1(x, y),
    }


def _reviewed_vicon_to_long(vicon: pd.DataFrame) -> pd.DataFrame:
    mappings = [
        (
            "walk",
            {
                "start_time_s": "walk_10_step_start_s",
                "end_time_s": "walk_10_step_end_s",
                "duration_s": "walk_10_step_duration_s",
                "step_count": "walk_10_step_count",
                "cadence_steps_min": "walk_10_step_cadence_steps_min",
                "mean_step_time_s": "walk_10_step_mean_step_time_s",
            },
        ),
        (
            "left_turn",
            {
                "start_time_s": "left_turn_start_s",
                "end_time_s": "left_turn_end_s",
                "duration_s": "left_turn_duration_s",
                "step_count": "left_turn_step_count",
                "time_to_peak_s": "left_turn_time_to_peak_s",
            },
        ),
        (
            "right_turn",
            {
                "start_time_s": "right_turn_start_s",
                "end_time_s": "right_turn_end_s",
                "duration_s": "right_turn_duration_s",
                "step_count": "right_turn_step_count",
                "time_to_peak_s": "right_turn_time_to_peak_s",
            },
        ),
        (
            "sit_to_stand",
            {
                "start_time_s": "sts_start_s",
                "end_time_s": "sts_end_s",
                "duration_s": "sts_duration_s",
                "time_to_peak_s": "sts_time_to_peak_s",
            },
        ),
    ]
    rows = []
    for _, source in vicon.iterrows():
        for activity, columns in mappings:
            row = {
                "subject_id": source["subject_id"],
                "visit_id": source["visit_id"],
                "activity": activity,
                "trial_id": 1,
            }
            for out_col in [
                "start_time_s",
                "end_time_s",
                "duration_s",
                "step_count",
                "cadence_steps_min",
                "mean_step_time_s",
                "time_to_peak_s",
            ]:
                row[out_col] = source.get(columns.get(out_col, ""), np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def load_vicon(path: Path | None = None):
    vicon = pd.read_csv(path or VICON_PATH)
    if "walk_start_s" in vicon.columns:
        vicon = _reviewed_vicon_to_long(vicon)
    vicon = vicon[vicon["subject_id"].map(keep_srs_participant)].copy()
    numeric_cols = [
        "start_time_s",
        "end_time_s",
        "duration_s",
        "step_count",
        "cadence_steps_min",
        "mean_step_time_s",
        "ten_step_start_time_s",
        "ten_step_end_time_s",
        "ten_step_duration_s",
        "ten_step_count",
        "ten_step_cadence_steps_min",
        "ten_step_mean_step_time_s",
        "time_to_peak_s",
    ]
    for col in numeric_cols:
        if col in vicon:
            vicon[col] = pd.to_numeric(vicon[col], errors="coerce")
    return vicon


@dataclass(frozen=True)
class AgreementSpec:
    title: str
    result_file: str
    xsens_col: str
    vicon_col: str
    unit: str
    activity: str
    transform: str | None = None
    xsens_slope: float = 1.0
    xsens_intercept: float = 0.0
    round_xsens: bool = False
    vicon_path: Path | None = None
    aggregation: str = "mean"


SPECS = [
    AgreementSpec(
        "Walking 10-Step Duration",
        "walking_results.csv",
        "sacrum_duration_s",
        "duration_s",
        "s",
        "walk",
        vicon_path=VICON_RAW_PATH,
        aggregation="first",
    ),
    AgreementSpec(
        "Walking 10-Step Cadence",
        "walking_results.csv",
        "sacrum_cadence_steps_min",
        "cadence_steps_min",
        "steps/min",
        "walk",
        vicon_path=VICON_RAW_PATH,
        aggregation="first",
    ),
    AgreementSpec(
        "Walking 10-Step Mean Step Time",
        "walking_results.csv",
        "sacrum_mean_step_time_s",
        "mean_step_time_s",
        "s",
        "walk",
        vicon_path=VICON_RAW_PATH,
        aggregation="first",
    ),
    AgreementSpec(
        "Left Turn Duration",
        "left_turn_results.csv",
        "sacrum_duration_s",
        "duration_s",
        "s",
        "left_turn",
        vicon_path=VICON_REVIEWED_PATH,
        aggregation="first",
    ),
    AgreementSpec(
        "Left Turn Step Count",
        "left_turn_results.csv",
        "sacrum_step_count",
        "step_count",
        "steps",
        "left_turn",
        vicon_path=VICON_REVIEWED_PATH,
        aggregation="first",
    ),
    AgreementSpec(
        "Right Turn Duration",
        "right_turn_results.csv",
        "sacrum_duration_s",
        "duration_s",
        "s",
        "right_turn",
        vicon_path=VICON_REVIEWED_PATH,
        aggregation="first",
    ),
    AgreementSpec(
        "Right Turn Step Count",
        "right_turn_results.csv",
        "sacrum_step_count",
        "step_count",
        "steps",
        "right_turn",
        vicon_path=VICON_REVIEWED_PATH,
        aggregation="first",
    ),
    AgreementSpec(
        "Sit-to-Stand Duration",
        "sit_to_stand_results.csv",
        "sacrum_duration_s",
        "duration_s",
        "s",
        "sit_to_stand",
        vicon_path=VICON_REVIEWED_PATH,
        aggregation="first",
    ),
]


def participant_agreement_table(spec: AgreementSpec):
    vicon = load_vicon(spec.vicon_path)
    vicon = vicon[vicon["activity"].eq(spec.activity)].copy()
    xsens = pd.read_csv(XSENS_RESULT_DIR / spec.result_file)
    xsens = xsens[xsens["subject_id"].map(keep_srs_participant)].copy()
    if VISIT_FILTER:
        vicon = vicon[vicon["visit_id"].isin(VISIT_FILTER)].copy()
        xsens = xsens[xsens["visit_id"].isin(VISIT_FILTER)].copy()
    vicon_col = spec.vicon_col
    xsens_col = spec.xsens_col
    vicon[vicon_col] = pd.to_numeric(vicon[vicon_col], errors="coerce")
    xsens[xsens_col] = pd.to_numeric(xsens[xsens_col], errors="coerce")
    keys = ["subject_id", "visit_id", "activity"]
    if "trial_id" in vicon and "trial_id" in xsens:
        keys.append("trial_id")
    if spec.aggregation == "first":
        vicon = vicon.sort_values("trial_id").groupby(["subject_id", "visit_id", "activity"], as_index=False).first()
        xsens = xsens.sort_values("trial_id").groupby(["subject_id", "visit_id", "activity"], as_index=False).first()
        keys = ["subject_id", "visit_id", "activity"]

    merged = vicon[keys + [vicon_col]].merge(
        xsens[keys + [xsens_col]],
        on=keys,
        how="inner",
        suffixes=("_vicon", "_xsens"),
    )
    if spec.aggregation != "first":
        vicon_named = vicon_col if vicon_col in merged else f"{vicon_col}_vicon"
        xsens_named = xsens_col if xsens_col in merged else f"{xsens_col}_xsens"
        merged = (
            merged.groupby(["subject_id", "visit_id", "activity"], as_index=False)[[vicon_named, xsens_named]]
            .mean()
        )
    vicon_col = spec.vicon_col if spec.vicon_col in merged else f"{spec.vicon_col}_vicon"
    xsens_col = spec.xsens_col if spec.xsens_col in merged else f"{spec.xsens_col}_xsens"
    table = merged[["subject_id", "visit_id", "activity", vicon_col, xsens_col]].rename(
        columns={vicon_col: "vicon", xsens_col: "xsens"}
    )
    table = table.dropna(subset=["vicon", "xsens"])
    table["xsens_raw"] = table["xsens"]
    table["xsens"] = table["xsens"] * spec.xsens_slope + spec.xsens_intercept
    if spec.round_xsens:
        table["xsens"] = table["xsens"].round().clip(lower=0)
    else:
        table["xsens"] = table["xsens"].clip(lower=0)
    table["error"] = table["xsens"] - table["vicon"]
    table["abs_error"] = table["error"].abs()
    table["percent_error"] = table["abs_error"] / table["vicon"].abs().replace(0, np.nan) * 100.0
    return table.sort_values(["subject_id", "visit_id"]).reset_index(drop=True)


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
    summary = pd.DataFrame(rows)
    for col in ["mae", "rmse", "bias", "pearson_r", "icc_2_1"]:
        if col in summary:
            summary[col] = summary[col].round(3)
    return summary


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
        f"ICC(2,1)={metrics['icc_2_1']:.3f}\nMAE={metrics['mae']:.3f}\nr={metrics['pearson_r']:.3f}",
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
        ax.annotate(
            row["subject_id"],
            ((row["xsens"] + row["vicon"]) / 2.0, row["error"]),
            fontsize=8,
            alpha=0.75,
        )
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
