"""Visualization and export helpers for final report assets.

These helpers consume saved pipeline outputs and write presentation-ready
figures and compact CSV tables. They do not fit models or change upstream
analysis results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from oulad_causal.config import FIGURES_DIR, REPORTS_DIR, TABLES_DIR
from oulad_causal.features import treatment_column_name


THRESHOLD_LABELS = {
    "median": "Median",
    "top_tertile": "Top tertile",
    "top_quartile": "Top quartile",
}


def ensure_report_dirs(
    *,
    reports_dir: Path = REPORTS_DIR,
    figures_dir: Path = FIGURES_DIR,
    tables_dir: Path = TABLES_DIR,
) -> dict[str, Path]:
    """Create standard report output directories and return their paths."""

    drafts_dir = reports_dir / "drafts"
    slides_dir = figures_dir / "slides"
    for path in (reports_dir, figures_dir, tables_dir, drafts_dir, slides_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "reports": reports_dir,
        "figures": figures_dir,
        "tables": tables_dir,
        "drafts": drafts_dir,
        "slides": slides_dir,
    }


def save_figure(fig: object, path: str | Path, *, dpi: int = 180) -> Path:
    """Save a matplotlib figure with stable report defaults."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path


def write_report_table(
    df: pd.DataFrame,
    path: str | Path,
    *,
    columns: list[str] | tuple[str, ...] | None = None,
    rename: Mapping[str, str] | None = None,
    numeric_formats: Mapping[str, str | Callable[[object], object]] | None = None,
) -> pd.DataFrame:
    """Write a curated CSV table and return the exported frame.

    ``numeric_formats`` keys may reference either original column names or
    renamed output column names. Format strings use standard Python formatting,
    for example ``"{:.3f}"``.
    """

    output = df.loc[:, list(columns)].copy() if columns is not None else df.copy()
    if rename:
        output = output.rename(columns=dict(rename))
    if numeric_formats:
        for column, formatter in numeric_formats.items():
            target = rename.get(column, column) if rename else column
            if target in output.columns:
                output[target] = output[target].map(lambda value: _format_value(value, formatter))
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


def cohort_flow_report_table(flow_table: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    """Write the final cohort-flow table."""

    table = flow_table.copy()
    table["stage"] = table["stage"].map(_clean_stage_label)
    return write_report_table(
        table,
        output_path,
        columns=("stage", "row_count", "excluded_count", "description"),
        rename={
            "stage": "Stage",
            "row_count": "Rows remaining",
            "excluded_count": "Excluded rows",
            "description": "Rule",
        },
    )


def main_effect_report_table(estimates: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    """Write the final main-effect estimate table."""

    table = estimates.copy()
    table["preferred"] = table["preferred"].map(lambda value: "Yes" if bool(value) else "No")
    table["estimator"] = table["estimator"].map(_clean_estimator_label)
    return write_report_table(
        table,
        output_path,
        columns=(
            "estimator",
            "preferred",
            "status",
            "estimate",
            "ci_lower",
            "ci_upper",
            "std_error",
            "treated_mean",
            "control_mean",
            "n",
            "notes",
        ),
        rename={
            "estimator": "Estimator",
            "preferred": "Preferred",
            "status": "Status",
            "estimate": "Risk difference",
            "ci_lower": "95% CI lower",
            "ci_upper": "95% CI upper",
            "std_error": "SE",
            "treated_mean": "Estimated treated mean",
            "control_mean": "Estimated control mean",
            "n": "Analysis rows",
            "notes": "Notes",
        },
        numeric_formats={
            "estimate": "{:.6f}",
            "ci_lower": "{:.6f}",
            "ci_upper": "{:.6f}",
            "std_error": "{:.6f}",
            "treated_mean": "{:.6f}",
            "control_mean": "{:.6f}",
            "n": "{:.0f}",
        },
    )


def robustness_report_table(window_summary: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    """Write the final compact robustness table."""

    table = window_summary.copy()
    table["threshold_name"] = table["threshold_name"].map(lambda value: THRESHOLD_LABELS.get(value, value))
    table["poor_overlap"] = table["poor_overlap"].map(_bool_label)
    return write_report_table(
        table,
        output_path,
        columns=(
            "window_days",
            "threshold_name",
            "scenario_status",
            "estimate",
            "n",
            "treated_count",
            "control_count",
            "poor_overlap",
        ),
        rename={
            "window_days": "Window days",
            "threshold_name": "Threshold",
            "scenario_status": "Status",
            "estimate": "AIPW risk difference",
            "n": "Analysis rows",
            "treated_count": "Treated rows",
            "control_count": "Control rows",
            "poor_overlap": "Overlap flag",
        },
        numeric_formats={
            "estimate": "{:.6f}",
            "n": "{:.0f}",
            "treated_count": "{:.0f}",
            "control_count": "{:.0f}",
        },
    )


def write_treatment_prevalence_figure(
    cohort: pd.DataFrame,
    output_path: str | Path,
    *,
    window_days: int = 14,
) -> Path:
    """Write a polished treatment-prevalence bar chart for one window."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rows = []
    for threshold in ("median", "top_tertile", "top_quartile"):
        column = treatment_column_name(window_days, threshold)
        if column not in cohort.columns:
            raise KeyError(f"Missing treatment column for prevalence figure: {column}")
        values = cohort[column].dropna()
        rows.append(
            {
                "threshold": THRESHOLD_LABELS[threshold],
                "prevalence": float(values.mean()) if not values.empty else np.nan,
                "n": int(values.shape[0]),
            }
        )
    plot_data = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(plot_data["threshold"], plot_data["prevalence"], color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share classified as high engagement")
    ax.set_xlabel("")
    ax.set_title(f"Treatment Prevalence, First {window_days} Days")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, prevalence in zip(bars, plot_data["prevalence"]):
        if pd.notna(prevalence):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                prevalence + 0.025,
                f"{prevalence:.1%}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    fig.tight_layout()
    saved = save_figure(fig, output_path)
    plt.close(fig)
    return saved


def write_discovery_comparison_figure(
    comparison: pd.DataFrame,
    stability_edges: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Write a compact comparison of discovery output and the hand-built DAG."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    methods = ["pc", "fci", "ges"]
    discovered = comparison[comparison["method"].isin(methods)].copy()
    rows = []
    for method in methods:
        method_edges = discovered[discovered["method"].eq(method)]
        rows.append(
            {
                "method": method.upper(),
                "in_hand": int(method_edges["in_hand_skeleton"].fillna(False).astype(bool).sum()),
                "other": int((~method_edges["in_hand_skeleton"].fillna(False).astype(bool)).sum()),
            }
        )
    plot_data = pd.DataFrame(rows)
    missing_count = int(comparison["method"].eq("hand_dag_missing_from_discovery").sum())
    stable_count = int((stability_edges.get("edge_frequency", pd.Series(dtype=float)) >= 0.70).sum())

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), gridspec_kw={"width_ratios": [2.3, 1]})
    x = np.arange(len(plot_data))
    axes[0].bar(x, plot_data["in_hand"], label="Matches hand-DAG skeleton", color="#4C78A8")
    axes[0].bar(
        x,
        plot_data["other"],
        bottom=plot_data["in_hand"],
        label="Other discovered skeleton edge",
        color="#F58518",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(plot_data["method"])
    axes[0].set_ylabel("Edge count")
    axes[0].set_title("Discovery Graph Comparison")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axes[0].set_axisbelow(True)

    summary_labels = ["Hand-DAG edges\nnot recovered", "Stable repeated\nsubsample edges"]
    summary_values = [missing_count, stable_count]
    axes[1].bar(summary_labels, summary_values, color=["#B279A2", "#54A24B"])
    axes[1].set_title("Summary Counts")
    axes[1].set_ylabel("Count")
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axes[1].set_axisbelow(True)
    for idx, value in enumerate(summary_values):
        axes[1].text(idx, value + 0.4, str(value), ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    saved = save_figure(fig, output_path)
    plt.close(fig)
    return saved


def write_subgroup_summary_figure(summary: pd.DataFrame, output_path: str | Path) -> Path:
    """Write a slide-friendly subgroup summary figure."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    subgroup = summary[
        summary["section"].eq("subgroup") & summary["scenario_status"].eq("success")
    ].copy()
    subgroup = subgroup.sort_values(["subgroup_variable", "estimate"])

    fig_height = max(4.2, 0.34 * len(subgroup) + 1.4)
    fig, ax = plt.subplots(figsize=(8.2, fig_height))
    if subgroup.empty:
        ax.text(0.5, 0.5, "No subgroup estimates passed adequacy gates", ha="center", va="center")
        ax.axis("off")
    else:
        labels = subgroup["subgroup_variable"].map(_clean_subgroup_label) + ": " + subgroup[
            "subgroup_level"
        ].astype(str)
        y = np.arange(len(subgroup))
        ax.scatter(subgroup["estimate"], y, color="#4C78A8", s=42)
        ax.axvline(0.0, color="#4B5563", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("AIPW risk difference")
        ax.set_title("Subgroup Estimates")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    saved = save_figure(fig, output_path)
    plt.close(fig)
    return saved


def _format_value(value: object, formatter: str | Callable[[object], object]) -> object:
    if pd.isna(value):
        return ""
    if callable(formatter):
        return formatter(value)
    return formatter.format(value)


def _clean_stage_label(value: object) -> str:
    labels = {
        "student_info_records": "Student info records",
        "after_required_table_joins": "After required table joins",
        "after_primary_treatment_eligibility": "After primary treatment eligibility",
    }
    return labels.get(str(value), str(value).replace("_", " ").title())


def _clean_estimator_label(value: object) -> str:
    labels = {
        "regression_adjustment": "Regression adjustment",
        "stabilized_iptw": "Stabilized IPTW",
        "aipw": "AIPW",
        "nearest_neighbor_matching": "Nearest-neighbor matching",
    }
    return labels.get(str(value), str(value).replace("_", " ").title())


def _clean_subgroup_label(value: object) -> str:
    labels = {
        "highest_education": "Highest education",
        "prior_attempts": "Prior attempts",
        "disability": "Disability",
    }
    return labels.get(str(value), str(value).replace("_", " ").title())


def _bool_label(value: object) -> str:
    if pd.isna(value):
        return ""
    return "Flagged" if bool(value) else "Not flagged"
