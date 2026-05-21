"""Robustness, environment, subgroup, placebo, and sensitivity analyses.

This module widens the primary OULAD early-engagement analysis without turning
it into an uncontrolled fishing grid. It reuses the primary estimation
machinery, saves compact report-ready summaries, and keeps detailed estimator
rows for audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from oulad_causal.config import DOCS_DIR, FIGURES_DIR, PROCESSED_DATA_DIR, TABLES_DIR
from oulad_causal.dag import ANALYTIC_COHORT_PATH, PRIMARY_OUTCOME_COLUMN, recommended_baseline_adjustment_set
from oulad_causal.estimation import EstimationConfig, estimate_effects
from oulad_causal.features import DEFAULT_THRESHOLDS, DEFAULT_WINDOWS, treatment_available_column, treatment_column_name


ROBUSTNESS_LONG_PATH = PROCESSED_DATA_DIR / "robustness_estimates_long.csv"
ROBUSTNESS_METADATA_PATH = PROCESSED_DATA_DIR / "robustness_run_metadata.json"
WINDOW_THRESHOLD_TABLE_PATH = TABLES_DIR / "robustness_window_threshold_summary.csv"
ENVIRONMENT_TABLE_PATH = TABLES_DIR / "robustness_environment_summary.csv"
SUBGROUP_PLACEBO_SENSITIVITY_TABLE_PATH = TABLES_DIR / "robustness_subgroup_placebo_sensitivity_summary.csv"
WINDOW_THRESHOLD_FIGURE_PATH = FIGURES_DIR / "robustness_window_threshold_heatmap.png"
MODULE_FIGURE_PATH = FIGURES_DIR / "robustness_module_presentation_estimates.png"
SUBGROUP_FIGURE_PATH = FIGURES_DIR / "robustness_subgroup_estimates.png"
PLACEBO_SENSITIVITY_FIGURE_PATH = FIGURES_DIR / "robustness_placebo_sensitivity.png"
ROBUSTNESS_SUMMARY_PATH = DOCS_DIR / "robustness_summary.md"

PRIMARY_WINDOW = 14
PRIMARY_THRESHOLD = "median"


@dataclass(frozen=True)
class RobustnessConfig:
    """Configuration for deterministic robustness checks."""

    cohort_path: Path = ANALYTIC_COHORT_PATH
    processed_dir: Path = PROCESSED_DATA_DIR
    tables_dir: Path = TABLES_DIR
    figures_dir: Path = FIGURES_DIR
    docs_dir: Path = DOCS_DIR
    windows: tuple[int, ...] = DEFAULT_WINDOWS
    thresholds: tuple[str, ...] = tuple(DEFAULT_THRESHOLDS.keys())
    primary_window: int = PRIMARY_WINDOW
    primary_threshold: str = PRIMARY_THRESHOLD
    outcome_column: str = PRIMARY_OUTCOME_COLUMN
    seed: int = 245
    module_min_n: int = 300
    module_min_treated: int = 50
    module_min_control: int = 50
    subgroup_min_n: int = 500
    subgroup_min_treated: int = 100
    subgroup_min_control: int = 100
    sensitivity_prevalence_differences: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)
    sensitivity_outcome_risk_differences: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40)


@dataclass
class RobustnessResult:
    """In-memory robustness outputs before writing artifacts."""

    window_threshold_summary: pd.DataFrame
    environment_summary: pd.DataFrame
    subgroup_placebo_sensitivity_summary: pd.DataFrame
    estimates_long: pd.DataFrame
    metadata: dict[str, Any]


def run_robustness_pipeline(config: RobustnessConfig | None = None) -> RobustnessResult:
    """Load the analytic cohort and run the robustness workflow."""

    config = config or RobustnessConfig()
    if not config.cohort_path.exists():
        raise FileNotFoundError(
            f"Missing analytic cohort at {config.cohort_path}. Run `make build-cohort` before robustness."
        )
    cohort = pd.read_parquet(config.cohort_path)
    return run_robustness_checks(cohort, config=config)


def run_robustness_checks(cohort: pd.DataFrame, *, config: RobustnessConfig | None = None) -> RobustnessResult:
    """Run pooled, stratified, subgroup, placebo, and sensitivity checks."""

    config = config or RobustnessConfig()
    analysis = _add_derived_check_columns(cohort)
    long_rows: list[dict[str, Any]] = []

    window_summary, window_long = _run_window_threshold_checks(analysis, config=config)
    long_rows.extend(window_long)
    environment_summary, environment_long = _run_environment_checks(analysis, config=config)
    long_rows.extend(environment_long)
    subgroup_summary, subgroup_long = _run_subgroup_checks(analysis, config=config)
    long_rows.extend(subgroup_long)
    placebo_summary, placebo_long = _run_placebo_checks(analysis, config=config)
    long_rows.extend(placebo_long)

    primary_estimate = _primary_aipw_estimate(window_summary, config=config)
    sensitivity_summary = _sensitivity_grid(primary_estimate, config=config)
    combined_summary = pd.concat(
        [subgroup_summary, placebo_summary, sensitivity_summary],
        ignore_index=True,
        sort=False,
    )
    estimates_long = pd.DataFrame(long_rows)
    metadata = _robustness_metadata(
        cohort=analysis,
        config=config,
        window_summary=window_summary,
        environment_summary=environment_summary,
        subgroup_summary=subgroup_summary,
        placebo_summary=placebo_summary,
        sensitivity_summary=sensitivity_summary,
    )

    return RobustnessResult(
        window_threshold_summary=window_summary,
        environment_summary=environment_summary,
        subgroup_placebo_sensitivity_summary=combined_summary,
        estimates_long=estimates_long,
        metadata=metadata,
    )


def write_robustness_artifacts(
    result: RobustnessResult,
    *,
    config: RobustnessConfig | None = None,
) -> dict[str, Path]:
    """Write robustness CSV, JSON, PNG, and markdown artifacts."""

    config = config or RobustnessConfig()
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.tables_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    config.docs_dir.mkdir(parents=True, exist_ok=True)

    paths = _artifact_paths(config)
    result.estimates_long.to_csv(paths["estimates_long"], index=False)
    result.window_threshold_summary.to_csv(paths["window_threshold_table"], index=False)
    result.environment_summary.to_csv(paths["environment_table"], index=False)
    result.subgroup_placebo_sensitivity_summary.to_csv(
        paths["subgroup_placebo_sensitivity_table"],
        index=False,
    )
    metadata = {**result.metadata, "artifact_paths": {name: str(path) for name, path in paths.items()}}
    paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    write_window_threshold_heatmap(result.window_threshold_summary, paths["window_threshold_figure"])
    write_module_presentation_figure(result.environment_summary, paths["module_figure"])
    write_subgroup_figure(result.subgroup_placebo_sensitivity_summary, paths["subgroup_figure"])
    write_placebo_sensitivity_figure(
        result.subgroup_placebo_sensitivity_summary,
        paths["placebo_sensitivity_figure"],
    )
    write_robustness_summary(result=result, metadata=metadata, summary_path=paths["summary"])
    return paths


def window_adjustment_columns(window: int, base_columns: Iterable[str] | None = None) -> tuple[str, ...]:
    """Return the baseline adjustment set with early-assessment columns matched to ``window``."""

    base = tuple(base_columns or recommended_baseline_adjustment_set())
    replacement = f"_{window}d"
    return tuple(
        column.replace("_14d", replacement) if column.startswith("early_assessment_") else column
        for column in base
    )


def treatment_spec(window: int, threshold: str) -> dict[str, str | int]:
    """Return treatment and availability columns for one robustness treatment definition."""

    return {
        "window_days": int(window),
        "threshold_name": threshold,
        "treatment_column": treatment_column_name(window, threshold),
        "availability_column": treatment_available_column(window),
    }


def subgroup_adjustment_columns(group_name: str, base_columns: Iterable[str]) -> tuple[str, ...]:
    """Remove the stratifying covariate family from a subgroup adjustment set."""

    remove_by_group = {
        "highest_education": {"baseline_highest_education"},
        "prior_attempts": {"baseline_num_of_prev_attempts"},
        "disability": {"baseline_disability"},
        "module_presentation": {"baseline_module_presentation"},
    }
    remove = remove_by_group.get(group_name, set())
    return tuple(column for column in base_columns if column not in remove)


def placebo_adjustment_columns(placebo_name: str, base_columns: Iterable[str]) -> tuple[str, ...]:
    """Remove the placebo outcome's covariate family from adjustment."""

    remove_by_placebo = {
        "placebo_registered_before_start": {
            "baseline_date_registration",
            "baseline_missing_date_registration",
            "baseline_registered_before_start",
        },
        "placebo_any_prior_attempts": {"baseline_num_of_prev_attempts"},
        "placebo_disability": {"baseline_disability"},
    }
    remove = remove_by_placebo.get(placebo_name, set())
    return tuple(column for column in base_columns if column not in remove)


def gate_reason(
    frame: pd.DataFrame,
    *,
    treatment_column: str,
    outcome_column: str,
    min_n: int,
    min_treated: int,
    min_control: int,
) -> str | None:
    """Return a skip reason if a scenario lacks adequate sample support."""

    data = frame.dropna(subset=[treatment_column, outcome_column]).copy()
    n = int(len(data))
    if n < min_n:
        return f"n={n} below minimum {min_n}"
    treatment = data[treatment_column].astype(int)
    treated = int(treatment.sum())
    control = int((1 - treatment).sum())
    if treated < min_treated:
        return f"treated_count={treated} below minimum {min_treated}"
    if control < min_control:
        return f"control_count={control} below minimum {min_control}"
    if treatment.nunique() != 2:
        return "treatment does not contain both treated and control records"
    if data[outcome_column].astype(int).nunique() != 2:
        return "outcome does not contain both event and non-event records"
    return None


def write_window_threshold_heatmap(summary: pd.DataFrame, output_path: Path) -> None:
    """Write a compact heatmap for pooled AIPW robustness estimates."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_data = summary[summary["scenario_status"].eq("success")].copy()
    thresholds = ["median", "top_tertile", "top_quartile"]
    pivot = plot_data.pivot(index="window_days", columns="threshold_name", values="estimate").reindex(
        index=[7, 14, 21],
        columns=thresholds,
    )
    fig, ax = plt.subplots(figsize=(7, 4.6))
    values = pivot.to_numpy(dtype=float)
    image = ax.imshow(values, cmap="RdYlBu", aspect="auto")
    ax.set_xticks(np.arange(len(thresholds)))
    ax.set_xticklabels(["Median", "Top tertile", "Top quartile"])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{int(value)}d" for value in pivot.index])
    ax.set_xlabel("Treatment threshold")
    ax.set_ylabel("Early engagement window")
    ax.set_title("Pooled AIPW Robustness Estimates")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            label = "" if np.isnan(values[i, j]) else f"{values[i, j]:.3f}"
            ax.text(j, i, label, ha="center", va="center", color="#111827", fontsize=10)
    fig.colorbar(image, ax=ax, label="Risk difference")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_module_presentation_figure(summary: pd.DataFrame, output_path: Path) -> None:
    """Write module-presentation primary-estimand estimates with pooled reference."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modules = summary[
        summary["section"].eq("module_presentation") & summary["scenario_status"].eq("success")
    ].copy()
    pooled = summary[summary["section"].eq("pooled_primary")]
    pooled_estimate = float(pooled["estimate"].iloc[0]) if not pooled.empty and pd.notna(pooled["estimate"].iloc[0]) else np.nan
    modules = modules.sort_values("estimate")
    fig_height = max(4.5, 0.25 * len(modules) + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    if modules.empty:
        ax.text(0.5, 0.5, "No adequate module-presentation strata", ha="center", va="center")
        ax.axis("off")
    else:
        y = np.arange(len(modules))
        ax.barh(y, modules["estimate"], color="#4C78A8")
        if pd.notna(pooled_estimate):
            ax.axvline(pooled_estimate, color="#F58518", linestyle="--", linewidth=1.5, label="Pooled primary")
            ax.legend(frameon=False)
        ax.set_yticks(y)
        ax.set_yticklabels(modules["stratum"])
        ax.set_xlabel("AIPW risk difference")
        ax.set_title("Module-Presentation Estimates")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_subgroup_figure(summary: pd.DataFrame, output_path: Path) -> None:
    """Write subgroup primary-estimand estimates."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subgroup = summary[summary["section"].eq("subgroup") & summary["scenario_status"].eq("success")].copy()
    subgroup = subgroup.sort_values(["subgroup_variable", "estimate"])
    fig_height = max(4.0, 0.32 * len(subgroup) + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    if subgroup.empty:
        ax.text(0.5, 0.5, "No adequate subgroup strata", ha="center", va="center")
        ax.axis("off")
    else:
        labels = subgroup["subgroup_variable"] + ": " + subgroup["subgroup_level"].astype(str)
        y = np.arange(len(subgroup))
        ax.scatter(subgroup["estimate"], y, color="#54A24B", s=38)
        ax.axvline(0.0, color="#555555", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("AIPW risk difference")
        ax.set_title("Subgroup Estimates")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_placebo_sensitivity_figure(summary: pd.DataFrame, output_path: Path) -> None:
    """Write placebo estimates and illustrative sensitivity grid together."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    placebo = summary[summary["section"].eq("placebo") & summary["scenario_status"].eq("success")].copy()
    sensitivity = summary[summary["section"].eq("sensitivity")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    if placebo.empty:
        axes[0].text(0.5, 0.5, "No placebo estimates", ha="center", va="center")
        axes[0].axis("off")
    else:
        placebo = placebo.sort_values("estimate")
        axes[0].barh(placebo["placebo_outcome"], placebo["estimate"], color="#B279A2")
        axes[0].axvline(0.0, color="#555555", linewidth=1)
        axes[0].set_xlabel("AIPW risk difference")
        axes[0].set_title("Pre-treatment Placebo Checks")

    if sensitivity.empty:
        axes[1].text(0.5, 0.5, "No sensitivity grid", ha="center", va="center")
        axes[1].axis("off")
    else:
        pivot = sensitivity.pivot(
            index="prevalence_difference",
            columns="outcome_risk_difference",
            values="corrected_estimate",
        )
        values = pivot.to_numpy(dtype=float)
        image = axes[1].imshow(values, cmap="RdYlBu", aspect="auto")
        axes[1].set_xticks(np.arange(len(pivot.columns)))
        axes[1].set_xticklabels([f"{value:.2f}" for value in pivot.columns])
        axes[1].set_yticks(np.arange(len(pivot.index)))
        axes[1].set_yticklabels([f"{value:.2f}" for value in pivot.index])
        axes[1].set_xlabel("Outcome risk difference")
        axes[1].set_ylabel("Prevalence difference")
        axes[1].set_title("Illustrative Bias Grid")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                axes[1].text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=axes[1], label="Corrected estimate")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_robustness_summary(
    *,
    result: RobustnessResult,
    metadata: dict[str, Any],
    summary_path: Path,
) -> None:
    """Generate markdown robustness summary from saved outputs."""

    primary = result.window_threshold_summary[
        (result.window_threshold_summary["window_days"] == metadata["primary_estimand"]["window_days"])
        & (result.window_threshold_summary["threshold_name"] == metadata["primary_estimand"]["threshold_name"])
    ]
    primary_estimate = float(primary["estimate"].iloc[0]) if not primary.empty else np.nan
    window_table = _markdown_rows(
        result.window_threshold_summary,
        ["window_days", "threshold_name", "scenario_status", "estimate", "n", "poor_overlap"],
        max_rows=12,
    )
    environment_comp = result.environment_summary[result.environment_summary["section"].isin(["pooled_primary", "stratified_weighted_mean"])]
    environment_table = _markdown_rows(
        environment_comp,
        ["section", "scenario_status", "estimate", "n", "estimate_difference_from_pooled"],
        max_rows=6,
    )
    skipped = pd.concat(
        [
            result.environment_summary[result.environment_summary["scenario_status"].eq("skipped")],
            result.subgroup_placebo_sensitivity_summary[
                result.subgroup_placebo_sensitivity_summary["scenario_status"].eq("skipped")
            ],
        ],
        ignore_index=True,
        sort=False,
    )
    skipped_text = (
        "- No strata or checks were skipped by sample-size gates."
        if skipped.empty
        else "\n".join(
            f"- {row.get('section', 'check')}: {row.get('stratum', row.get('subgroup_level', row.get('placebo_outcome', '')))} ({row.get('skip_reason')})"
            for row in skipped.head(12).to_dict(orient="records")
        )
    )
    text = f"""# Robustness Summary

This summary is generated from saved robustness artifacts. Interpret these checks as sensitivity and diagnostic evidence, not as new primary causal claims.

## Primary Reference

- Primary estimand retained: high 14-day median engagement on course success.
- Primary AIPW risk-difference estimate in the robustness stage: {primary_estimate:.6f}.
- Matching is disabled in robustness grids; regression adjustment and IPTW companion rows remain in `data/processed/robustness_estimates_long.csv`.

## Treatment Window and Threshold Checks

| window_days | threshold_name | status | estimate | n | poor_overlap |
| --- | --- | --- | ---: | ---: | --- |
{window_table}

## Pooled Versus Stratified Environment Check

| section | status | estimate | n | difference_from_pooled |
| --- | --- | ---: | ---: | ---: |
{environment_table}

## Subgroups, Placebos, and Sensitivity

- Subgroups are reported only where sample size, treatment variation, and outcome variation are adequate.
- Placebo outcomes use pre-treatment quantities: registered before start, any prior attempts, and disability.
- The sensitivity grid is an illustrative additive bias calculation, not a formal Rosenbaum bound or E-value.
- Combinations explaining away the primary point estimate: {metadata["sensitivity"]["explains_away_count"]}.

## Skipped Checks

{skipped_text}

## Cautions

- These estimates remain observational and rely on measured baseline adjustment.
- Unmeasured motivation, available study time, outside support, and competing obligations remain plausible confounders.
- Module-presentation and subgroup estimates are descriptive robustness checks and should not be overinterpreted as definitive heterogeneity.
"""
    summary_path.write_text(text, encoding="utf-8")


def _run_window_threshold_checks(
    cohort: pd.DataFrame,
    *,
    config: RobustnessConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    for window in config.windows:
        for threshold in config.thresholds:
            spec = treatment_spec(window, threshold)
            summary, rows = _estimate_scenario(
                cohort,
                config=config,
                section="window_threshold",
                scenario_id=f"pooled_{window}d_{threshold}",
                window_days=window,
                threshold_name=threshold,
                treatment_column=str(spec["treatment_column"]),
                availability_column=str(spec["availability_column"]),
                outcome_column=config.outcome_column,
                adjustment_columns=window_adjustment_columns(window),
                min_n=1,
                min_treated=1,
                min_control=1,
            )
            summary_rows.append(summary)
            long_rows.extend(rows)
    return pd.DataFrame(summary_rows), long_rows


def _run_environment_checks(
    cohort: pd.DataFrame,
    *,
    config: RobustnessConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    spec = treatment_spec(config.primary_window, config.primary_threshold)
    treatment_column = str(spec["treatment_column"])
    availability_column = str(spec["availability_column"])
    base_adjustment = window_adjustment_columns(config.primary_window)
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    pooled, pooled_rows = _estimate_scenario(
        cohort,
        config=config,
        section="pooled_primary",
        scenario_id="pooled_primary_14d_median",
        window_days=config.primary_window,
        threshold_name=config.primary_threshold,
        treatment_column=treatment_column,
        availability_column=availability_column,
        outcome_column=config.outcome_column,
        adjustment_columns=base_adjustment,
        min_n=1,
        min_treated=1,
        min_control=1,
    )
    pooled["estimate_difference_from_pooled"] = 0.0
    summary_rows.append(pooled)
    rows.extend(pooled_rows)
    pooled_estimate = pooled.get("estimate", np.nan)

    module_adjustment = subgroup_adjustment_columns("module_presentation", base_adjustment)
    for module, group in cohort.groupby("baseline_module_presentation", dropna=False):
        summary, scenario_rows = _estimate_scenario(
            group,
            config=config,
            section="module_presentation",
            scenario_id=f"module_{module}",
            window_days=config.primary_window,
            threshold_name=config.primary_threshold,
            treatment_column=treatment_column,
            availability_column=availability_column,
            outcome_column=config.outcome_column,
            adjustment_columns=module_adjustment,
            min_n=config.module_min_n,
            min_treated=config.module_min_treated,
            min_control=config.module_min_control,
            stratum=str(module),
        )
        summary["estimate_difference_from_pooled"] = (
            float(summary["estimate"] - pooled_estimate) if pd.notna(summary["estimate"]) and pd.notna(pooled_estimate) else np.nan
        )
        summary_rows.append(summary)
        rows.extend(scenario_rows)

    summary_frame = pd.DataFrame(summary_rows)
    successful_modules = summary_frame[
        summary_frame["section"].eq("module_presentation") & summary_frame["scenario_status"].eq("success")
    ]
    if successful_modules.empty:
        weighted = _empty_summary_row(
            section="stratified_weighted_mean",
            scenario_id="module_weighted_mean",
            treatment_column=treatment_column,
            outcome_column=config.outcome_column,
            availability_column=availability_column,
            adjustment_columns=module_adjustment,
            skip_reason="no module-presentation strata passed adequacy gates",
        )
    else:
        total_n = float(successful_modules["n"].sum())
        weighted_estimate = float(np.sum(successful_modules["estimate"] * successful_modules["n"]) / total_n)
        weighted = {
            "section": "stratified_weighted_mean",
            "scenario_id": "module_weighted_mean",
            "scenario_status": "success",
            "window_days": config.primary_window,
            "threshold_name": config.primary_threshold,
            "treatment_column": treatment_column,
            "outcome_column": config.outcome_column,
            "availability_column": availability_column,
            "stratum": "adequate module-presentation strata",
            "n": int(successful_modules["n"].sum()),
            "treated_count": int(successful_modules["treated_count"].sum()),
            "control_count": int(successful_modules["control_count"].sum()),
            "outcome_event_count": int(successful_modules["outcome_event_count"].sum()),
            "estimate": weighted_estimate,
            "treated_mean": np.nan,
            "control_mean": np.nan,
            "effective_sample_size": np.nan,
            "common_support_outside_share": np.nan,
            "max_abs_smd_after_weighting": np.nan,
            "poor_overlap": np.nan,
            "skip_reason": "",
            "adjustment_columns": "|".join(module_adjustment),
        }
    weighted["estimate_difference_from_pooled"] = (
        float(weighted["estimate"] - pooled_estimate) if pd.notna(weighted["estimate"]) and pd.notna(pooled_estimate) else np.nan
    )
    summary_frame = pd.concat([summary_frame, pd.DataFrame([weighted])], ignore_index=True, sort=False)
    return summary_frame, rows


def _run_subgroup_checks(
    cohort: pd.DataFrame,
    *,
    config: RobustnessConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    spec = treatment_spec(config.primary_window, config.primary_threshold)
    base_adjustment = window_adjustment_columns(config.primary_window)
    subgroup_specs = {
        "highest_education": "baseline_highest_education",
        "prior_attempts": "robustness_prior_attempt_group",
        "disability": "baseline_disability",
    }
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for subgroup_name, column in subgroup_specs.items():
        adjustment = subgroup_adjustment_columns(subgroup_name, base_adjustment)
        for level, group in cohort.groupby(column, dropna=False):
            summary, scenario_rows = _estimate_scenario(
                group,
                config=config,
                section="subgroup",
                scenario_id=f"subgroup_{subgroup_name}_{level}",
                window_days=config.primary_window,
                threshold_name=config.primary_threshold,
                treatment_column=str(spec["treatment_column"]),
                availability_column=str(spec["availability_column"]),
                outcome_column=config.outcome_column,
                adjustment_columns=adjustment,
                min_n=config.subgroup_min_n,
                min_treated=config.subgroup_min_treated,
                min_control=config.subgroup_min_control,
                subgroup_variable=subgroup_name,
                subgroup_level=str(level),
                stratum=str(level),
            )
            summary_rows.append(summary)
            rows.extend(scenario_rows)
    return pd.DataFrame(summary_rows), rows


def _run_placebo_checks(
    cohort: pd.DataFrame,
    *,
    config: RobustnessConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    spec = treatment_spec(config.primary_window, config.primary_threshold)
    base_adjustment = window_adjustment_columns(config.primary_window)
    placebo_outcomes = [
        "placebo_registered_before_start",
        "placebo_any_prior_attempts",
        "placebo_disability",
    ]
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for outcome in placebo_outcomes:
        adjustment = placebo_adjustment_columns(outcome, base_adjustment)
        summary, scenario_rows = _estimate_scenario(
            cohort,
            config=config,
            section="placebo",
            scenario_id=outcome,
            window_days=config.primary_window,
            threshold_name=config.primary_threshold,
            treatment_column=str(spec["treatment_column"]),
            availability_column=str(spec["availability_column"]),
            outcome_column=outcome,
            adjustment_columns=adjustment,
            min_n=1,
            min_treated=1,
            min_control=1,
            placebo_outcome=outcome,
        )
        summary_rows.append(summary)
        rows.extend(scenario_rows)
    return pd.DataFrame(summary_rows), rows


def _estimate_scenario(
    frame: pd.DataFrame,
    *,
    config: RobustnessConfig,
    section: str,
    scenario_id: str,
    window_days: int | None,
    threshold_name: str | None,
    treatment_column: str,
    availability_column: str,
    outcome_column: str,
    adjustment_columns: tuple[str, ...],
    min_n: int,
    min_treated: int,
    min_control: int,
    stratum: str = "",
    subgroup_variable: str = "",
    subgroup_level: str = "",
    placebo_outcome: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    available = frame[availability_column].fillna(False).astype(bool) if availability_column in frame else pd.Series(False, index=frame.index)
    analysis = frame.loc[available].copy()
    counts = _scenario_counts(analysis, treatment_column=treatment_column, outcome_column=outcome_column)
    reason = _missing_columns_reason(analysis, [treatment_column, outcome_column, availability_column, *adjustment_columns])
    reason = reason or gate_reason(
        analysis,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
        min_n=min_n,
        min_treated=min_treated,
        min_control=min_control,
    )
    common = {
        "section": section,
        "scenario_id": scenario_id,
        "window_days": window_days,
        "threshold_name": threshold_name,
        "treatment_column": treatment_column,
        "outcome_column": outcome_column,
        "availability_column": availability_column,
        "stratum": stratum,
        "subgroup_variable": subgroup_variable,
        "subgroup_level": subgroup_level,
        "placebo_outcome": placebo_outcome,
        "adjustment_columns": "|".join(adjustment_columns),
        **counts,
    }
    if reason:
        summary = {
            **common,
            "scenario_status": "skipped",
            "estimate": np.nan,
            "treated_mean": np.nan,
            "control_mean": np.nan,
            "effective_sample_size": np.nan,
            "common_support_outside_share": np.nan,
            "max_abs_smd_after_weighting": np.nan,
            "poor_overlap": np.nan,
            "skip_reason": reason,
        }
        return summary, [{**summary, "estimator": "aipw", "preferred": True, "estimator_status": "skipped", "notes": reason}]

    est_config = EstimationConfig(
        cohort_path=config.cohort_path,
        processed_dir=config.processed_dir,
        figures_dir=config.figures_dir,
        docs_dir=config.docs_dir,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
        availability_column=availability_column,
        adjustment_columns=adjustment_columns,
        seed=config.seed,
        matching_enabled=False,
    )
    result = estimate_effects(frame, config=est_config)
    diagnostics = result.metadata["diagnostics"]
    estimates = result.effect_estimates.copy()
    estimates = estimates[estimates["estimator"].ne("nearest_neighbor_matching")]
    preferred = estimates[estimates["preferred"].eq(True)].iloc[0]
    summary = {
        **common,
        "scenario_status": "success",
        "estimate": float(preferred["estimate"]),
        "treated_mean": float(preferred["treated_mean"]),
        "control_mean": float(preferred["control_mean"]),
        "effective_sample_size": float(diagnostics["effective_sample_size"]),
        "common_support_outside_share": float(diagnostics["common_support"]["outside_share"]),
        "max_abs_smd_after_weighting": float(diagnostics["max_abs_smd_after_weighting"]),
        "poor_overlap": bool(diagnostics["poor_overlap"]),
        "skip_reason": "",
    }
    long_rows = []
    for row in estimates.to_dict(orient="records"):
        long_rows.append(
            {
                **common,
                "scenario_status": "success",
                "estimator": row["estimator"],
                "preferred": bool(row["preferred"]),
                "estimator_status": row["status"],
                "estimate": row["estimate"],
                "treated_mean": row["treated_mean"],
                "control_mean": row["control_mean"],
                "matched_pairs": row["matched_pairs"],
                "matched_retention": row["matched_retention"],
                "effective_sample_size": float(diagnostics["effective_sample_size"]),
                "common_support_outside_share": float(diagnostics["common_support"]["outside_share"]),
                "max_abs_smd_after_weighting": float(diagnostics["max_abs_smd_after_weighting"]),
                "poor_overlap": bool(diagnostics["poor_overlap"]),
                "skip_reason": "",
                "notes": row["notes"],
            }
        )
    return summary, long_rows


def _add_derived_check_columns(cohort: pd.DataFrame) -> pd.DataFrame:
    result = cohort.copy()
    result["robustness_prior_attempt_group"] = pd.cut(
        pd.to_numeric(result["baseline_num_of_prev_attempts"], errors="coerce"),
        bins=[-1, 0, 1, np.inf],
        labels=["0", "1", "2+"],
    ).astype("object")
    result["robustness_prior_attempt_group"] = result["robustness_prior_attempt_group"].fillna("<MISSING>")
    result["placebo_registered_before_start"] = (
        pd.to_numeric(result["baseline_registered_before_start"], errors="coerce").fillna(0).astype(int)
    )
    result["placebo_any_prior_attempts"] = (
        pd.to_numeric(result["baseline_num_of_prev_attempts"], errors="coerce").fillna(0).gt(0).astype(int)
    )
    result["placebo_disability"] = result["baseline_disability"].eq("Y").astype(int)
    return result


def _scenario_counts(frame: pd.DataFrame, *, treatment_column: str, outcome_column: str) -> dict[str, int]:
    if treatment_column not in frame or outcome_column not in frame:
        return {"n": int(len(frame)), "treated_count": 0, "control_count": 0, "outcome_event_count": 0}
    data = frame.dropna(subset=[treatment_column, outcome_column]).copy()
    if data.empty:
        return {"n": 0, "treated_count": 0, "control_count": 0, "outcome_event_count": 0}
    treatment = data[treatment_column].astype(int)
    outcome = data[outcome_column].astype(int)
    return {
        "n": int(len(data)),
        "treated_count": int(treatment.sum()),
        "control_count": int((1 - treatment).sum()),
        "outcome_event_count": int(outcome.sum()),
    }


def _missing_columns_reason(frame: pd.DataFrame, required: Iterable[str]) -> str | None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return f"missing required columns: {missing}"
    return None


def _empty_summary_row(
    *,
    section: str,
    scenario_id: str,
    treatment_column: str,
    outcome_column: str,
    availability_column: str,
    adjustment_columns: tuple[str, ...],
    skip_reason: str,
) -> dict[str, Any]:
    return {
        "section": section,
        "scenario_id": scenario_id,
        "scenario_status": "skipped",
        "window_days": np.nan,
        "threshold_name": "",
        "treatment_column": treatment_column,
        "outcome_column": outcome_column,
        "availability_column": availability_column,
        "stratum": "",
        "n": 0,
        "treated_count": 0,
        "control_count": 0,
        "outcome_event_count": 0,
        "estimate": np.nan,
        "treated_mean": np.nan,
        "control_mean": np.nan,
        "effective_sample_size": np.nan,
        "common_support_outside_share": np.nan,
        "max_abs_smd_after_weighting": np.nan,
        "poor_overlap": np.nan,
        "skip_reason": skip_reason,
        "adjustment_columns": "|".join(adjustment_columns),
    }


def _primary_aipw_estimate(summary: pd.DataFrame, *, config: RobustnessConfig) -> float:
    primary = summary[
        summary["window_days"].eq(config.primary_window) & summary["threshold_name"].eq(config.primary_threshold)
    ]
    if primary.empty or primary["scenario_status"].iloc[0] != "success":
        return np.nan
    return float(primary["estimate"].iloc[0])


def _sensitivity_grid(primary_estimate: float, *, config: RobustnessConfig) -> pd.DataFrame:
    rows = []
    for prevalence_difference in config.sensitivity_prevalence_differences:
        for outcome_risk_difference in config.sensitivity_outcome_risk_differences:
            bias = float(prevalence_difference * outcome_risk_difference)
            corrected = float(primary_estimate - bias) if pd.notna(primary_estimate) else np.nan
            rows.append(
                {
                    "section": "sensitivity",
                    "scenario_id": f"bias_{prevalence_difference:.2f}_{outcome_risk_difference:.2f}",
                    "scenario_status": "illustrative",
                    "primary_estimate": primary_estimate,
                    "prevalence_difference": float(prevalence_difference),
                    "outcome_risk_difference": float(outcome_risk_difference),
                    "bias": bias,
                    "corrected_estimate": corrected,
                    "explains_away": bool(pd.notna(corrected) and corrected <= 0),
                    "notes": "Additive unmeasured-confounding placeholder; not a formal sensitivity bound.",
                }
            )
    return pd.DataFrame(rows)


def _robustness_metadata(
    *,
    cohort: pd.DataFrame,
    config: RobustnessConfig,
    window_summary: pd.DataFrame,
    environment_summary: pd.DataFrame,
    subgroup_summary: pd.DataFrame,
    placebo_summary: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
) -> dict[str, Any]:
    skipped = int(
        pd.concat([environment_summary, subgroup_summary, placebo_summary], ignore_index=True, sort=False)[
            "scenario_status"
        ].eq("skipped").sum()
    )
    return {
        "stage": "robustness",
        "cohort_rows": int(len(cohort)),
        "primary_estimand": {
            "window_days": config.primary_window,
            "threshold_name": config.primary_threshold,
            "treatment_column": treatment_column_name(config.primary_window, config.primary_threshold),
            "outcome_column": config.outcome_column,
        },
        "grid": {
            "windows": list(config.windows),
            "thresholds": list(config.thresholds),
            "pooled_scenarios": int(len(window_summary)),
        },
        "gates": {
            "module_min_n": config.module_min_n,
            "module_min_treated": config.module_min_treated,
            "module_min_control": config.module_min_control,
            "subgroup_min_n": config.subgroup_min_n,
            "subgroup_min_treated": config.subgroup_min_treated,
            "subgroup_min_control": config.subgroup_min_control,
        },
        "scenario_counts": {
            "window_threshold_success": int(window_summary["scenario_status"].eq("success").sum()),
            "module_success": int(
                (
                    environment_summary["section"].eq("module_presentation")
                    & environment_summary["scenario_status"].eq("success")
                ).sum()
            ),
            "subgroup_success": int(subgroup_summary["scenario_status"].eq("success").sum()),
            "placebo_success": int(placebo_summary["scenario_status"].eq("success").sum()),
            "skipped_checks": skipped,
        },
        "sensitivity": {
            "policy": "corrected_estimate = primary_estimate - prevalence_difference * outcome_risk_difference",
            "prevalence_differences": list(config.sensitivity_prevalence_differences),
            "outcome_risk_differences": list(config.sensitivity_outcome_risk_differences),
            "explains_away_count": int(sensitivity_summary["explains_away"].sum()),
            "formal_bound": False,
        },
        "warnings": [
            "Robustness, subgroup, and placebo checks are diagnostic and do not remove unmeasured-confounding concerns.",
            "Matching is disabled in robustness grids; detailed rows retain regression adjustment, IPTW, and AIPW.",
            "The sensitivity grid is illustrative and additive, not a formal Rosenbaum bound or E-value.",
        ],
    }


def _artifact_paths(config: RobustnessConfig) -> dict[str, Path]:
    return {
        "estimates_long": config.processed_dir / ROBUSTNESS_LONG_PATH.name,
        "metadata": config.processed_dir / ROBUSTNESS_METADATA_PATH.name,
        "window_threshold_table": config.tables_dir / WINDOW_THRESHOLD_TABLE_PATH.name,
        "environment_table": config.tables_dir / ENVIRONMENT_TABLE_PATH.name,
        "subgroup_placebo_sensitivity_table": config.tables_dir / SUBGROUP_PLACEBO_SENSITIVITY_TABLE_PATH.name,
        "window_threshold_figure": config.figures_dir / WINDOW_THRESHOLD_FIGURE_PATH.name,
        "module_figure": config.figures_dir / MODULE_FIGURE_PATH.name,
        "subgroup_figure": config.figures_dir / SUBGROUP_FIGURE_PATH.name,
        "placebo_sensitivity_figure": config.figures_dir / PLACEBO_SENSITIVITY_FIGURE_PATH.name,
        "summary": config.docs_dir / ROBUSTNESS_SUMMARY_PATH.name,
    }


def _markdown_rows(frame: pd.DataFrame, columns: list[str], *, max_rows: int) -> str:
    if frame.empty:
        return "| " + " | ".join("" for _ in columns) + " |"
    rows = []
    for row in frame.head(max_rows).to_dict(orient="records"):
        values = []
        for column in columns:
            value = row.get(column, "")
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)
