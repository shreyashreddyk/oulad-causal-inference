"""Treatment effect estimation routines for the primary OULAD estimand.

The functions in this module estimate the average treatment effect of high
14-day online engagement on course success using the adjustment set documented
in the hand-built DAG. Estimation outputs are designed to be saved as pipeline
artifacts, not copied from notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from oulad_causal.config import DOCS_DIR, FIGURES_DIR, PROCESSED_DATA_DIR
from oulad_causal.dag import (
    ANALYTIC_COHORT_PATH,
    PRIMARY_OUTCOME_COLUMN,
    PRIMARY_TREATMENT_COLUMN,
    recommended_baseline_adjustment_set,
)


PRIMARY_AVAILABILITY_COLUMN = "treatment_available_14d"
EFFECT_ESTIMATES_PATH = PROCESSED_DATA_DIR / "effect_estimates_main.csv"
BALANCE_TABLE_PATH = PROCESSED_DATA_DIR / "balance_table_main.csv"
ESTIMATION_METADATA_PATH = PROCESSED_DATA_DIR / "estimation_run_metadata.json"
OVERLAP_PLOT_PATH = FIGURES_DIR / "overlap_plot.png"
LOVE_PLOT_PATH = FIGURES_DIR / "love_plot_main.png"
ESTIMATION_SUMMARY_PATH = DOCS_DIR / "estimation_summary.md"


@dataclass(frozen=True)
class EstimationConfig:
    """Configuration for the primary observational estimation workflow."""

    cohort_path: Path = ANALYTIC_COHORT_PATH
    processed_dir: Path = PROCESSED_DATA_DIR
    figures_dir: Path = FIGURES_DIR
    docs_dir: Path = DOCS_DIR
    treatment_column: str = PRIMARY_TREATMENT_COLUMN
    outcome_column: str = PRIMARY_OUTCOME_COLUMN
    availability_column: str = PRIMARY_AVAILABILITY_COLUMN
    adjustment_columns: tuple[str, ...] = field(default_factory=recommended_baseline_adjustment_set)
    seed: int = 245
    propensity_clip: float = 1e-3
    poor_overlap_propensity_lower: float = 0.05
    poor_overlap_propensity_upper: float = 0.95
    poor_overlap_outside_common_support_share: float = 0.05
    poor_overlap_max_abs_smd: float = 0.25
    matching_enabled: bool = True
    matching_min_retention: float = 0.80
    matching_max_abs_logit_distance: float = 0.50


@dataclass
class EstimationResult:
    """In-memory result from the estimation workflow."""

    effect_estimates: pd.DataFrame
    balance_table: pd.DataFrame
    metadata: dict[str, Any]
    propensity_scores: pd.Series
    stabilized_weights: pd.Series
    treatment: pd.Series


class ConstantProbabilityModel:
    """Small fallback model for single-class outcome strata."""

    def __init__(self, probability: float):
        self.probability = float(np.clip(probability, 0.0, 1.0))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p1 = np.full(x.shape[0], self.probability, dtype=float)
        return np.column_stack([1.0 - p1, p1])


def run_estimation_pipeline(config: EstimationConfig | None = None) -> EstimationResult:
    """Load the analytic cohort, run primary estimators, and return artifacts."""

    config = config or EstimationConfig()
    if not config.cohort_path.exists():
        raise FileNotFoundError(
            f"Missing analytic cohort at {config.cohort_path}. Run `make build-cohort` before estimation."
        )

    cohort = pd.read_parquet(config.cohort_path)
    return estimate_effects(cohort, config=config)


def estimate_effects(cohort: pd.DataFrame, *, config: EstimationConfig | None = None) -> EstimationResult:
    """Estimate the primary ATE and diagnostics from an analytic cohort frame."""

    config = config or EstimationConfig()
    validate_estimation_columns(cohort, config=config)
    analysis = _prepare_analysis_frame(cohort, config=config)

    treatment = analysis[config.treatment_column].astype(int).to_numpy()
    outcome = analysis[config.outcome_column].astype(int).to_numpy()
    if len(np.unique(treatment)) != 2:
        raise ValueError("Primary treatment must contain both treated and control records.")
    if len(np.unique(outcome)) != 2:
        raise ValueError("Primary outcome must contain both success and non-success records.")

    x_raw = _design_input_frame(analysis, config.adjustment_columns)
    transformer, x_design, feature_names, preprocessing_metadata = fit_covariate_transformer(x_raw)
    treatment_model = fit_treatment_model(x_design, treatment, seed=config.seed)
    propensity = np.asarray(treatment_model.predict_proba(x_design)[:, 1], dtype=float)
    propensity_for_computation = clip_propensity_scores(propensity, epsilon=config.propensity_clip)
    stabilized_weights = stabilized_iptw_weights(treatment, propensity_for_computation)

    outcome_models = fit_outcome_models(x_design, treatment, outcome, seed=config.seed)
    mu0 = np.asarray(outcome_models[0].predict_proba(x_design)[:, 1], dtype=float)
    mu1 = np.asarray(outcome_models[1].predict_proba(x_design)[:, 1], dtype=float)

    positivity = positivity_diagnostics(
        treatment=treatment,
        propensity=propensity,
        propensity_for_computation=propensity_for_computation,
        stabilized_weights=stabilized_weights,
        config=config,
    )
    balance = standardized_mean_difference_table(
        pd.DataFrame(x_design, columns=feature_names),
        treatment=treatment,
        weights=stabilized_weights,
    )
    positivity["max_abs_smd_after_weighting"] = float(balance["abs_smd_weighted"].max())
    positivity["poor_balance_after_weighting"] = bool(
        positivity["max_abs_smd_after_weighting"] > config.poor_overlap_max_abs_smd
    )
    positivity["poor_overlap"] = bool(
        positivity["poor_overlap_by_propensity"]
        or positivity["poor_overlap_by_common_support"]
        or positivity["poor_balance_after_weighting"]
    )

    estimates = _effect_estimates(
        treatment=treatment,
        outcome=outcome,
        propensity=propensity_for_computation,
        stabilized_weights=stabilized_weights,
        mu0=mu0,
        mu1=mu1,
        config=config,
    )
    matching = nearest_neighbor_matching_estimate(
        treatment=treatment,
        outcome=outcome,
        propensity=propensity_for_computation,
        poor_overlap=positivity["poor_overlap"],
        config=config,
    )
    estimates = pd.concat([estimates, pd.DataFrame([matching])], ignore_index=True)

    metadata = _build_metadata(
        config=config,
        cohort=cohort,
        analysis=analysis,
        transformer=transformer,
        preprocessing_metadata=preprocessing_metadata,
        treatment_model=treatment_model,
        outcome_models=outcome_models,
        positivity=positivity,
        matching=matching,
    )

    return EstimationResult(
        effect_estimates=estimates,
        balance_table=balance,
        metadata=metadata,
        propensity_scores=pd.Series(propensity, index=analysis.index, name="propensity_score"),
        stabilized_weights=pd.Series(stabilized_weights, index=analysis.index, name="stabilized_weight"),
        treatment=pd.Series(treatment, index=analysis.index, name=config.treatment_column),
    )


def validate_estimation_columns(cohort: pd.DataFrame, *, config: EstimationConfig | None = None) -> None:
    """Raise a clear error when required primary estimation columns are absent."""

    config = config or EstimationConfig()
    required = [
        config.treatment_column,
        config.outcome_column,
        config.availability_column,
        *config.adjustment_columns,
    ]
    missing = [column for column in required if column not in cohort.columns]
    if missing:
        raise KeyError(f"Missing required estimation columns: {missing}")


def fit_covariate_transformer(x_raw: pd.DataFrame) -> tuple[ColumnTransformer, np.ndarray, list[str], dict[str, Any]]:
    """Fit the shared covariate preprocessing transformer."""

    numeric_columns = [column for column in x_raw.columns if is_numeric_dtype(x_raw[column])]
    categorical_columns = [column for column in x_raw.columns if column not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="<MISSING>")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    x_design = transformer.fit_transform(x_raw)
    feature_names = [str(name) for name in transformer.get_feature_names_out()]
    metadata = {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "feature_count_after_encoding": int(len(feature_names)),
        "categorical_missing_fill_value": "<MISSING>",
        "numeric_imputation": "median",
        "numeric_scaling": "standard",
        "categorical_encoding": "one_hot_drop_none",
    }
    return transformer, np.asarray(x_design, dtype=float), feature_names, metadata


def fit_treatment_model(x_design: np.ndarray, treatment: np.ndarray, *, seed: int) -> LogisticRegression:
    """Fit the propensity-score model."""

    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(x_design, treatment)
    return model


def fit_outcome_models(
    x_design: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    *,
    seed: int,
) -> dict[int, LogisticRegression | ConstantProbabilityModel]:
    """Fit separate outcome regressions for treated and control records."""

    models: dict[int, LogisticRegression | ConstantProbabilityModel] = {}
    for value in (0, 1):
        mask = treatment == value
        y_group = outcome[mask]
        if len(np.unique(y_group)) == 1:
            models[value] = ConstantProbabilityModel(float(y_group[0]))
            continue
        model = LogisticRegression(max_iter=1000, random_state=seed)
        model.fit(x_design[mask], y_group)
        models[value] = model
    return models


def clip_propensity_scores(propensity: np.ndarray, *, epsilon: float) -> np.ndarray:
    """Bound propensity scores away from exact 0 and 1 for finite arithmetic."""

    if not 0 < epsilon < 0.5:
        raise ValueError("propensity_clip must be between 0 and 0.5.")
    return np.clip(np.asarray(propensity, dtype=float), epsilon, 1.0 - epsilon)


def stabilized_iptw_weights(treatment: np.ndarray, propensity: np.ndarray) -> np.ndarray:
    """Compute stabilized inverse-probability treatment weights."""

    treatment = np.asarray(treatment, dtype=int)
    propensity = np.asarray(propensity, dtype=float)
    marginal_treated = float(treatment.mean())
    return np.where(
        treatment == 1,
        marginal_treated / propensity,
        (1.0 - marginal_treated) / (1.0 - propensity),
    )


def positivity_diagnostics(
    *,
    treatment: np.ndarray,
    propensity: np.ndarray,
    propensity_for_computation: np.ndarray,
    stabilized_weights: np.ndarray,
    config: EstimationConfig,
) -> dict[str, Any]:
    """Compute overlap and weight diagnostics."""

    treatment = np.asarray(treatment, dtype=int)
    propensity = np.asarray(propensity, dtype=float)
    treated_propensity = propensity[treatment == 1]
    control_propensity = propensity[treatment == 0]
    common_lower = float(max(treated_propensity.min(), control_propensity.min()))
    common_upper = float(min(treated_propensity.max(), control_propensity.max()))
    outside_common = (propensity < common_lower) | (propensity > common_upper)
    clipped_count = int(np.sum(np.abs(propensity_for_computation - propensity) > 0))

    diagnostics = {
        "propensity_summary": {
            "all": _series_summary(propensity),
            "treated": _series_summary(treated_propensity),
            "control": _series_summary(control_propensity),
        },
        "common_support": {
            "lower": common_lower,
            "upper": common_upper,
            "outside_count": int(outside_common.sum()),
            "outside_share": float(outside_common.mean()),
        },
        "stabilized_weight_summary": _series_summary(stabilized_weights),
        "effective_sample_size": float((stabilized_weights.sum() ** 2) / np.square(stabilized_weights).sum()),
        "propensity_clip_epsilon": config.propensity_clip,
        "propensity_scores_clipped_for_computation_count": clipped_count,
    }
    diagnostics["poor_overlap_by_propensity"] = bool(
        propensity.min() < config.poor_overlap_propensity_lower
        or propensity.max() > config.poor_overlap_propensity_upper
    )
    diagnostics["poor_overlap_by_common_support"] = bool(
        diagnostics["common_support"]["outside_share"] > config.poor_overlap_outside_common_support_share
    )
    return diagnostics


def standardized_mean_difference_table(
    covariates: pd.DataFrame,
    *,
    treatment: np.ndarray | pd.Series,
    weights: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """Return SMD balance before and after weighting for each covariate column."""

    unweighted = standardized_mean_differences(covariates, treatment=treatment)
    weighted = standardized_mean_differences(covariates, treatment=treatment, weights=weights)
    table = pd.DataFrame(
        {
            "variable": list(unweighted.index),
            "smd_unweighted": unweighted.to_numpy(dtype=float),
            "smd_weighted": weighted.to_numpy(dtype=float),
        }
    )
    table["abs_smd_unweighted"] = table["smd_unweighted"].abs()
    table["abs_smd_weighted"] = table["smd_weighted"].abs()
    table["improved_after_weighting"] = table["abs_smd_weighted"] <= table["abs_smd_unweighted"]
    return table.sort_values("abs_smd_unweighted", ascending=False).reset_index(drop=True)


def standardized_mean_differences(
    covariates: pd.DataFrame,
    *,
    treatment: np.ndarray | pd.Series,
    weights: np.ndarray | pd.Series | None = None,
) -> pd.Series:
    """Compute column-wise standardized mean differences for binary treatment."""

    treatment_array = np.asarray(treatment, dtype=int)
    if set(np.unique(treatment_array)) != {0, 1}:
        raise ValueError("SMD treatment input must contain binary values 0 and 1.")
    if weights is None:
        weight_array = np.ones_like(treatment_array, dtype=float)
    else:
        weight_array = np.asarray(weights, dtype=float)
    if covariates.shape[0] != treatment_array.shape[0] or covariates.shape[0] != weight_array.shape[0]:
        raise ValueError("Covariates, treatment, and weights must have the same row count.")

    smds: dict[str, float] = {}
    values = covariates.astype(float)
    for column in values.columns:
        x = values[column].to_numpy(dtype=float)
        treated = treatment_array == 1
        control = ~treated
        mean_t = _weighted_mean(x[treated], weight_array[treated])
        mean_c = _weighted_mean(x[control], weight_array[control])
        var_t = _weighted_variance(x[treated], weight_array[treated], mean_t)
        var_c = _weighted_variance(x[control], weight_array[control], mean_c)
        denominator = float(np.sqrt((var_t + var_c) / 2.0))
        smds[column] = 0.0 if denominator == 0 else float((mean_t - mean_c) / denominator)
    return pd.Series(smds, dtype=float)


def nearest_neighbor_matching_estimate(
    *,
    treatment: np.ndarray,
    outcome: np.ndarray,
    propensity: np.ndarray,
    poor_overlap: bool,
    config: EstimationConfig,
) -> dict[str, Any]:
    """Optionally estimate a nearest-neighbor ATT-style matched contrast."""

    if not config.matching_enabled:
        return _skipped_matching_row("matching disabled in EstimationConfig", config=config)
    if poor_overlap:
        return _skipped_matching_row("poor overlap diagnostics; matching estimate not reported", config=config)

    treated_idx = np.flatnonzero(treatment == 1)
    control_idx = np.flatnonzero(treatment == 0)
    if len(treated_idx) == 0 or len(control_idx) == 0:
        return _skipped_matching_row("treated or control group is empty", config=config)

    treated_scores = _logit(propensity[treated_idx]).reshape(-1, 1)
    control_scores = _logit(propensity[control_idx]).reshape(-1, 1)
    neighbors = NearestNeighbors(n_neighbors=1, algorithm="auto")
    neighbors.fit(control_scores)
    distances, indices = neighbors.kneighbors(treated_scores)
    retained = distances[:, 0] <= config.matching_max_abs_logit_distance
    retention = float(retained.mean())
    if retention < config.matching_min_retention:
        return _skipped_matching_row(
            f"matched retention {retention:.3f} below threshold {config.matching_min_retention:.3f}",
            config=config,
            matched_pairs=int(retained.sum()),
            retention=retention,
        )

    matched_treated = treated_idx[retained]
    matched_controls = control_idx[indices[retained, 0]]
    treated_mean = float(outcome[matched_treated].mean())
    control_mean = float(outcome[matched_controls].mean())
    return {
        "estimator": "nearest_neighbor_matching",
        "preferred": False,
        "estimand": "ATT-style matched risk difference for treated records retained by the matching gate",
        "status": "success",
        "estimate": treated_mean - control_mean,
        "treated_mean": treated_mean,
        "control_mean": control_mean,
        "std_error": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "uncertainty_method": "",
        "n": int(len(treatment)),
        "matched_pairs": int(len(matched_treated)),
        "matched_retention": retention,
        "notes": "nearest-neighbor matching on logit propensity score without replacement constraints",
    }


def write_estimation_artifacts(result: EstimationResult, *, config: EstimationConfig | None = None) -> dict[str, Path]:
    """Write CSV, JSON, PNG, and markdown estimation artifacts."""

    config = config or EstimationConfig()
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    config.docs_dir.mkdir(parents=True, exist_ok=True)

    paths = _artifact_paths(config)
    result.effect_estimates.to_csv(paths["effect_estimates"], index=False)
    result.balance_table.to_csv(paths["balance_table"], index=False)
    metadata = {**result.metadata, "artifact_paths": {name: str(path) for name, path in paths.items()}}
    paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_overlap_plot(result.propensity_scores, result.treatment, paths["overlap_plot"])
    write_love_plot(result.balance_table, paths["love_plot"])
    write_estimation_summary(
        effect_estimates=result.effect_estimates,
        metadata=metadata,
        summary_path=paths["summary"],
    )
    return paths


def write_overlap_plot(propensity_scores: pd.Series, treatment: list[int] | np.ndarray, output_path: Path) -> None:
    """Write the propensity-score overlap figure."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    treatment_array = np.asarray(treatment, dtype=int)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        propensity_scores[treatment_array == 0],
        bins=30,
        alpha=0.60,
        density=True,
        label="Lower engagement",
        color="#4C78A8",
    )
    ax.hist(
        propensity_scores[treatment_array == 1],
        bins=30,
        alpha=0.55,
        density=True,
        label="High engagement",
        color="#F58518",
    )
    ax.set_xlabel("Estimated propensity score")
    ax.set_ylabel("Density")
    ax.set_title("Treatment Model Overlap")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_love_plot(balance_table: pd.DataFrame, output_path: Path, *, max_variables: int = 30) -> None:
    """Write an absolute-SMD balance figure."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_data = (
        balance_table.sort_values("abs_smd_unweighted", ascending=False)
        .head(max_variables)
        .sort_values("abs_smd_unweighted", ascending=True)
    )
    fig_height = max(4.0, 0.22 * len(plot_data) + 1.2)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    y = np.arange(len(plot_data))
    ax.scatter(plot_data["abs_smd_unweighted"], y, label="Before weighting", color="#4C78A8", s=28)
    ax.scatter(plot_data["abs_smd_weighted"], y, label="After IPTW", color="#F58518", s=28)
    ax.axvline(0.10, color="#555555", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_data["variable"])
    ax.set_xlabel("Absolute standardized mean difference")
    ax.set_title("Covariate Balance")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_estimation_summary(
    *,
    effect_estimates: pd.DataFrame,
    metadata: dict[str, Any],
    summary_path: Path,
) -> None:
    """Generate the markdown estimation summary from saved results."""

    main = effect_estimates.loc[effect_estimates["preferred"] == True]  # noqa: E712
    main_estimate = main.iloc[0] if not main.empty else effect_estimates.iloc[0]
    diagnostics = metadata["diagnostics"]
    warnings = metadata.get("warnings", [])
    warning_text = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- No automatic warnings were raised."
    text = f"""# Estimation Summary

This summary is generated from the saved estimation artifacts. It should not be edited to hardcode conclusions.

## Main Estimand

- Estimand: population average treatment effect risk difference.
- Treatment: `{metadata["estimand"]["treatment_column"]}`.
- Outcome: `{metadata["estimand"]["outcome_column"]}`.
- Adjustment set source: `{metadata["estimand"]["adjustment_set_source"]}`.
- Analysis rows: {metadata["analysis"]["analysis_rows"]}.

## Main Estimators

| estimator | preferred | status | estimate | 95% CI | SE | notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
{_markdown_estimate_rows(effect_estimates)}

Preferred main estimate: `{main_estimate["estimator"]}` with risk difference {float(main_estimate["estimate"]):.6f}.

## Diagnostics Produced

- Effect estimates table: `data/processed/effect_estimates_main.csv`.
- Balance table: `data/processed/balance_table_main.csv`.
- Overlap plot: `reports/figures/overlap_plot.png`.
- Love plot: `reports/figures/love_plot_main.png`.
- Effective sample size after stabilized weighting: {diagnostics["effective_sample_size"]:.2f}.
- Propensity score range: {diagnostics["propensity_summary"]["all"]["min"]:.6f} to {diagnostics["propensity_summary"]["all"]["max"]:.6f}.
- Propensity scores clipped for finite arithmetic: {diagnostics["propensity_scores_clipped_for_computation_count"]}.
- Common support outside share: {diagnostics["common_support"]["outside_share"]:.6f}.
- Maximum absolute SMD after weighting: {diagnostics["max_abs_smd_after_weighting"]:.6f}.
- Poor overlap flag: {diagnostics["poor_overlap"]}.

## Known Limitations

- This is an observational estimate and relies on conditional exchangeability after the documented baseline adjustment set.
- Motivation, time availability, outside support, employment, and competing obligations remain unmeasured.
- VLE clicks measure platform interaction quantity, not necessarily learning quality.
- Stabilized weights are not truncated by default; poor overlap is detected and reported instead.
- Later assessment behavior and later VLE activity are intentionally excluded from the primary adjustment set because they may be post-treatment mediators.

## Automatic Warnings

{warning_text}
"""
    summary_path.write_text(text, encoding="utf-8")


def _effect_estimates(
    *,
    treatment: np.ndarray,
    outcome: np.ndarray,
    propensity: np.ndarray,
    stabilized_weights: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    config: EstimationConfig,
) -> pd.DataFrame:
    treated = treatment == 1
    control = treatment == 0

    ra_treated = float(mu1.mean())
    ra_control = float(mu0.mean())
    ra_scores = mu1 - mu0
    iptw_treated = _weighted_mean(outcome[treated], stabilized_weights[treated])
    iptw_control = _weighted_mean(outcome[control], stabilized_weights[control])
    iptw_scores = (
        len(treatment)
        * (
            treatment * stabilized_weights * outcome / stabilized_weights[treated].sum()
            - (1 - treatment) * stabilized_weights * outcome / stabilized_weights[control].sum()
        )
    )
    aipw_scores = (
        mu1
        - mu0
        + treatment * (outcome - mu1) / propensity
        - (1 - treatment) * (outcome - mu0) / (1.0 - propensity)
    )
    aipw_estimate = float(aipw_scores.mean())
    aipw_treated = float((mu1 + treatment * (outcome - mu1) / propensity).mean())
    aipw_control = float((mu0 + (1 - treatment) * (outcome - mu0) / (1.0 - propensity)).mean())

    rows = [
        {
            "estimator": "regression_adjustment",
            "preferred": False,
            "estimand": "ATE risk difference",
            "status": "success",
            "estimate": ra_treated - ra_control,
            "treated_mean": ra_treated,
            "control_mean": ra_control,
            **_uncertainty_fields(ra_scores, "empirical predicted-contrast score"),
            "n": int(len(treatment)),
            "matched_pairs": np.nan,
            "matched_retention": np.nan,
            "notes": "mean predicted potential outcome under separate logistic outcome regressions",
        },
        {
            "estimator": "stabilized_iptw",
            "preferred": False,
            "estimand": "ATE risk difference",
            "status": "success",
            "estimate": iptw_treated - iptw_control,
            "treated_mean": iptw_treated,
            "control_mean": iptw_control,
            **_uncertainty_fields(iptw_scores, "empirical stabilized-IPTW contrast score"),
            "n": int(len(treatment)),
            "matched_pairs": np.nan,
            "matched_retention": np.nan,
            "notes": "stabilized IPTW Hajek-style weighted mean contrast; no default truncation",
        },
        {
            "estimator": "aipw",
            "preferred": True,
            "estimand": "ATE risk difference",
            "status": "success",
            "estimate": aipw_estimate,
            "treated_mean": aipw_treated,
            "control_mean": aipw_control,
            **_uncertainty_fields(aipw_scores, "empirical AIPW score"),
            "n": int(len(treatment)),
            "matched_pairs": np.nan,
            "matched_retention": np.nan,
            "notes": "doubly robust AIPW estimator using propensity and separate outcome nuisance models",
        },
    ]
    return pd.DataFrame(rows)


def _prepare_analysis_frame(cohort: pd.DataFrame, *, config: EstimationConfig) -> pd.DataFrame:
    keep_columns = [
        config.treatment_column,
        config.outcome_column,
        config.availability_column,
        *config.adjustment_columns,
    ]
    analysis = cohort.loc[:, keep_columns].copy()
    availability = analysis[config.availability_column].astype(bool)
    analysis = analysis.loc[availability].copy()
    analysis = analysis.dropna(subset=[config.treatment_column, config.outcome_column])
    return analysis


def _design_input_frame(cohort: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    x = cohort.loc[:, list(columns)].copy()
    for column in x.columns:
        if is_numeric_dtype(x[column]):
            x[column] = pd.to_numeric(x[column], errors="coerce")
        else:
            x[column] = x[column].astype("object").where(x[column].notna(), np.nan)
    return x


def _build_metadata(
    *,
    config: EstimationConfig,
    cohort: pd.DataFrame,
    analysis: pd.DataFrame,
    transformer: ColumnTransformer,
    preprocessing_metadata: dict[str, Any],
    treatment_model: LogisticRegression,
    outcome_models: dict[int, LogisticRegression | ConstantProbabilityModel],
    positivity: dict[str, Any],
    matching: dict[str, Any],
) -> dict[str, Any]:
    warnings = []
    if positivity["poor_overlap"]:
        warnings.append("Overlap or post-weighting balance diagnostics were flagged as poor.")
    if positivity["propensity_scores_clipped_for_computation_count"]:
        warnings.append(
            "Some propensity scores were clipped only for finite arithmetic; weights were not otherwise truncated."
        )
    if matching["status"] == "skipped":
        warnings.append(f"Nearest-neighbor matching skipped: {matching['notes']}.")

    return {
        "estimand": {
            "name": "ATE of high 14-day early engagement on course success",
            "scale": "risk_difference",
            "treatment_column": config.treatment_column,
            "outcome_column": config.outcome_column,
            "availability_column": config.availability_column,
            "adjustment_set_source": "oulad_causal.dag.recommended_baseline_adjustment_set",
            "adjustment_columns": list(config.adjustment_columns),
        },
        "analysis": {
            "cohort_rows": int(len(cohort)),
            "analysis_rows": int(len(analysis)),
            "excluded_unavailable_or_missing_rows": int(len(cohort) - len(analysis)),
            "treated_count": int(analysis[config.treatment_column].sum()),
            "control_count": int((1 - analysis[config.treatment_column].astype(int)).sum()),
            "outcome_success_count": int(analysis[config.outcome_column].sum()),
        },
        "model_settings": {
            "seed": config.seed,
            "treatment_model": {
                "class": "sklearn.linear_model.LogisticRegression",
                "max_iter": treatment_model.max_iter,
                "random_state": config.seed,
            },
            "outcome_models": {
                str(value): {
                    "class": model.__class__.__name__,
                    "max_iter": getattr(model, "max_iter", None),
                    "random_state": getattr(model, "random_state", None),
                }
                for value, model in outcome_models.items()
            },
            "preprocessing": preprocessing_metadata,
            "weight_policy": "stabilized IPTW; no default truncation",
            "propensity_clip_for_computation_only": config.propensity_clip,
            "matching_policy": {
                "enabled": config.matching_enabled,
                "min_retention": config.matching_min_retention,
                "max_abs_logit_distance": config.matching_max_abs_logit_distance,
            },
        },
        "diagnostics": positivity,
        "warnings": warnings,
    }


def _artifact_paths(config: EstimationConfig) -> dict[str, Path]:
    return {
        "effect_estimates": config.processed_dir / EFFECT_ESTIMATES_PATH.name,
        "balance_table": config.processed_dir / BALANCE_TABLE_PATH.name,
        "metadata": config.processed_dir / ESTIMATION_METADATA_PATH.name,
        "overlap_plot": config.figures_dir / OVERLAP_PLOT_PATH.name,
        "love_plot": config.figures_dir / LOVE_PLOT_PATH.name,
        "summary": config.docs_dir / ESTIMATION_SUMMARY_PATH.name,
    }


def _skipped_matching_row(
    reason: str,
    *,
    config: EstimationConfig,
    matched_pairs: int = 0,
    retention: float = 0.0,
) -> dict[str, Any]:
    return {
        "estimator": "nearest_neighbor_matching",
        "preferred": False,
        "estimand": "ATT-style matched risk difference for treated records retained by the matching gate",
        "status": "skipped",
        "estimate": np.nan,
        "treated_mean": np.nan,
        "control_mean": np.nan,
        "std_error": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "uncertainty_method": "",
        "n": np.nan,
        "matched_pairs": matched_pairs,
        "matched_retention": retention,
        "notes": reason,
    }


def _series_summary(values: np.ndarray | pd.Series) -> dict[str, float]:
    series = pd.Series(values, dtype=float)
    return {
        "min": float(series.min()),
        "p01": float(series.quantile(0.01)),
        "p05": float(series.quantile(0.05)),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
        "max": float(series.max()),
    }


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def _weighted_variance(values: np.ndarray, weights: np.ndarray, mean: float) -> float:
    return float(np.sum(weights * np.square(values - mean)) / np.sum(weights))


def _uncertainty_fields(scores: np.ndarray, method: str, *, z_value: float = 1.96) -> dict[str, float | str]:
    scores = np.asarray(scores, dtype=float)
    estimate = float(scores.mean())
    std_error = float(scores.std(ddof=1) / np.sqrt(scores.shape[0])) if scores.shape[0] > 1 else np.nan
    return {
        "std_error": std_error,
        "ci_lower": estimate - z_value * std_error if np.isfinite(std_error) else np.nan,
        "ci_upper": estimate + z_value * std_error if np.isfinite(std_error) else np.nan,
        "uncertainty_method": method,
    }


def _logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 1e-12, 1 - 1e-12)
    return np.log(values / (1.0 - values))


def _markdown_estimate_rows(effect_estimates: pd.DataFrame) -> str:
    rows = []
    for row in effect_estimates.to_dict(orient="records"):
        estimate = "" if pd.isna(row["estimate"]) else f"{float(row['estimate']):.6f}"
        interval = (
            ""
            if pd.isna(row.get("ci_lower")) or pd.isna(row.get("ci_upper"))
            else f"[{float(row['ci_lower']):.6f}, {float(row['ci_upper']):.6f}]"
        )
        std_error = "" if pd.isna(row.get("std_error")) else f"{float(row['std_error']):.6f}"
        notes = str(row["notes"]).replace("|", "\\|")
        rows.append(
            f"| {row['estimator']} | {row['preferred']} | {row['status']} | {estimate} | {interval} | {std_error} | {notes} |"
        )
    return "\n".join(rows)
