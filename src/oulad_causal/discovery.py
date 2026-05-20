"""Bounded causal discovery workflow for the OULAD project.

The routines in this module are intentionally conservative. They run PC, FCI,
and GES on a small, documented, discretized variable set and write artifacts for
comparison with the hand-built DAG. The output is exploratory support only; it
does not replace the identification plan or primary adjustment set.
"""

from __future__ import annotations

from dataclasses import dataclass
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from oulad_causal.config import DOCS_DIR, FIGURES_DIR, PROCESSED_DATA_DIR
from oulad_causal.dag import ANALYTIC_COHORT_PATH, primary_dag_spec


DISCOVERY_VARIABLES: tuple[str, ...] = (
    "baseline_gender",
    "baseline_age_band",
    "baseline_highest_education",
    "baseline_imd_band",
    "baseline_disability",
    "baseline_num_of_prev_attempts",
    "baseline_studied_credits",
    "baseline_registered_before_start",
    "baseline_module_presentation_length",
    "early_assessment_weight_14d",
    "treatment_high_engagement_14d_median",
    "outcome_success",
)

PRIMARY_TREATMENT_COLUMN = "treatment_high_engagement_14d_median"
PRIMARY_OUTCOME_COLUMN = "outcome_success"

DISCOVERY_ANALYSIS_DATA_PATH = PROCESSED_DATA_DIR / "discovery_analysis_data.parquet"
DISCOVERY_PREPROCESSING_MAP_PATH = PROCESSED_DATA_DIR / "discovery_preprocessing_map.json"
DISCOVERY_COMBINED_EDGES_PATH = PROCESSED_DATA_DIR / "discovery_edges.csv"
DISCOVERY_STABILITY_PATH = PROCESSED_DATA_DIR / "discovery_stability_edges.csv"
DISCOVERY_COMPARISON_PATH = PROCESSED_DATA_DIR / "discovery_hand_dag_comparison.csv"
DISCOVERY_METADATA_PATH = PROCESSED_DATA_DIR / "discovery_run_metadata.json"
DISCOVERY_SUMMARY_PATH = DOCS_DIR / "discovery_summary.md"

AGE_BAND_ORDER = {"0-35": 0, "35-55": 1, "55<=": 2}
EDUCATION_ORDER = {
    "No Formal quals": 0,
    "Lower Than A Level": 1,
    "A Level or Equivalent": 2,
    "HE Qualification": 3,
    "Post Graduate Qualification": 4,
}
BINARY_01_COLUMNS = (
    "baseline_registered_before_start",
    PRIMARY_TREATMENT_COLUMN,
    PRIMARY_OUTCOME_COLUMN,
)
NOMINAL_COLUMNS = ("baseline_gender", "baseline_disability")
LOW_CARDINALITY_NUMERIC_COLUMNS = (
    "baseline_num_of_prev_attempts",
    "baseline_module_presentation_length",
    "early_assessment_weight_14d",
)
MISSING_CATEGORY_LABELS = {"__MISSING__", "<MISSING>", "", "nan", "None", "NA"}

COLUMN_TO_DAG_NODE = {
    "baseline_gender": "gender",
    "baseline_age_band": "age_band",
    "baseline_highest_education": "highest_education",
    "baseline_imd_band": "imd_band",
    "baseline_disability": "disability",
    "baseline_num_of_prev_attempts": "prior_attempts",
    "baseline_studied_credits": "studied_credits",
    "baseline_registered_before_start": "registration_timing",
    "baseline_module_presentation_length": "module_presentation",
    "early_assessment_weight_14d": "early_assessment_load",
    PRIMARY_TREATMENT_COLUMN: "early_engagement_14d",
    PRIMARY_OUTCOME_COLUMN: "final_result_success",
}

EDGE_COLUMNS = (
    "method",
    "source",
    "target",
    "endpoint_source",
    "endpoint_target",
    "edge_type",
    "directed_source",
    "directed_target",
    "is_directed",
    "skeleton_key",
)


@dataclass(frozen=True)
class DiscoveryConfig:
    """Runtime configuration for the discovery pipeline."""

    cohort_path: Path = ANALYTIC_COHORT_PATH
    processed_dir: Path = PROCESSED_DATA_DIR
    figures_dir: Path = FIGURES_DIR
    docs_dir: Path = DOCS_DIR
    alpha: float = 0.01
    seed: int = 245
    stability_reps: int = 20
    stability_sample_size: int = 3000
    skip_fci: bool = False
    fci_stability_max_seconds: float = 10.0


def select_discovery_variables() -> tuple[str, ...]:
    """Return the bounded, documented discovery variable set."""

    return DISCOVERY_VARIABLES


def preprocess_discovery_data(
    cohort: pd.DataFrame,
    variables: tuple[str, ...] = DISCOVERY_VARIABLES,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Discretize mixed OULAD variables and return analysis data plus metadata."""

    missing = [column for column in variables if column not in cohort.columns]
    if missing:
        raise KeyError(f"Missing discovery columns in cohort: {missing}")

    encoded = pd.DataFrame(index=cohort.index)
    metadata: dict[str, Any] = {
        "variables": list(variables),
        "row_count": int(cohort.shape[0]),
        "preprocessing_note": (
            "Causal-learn PC/FCI/GES are run on a reduced discrete matrix. "
            "Ordinal fields use explicit orderings, nominal/binary fields use "
            "documented category codes, and studied credits are quantile-binned."
        ),
        "columns": {},
    }

    for column in variables:
        series = cohort[column]
        missing_count = int(series.isna().sum())
        if column == "baseline_age_band":
            encoded[column] = _map_with_metadata(series, AGE_BAND_ORDER, missing_code=-1)
            column_meta = _mapping_metadata("explicit_ordinal", AGE_BAND_ORDER, missing_count)
        elif column == "baseline_highest_education":
            encoded[column] = _map_with_metadata(series, EDUCATION_ORDER, missing_code=-1)
            column_meta = _mapping_metadata("explicit_ordinal", EDUCATION_ORDER, missing_count)
        elif column == "baseline_imd_band":
            encoded[column], mapping = _encode_imd_band(series)
            column_meta = _mapping_metadata("explicit_ordinal_imd_lower_bound", mapping, missing_count)
        elif column == "baseline_studied_credits":
            encoded[column], column_meta = _quantile_bin(series, q=4, column=column)
        elif column in NOMINAL_COLUMNS:
            encoded[column], mapping = _category_codes(series)
            column_meta = _mapping_metadata("nominal_category_codes", mapping, missing_count)
        elif column in BINARY_01_COLUMNS:
            numeric = pd.to_numeric(series, errors="coerce")
            encoded[column] = numeric.fillna(-1).astype("int64")
            values = sorted(int(value) for value in encoded[column].dropna().unique())
            column_meta = {
                "preprocessing": "binary_integer_state",
                "mapping": {str(value): value for value in values},
                "missing_count": missing_count,
                "missing_code": -1,
            }
        elif column in LOW_CARDINALITY_NUMERIC_COLUMNS:
            encoded[column], mapping = _ordered_numeric_codes(series)
            column_meta = _mapping_metadata("ordered_numeric_state_codes", mapping, missing_count)
        else:
            raise ValueError(f"No discovery preprocessing rule defined for {column!r}.")

        metadata["columns"][column] = column_meta

    encoded = encoded.astype("int64")
    if not np.isfinite(encoded.to_numpy(dtype=float)).all():
        raise ValueError("Discovery preprocessing produced non-finite values.")
    return encoded, metadata


def run_discovery_pipeline(config: DiscoveryConfig | None = None) -> dict[str, Path]:
    """Run the complete causal discovery pipeline and write all artifacts."""

    config = config or DiscoveryConfig()
    _prepare_runtime_cache()
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    config.docs_dir.mkdir(parents=True, exist_ok=True)

    variables = select_discovery_variables()
    cohort = pd.read_parquet(config.cohort_path, columns=list(variables))
    analysis_data, preprocessing = preprocess_discovery_data(cohort, variables)

    paths = _artifact_paths(config)
    analysis_data.to_parquet(paths["analysis_data"], index=False)
    _write_json(paths["preprocessing_map"], preprocessing)

    metadata: dict[str, Any] = {
        "cohort_path": str(config.cohort_path),
        "row_count": int(analysis_data.shape[0]),
        "variables": list(variables),
        "alpha": config.alpha,
        "seed": config.seed,
        "stability_reps": config.stability_reps,
        "stability_sample_size": config.stability_sample_size,
        "methods": {},
    }

    edge_frames: list[pd.DataFrame] = []
    method_results: dict[str, Any] = {}
    for method in ("pc", "fci", "ges"):
        if method == "fci" and config.skip_fci:
            metadata["methods"][method] = {"status": "skipped", "reason": "skip_fci=True"}
            _write_empty_edges(paths[f"edges_{method}"], method)
            continue
        result = _run_method_safely(method, analysis_data, variables, config)
        metadata["methods"][method] = result["metadata"]
        if result["metadata"]["status"] != "success":
            _write_empty_edges(paths[f"edges_{method}"], method)
            continue
        method_results[method] = result
        edge_df = result["edges"]
        edge_df.to_csv(paths[f"edges_{method}"], index=False)
        result["adjacency"].to_csv(paths[f"adjacency_{method}"])
        render_discovery_graph(edge_df, variables, paths[f"figure_{method}"], title=f"{method.upper()} discovery graph")
        edge_frames.append(edge_df)

    combined_edges = pd.concat(edge_frames, ignore_index=True) if edge_frames else _empty_edges()
    combined_edges.to_csv(paths["combined_edges"], index=False)

    stability = run_stability_checks(analysis_data, variables, config, metadata)
    stability.to_csv(paths["stability_edges"], index=False)

    comparison = compare_with_hand_built_dag(combined_edges, variables)
    comparison.to_csv(paths["hand_dag_comparison"], index=False)

    metadata["artifacts"] = {name: str(path) for name, path in paths.items()}
    _write_json(paths["metadata"], metadata)
    write_discovery_summary(
        summary_path=paths["summary"],
        preprocessing_metadata=preprocessing,
        run_metadata=metadata,
        comparison=comparison,
        stability=stability,
    )

    return paths


def run_stability_checks(
    analysis_data: pd.DataFrame,
    variables: tuple[str, ...],
    config: DiscoveryConfig,
    run_metadata: dict[str, Any],
) -> pd.DataFrame:
    """Run repeated-subsample edge stability checks for feasible methods."""

    methods = ["pc", "ges"]
    fci_status = run_metadata["methods"].get("fci", {})
    if (
        fci_status.get("status") == "success"
        and fci_status.get("seconds", float("inf")) <= config.fci_stability_max_seconds
    ):
        methods.append("fci")
    elif fci_status.get("status") == "success":
        run_metadata["methods"]["fci"]["stability_status"] = "skipped_slow_primary_run"
    else:
        run_metadata["methods"].setdefault("fci", {})["stability_status"] = "skipped_no_successful_primary_run"

    rng = np.random.default_rng(config.seed)
    records: list[dict[str, Any]] = []
    sample_size = min(config.stability_sample_size, analysis_data.shape[0])

    for method in methods:
        successes = 0
        failures = 0
        observed_edges: list[pd.DataFrame] = []
        for rep in range(config.stability_reps):
            sampled_index = rng.choice(analysis_data.index.to_numpy(), size=sample_size, replace=False)
            sample = analysis_data.loc[sampled_index].reset_index(drop=True)
            result = _run_method_safely(method, sample, variables, config, stability_run=True)
            if result["metadata"]["status"] != "success":
                failures += 1
                continue
            successes += 1
            edges = result["edges"].copy()
            edges["rep"] = rep
            observed_edges.append(edges)

        if not observed_edges:
            records.append(
                {
                    "method": method,
                    "var_a": "",
                    "var_b": "",
                    "edge_frequency": 0.0,
                    "directed_a_to_b_frequency": 0.0,
                    "directed_b_to_a_frequency": 0.0,
                    "repetitions_attempted": config.stability_reps,
                    "repetitions_succeeded": successes,
                    "failures": failures,
                }
            )
            continue

        method_edges = pd.concat(observed_edges, ignore_index=True)
        for skeleton_key, group in method_edges.groupby("skeleton_key"):
            var_a, var_b = skeleton_key.split("||", maxsplit=1)
            directed_a_to_b = int(((group["directed_source"] == var_a) & (group["directed_target"] == var_b)).sum())
            directed_b_to_a = int(((group["directed_source"] == var_b) & (group["directed_target"] == var_a)).sum())
            records.append(
                {
                    "method": method,
                    "var_a": var_a,
                    "var_b": var_b,
                    "edge_frequency": float(group["rep"].nunique() / max(successes, 1)),
                    "directed_a_to_b_frequency": float(directed_a_to_b / max(successes, 1)),
                    "directed_b_to_a_frequency": float(directed_b_to_a / max(successes, 1)),
                    "repetitions_attempted": config.stability_reps,
                    "repetitions_succeeded": successes,
                    "failures": failures,
                }
            )

    return pd.DataFrame.from_records(records)


def compare_with_hand_built_dag(
    discovered_edges: pd.DataFrame,
    variables: tuple[str, ...] = DISCOVERY_VARIABLES,
) -> pd.DataFrame:
    """Compare discovered edges with the hand-built DAG among selected variables."""

    hand_edges = _hand_dag_edges_for_variables(variables)
    hand_skeletons = {_skeleton_key(source, target) for source, target in hand_edges}
    rows: list[dict[str, Any]] = []

    for row in discovered_edges.to_dict(orient="records"):
        source = row.get("source", "")
        target = row.get("target", "")
        if not source or not target:
            continue
        skeleton_key = _skeleton_key(source, target)
        hand_direction = _hand_direction_for_pair(source, target, hand_edges)
        rows.append(
            {
                **row,
                "source_dag_node": COLUMN_TO_DAG_NODE.get(source, ""),
                "target_dag_node": COLUMN_TO_DAG_NODE.get(target, ""),
                "source_role": _dag_role_for_column(source),
                "target_role": _dag_role_for_column(target),
                "in_hand_skeleton": skeleton_key in hand_skeletons,
                "hand_direction": hand_direction,
                "in_hand_directed": (
                    bool(row.get("is_directed"))
                    and (row.get("directed_source"), row.get("directed_target")) in hand_edges
                ),
            }
        )

    discovered_skeletons = set(discovered_edges["skeleton_key"]) if "skeleton_key" in discovered_edges else set()
    for source, target in sorted(hand_edges):
        skeleton_key = _skeleton_key(source, target)
        if skeleton_key in discovered_skeletons:
            continue
        rows.append(
            {
                "method": "hand_dag_missing_from_discovery",
                "source": source,
                "target": target,
                "endpoint_source": "TAIL",
                "endpoint_target": "ARROW",
                "edge_type": f"{source} --> {target}",
                "directed_source": source,
                "directed_target": target,
                "is_directed": True,
                "skeleton_key": skeleton_key,
                "source_dag_node": COLUMN_TO_DAG_NODE.get(source, ""),
                "target_dag_node": COLUMN_TO_DAG_NODE.get(target, ""),
                "source_role": _dag_role_for_column(source),
                "target_role": _dag_role_for_column(target),
                "in_hand_skeleton": True,
                "hand_direction": f"{source} -> {target}",
                "in_hand_directed": False,
            }
        )

    return pd.DataFrame.from_records(rows)


def write_discovery_summary(
    *,
    summary_path: Path,
    preprocessing_metadata: dict[str, Any],
    run_metadata: dict[str, Any],
    comparison: pd.DataFrame,
    stability: pd.DataFrame,
) -> None:
    """Write a markdown summary grounded in saved discovery artifacts."""

    stable = stability[stability["edge_frequency"] >= 0.6].copy() if not stability.empty else pd.DataFrame()
    supported = comparison[
        (comparison["method"] != "hand_dag_missing_from_discovery")
        & (comparison["in_hand_skeleton"].astype(bool))
    ].copy() if not comparison.empty else pd.DataFrame()
    missing = comparison[comparison["method"] == "hand_dag_missing_from_discovery"].copy() if not comparison.empty else pd.DataFrame()
    noisy = stability[(stability["edge_frequency"] > 0) & (stability["edge_frequency"] < 0.6)].copy() if not stability.empty else pd.DataFrame()

    lines = [
        "# Causal Discovery Summary",
        "",
        "This file is generated from the saved causal discovery artifacts. Discovery is exploratory support for the hand-built DAG, not a replacement for the identification plan.",
        "",
        "## Scope and preprocessing",
        "",
        f"- Analytic rows: {run_metadata.get('row_count', 'unknown')}.",
        f"- Variables: {', '.join(preprocessing_metadata.get('variables', []))}.",
        "- Mixed variables were discretized before discovery: ordinal mappings for age, education, and IMD; category codes for nominal/binary fields; quantile bins for studied credits; ordered state codes for low-cardinality numeric fields.",
        f"- PC and FCI used chi-square conditional independence tests with alpha {run_metadata.get('alpha')}; GES used BDeu scoring on the same discrete matrix.",
        "",
        "## Method status",
        "",
        "| method | status | seconds | edges | note |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for method, meta in run_metadata.get("methods", {}).items():
        lines.append(
            "| {method} | {status} | {seconds} | {edges} | {note} |".format(
                method=method,
                status=meta.get("status", ""),
                seconds=_format_seconds(meta.get("seconds")),
                edges=meta.get("edge_count", ""),
                note=meta.get("reason", meta.get("error", meta.get("stability_status", ""))),
            )
        )

    lines.extend(
        [
            "",
            "## What discovery supports",
            "",
        ]
    )
    if supported.empty:
        lines.append("- No discovered skeleton edge among the selected variables matched the hand-built DAG in the completed methods.")
    else:
        support_counts = (
            supported.groupby("skeleton_key", as_index=False)
            .agg(methods=("method", lambda values: ", ".join(sorted(set(values)))))
            .sort_values("skeleton_key")
        )
        for row in support_counts.head(12).to_dict(orient="records"):
            var_a, var_b = row["skeleton_key"].split("||", maxsplit=1)
            stability_note = _stability_note(stable, row["skeleton_key"])
            lines.append(
                f"- `{var_a}` -- `{var_b}` appeared in {row['methods']} and matches the hand-built DAG skeleton{stability_note}."
            )

    lines.extend(
        [
            "",
            "## What discovery does not establish",
            "",
            "- It does not establish causal truth, because the OULAD cohort remains observational and motivation, time availability, outside support, and similar constructs are unmeasured.",
            "- It does not justify adjusting for post-treatment variables or changing the primary adjustment set.",
            "- It does not make orientations definitive; PC, FCI, and GES orientations here are treated as exploratory, especially for undirected, circle, or bidirected endpoints.",
            "- It does not remove sensitivity to preprocessing choices; the algorithms used a reduced, discretized representation of mixed variables.",
            "",
            "## Noisy or unstable findings",
            "",
        ]
    )
    if noisy.empty:
        lines.append("- No nonzero low-stability edges were recorded in the repeated-subsample checks.")
    else:
        for row in noisy.sort_values("edge_frequency", ascending=False).head(12).to_dict(orient="records"):
            lines.append(
                f"- `{row['var_a']}` -- `{row['var_b']}` in {row['method']} had edge frequency {row['edge_frequency']:.2f}."
            )

    lines.extend(["", "## Hand-built DAG edges not recovered", ""])
    if missing.empty:
        lines.append("- Every selected hand-built DAG skeleton edge appeared in at least one completed discovery output.")
    else:
        for row in missing.head(16).to_dict(orient="records"):
            lines.append(f"- `{row['source']}` -> `{row['target']}` was in the hand-built DAG but not recovered as a skeleton edge.")

    lines.extend(
        [
            "",
            "## Artifact inventory",
            "",
        ]
    )
    for name, path in run_metadata.get("artifacts", {}).items():
        lines.append(f"- `{name}`: `{path}`")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_discovery_graph(edge_df: pd.DataFrame, variables: tuple[str, ...], output_path: Path, *, title: str) -> None:
    """Render a compact discovery graph figure."""

    _prepare_runtime_cache()
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    graph = nx.DiGraph()
    graph.add_nodes_from(variables)
    undirected_edges = []
    directed_edges = []
    for row in edge_df.to_dict(orient="records"):
        if row.get("is_directed") and row.get("directed_source") and row.get("directed_target"):
            graph.add_edge(row["directed_source"], row["directed_target"])
            directed_edges.append((row["directed_source"], row["directed_target"]))
        else:
            source = row.get("source")
            target = row.get("target")
            if source and target:
                graph.add_edge(source, target)
                undirected_edges.append((source, target))

    positions = nx.spring_layout(graph, seed=245, k=1.5)
    role_colors = {
        "confounder": "#D9E8F5",
        "treatment": "#F2C879",
        "outcome": "#A9D8B8",
    }
    colors = [role_colors.get(_dag_role_for_column(node), "#E5E7EB") for node in graph.nodes]

    plt.figure(figsize=(14, 9))
    nx.draw_networkx_nodes(graph, positions, node_color=colors, node_size=1700, edgecolors="#374151", linewidths=1.2)
    nx.draw_networkx_labels(graph, positions, font_size=8)
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=directed_edges,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=16,
        width=1.8,
        edge_color="#374151",
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=undirected_edges,
        arrows=False,
        width=1.4,
        style="dashed",
        edge_color="#6B7280",
    )
    plt.title(title)
    plt.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _run_method_safely(
    method: str,
    analysis_data: pd.DataFrame,
    variables: tuple[str, ...],
    config: DiscoveryConfig,
    *,
    stability_run: bool = False,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        graph = _run_method(method, analysis_data, variables, config)
        edges = _edges_from_graph(method, graph)
        adjacency = pd.DataFrame(np.asarray(graph.graph), index=variables, columns=variables)
        seconds = time.perf_counter() - start
        return {
            "graph": graph,
            "edges": edges,
            "adjacency": adjacency,
            "metadata": {
                "status": "success",
                "seconds": round(seconds, 3),
                "edge_count": int(edges.shape[0]),
                "stability_run": stability_run,
            },
        }
    except Exception as exc:  # pragma: no cover - exercised by failure-handling tests via monkeypatch
        seconds = time.perf_counter() - start
        return {
            "metadata": {
                "status": "failed",
                "seconds": round(seconds, 3),
                "error": f"{type(exc).__name__}: {exc}",
                "stability_run": stability_run,
            }
        }


def _run_method(
    method: str,
    analysis_data: pd.DataFrame,
    variables: tuple[str, ...],
    config: DiscoveryConfig,
) -> Any:
    _prepare_runtime_cache()
    data = analysis_data.to_numpy(dtype="int64")
    if method == "pc":
        from causallearn.search.ConstraintBased.PC import pc

        return pc(
            data,
            alpha=config.alpha,
            indep_test="chisq",
            stable=True,
            show_progress=False,
            node_names=list(variables),
        ).G
    if method == "fci":
        from causallearn.search.ConstraintBased.FCI import fci

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            graph, _ = fci(
                data,
                independence_test_method="chisq",
                alpha=config.alpha,
                depth=3,
                max_path_length=3,
                verbose=False,
                show_progress=False,
                node_names=list(variables),
            )
        return graph
    if method == "ges":
        from causallearn.search.ScoreBased.GES import ges

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = ges(
                data,
                score_func="local_score_BDeu",
                maxP=3,
                node_names=list(variables),
            )
        return result["G"]
    raise ValueError(f"Unknown discovery method: {method}")


def _edges_from_graph(method: str, graph: Any) -> pd.DataFrame:
    rows = []
    for edge in graph.get_graph_edges():
        node1 = str(edge.get_node1())
        node2 = str(edge.get_node2())
        endpoint1 = _endpoint_name(edge.get_endpoint1())
        endpoint2 = _endpoint_name(edge.get_endpoint2())
        directed_source = ""
        directed_target = ""
        if endpoint1 == "TAIL" and endpoint2 == "ARROW":
            directed_source, directed_target = node1, node2
        elif endpoint1 == "ARROW" and endpoint2 == "TAIL":
            directed_source, directed_target = node2, node1

        rows.append(
            {
                "method": method,
                "source": node1,
                "target": node2,
                "endpoint_source": endpoint1,
                "endpoint_target": endpoint2,
                "edge_type": str(edge),
                "directed_source": directed_source,
                "directed_target": directed_target,
                "is_directed": bool(directed_source and directed_target),
                "skeleton_key": _skeleton_key(node1, node2),
            }
        )
    return pd.DataFrame(rows, columns=EDGE_COLUMNS)


def _artifact_paths(config: DiscoveryConfig) -> dict[str, Path]:
    return {
        "analysis_data": config.processed_dir / "discovery_analysis_data.parquet",
        "preprocessing_map": config.processed_dir / "discovery_preprocessing_map.json",
        "combined_edges": config.processed_dir / "discovery_edges.csv",
        "edges_pc": config.processed_dir / "discovery_edges_pc.csv",
        "edges_fci": config.processed_dir / "discovery_edges_fci.csv",
        "edges_ges": config.processed_dir / "discovery_edges_ges.csv",
        "adjacency_pc": config.processed_dir / "discovery_adjacency_pc.csv",
        "adjacency_fci": config.processed_dir / "discovery_adjacency_fci.csv",
        "adjacency_ges": config.processed_dir / "discovery_adjacency_ges.csv",
        "figure_pc": config.figures_dir / "discovery_pc.png",
        "figure_fci": config.figures_dir / "discovery_fci.png",
        "figure_ges": config.figures_dir / "discovery_ges.png",
        "stability_edges": config.processed_dir / "discovery_stability_edges.csv",
        "hand_dag_comparison": config.processed_dir / "discovery_hand_dag_comparison.csv",
        "metadata": config.processed_dir / "discovery_run_metadata.json",
        "summary": config.docs_dir / "discovery_summary.md",
    }


def _category_codes(series: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    values = series.astype("string").fillna("__MISSING__")
    categories = sorted(values.unique().tolist())
    mapping = {category: index for index, category in enumerate(categories)}
    return values.map(mapping).astype("int64"), mapping


def _ordered_numeric_codes(series: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    numeric = pd.to_numeric(series, errors="coerce")
    values = sorted(float(value) for value in numeric.dropna().unique())
    mapping = {str(_clean_number(value)): index for index, value in enumerate(values)}
    encoded = numeric.map({_clean_number(value): index for index, value in enumerate(values)}).fillna(-1)
    return encoded.astype("int64"), mapping


def _map_with_metadata(series: pd.Series, mapping: dict[str, int], *, missing_code: int) -> pd.Series:
    values = series.astype("string").fillna("__MISSING__")
    unknown = sorted(set(values.unique()) - set(mapping) - MISSING_CATEGORY_LABELS)
    if unknown:
        raise ValueError(f"Unexpected categories for ordinal mapping: {unknown}")
    normalized = values.where(~values.isin(MISSING_CATEGORY_LABELS), "__MISSING__")
    return normalized.map(mapping).fillna(missing_code).astype("int64")


def _encode_imd_band(series: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    values = series.astype("string").fillna("__MISSING__")
    mapping: dict[str, int] = {}
    for value in sorted(values.unique()):
        mapping[value] = _imd_code(value)
    return values.map(mapping).fillna(-1).astype("int64"), mapping


def _imd_code(value: str) -> int:
    if value in MISSING_CATEGORY_LABELS:
        return -1
    lower = value.split("-", maxsplit=1)[0].strip()
    try:
        return int(float(lower) // 10)
    except ValueError as exc:
        raise ValueError(f"Unexpected IMD band value {value!r}") from exc


def _quantile_bin(series: pd.Series, *, q: int, column: str) -> tuple[pd.Series, dict[str, Any]]:
    numeric = pd.to_numeric(series, errors="coerce")
    binned = pd.qcut(numeric, q=q, labels=False, duplicates="drop")
    encoded = pd.Series(binned, index=series.index).fillna(-1).astype("int64")
    intervals = pd.qcut(numeric, q=q, duplicates="drop").cat.categories
    return encoded, {
        "preprocessing": "quantile_bins",
        "requested_bins": q,
        "actual_bins": int(len(intervals)),
        "bin_intervals": [str(interval) for interval in intervals],
        "missing_count": int(series.isna().sum()),
        "missing_code": -1,
        "column": column,
    }


def _mapping_metadata(preprocessing: str, mapping: dict[Any, int], missing_count: int) -> dict[str, Any]:
    return {
        "preprocessing": preprocessing,
        "mapping": {str(key): int(value) for key, value in mapping.items()},
        "missing_count": missing_count,
        "missing_code": -1,
    }


def _hand_dag_edges_for_variables(variables: tuple[str, ...]) -> set[tuple[str, str]]:
    node_to_columns: dict[str, list[str]] = {}
    for column in variables:
        node_to_columns.setdefault(COLUMN_TO_DAG_NODE[column], []).append(column)

    edges: set[tuple[str, str]] = set()
    for edge in primary_dag_spec()["edges"]:
        source_columns = node_to_columns.get(edge["source"], [])
        target_columns = node_to_columns.get(edge["target"], [])
        for source in source_columns:
            for target in target_columns:
                edges.add((source, target))
    return edges


def _hand_direction_for_pair(source: str, target: str, hand_edges: set[tuple[str, str]]) -> str:
    if (source, target) in hand_edges:
        return f"{source} -> {target}"
    if (target, source) in hand_edges:
        return f"{target} -> {source}"
    return ""


def _dag_role_for_column(column: str) -> str:
    node_id = COLUMN_TO_DAG_NODE.get(column)
    if node_id is None:
        return ""
    for node in primary_dag_spec()["nodes"]:
        if node["id"] == node_id:
            return str(node["role"])
    return ""


def _skeleton_key(source: str, target: str) -> str:
    return "||".join(sorted((source, target)))


def _endpoint_name(endpoint: Any) -> str:
    return str(getattr(endpoint, "name", endpoint))


def _format_seconds(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _stability_note(stable: pd.DataFrame, skeleton_key: str) -> str:
    if stable.empty:
        return ""
    matches = stable[(stable["var_a"] + "||" + stable["var_b"]) == skeleton_key]
    if matches.empty:
        return ""
    best = matches.sort_values("edge_frequency", ascending=False).iloc[0]
    return f"; repeated-subsample {best['method']} frequency {best['edge_frequency']:.2f}"


def _empty_edges() -> pd.DataFrame:
    return pd.DataFrame(columns=EDGE_COLUMNS)


def _write_empty_edges(path: Path, method: str) -> None:
    edges = _empty_edges()
    edges["method"] = edges.get("method", pd.Series(dtype=object))
    path.parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(path, index=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def _clean_number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else value


def _prepare_runtime_cache() -> None:
    cache_dir = Path(tempfile.gettempdir()) / "oulad_causal_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
