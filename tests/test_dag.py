"""Tests for the hand-built DAG specification."""

import networkx as nx

from oulad_causal import dag


REQUIRED_NODE_IDS = {
    "age_band",
    "highest_education",
    "imd_band",
    "disability",
    "prior_attempts",
    "studied_credits",
    "registration_timing",
    "module_presentation",
    "early_assessment_load",
    "early_engagement_14d",
    "later_assessment_behavior",
    "final_result_success",
}


def test_primary_dag_spec_contains_required_nodes_and_roles() -> None:
    spec = dag.primary_dag_spec()
    nodes = {node["id"]: node for node in spec["nodes"]}

    assert REQUIRED_NODE_IDS.issubset(nodes)
    assert nodes["early_engagement_14d"]["role"] == "treatment"
    assert nodes["later_assessment_behavior"]["role"] == "mediator"
    assert nodes["final_result_success"]["role"] == "outcome"
    assert spec["metadata"]["latent_placeholders"][0]["id"] == "motivation_time_availability"
    assert spec["metadata"]["latent_placeholders"][0]["observed"] is False


def test_primary_dag_graph_is_acyclic() -> None:
    graph = dag.primary_dag_graph()

    assert nx.is_directed_acyclic_graph(graph)


def test_recommended_adjustment_set_uses_baseline_or_scheduled_load_columns() -> None:
    adjustment_set = dag.recommended_baseline_adjustment_set()

    assert adjustment_set
    assert all(column.startswith(("baseline_", "early_assessment_")) for column in adjustment_set)
    assert "early_assessment_count_14d" in adjustment_set
    assert "early_assessment_weight_14d" in adjustment_set


def test_recommended_adjustment_set_excludes_treatment_outcome_and_post_treatment_columns() -> None:
    adjustment_set = set(dag.recommended_baseline_adjustment_set())
    forbidden = {
        dag.PRIMARY_TREATMENT_COLUMN,
        dag.PRIMARY_TREATMENT_SCORE_COLUMN,
        dag.PRIMARY_OUTCOME_COLUMN,
        "outcome_withdrawn",
        "final_result",
        "date_submitted",
        "score",
        "is_banked",
        "later_assessment_behavior",
    }

    assert adjustment_set.isdisjoint(forbidden)
    assert not any(column.startswith("early_clicks_") for column in adjustment_set)


def test_discovery_variable_list_is_available_in_processed_cohort_metadata() -> None:
    available_columns = {
        *dag.recommended_baseline_adjustment_set(),
        dag.PRIMARY_TREATMENT_SCORE_COLUMN,
        dag.PRIMARY_TREATMENT_COLUMN,
        dag.PRIMARY_OUTCOME_COLUMN,
    }

    assert set(dag.discovery_variable_list()).issubset(available_columns)
