"""Domain-informed DAG specification utilities.

This module encodes the hand-built DAG for the OULAD early-engagement project.
The DAG is intentionally modest: it separates baseline covariates, the primary
14-day engagement treatment, post-treatment assessment behavior, and course
success without pretending that unobserved motivation or time availability are
measured in the analytic cohort.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import yaml

from oulad_causal.config import FIGURES_DIR, PROCESSED_DATA_DIR


PRIMARY_TREATMENT_COLUMN = "treatment_high_engagement_14d_median"
PRIMARY_TREATMENT_SCORE_COLUMN = "early_clicks_14d_z"
PRIMARY_OUTCOME_COLUMN = "outcome_success"

DAG_YAML_PATH = PROCESSED_DATA_DIR / "primary_dag.yaml"
DAG_FIGURE_PATH = FIGURES_DIR / "primary_dag.png"
DAG_VARIABLE_AVAILABILITY_PATH = PROCESSED_DATA_DIR / "dag_variable_availability.csv"
ANALYTIC_COHORT_PATH = PROCESSED_DATA_DIR / "oulad_analytic_cohort.parquet"


def primary_dag_spec() -> dict[str, Any]:
    """Return the canonical machine-readable hand-built DAG specification."""

    return {
        "metadata": {
            "name": "oulad_primary_early_engagement_dag",
            "description": (
                "Domain-informed DAG for the effect of high 14-day online "
                "engagement on course success in the OULAD analytic cohort."
            ),
            "primary_treatment": {
                "node": "early_engagement_14d",
                "column": PRIMARY_TREATMENT_COLUMN,
                "score_column": PRIMARY_TREATMENT_SCORE_COLUMN,
                "window_days": 14,
                "threshold": "within-module-presentation median",
            },
            "primary_outcome": {
                "node": "final_result_success",
                "column": PRIMARY_OUTCOME_COLUMN,
            },
            "latent_placeholders": [
                {
                    "id": "motivation_time_availability",
                    "label": "Motivation / time availability",
                    "observed": False,
                    "estimable": False,
                    "description": (
                        "Unmeasured student motivation, available study time, "
                        "outside support, and competing obligations may affect "
                        "early engagement and course success."
                    ),
                }
            ],
        },
        "nodes": [
            _node(
                "gender",
                "Gender",
                "confounder",
                ("baseline_gender",),
                "Pre-treatment student demographic field.",
            ),
            _node(
                "region",
                "Region",
                "confounder",
                ("baseline_region",),
                "Pre-treatment student regional context.",
            ),
            _node(
                "age_band",
                "Age band",
                "confounder",
                ("baseline_age_band",),
                "Pre-treatment student age category.",
            ),
            _node(
                "highest_education",
                "Highest education",
                "confounder",
                ("baseline_highest_education",),
                "Pre-treatment educational attainment.",
            ),
            _node(
                "imd_band",
                "IMD band",
                "confounder",
                ("baseline_imd_band",),
                "Area-level deprivation band recorded before treatment.",
            ),
            _node(
                "disability",
                "Disability",
                "confounder",
                ("baseline_disability",),
                "Pre-treatment disability indicator.",
            ),
            _node(
                "prior_attempts",
                "Prior attempts",
                "confounder",
                ("baseline_num_of_prev_attempts",),
                "Number of previous attempts before the current presentation.",
            ),
            _node(
                "studied_credits",
                "Studied credits",
                "confounder",
                ("baseline_studied_credits",),
                "Planned study load at registration.",
            ),
            _node(
                "registration_timing",
                "Registration timing",
                "confounder",
                (
                    "baseline_date_registration",
                    "baseline_missing_date_registration",
                    "baseline_registered_before_start",
                ),
                "Registration timing features available before the engagement window.",
            ),
            _node(
                "module_presentation",
                "Module presentation",
                "confounder",
                ("baseline_module_presentation", "baseline_module_presentation_length"),
                "Course and presentation context, including presentation length.",
            ),
            _node(
                "early_assessment_load",
                "Scheduled early assessment load",
                "confounder",
                (
                    "early_assessment_count_14d",
                    "early_assessment_weight_14d",
                    "early_assessment_cma_count_14d",
                    "early_assessment_tma_count_14d",
                    "early_assessment_exam_count_14d",
                ),
                "Assessment due dates and weights scheduled during days 0 through 13.",
            ),
            _node(
                "early_engagement_14d",
                "High early engagement, 14d",
                "treatment",
                (PRIMARY_TREATMENT_COLUMN, PRIMARY_TREATMENT_SCORE_COLUMN),
                "Within-module-presentation normalized VLE activity in days 0 through 13.",
            ),
            _node(
                "later_assessment_behavior",
                "Later assessment behavior",
                "mediator",
                (),
                (
                    "Post-treatment submissions, scores, and banked status; intentionally "
                    "not included in the baseline adjustment set."
                ),
            ),
            _node(
                "final_result_success",
                "Final result success",
                "outcome",
                (PRIMARY_OUTCOME_COLUMN,),
                "Pass or distinction versus fail or withdrawn.",
            ),
        ],
        "edges": [
            *_baseline_edges(
                (
                    "gender",
                    "region",
                    "age_band",
                    "highest_education",
                    "imd_band",
                    "disability",
                    "prior_attempts",
                    "studied_credits",
                    "registration_timing",
                )
            ),
            {
                "source": "module_presentation",
                "target": "early_assessment_load",
                "description": "Assessment schedules vary by module presentation.",
            },
            {
                "source": "module_presentation",
                "target": "early_engagement_14d",
                "description": "Presentation structure affects expected early activity.",
            },
            {
                "source": "module_presentation",
                "target": "final_result_success",
                "description": "Course context affects success rates.",
            },
            {
                "source": "early_assessment_load",
                "target": "early_engagement_14d",
                "description": "Scheduled early deadlines can increase early VLE activity.",
            },
            {
                "source": "early_assessment_load",
                "target": "final_result_success",
                "description": "Early workload may affect success independent of VLE clicks.",
            },
            {
                "source": "early_engagement_14d",
                "target": "later_assessment_behavior",
                "description": "Early engagement may shape later submission and score behavior.",
            },
            {
                "source": "later_assessment_behavior",
                "target": "final_result_success",
                "description": "Later submissions and scores are on the post-treatment pathway.",
            },
            {
                "source": "early_engagement_14d",
                "target": "final_result_success",
                "description": "Primary total-effect path from early engagement to success.",
            },
        ],
        "recommended_adjustment_set": list(recommended_baseline_adjustment_set()),
        "discovery_variable_list": list(discovery_variable_list()),
    }


def primary_dag_graph(spec: dict[str, Any] | None = None) -> nx.DiGraph:
    """Build a NetworkX directed graph from the primary DAG spec."""

    spec = spec or primary_dag_spec()
    graph = nx.DiGraph(name=spec["metadata"]["name"])
    for node in spec["nodes"]:
        graph.add_node(node["id"], **{key: value for key, value in node.items() if key != "id"})
    for edge in spec["edges"]:
        graph.add_edge(edge["source"], edge["target"], description=edge.get("description", ""))
    return graph


def recommended_baseline_adjustment_set() -> tuple[str, ...]:
    """Return the primary adjustment columns for the 14-day engagement effect."""

    return (
        "baseline_gender",
        "baseline_region",
        "baseline_age_band",
        "baseline_highest_education",
        "baseline_imd_band",
        "baseline_disability",
        "baseline_num_of_prev_attempts",
        "baseline_studied_credits",
        "baseline_date_registration",
        "baseline_missing_date_registration",
        "baseline_registered_before_start",
        "baseline_module_presentation",
        "baseline_module_presentation_length",
        "early_assessment_count_14d",
        "early_assessment_weight_14d",
        "early_assessment_cma_count_14d",
        "early_assessment_tma_count_14d",
        "early_assessment_exam_count_14d",
    )


def discovery_variable_list() -> tuple[str, ...]:
    """Return a reduced documented variable list for later discovery review.

    This list prepares inputs for PC/FCI/GES-style review but does not run
    discovery. Downstream discovery code should decide encoding and method
    settings explicitly before using these columns.
    """

    return (
        "baseline_gender",
        "baseline_region",
        "baseline_age_band",
        "baseline_highest_education",
        "baseline_imd_band",
        "baseline_disability",
        "baseline_num_of_prev_attempts",
        "baseline_studied_credits",
        "baseline_registered_before_start",
        "baseline_module_presentation",
        "baseline_module_presentation_length",
        "early_assessment_count_14d",
        "early_assessment_weight_14d",
        PRIMARY_TREATMENT_SCORE_COLUMN,
        PRIMARY_TREATMENT_COLUMN,
        PRIMARY_OUTCOME_COLUMN,
    )


def write_dag_artifacts(
    *,
    spec_path: str | Path = DAG_YAML_PATH,
    figure_path: str | Path = DAG_FIGURE_PATH,
    availability_path: str | Path = DAG_VARIABLE_AVAILABILITY_PATH,
    cohort_path: str | Path = ANALYTIC_COHORT_PATH,
) -> dict[str, Path]:
    """Write the DAG YAML, figure, and variable-availability table."""

    spec = primary_dag_spec()
    spec_path = Path(spec_path)
    figure_path = Path(figure_path)
    availability_path = Path(availability_path)
    cohort_path = Path(cohort_path)

    spec_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    availability_path.parent.mkdir(parents=True, exist_ok=True)

    with spec_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(spec, handle, sort_keys=False, allow_unicode=False)

    _render_dag_figure(primary_dag_graph(spec), figure_path)
    _write_variable_availability(spec, availability_path, cohort_path)

    return {
        "dag_yaml": spec_path,
        "dag_figure": figure_path,
        "variable_availability": availability_path,
    }


def _node(
    node_id: str,
    label: str,
    role: str,
    cohort_columns: tuple[str, ...],
    description: str,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "label": label,
        "role": role,
        "observed": bool(cohort_columns),
        "cohort_columns": list(cohort_columns),
        "description": description,
    }


def _baseline_edges(node_ids: tuple[str, ...]) -> list[dict[str, str]]:
    return [
        {
            "source": node_id,
            "target": target,
            "description": "Baseline variable may affect engagement and success.",
        }
        for node_id in node_ids
        for target in ("early_engagement_14d", "final_result_success")
    ]


def _render_dag_figure(graph: nx.DiGraph, output_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    positions = {
        "gender": (-2.8, 1.8),
        "region": (-2.8, 1.2),
        "age_band": (-2.8, 0.6),
        "highest_education": (-2.8, 0.0),
        "imd_band": (-2.8, -0.6),
        "disability": (-2.8, -1.2),
        "prior_attempts": (-1.2, 1.2),
        "studied_credits": (-1.2, 0.4),
        "registration_timing": (-1.2, -0.4),
        "module_presentation": (-1.2, -1.2),
        "early_assessment_load": (0.3, -1.1),
        "early_engagement_14d": (0.5, 0.4),
        "later_assessment_behavior": (2.0, 0.0),
        "final_result_success": (3.5, 0.4),
    }
    role_colors = {
        "confounder": "#D9E8F5",
        "treatment": "#F2C879",
        "mediator": "#D8C7E8",
        "outcome": "#A9D8B8",
    }

    width, height = 2700, 1500
    margin_x, margin_y = 180, 190
    x_min, x_max = -3.35, 4.15
    y_min, y_max = -1.75, 2.05
    box_w, box_h = 360, 118

    def to_canvas(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        canvas_x = margin_x + (x - x_min) / (x_max - x_min) * (width - 2 * margin_x)
        canvas_y = margin_y + (y_max - y) / (y_max - y_min) * (height - 2 * margin_y)
        return canvas_x, canvas_y

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _load_font(48, bold=True)
    note_font = _load_font(25)
    label_font = _load_font(24, bold=True)

    title = "Domain-informed DAG: early online engagement and course success"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - (title_box[2] - title_box[0])) / 2, 48), title, fill="#111827", font=title_font)

    for source, target in graph.edges:
        start = to_canvas(positions[source])
        end = to_canvas(positions[target])
        clipped_start = _clip_to_box(start, end, box_w, box_h)
        clipped_end = _clip_to_box(end, start, box_w, box_h)
        _draw_arrow(draw, clipped_start, clipped_end, fill="#5B6673", width=3)

    for node in graph.nodes:
        center = to_canvas(positions[node])
        left = center[0] - box_w / 2
        top = center[1] - box_h / 2
        right = center[0] + box_w / 2
        bottom = center[1] + box_h / 2
        fill = role_colors[graph.nodes[node]["role"]]
        draw.rounded_rectangle((left, top, right, bottom), radius=22, fill=fill, outline="#1F2933", width=3)
        _draw_centered_text(
            draw,
            graph.nodes[node]["label"],
            box=(left + 18, top + 12, right - 18, bottom - 12),
            font=label_font,
            fill="#111827",
        )

    note = (
        "Primary adjustment uses pre-treatment student, registration, module, "
        "and scheduled early-assessment variables only."
    )
    note_box = draw.textbbox((0, 0), note, font=note_font)
    draw.text(((width - (note_box[2] - note_box[0])) / 2, height - 82), note, fill="#374151", font=note_font)

    latent_note = "Latent threat documented, not estimable: motivation / time availability."
    latent_box = draw.textbbox((0, 0), latent_note, font=note_font)
    draw.text(
        ((width - (latent_box[2] - latent_box[0])) / 2, height - 44),
        latent_note,
        fill="#6B7280",
        font=note_font,
    )
    image.save(output_path)


def _load_font(size: int, *, bold: bool = False) -> Any:
    from PIL import ImageFont

    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def _clip_to_box(
    box_center: tuple[float, float],
    other: tuple[float, float],
    box_width: int,
    box_height: int,
) -> tuple[float, float]:
    dx = other[0] - box_center[0]
    dy = other[1] - box_center[1]
    if dx == 0 and dy == 0:
        return box_center
    scale_x = (box_width / 2) / abs(dx) if dx else float("inf")
    scale_y = (box_height / 2) / abs(dy) if dy else float("inf")
    scale = min(scale_x, scale_y)
    return box_center[0] + dx * scale, box_center[1] + dy * scale


def _draw_arrow(
    draw: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: str,
    width: int,
) -> None:
    import math

    draw.line((start, end), fill=fill, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 17
    left = (
        end[0] - size * math.cos(angle - math.pi / 6),
        end[1] - size * math.sin(angle - math.pi / 6),
    )
    right = (
        end[0] - size * math.cos(angle + math.pi / 6),
        end[1] - size * math.sin(angle + math.pi / 6),
    )
    draw.polygon((end, left, right), fill=fill)


def _draw_centered_text(
    draw: Any,
    text: str,
    *,
    box: tuple[float, float, float, float],
    font: Any,
    fill: str,
) -> None:
    lines = _wrap_text(draw, text, font, max_width=box[2] - box[0])
    line_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    line_heights = [line_box[3] - line_box[1] for line_box in line_boxes]
    total_height = sum(line_heights) + max(0, len(lines) - 1) * 8
    y = box[1] + ((box[3] - box[1]) - total_height) / 2
    for line, line_box, line_height in zip(lines, line_boxes, line_heights):
        line_width = line_box[2] - line_box[0]
        x = box[0] + ((box[2] - box[0]) - line_width) / 2
        draw.text((x, y), line, fill=fill, font=font)
        y += line_height + 8


def _wrap_text(draw: Any, text: str, font: Any, max_width: float) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        candidate_box = draw.textbbox((0, 0), candidate, font=font)
        if candidate_box[2] - candidate_box[0] <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)
    return lines


def _write_variable_availability(spec: dict[str, Any], output_path: Path, cohort_path: Path) -> None:
    available_columns = _read_cohort_columns(cohort_path)
    rows: list[dict[str, object]] = []

    for node in spec["nodes"]:
        columns = list(node["cohort_columns"])
        missing = [column for column in columns if column not in available_columns]
        rows.append(
            {
                "source": "dag_node",
                "name": node["id"],
                "role": node["role"],
                "cohort_columns": ", ".join(columns),
                "available": bool(columns) and not missing,
                "missing_columns": ", ".join(missing),
            }
        )

    for item in spec["metadata"]["latent_placeholders"]:
        rows.append(
            {
                "source": "latent_placeholder",
                "name": item["id"],
                "role": "unobserved",
                "cohort_columns": "",
                "available": False,
                "missing_columns": "not estimable in OULAD cohort",
            }
        )

    for column in discovery_variable_list():
        rows.append(
            {
                "source": "discovery_variable",
                "name": column,
                "role": "prepared_for_later_review",
                "cohort_columns": column,
                "available": column in available_columns,
                "missing_columns": "" if column in available_columns else column,
            }
        )

    pd.DataFrame(rows).to_csv(output_path, index=False)


def _read_cohort_columns(cohort_path: Path) -> set[str]:
    if not cohort_path.exists():
        return set()
    return set(pd.read_parquet(cohort_path).columns)
