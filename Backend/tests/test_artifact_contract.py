"""The backend's content models must accept the frontend's REAL payloads (copied
verbatim from the Lovable audit) and reject what the frontend can't render.

Run:  python -m pytest tests/test_artifact_contract.py
"""

import pytest
from pydantic import ValidationError

from Backend.app.schemas.artifact_content import (
    ChartContent,
    DiagramContent,
    QuizContent,
    parse_content,
)

# ---- payloads exactly as the frontend documented them ---------------------------------
FRONTEND_SAMPLES = {
    "summary": {"type": "summary", "body": "## Overview\n\nAlphaFold 2 predicts protein structure end-to-end...\n\n### Key points\n\n- The primary claim is stated directly..."},
    "chart-bar": {"type": "chart", "chart_type": "bar", "x_key": "chunk", "y_label": "similarity",
                  "data": [{"chunk": "[1]", "score": 0.91}, {"chunk": "[2]", "score": 0.77}, {"chunk": "other", "score": 0.34}],
                  "series": [{"key": "score", "label": "cosine similarity", "color": "var(--primary)"}]},
    "chart-line": {"type": "chart", "chart_type": "line", "x_key": "iter", "x_label": "recycling iteration", "y_label": "score",
                   "caption": "Backbone RMSD and pLDDT over recycling rounds",
                   "data": [{"iter": "0", "rmsd": 6.4, "plddt": 62}, {"iter": "1", "rmsd": 3.1, "plddt": 78},
                            {"iter": "2", "rmsd": 1.9, "plddt": 86}, {"iter": "3", "rmsd": 1.4, "plddt": 91}],
                   "series": [{"key": "rmsd", "label": "RMSD (Å)", "color": "var(--primary)"},
                              {"key": "plddt", "label": "pLDDT", "color": "var(--trace)"}]},
    "chart-pie": {"type": "chart", "chart_type": "pie", "x_key": "category", "caption": "Token distribution across categories",
                  "data": [{"category": "Structure", "count": 45}, {"category": "Sequence", "count": 30}, {"category": "Function", "count": 25}],
                  "series": [{"key": "count", "label": "Tokens", "color": "var(--primary)"}]},
    "chart-scatter": {"type": "chart", "chart_type": "scatter", "x_key": "x", "x_label": "embedding dim 1", "y_label": "embedding dim 2",
                      "data": [{"x": 1.2, "cluster_a": 3.4}, {"x": 2.5, "cluster_a": 4.1}, {"x": 1.8, "cluster_b": 1.2}],
                      "series": [{"key": "cluster_a", "label": "Cluster A", "color": "var(--primary)"},
                                 {"key": "cluster_b", "label": "Cluster B", "color": "var(--trace)"}]},
    "diagram": {"type": "diagram",
                "nodes": [{"id": "cm", "label": "Membrane capacitance Cm", "x": 0, "y": 0}, {"id": "na", "label": "Na+ conductance gNa", "x": 230, "y": -90}],
                "edges": [{"id": "e1", "source": "cm", "target": "na", "label": "dV/dt"}]},
    "keywords": {"type": "keywords", "items": [{"term": "warehouse migration", "weight": 1}, {"term": "columnar storage", "weight": 0.86}]},
    "quiz": {"type": "quiz", "questions": [{"id": "q1", "question": "What two representations does the Evoformer operate on?",
                                            "answer": "The MSA representation and the pair representation."}]},
    "mcq": {"type": "mcq", "questions": [{"id": "mq1", "question": "Which gating variables govern the sodium conductance?",
                                          "options": ["n only", "m and h", "m and n", "h only"], "correct_index": 1}]},
    "flashcards": {"type": "flashcards", "cards": [{"id": "fc1", "front": "Workspace", "back": "Top-level container grouping projects and members."}]},
}


@pytest.mark.parametrize("name", sorted(FRONTEND_SAMPLES))
def test_every_frontend_sample_is_accepted_and_round_trips_unchanged(name):
    sample = FRONTEND_SAMPLES[name]
    parsed = parse_content(sample)
    dumped = parsed.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
    dumped.pop("type", None)
    expected = {k: v for k, v in sample.items() if k != "type"}
    assert dumped == expected  # nothing added, nothing lost


def test_the_old_open_quiz_shape_still_works_and_graded_kinds_are_additive():
    q = QuizContent.model_validate(FRONTEND_SAMPLES["quiz"]).questions[0]
    assert q.kind == "open" and q.options is None
    graded = QuizContent.model_validate({"type": "quiz", "questions": [
        {"id": "a", "kind": "true_false", "question": "Evoformer uses MSAs.", "answer": "True"},
        {"id": "b", "kind": "mcq", "question": "Q?", "answer": "m and h", "options": ["n", "m and h"], "correct_index": 1},
        {"id": "c", "kind": "fill_blank", "question": "The ____ module refines frames.", "answer": "structure",
         "accepted_answers": ["structure", "Structure"]},
    ]})
    assert [q.kind for q in graded.questions] == ["true_false", "mcq", "fill_blank"]


@pytest.mark.parametrize("bad", [
    {"id": "a", "kind": "true_false", "question": "x", "answer": "Maybe"},
    {"id": "a", "kind": "mcq", "question": "x", "answer": "y", "options": ["only one"], "correct_index": 0},
    {"id": "a", "kind": "mcq", "question": "x", "answer": "y", "options": ["a", "b"], "correct_index": 5},
    {"id": "a", "kind": "fill_blank", "question": "no blank here", "answer": "y", "accepted_answers": ["y"]},
    {"id": "a", "kind": "fill_blank", "question": "two ____ and ____", "answer": "y", "accepted_answers": ["y"]},
    {"id": "a", "kind": "fill_blank", "question": "one ____", "answer": "y"},
])
def test_ungradable_quiz_questions_are_rejected(bad):
    with pytest.raises(ValidationError):
        QuizContent.model_validate({"type": "quiz", "questions": [bad]})


def test_mcq_options_must_be_distinct_and_the_answer_in_range():
    base = {"type": "mcq", "questions": [{"id": "1", "question": "q", "options": ["a", "A "], "correct_index": 0}]}
    with pytest.raises(ValidationError):
        parse_content(base)


def test_pie_has_exactly_one_series_and_series_must_have_values():
    pie = dict(FRONTEND_SAMPLES["chart-pie"])
    pie["series"] = pie["series"] + [{"key": "other", "label": "x", "color": "var(--trace)"}]
    with pytest.raises(ValidationError):
        ChartContent.model_validate(pie)
    ghost = dict(FRONTEND_SAMPLES["chart-bar"])
    ghost["series"] = [{"key": "nope", "label": "x", "color": "var(--primary)"}]
    with pytest.raises(ValidationError):
        ChartContent.model_validate(ghost)


def test_chart_rows_need_the_x_key_and_series_keys_must_be_unique():
    bar = dict(FRONTEND_SAMPLES["chart-bar"])
    bar["data"] = [{"score": 1}]
    with pytest.raises(ValidationError):
        ChartContent.model_validate(bar)
    dup = dict(FRONTEND_SAMPLES["chart-line"])
    dup["series"] = [dup["series"][0], dup["series"][0]]
    with pytest.raises(ValidationError):
        ChartContent.model_validate(dup)


def test_diagram_edges_must_point_at_existing_nodes_and_ids_be_unique():
    d = dict(FRONTEND_SAMPLES["diagram"])
    with pytest.raises(ValidationError):
        DiagramContent.model_validate({**d, "edges": [{"id": "e1", "source": "cm", "target": "ghost"}]})
    with pytest.raises(ValidationError):
        DiagramContent.model_validate({**d, "nodes": d["nodes"] + [d["nodes"][0]]})


def test_unknown_types_and_missing_discriminator_are_rejected():
    with pytest.raises(ValidationError):
        parse_content({"type": "radar", "items": []})
    with pytest.raises(ValidationError):
        parse_content({"body": "no type"})


def test_content_saved_by_the_frontend_with_extra_fields_is_kept():
    parsed = parse_content({"type": "summary", "body": "x", "future_field": 7})
    assert parsed.model_dump()["future_field"] == 7
