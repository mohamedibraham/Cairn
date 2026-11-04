"""The generic engine: profile gate, forced generation, grounded-failure mismatches, and all
seven types end to end.   Run:  python -m pytest tests/test_artifact_engine.py"""

import pytest

from Backend.app.enums import ArtifactType as T
from Backend.app.llm.providers.fake import FakeLLMClient
from Backend.app.schemas.artifact_content import parse_content
from Backend.app.services.artifacts.base import ArtifactGenerationError, ArtifactOptions, ArtifactResult, Mismatch
from Backend.app.services.artifacts.engine import generate_from_chunks
from Backend.app.services.artifacts.profile import MISMATCH_REASON, compute_profile
from test_artifact_analysis_types import FLOW, GOOD_EDGES, GOOD_NODES, RMSD, T1, chart_draft, diagram_draft
from test_artifact_grounding_profile import PROSE
from test_artifact_study_types import GOOD_CARDS, GOOD_MCQ, GOOD_QUIZ, P1, P2, P3, chunk

RICH = [chunk(1, P1), chunk(2, P2), chunk(3, P3), chunk(4, T1), chunk(5, FLOW)]
PROSE_ONLY = [chunk(1, PROSE)]  # prose and terminology, but no numbers and no flow

KEYWORDS = {"items": [{"term": t, "definition": "d", "passage": 1} for t in
                      ("Evoformer", "pair representation", "invariant point attention", "recycling", "pLDDT", "self distillation")]}
SUMMARY = ("## Overview\n\nAlphaFold predicts protein structure with an Evoformer stack and refines it by recycling. "
           "RMSD fell from 6.4 to 1.4 over four iterations.\n\n## Key points\n\n- The structure module refines backbone frames.\n"
           "- Recycling feeds the predicted structure back into the network.\n- pLDDT reports the expected accuracy per residue.")

DRAFTS = {
    T.KEYWORDS: [KEYWORDS], T.FLASHCARDS: [{"cards": GOOD_CARDS}], T.MCQ: [{"questions": GOOD_MCQ}],
    T.QUIZ: [{"questions": GOOD_QUIZ}], T.SUMMARY: [SUMMARY], T.CHART: [chart_draft(RMSD)],
    T.DIAGRAM: [diagram_draft(GOOD_NODES, GOOD_EDGES)],
}


def test_the_rich_fixture_really_supports_every_type():
    profile = compute_profile(RICH)
    assert profile.has_numeric_data and profile.has_relationships and profile.has_distinct_terms and profile.has_prose


@pytest.mark.parametrize("artifact_type", list(T))
def test_every_type_is_generated_end_to_end_and_satisfies_the_frontend_contract(artifact_type):
    fake = FakeLLMClient(DRAFTS[artifact_type])
    result = generate_from_chunks(RICH, artifact_type, ArtifactOptions(), fake)
    assert isinstance(result, ArtifactResult) and result.type == artifact_type
    parsed = parse_content(result.content)
    assert parsed.type == artifact_type.value and result.title
    assert result.content["sources"]  # provenance travels with the artifact


def test_an_empty_scope_is_an_error_not_an_artifact():
    with pytest.raises(ArtifactGenerationError) as exc:
        generate_from_chunks([], T.SUMMARY, ArtifactOptions(), FakeLLMClient())
    assert exc.value.code == "NO_CONTENT"


def test_the_profile_gate_returns_the_ui_mismatch_without_spending_an_llm_call():
    fake = FakeLLMClient()
    outcome = generate_from_chunks(PROSE_ONLY, T.CHART, ArtifactOptions(), fake)
    assert isinstance(outcome, Mismatch) and outcome.requested == T.CHART
    assert outcome.reason == MISMATCH_REASON[T.CHART]
    assert outcome.alternatives == [T.KEYWORDS, T.SUMMARY, T.QUIZ, T.MCQ, T.FLASHCARDS]  # same list the frontend would compute
    assert fake.calls == []


def test_force_skips_the_gate_and_a_grounded_failure_is_then_an_error():
    fake = FakeLLMClient([chart_draft([], feasible=False)])
    with pytest.raises(ArtifactGenerationError) as exc:
        generate_from_chunks(PROSE_ONLY, T.CHART, ArtifactOptions(), fake, force=True)
    assert exc.value.code == "ARTIFACT_INSUFFICIENT_DATA" and len(fake.calls) == 1  # the model WAS asked


def test_when_the_profile_passes_but_the_model_cannot_ground_it_the_user_gets_a_mismatch():
    fake = FakeLLMClient([chart_draft([], feasible=False)])
    outcome = generate_from_chunks(RICH, T.CHART, ArtifactOptions(), fake)
    assert isinstance(outcome, Mismatch) and outcome.requested == T.CHART
    assert T.CHART not in outcome.alternatives and T.SUMMARY in outcome.alternatives


def test_force_on_good_data_just_generates():
    fake = FakeLLMClient(DRAFTS[T.CHART])
    assert isinstance(generate_from_chunks(RICH, T.CHART, ArtifactOptions(), fake, force=True), ArtifactResult)


def test_numbers_only_text_gets_chart_but_not_quiz():
    numbers = [chunk(1, "Results: RMSD: 6.4 Å at iteration 0. RMSD: 3.1 Å at iteration 1. RMSD: 1.9 Å. pLDDT: 91.")]
    fake = FakeLLMClient()
    outcome = generate_from_chunks(numbers, T.QUIZ, ArtifactOptions(), fake)
    assert isinstance(outcome, Mismatch) and T.CHART in outcome.alternatives and fake.calls == []


def test_llm_errors_are_not_swallowed_by_the_engine():
    from Backend.app.llm import LLMRateLimitError

    with pytest.raises(LLMRateLimitError):
        generate_from_chunks(RICH, T.KEYWORDS, ArtifactOptions(), FakeLLMClient([LLMRateLimitError("slow", retry_after=3)]))
