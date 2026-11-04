import dataclasses
from enums import ArtifactType
from llm import LLMClient
from services.artifacts.diagram_layout import apply_layout_if_needed
from services.artifacts.base import (
    ArtifactGenerationError,
    ArtifactOptions,
    ArtifactResult,
    Insufficient,
    Mismatch,
)
from services.artifacts.profile import MISMATCH_REASON, SUPPORTED_BY, alternatives_for, compute_profile
from services.artifacts.registry import get_strategy


def generate_from_chunks(
    chunks: list,
    artifact_type: ArtifactType,
    options: ArtifactOptions,
    llm: LLMClient,
    *,
    force: bool = False,
):
    if not chunks:
        raise ArtifactGenerationError("NO_CONTENT", "No ready (embedded) content found in scope to generate from")

    profile = compute_profile(chunks)
    if not force and not SUPPORTED_BY[artifact_type](profile):
        return Mismatch(artifact_type, MISMATCH_REASON[artifact_type], alternatives_for(artifact_type, profile))

    strategy = get_strategy(artifact_type)
    outcome = strategy.generate(llm, strategy.select(chunks, options), options)

    if isinstance(outcome, Insufficient):
        if force:
            raise ArtifactGenerationError("ARTIFACT_INSUFFICIENT_DATA", outcome.reason)
        return Mismatch(artifact_type, outcome.reason, alternatives_for(artifact_type, profile))

    assert isinstance(outcome, ArtifactResult)

    if artifact_type == ArtifactType.DIAGRAM and isinstance(outcome.content, dict):
        outcome = dataclasses.replace(outcome, content=apply_layout_if_needed(outcome.content))

    return outcome