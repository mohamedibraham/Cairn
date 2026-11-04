"""Grounding helpers and ContentProfile.   Run: python -m pytest tests/test_artifact_grounding_profile.py"""

from types import SimpleNamespace

import pytest

from Backend.app.enums import ArtifactType as T
from Backend.app.services.artifacts.grounding import (
    contains_quote,
    contains_term,
    count_term,
    number_in_text,
    numbers_in,
    squash,
)
from Backend.app.services.artifacts.profile import (
    HEAVY_TYPES,
    LIGHT_TYPES,
    MISMATCH_REASON,
    SUPPORTED_BY,
    ContentProfile,
    alternatives_for,
    compute_profile,
)


def chunk(content, metadata=None):
    return SimpleNamespace(content=content, metadata=metadata)


# ---- grounding ---------------------------------------------------------------------

def test_quote_matching_ignores_case_punctuation_whitespace_and_hyphenation():
    passage = "AlphaFold predicts protein structure using an Evoformer-based attention stack.\nIt was published in 2021."
    assert contains_quote(passage, "alphafold predicts   protein structure, using an evoformer based attention stack")
    assert contains_quote(passage, '"It was published in 2021"')


def test_quote_matching_still_requires_the_same_content():
    passage = "AlphaFold predicts protein structure using an Evoformer."
    assert not contains_quote(passage, "AlphaFold predicts protein function using an Evoformer")
    assert not contains_quote(passage, "short")  # too short to be evidence
    assert not contains_quote(passage, "")


def test_arabic_diacritics_tatweel_and_digit_forms_are_normalised():
    passage = "يُعَدُّ ألفافولد نموذجًا لـتنبؤ بنية البروتين بدقة ٩٢ بالمئة"
    assert contains_quote(passage, "يعد ألفافولد نموذجا لتنبؤ بنية البروتين بدقة 92 بالمئة")
    assert squash("ـــ") == ""  # tatweel only


def test_term_checks():
    text = "The Evoformer refines the MSA representation; the evoformer is iterated."
    assert contains_term(text, "MSA representation") and not contains_term(text, "Transformer-XL")
    assert count_term(text, "evoformer") == 2 and count_term(text, "zz") == 0


@pytest.mark.parametrize("text, expected", [
    ("RMSD fell from 6.4 Å to 1.4 Å", [6.4, 1.4]),
    ("revenue was 1,234.5 and 12,000", [1234.5, 12000.0]),
    ("accuracy 92%", [92.0]),
    ("سجلنا ٣٫٥ ثم ١٢", [3.5, 12.0]),
    ("temperature −4 degrees", [-4.0]),
    ("ratio 3,5 here", [3.5]),
])
def test_numbers_are_extracted_across_formats(text, expected):
    assert numbers_in(text) == expected


def test_number_in_text_is_exact_not_approximate():
    assert number_in_text(6.4, "RMSD 6.4 Å") and number_in_text(1234.5, "total 1,234.5")
    assert not number_in_text(6.5, "RMSD 6.4 Å") and not number_in_text(64, "RMSD 6.4 Å")


# ---- profile ------------------------------------------------------------------------

PROSE = (
    "AlphaFold predicts protein structure using an Evoformer stack. The Evoformer processes the multiple sequence alignment "
    "and the pair representation. The structure module then refines backbone frames iteratively. Recycling feeds the output "
    "back as input, which improves accuracy. Invariant point attention keeps the update equivariant. Protein folding "
    "residues, distances and templates all contribute to the final prediction quality. "
    "The alignment provides evolutionary signal, and the representation of each residue pair encodes distance information. "
    "Attention over the alignment lets the Evoformer exchange information between sequences, while attention over the pair "
    "representation enforces geometric consistency. Each recycling round reuses the previous structure prediction as a template, "
    "so accuracy improves with further recycling. The final prediction reports a confidence score for every residue."
)
NUMBERS = "Results: RMSD: 6.4 Å at iteration 0. RMSD: 3.1 Å at iteration 1. RMSD: 1.9 Å at iteration 2. pLDDT: 91 at iteration 3."


def test_prose_makes_summary_and_study_types_possible_but_not_charts():
    p = compute_profile([chunk(PROSE)])
    assert p.has_prose and p.has_distinct_terms and not p.has_numeric_data
    assert SUPPORTED_BY[T.SUMMARY](p) and SUPPORTED_BY[T.QUIZ](p) and SUPPORTED_BY[T.MCQ](p) and SUPPORTED_BY[T.FLASHCARDS](p)
    assert not SUPPORTED_BY[T.CHART](p)


def test_labelled_numbers_and_unit_numbers_make_charts_possible():
    assert compute_profile([chunk(NUMBERS)]).has_numeric_data
    assert compute_profile([chunk("Latency fell 40% then 25% then 12% and finally 8%.")]).has_numeric_data


def test_a_one_paragraph_note_is_too_short_to_count_as_prose():
    short = "AlphaFold predicts protein structure using an Evoformer. It refines frames. Recycling helps accuracy."
    p = compute_profile([chunk(short)])
    assert not p.has_prose and not p.has_distinct_terms


def test_incidental_numbers_do_not_count_as_data():
    text = "See [1] and [2] on page 4, section 3.2, published in 2021 by the team in room 12."
    assert not compute_profile([chunk(text)]).has_numeric_data


def test_relation_cues_make_diagrams_possible():
    flow = "Raw files flow to the extractor, which feeds the chunker. The chunker produces chunks that depend on the embedder. Output is stored."
    assert compute_profile([chunk(flow)]).has_relationships
    assert not compute_profile([chunk("A short note about lunch.")]).has_relationships


def test_python_code_is_not_prose_and_its_constants_are_not_data():
    code = chunk("def run(self):\n    LIMIT = 1024\n    MAX = 2048\n    A = 1\n    B = 2\n", {
        "parse_status": "ok", "chunk_type": "method", "docstring": "Run it.", "calls": ["Folder.helper"],
    })
    p = compute_profile([code])
    assert not p.has_prose and not p.has_numeric_data


def test_python_structure_counts_as_relationships():
    chunks = [chunk("x", {"parse_status": "ok", "chunk_type": "class", "bases": ["Base"]}),
              chunk("y", {"parse_status": "ok", "chunk_type": "method", "calls": ["a", "b", "c"]})]
    assert compute_profile(chunks).has_relationships


def test_empty_scope_supports_nothing():
    p = compute_profile([])
    assert not any([p.has_numeric_data, p.has_relationships, p.has_distinct_terms, p.has_prose])


# ---- mismatch data, identical to the frontend's tables -----------------------------

def test_alternatives_follow_the_frontend_rule_and_order():
    profile = ContentProfile(has_numeric_data=True, has_relationships=False, has_distinct_terms=True, has_prose=True)
    assert alternatives_for(T.DIAGRAM, profile) == [T.CHART, T.KEYWORDS, T.SUMMARY, T.QUIZ, T.MCQ, T.FLASHCARDS]
    assert alternatives_for(T.CHART, profile) == [T.KEYWORDS, T.SUMMARY, T.QUIZ, T.MCQ, T.FLASHCARDS]
    nothing = ContentProfile(False, False, False, False)
    assert alternatives_for(T.CHART, nothing) == []


def test_type_lists_and_reasons_match_the_frontend():
    assert [t.value for t in LIGHT_TYPES] == ["chart", "diagram", "keywords"]
    assert [t.value for t in HEAVY_TYPES] == ["summary", "quiz", "mcq", "flashcards"]
    assert set(MISMATCH_REASON) == set(T) and MISMATCH_REASON[T.CHART].startswith("No numeric or comparable values")
