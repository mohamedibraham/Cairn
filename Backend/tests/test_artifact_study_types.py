"""Keywords / Flashcards / MCQ / Quiz strategies, with a scripted fake LLM.
Run:  python -m pytest tests/test_artifact_study_types.py"""

import pytest

from Backend.app.llm.providers.fake import FakeLLMClient
from Backend.app.schemas.artifact_content import parse_content
from Backend.app.services.artifacts.base import ArtifactOptions, ArtifactResult, Insufficient
from Backend.app.services.artifacts.flashcards import FlashcardsStrategy
from Backend.app.services.artifacts.keywords import KeywordsStrategy
from Backend.app.services.artifacts.mcq import McqStrategy
from Backend.app.services.artifacts.quiz import QuizStrategy
from Backend.app.services.retrieval_service import ScopedChunk

P1 = ("AlphaFold predicts protein structure using an Evoformer stack that processes the multiple sequence alignment "
      "and the pair representation together. The Evoformer repeatedly exchanges information between the two representations.")
P2 = ("The structure module refines backbone frames using invariant point attention. Recycling feeds the predicted structure "
      "back into the network as additional input, which improves accuracy over several rounds of refinement.")
P3 = ("Training used the Protein Data Bank together with a self distillation set. The confidence score pLDDT reports the "
      "expected accuracy for every residue, so low confidence regions can be identified quickly by the user.")


def chunk(n, text, **kw):
    return ScopedChunk(chunk_id=f"c{n}", file_id="f1", filename="alphafold.pdf", file_type="pdf",
                       chunk_index=n, content=text, page_number=n, **kw)


CHUNKS = [chunk(1, P1), chunk(2, P2), chunk(3, P3)]
Q1 = "processes the multiple sequence alignment and the pair representation together"
Q2 = "Recycling feeds the predicted structure back into the network as additional input"
Q3 = "The confidence score pLDDT reports the expected accuracy for every residue"


def run(strategy, drafts, options=None, chunks=CHUNKS):
    fake = FakeLLMClient(drafts)
    return strategy.generate(fake, chunks, options or ArtifactOptions()), fake


def valid_contract(result):
    assert isinstance(result, ArtifactResult)
    parse_content(result.content)  # must satisfy the frontend contract
    return result.content


# ============================= keywords ==============================================

def kw(term, passage=1, definition="a definition"):
    return {"term": term, "definition": definition, "passage": passage}


def test_keywords_are_grounded_weighted_by_frequency_and_carry_their_source():
    draft = {"items": [kw("pLDDT", 3), kw("Evoformer", 1), kw("invariant point attention", 2), kw("recycling", 2),
                       kw("pair representation", 1), kw("self distillation", 3)]}
    result, _ = run(KeywordsStrategy(), [draft])
    content = valid_contract(result)
    items = {i["term"]: i for i in content["items"]}
    assert items["Evoformer"]["weight"] == 1.0  # most frequent term (2x) anchors the scale
    assert items["pLDDT"]["weight"] == 0.5 and items["pLDDT"]["source"] == "c3"
    assert [i["term"] for i in content["items"]][0] == "Evoformer"  # sorted by weight
    assert result.title == "Key terms" and content["sources"]


def test_keywords_reject_invented_generic_overlong_and_duplicate_terms():
    draft = {"items": [kw("Evoformer"), kw("evoformer"), kw("quantum teleportation"), kw("system"),
                       kw("a very long term with way too many words"), kw("pLDDT", 3), kw("recycling", 2),
                       kw("invariant point attention", 2), kw("pair representation", 1), kw("self distillation", 3)]}
    result, _ = run(KeywordsStrategy(), [draft])
    terms = [i["term"] for i in valid_contract(result)["items"]]
    assert "quantum teleportation" not in terms and "system" not in terms
    assert sum(t.casefold() == "evoformer" for t in terms) == 1 and len(terms) == 6


def test_a_term_cited_to_the_wrong_passage_is_still_found_in_the_right_one():
    result, _ = run(KeywordsStrategy(), [{"items": [kw("pLDDT", 1), kw("Evoformer", 1), kw("recycling", 1),
                                                     kw("self distillation", 1), kw("pair representation", 1)]}])
    assert {i["term"]: i["source"] for i in valid_contract(result)["items"]}["pLDDT"] == "c3"


def test_rejected_output_triggers_one_corrective_retry_that_lists_the_problems():
    bad = {"items": [kw("hallucinated one"), kw("hallucinated two"), kw("Evoformer")]}
    good = {"items": [kw("pLDDT", 3), kw("Evoformer", 1), kw("recycling", 2), kw("self distillation", 3), kw("pair representation", 1)]}
    result, fake = run(KeywordsStrategy(), [bad, good])
    assert len(fake.calls) == 2 and len(valid_contract(result)["items"]) == 5
    retry_prompt = fake.calls[1].messages[0].content
    assert "REJECTED" in retry_prompt and "'hallucinated one' does not appear in the passages" in retry_prompt


def test_too_few_grounded_terms_means_insufficient_not_a_made_up_artifact():
    result, fake = run(KeywordsStrategy(), [{"items": [kw("nonsense a"), kw("nonsense b")]}] * 2)
    assert isinstance(result, Insufficient) and len(fake.calls) == 2


def test_keywords_count_is_clamped_and_topic_reaches_the_prompt_and_title():
    result, fake = run(KeywordsStrategy(), [{"items": [kw("Evoformer")] * 1}], ArtifactOptions(count=500, topic="protein folding"))
    prompt = fake.calls[0].messages[0].content
    assert "Extract the 30 most important" in prompt and "Focus on: protein folding" in prompt
    run2, _ = run(KeywordsStrategy(), [{"items": [kw("pLDDT", 3), kw("Evoformer"), kw("recycling", 2), kw("self distillation", 3), kw("pair representation")]}],
                  ArtifactOptions(topic="protein folding"))
    assert run2.title == "Key terms: protein folding"


# ============================= flashcards ============================================

def card(front, back, passage, quote):
    return {"front": front, "back": back, "passage": passage, "quote": quote}


GOOD_CARDS = [
    card("Evoformer", "Stack that processes the alignment and pair representation together", 1, Q1),
    card("Recycling", "Feeding the predicted structure back as extra input", 2, Q2),
    card("pLDDT", "Per-residue confidence score", 3, Q3),
    card("Structure module", "Refines backbone frames with invariant point attention", 2, "The structure module refines backbone frames using invariant point attention"),
]


def test_flashcards_valid_output_follows_the_contract():
    result, _ = run(FlashcardsStrategy(), [{"cards": GOOD_CARDS}])
    content = valid_contract(result)
    assert [c["id"] for c in content["cards"]] == ["fc1", "fc2", "fc3", "fc4"]
    assert content["cards"][2]["source"] == "c3" and result.title == "Flashcards"


def test_flashcards_with_a_quote_that_is_not_in_the_sources_are_dropped():
    fake_card = card("Teleportation", "Moves matter instantly", 1, "Quantum teleportation moves matter instantly across space")
    result, _ = run(FlashcardsStrategy(), [{"cards": GOOD_CARDS + [fake_card]}])
    assert "Teleportation" not in [c["front"] for c in valid_contract(result)["cards"]]


def test_a_real_quote_cited_to_the_wrong_passage_is_accepted():
    shifted = [dict(c, passage=1) for c in GOOD_CARDS]
    result, _ = run(FlashcardsStrategy(), [{"cards": shifted}])
    assert len(valid_contract(result)["cards"]) == 4


def test_flashcard_duplicates_and_degenerate_cards_are_removed():
    cards = GOOD_CARDS + [dict(GOOD_CARDS[0]), card("Same", "same", 1, Q1), card("x" * 200, "y", 1, Q1)]
    result, _ = run(FlashcardsStrategy(), [{"cards": cards}])
    assert len(valid_contract(result)["cards"]) == 4


def test_flashcards_need_at_least_four_grounded_cards():
    result, _ = run(FlashcardsStrategy(), [{"cards": GOOD_CARDS[:3]}] * 2)
    assert isinstance(result, Insufficient)


# ============================= mcq ===================================================

def mcq(question, correct, wrong, passage, quote, explanation="Because the passage says so."):
    options = [correct] + wrong
    return {"question": question, "options": options, "correct_index": 0, "explanation": explanation, "passage": passage, "quote": quote}


GOOD_MCQ = [
    mcq("What does recycling feed back into the network?", "The predicted structure", ["The raw sequence", "The training set", "The confidence score"], 2, Q2),
    mcq("What does pLDDT report?", "Expected accuracy per residue", ["Training loss", "Sequence length", "Alignment depth"], 3, Q3),
    mcq("What does the Evoformer process together?", "The alignment and the pair representation", ["Only the sequence", "Only the templates", "The final coordinates"], 1, Q1),
]


def test_mcq_keeps_the_correct_answer_and_mixes_positions_deterministically():
    result, _ = run(McqStrategy(), [{"questions": GOOD_MCQ}])
    content = valid_contract(result)
    answers = {q["question"]: q["options"][q["correct_index"]] for q in content["questions"]}
    assert answers["What does pLDDT report?"] == "Expected accuracy per residue"
    assert answers["What does recycling feed back into the network?"] == "The predicted structure"
    assert any(q["correct_index"] != 0 for q in content["questions"])  # the model's "always first" bias is broken up
    again, _ = run(McqStrategy(), [{"questions": GOOD_MCQ}])
    assert again.content == result.content  # reproducible
    assert all(q["source"] and q["explanation"] for q in content["questions"])


@pytest.mark.parametrize("break_it", [
    lambda q: q.update(options=["A", "a", "B", "C"]),  # duplicates
    lambda q: q.update(options=["A", "B", "All of the above", "C"]),  # vague
    lambda q: q.update(correct_index=9),  # out of range
    lambda q: q.update(options=["only", "two"]),  # too few
    lambda q: q.update(quote="This sentence was never in any passage at all"),  # ungrounded
])
def test_unusable_mcq_questions_are_dropped(break_it):
    broken = dict(GOOD_MCQ[0])
    break_it(broken)
    result, _ = run(McqStrategy(), [{"questions": GOOD_MCQ[1:] + [broken, dict(GOOD_MCQ[0], question="Another question?")]}])
    questions = [q["question"] for q in valid_contract(result)["questions"]]
    assert GOOD_MCQ[0]["question"] not in questions and len(questions) == 3


def test_mcq_needs_three_valid_questions_and_clamps_the_count():
    result, fake = run(McqStrategy(), [{"questions": GOOD_MCQ[:2]}] * 2, ArtifactOptions(count=99))
    assert isinstance(result, Insufficient) and "Write 8 multiple-choice" in fake.calls[0].messages[0].content


# ============================= quiz ==================================================

TF = {"kind": "true_false", "question": "Recycling feeds the predicted structure back into the network.", "answer_bool": True,
      "explanation": "Stated directly.", "passage": 2, "quote": Q2}
TF_FALSE = {"kind": "true_false", "question": "Recycling removes the predicted structure from the network.", "answer_bool": False,
            "explanation": "The opposite is stated.", "passage": 2, "quote": Q2}
QMCQ = {"kind": "mcq", "question": "What does pLDDT report?", "options": ["Expected accuracy per residue", "Loss", "Depth", "Length"],
        "correct_index": 0, "explanation": "Stated directly.", "passage": 3, "quote": Q3}
FILL = {"kind": "fill_blank", "question": "The _____ module refines backbone frames.", "accepted_answers": ["structure", "Structure"],
        "explanation": "Stated directly.", "passage": 2, "quote": "The structure module refines backbone frames using invariant point attention"}
GOOD_QUIZ = [TF, TF_FALSE, QMCQ, FILL]


def test_quiz_mixes_gradable_kinds_and_every_question_has_a_displayable_answer():
    result, _ = run(QuizStrategy(), [{"questions": GOOD_QUIZ}])
    content = valid_contract(result)
    by_kind = {q["kind"]: q for q in content["questions"]}
    assert set(by_kind) == {"true_false", "mcq", "fill_blank"}
    assert {q["answer"] for q in content["questions"] if q["kind"] == "true_false"} == {"True", "False"}
    assert by_kind["mcq"]["answer"] == by_kind["mcq"]["options"][by_kind["mcq"]["correct_index"]] == "Expected accuracy per residue"
    assert by_kind["fill_blank"]["question"] == "The ____ module refines backbone frames."  # blank normalised
    assert by_kind["fill_blank"]["accepted_answers"] == ["structure", "Structure"] and by_kind["fill_blank"]["answer"] == "structure"
    assert [q["id"] for q in content["questions"]] == ["q1", "q2", "q3", "q4"] and result.title == "Quiz"


def test_quiz_questions_that_cannot_be_graded_or_grounded_are_dropped():
    bad = [
        dict(TF, question="Statement without answer?", answer_bool=None),
        dict(FILL, question="No blank in this sentence at all."),
        dict(FILL, question="Two ____ and ____ blanks.", accepted_answers=["x"]),
        dict(FILL, question="The ____ module refines frames.", accepted_answers=["quantum"]),  # answer not in the quote
        dict(FILL, question="The ____ module again.", accepted_answers=["one two three four five six"]),  # too long
        dict(QMCQ, question="Bad mcq?", correct_index=7),
        dict(TF, question="Ungrounded statement.", quote="a quote that is nowhere in the passages"),
    ]
    result, _ = run(QuizStrategy(), [{"questions": GOOD_QUIZ + bad}])
    assert len(valid_contract(result)["questions"]) == 4


def test_requested_question_types_restrict_the_quiz():
    result, fake = run(QuizStrategy(), [{"questions": GOOD_QUIZ}], ArtifactOptions(question_types=("true_false", "mcq")))
    kinds = {q["kind"] for q in valid_contract(result)["questions"]}
    assert kinds == {"true_false", "mcq"}
    assert "Use only these kinds, mixed roughly evenly: true_false, mcq" in fake.calls[0].messages[0].content


def test_quiz_with_too_few_valid_questions_is_insufficient():
    result, _ = run(QuizStrategy(), [{"questions": [TF, FILL]}] * 2)
    assert isinstance(result, Insufficient)


def test_quiz_selects_distinct_prompts_per_option_and_never_exceeds_the_limit():
    many = [dict(TF, question=f"Distinct statement number {i} about recycling.") for i in range(12)]
    result, _ = run(QuizStrategy(), [{"questions": many}], ArtifactOptions(count=6))
    assert len(valid_contract(result)["questions"]) == 6
