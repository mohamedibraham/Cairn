import re
from typing import Literal

from pydantic import BaseModel

from enums import ArtifactType
from schemas.artifact_content import BLANK, QuizContent
from services.artifacts.base import (
    ArtifactOptions,
    ArtifactResult,
    ArtifactStrategy,
    Insufficient,
    ask_validated,
    build_prompt,
    clamp,
    source_refs,
)
from services.artifacts.grounding import contains_term, squash
from services.artifacts.profile import MISMATCH_REASON
from services.artifacts.questions import (
    clean_options,
    difficulty_hint,
    evidence_chunk,
    shuffled,
    study_chunks,
    topic_line,
)

KINDS = ("true_false", "mcq", "fill_blank")
_BLANK_RUN = re.compile(r"_{3,}")


class _Q(BaseModel):
    kind: Literal["true_false", "mcq", "fill_blank"]
    question: str
    answer_bool: bool | None = None  # true_false
    options: list[str] | None = None  # mcq
    correct_index: int | None = None  # mcq
    accepted_answers: list[str] | None = None  # fill_blank
    explanation: str
    passage: int
    quote: str


class _Draft(BaseModel):
    questions: list[_Q]


class QuizStrategy(ArtifactStrategy):
    type = ArtifactType.QUIZ

    def select(self, chunks: list, options: ArtifactOptions) -> list:
        return study_chunks(chunks, 12)

    def generate(self, llm, chunks: list, options: ArtifactOptions):
        count = clamp(options.count, 6, 3, 8)
        kinds = tuple(k for k in options.question_types if k in KINDS) or KINDS
        task = (
            f"Write a quiz of {count} questions about the passages.{topic_line(options.topic)}\n"
            f"- {difficulty_hint(options.difficulty)}\n"
            f"- Use only these kinds, mixed roughly evenly: {', '.join(kinds)}.\n"
            "- true_false: a single statement (kind=true_false, answer_bool). About half should be false: make a false "
            "statement by changing one fact of a true one.\n"
            f"- mcq: 4 options, one correct (options, correct_index 0-based); never 'all of the above'.\n"
            f"- fill_blank: one sentence with exactly one blank written as {BLANK}; accepted_answers = 1 to 3 acceptable "
            "short fills (max 5 words each) that appear in the quote.\n"
            "- explanation: one sentence based on the passage.\n"
            "- passage: the passage number; quote: verbatim span of that passage that supports the correct answer."
        )

        def validate(draft: _Draft):
            problems, seen, rows = [], set(), []
            for q in draft.questions:
                question = q.question.strip()
                label = f"'{question[:40]}'"
                if q.kind not in kinds or not question or squash(question) in seen:
                    continue
                source = evidence_chunk(chunks, q.passage, q.quote)
                if source is None:
                    problems.append(f"{label}: the quote is not in the passages")
                    continue
                item = {"kind": q.kind, "explanation": q.explanation.strip()[:300] or None, "source": source.chunk_id}

                if q.kind == "true_false":
                    if q.answer_bool is None:
                        problems.append(f"{label}: true_false needs answer_bool")
                        continue
                    item.update(question=question, answer="True" if q.answer_bool else "False")

                elif q.kind == "mcq":
                    options = clean_options(q.options or [])
                    ok_index = q.correct_index is not None and q.options and 0 <= q.correct_index < len(q.options)
                    if not options or not ok_index or q.options[q.correct_index].strip() not in options:
                        problems.append(f"{label}: mcq needs 3-5 distinct options and a valid correct_index")
                        continue
                    mixed, index = shuffled(question, options, options.index(q.options[q.correct_index].strip()))
                    item.update(question=question, options=mixed, correct_index=index, answer=mixed[index])

                else:  
                    text = _BLANK_RUN.sub(BLANK, question)
                    answers = list(dict.fromkeys(a.strip() for a in (q.accepted_answers or []) if a and a.strip()))
                    if text.count(BLANK) != 1:
                        problems.append(f"{label}: fill_blank needs exactly one {BLANK}")
                        continue
                    if not answers or any(len(a.split()) > 5 for a in answers):
                        problems.append(f"{label}: accepted_answers must be 1-3 short answers")
                        continue
                    if not contains_term(q.quote, answers[0], min_chars=2):
                        problems.append(f"{label}: the answer '{answers[0]}' is not in the quote")
                        continue
                    item.update(question=text, answer=answers[0], accepted_answers=answers[:3])

                seen.add(squash(question))
                rows.append((item, source))
            return rows, problems

        rows, _ = ask_validated(llm, prompt=build_prompt(chunks, task), draft_model=_Draft,
                                validate=validate, min_valid=max(3, count // 2 + 1))
        if len(rows) < 3:
            return Insufficient(MISMATCH_REASON[self.type])

        rows = rows[:count]
        questions = [{"id": f"q{i}", **item} for i, (item, _) in enumerate(rows, 1)]
        content = QuizContent(questions=questions, sources=source_refs([s for _, s in rows]))
        return ArtifactResult(type=self.type, title="Quiz", content=content.model_dump(mode="json", exclude_none=True))
