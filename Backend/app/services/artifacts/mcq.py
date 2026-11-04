from pydantic import BaseModel

from enums import ArtifactType
from schemas.artifact_content import McqContent
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
from services.artifacts.grounding import squash
from services.artifacts.profile import MISMATCH_REASON
from services.artifacts.questions import (
    clean_options,
    difficulty_hint,
    evidence_chunk,
    shuffled,
    study_chunks,
    topic_line,
)


class _Q(BaseModel):
    question: str
    options: list[str]
    correct_index: int
    explanation: str
    passage: int
    quote: str


class _Draft(BaseModel):
    questions: list[_Q]


class McqStrategy(ArtifactStrategy):
    type = ArtifactType.MCQ

    def select(self, chunks: list, options: ArtifactOptions) -> list:
        return study_chunks(chunks, 12)

    def generate(self, llm, chunks: list, options: ArtifactOptions):
        count = clamp(options.count, 5, 3, 8)
        task = (
            f"Write {count} multiple-choice questions testing understanding of the passages.{topic_line(options.topic)}\n"
            f"- {difficulty_hint(options.difficulty)}\n"
            "- Exactly 4 options; exactly one is correct according to the passages; the others are plausible but wrong.\n"
            "- Never use 'all of the above' or 'none of the above'.\n"
            "- correct_index: 0-based index of the correct option.\n"
            "- explanation: one sentence, based on the passage.\n"
            "- passage: the passage number; quote: verbatim span of that passage that proves the correct answer."
        )

        def validate(draft: _Draft):
            problems, seen, rows = [], set(), []
            for q in draft.questions:
                question = q.question.strip()
                options = clean_options(q.options)
                if not question or squash(question) in seen:
                    continue
                if not options:
                    problems.append(f"'{question[:40]}': need 3-5 distinct, specific options (no 'all of the above')")
                    continue
                if not 0 <= q.correct_index < len(q.options) or not q.options[q.correct_index].strip():
                    problems.append(f"'{question[:40]}': correct_index does not point at an option")
                    continue
                correct = q.options[q.correct_index].strip()
                if correct not in options:
                    problems.append(f"'{question[:40]}': the correct option was removed during cleaning")
                    continue
                source = evidence_chunk(chunks, q.passage, q.quote)
                if source is None:
                    problems.append(f"'{question[:40]}': the quote is not in the passages")
                    continue
                seen.add(squash(question))
                mixed, index = shuffled(question, options, options.index(correct))
                rows.append((question, mixed, index, q.explanation.strip()[:300] or None, source))
            return rows, problems

        rows, _ = ask_validated(llm, prompt=build_prompt(chunks, task), draft_model=_Draft,
                                validate=validate, min_valid=max(3, count // 2 + 1))
        if len(rows) < 3:
            return Insufficient(MISMATCH_REASON[self.type])

        rows = rows[:count]
        questions = [
            {"id": f"mq{i}", "question": q, "options": o, "correct_index": ci, "explanation": ex, "source": s.chunk_id}
            for i, (q, o, ci, ex, s) in enumerate(rows, 1)
        ]
        content = McqContent(questions=questions, sources=source_refs([r[4] for r in rows]))
        return ArtifactResult(type=self.type, title="Multiple-choice questions",
                              content=content.model_dump(mode="json", exclude_none=True))
