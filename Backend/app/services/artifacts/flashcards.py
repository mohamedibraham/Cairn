from pydantic import BaseModel

from enums import ArtifactType
from schemas.artifact_content import FlashcardsContent
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
from services.artifacts.questions import evidence_chunk, study_chunks, topic_line


class _Card(BaseModel):
    front: str
    back: str
    passage: int
    quote: str


class _Draft(BaseModel):
    cards: list[_Card]


class FlashcardsStrategy(ArtifactStrategy):
    type = ArtifactType.FLASHCARDS

    def select(self, chunks: list, options: ArtifactOptions) -> list:
        return study_chunks(chunks, 12)

    def generate(self, llm, chunks: list, options: ArtifactOptions):
        count = clamp(options.count, 10, 4, 15)
        task = (
            f"Write {count} flashcards for revising the passages.{topic_line(options.topic)}\n"
            "- front: a term, concept or short question (max 15 words).\n"
            "- back: the answer or explanation (max 40 words), based only on the passages.\n"
            "- Each card must stand alone (no 'as mentioned above').\n"
            "- passage: the passage number; quote: verbatim span of that passage that supports the back."
        )

        def validate(draft: _Draft):
            problems, seen, rows = [], set(), []
            for card in draft.cards:
                front, back = card.front.strip(), card.back.strip()
                if not front or not back or len(front) > 140 or len(back) > 320:
                    problems.append(f"card '{front[:30]}': front/back missing or too long")
                    continue
                if squash(front) == squash(back) or squash(front) in seen:
                    continue
                source = evidence_chunk(chunks, card.passage, card.quote)
                if source is None:
                    problems.append(f"card '{front[:30]}': the quote is not in the passages")
                    continue
                seen.add(squash(front))
                rows.append((front, back, source))
            return rows, problems

        rows, _ = ask_validated(llm, prompt=build_prompt(chunks, task), draft_model=_Draft,
                                validate=validate, min_valid=max(4, count // 2))
        if len(rows) < 4:
            return Insufficient(MISMATCH_REASON[self.type])

        rows = rows[:count]
        cards = [{"id": f"fc{i}", "front": f, "back": b, "source": s.chunk_id} for i, (f, b, s) in enumerate(rows, 1)]
        content = FlashcardsContent(cards=cards, sources=source_refs([r[2] for r in rows]))
        return ArtifactResult(type=self.type, title="Flashcards", content=content.model_dump(mode="json", exclude_none=True))
