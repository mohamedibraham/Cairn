from pydantic import BaseModel

from enums import ArtifactType
from schemas.artifact_content import KeywordsContent
from services.artifacts.base import (
    ArtifactOptions,
    ArtifactResult,
    ArtifactStrategy,
    Insufficient,
    ask_validated,
    build_prompt,
    clamp,
    fit_to_budget,
    passage_at,
    source_refs,
)
from services.artifacts.grounding import contains_term, count_term, squash
from services.artifacts.profile import MISMATCH_REASON
from services.retrieval_service import stride_sample

_GENERIC = {
    "system", "data", "method", "methods", "approach", "model", "models", "result", "results", "analysis", "process",
    "information", "work", "paper", "study", "example", "section", "figure", "table", "نظام", "بيانات", "طريقة", "نتائج",
}


class _Item(BaseModel):
    term: str
    definition: str
    passage: int


class _Draft(BaseModel):
    items: list[_Item]


class KeywordsStrategy(ArtifactStrategy):
    type = ArtifactType.KEYWORDS

    def select(self, chunks: list, options: ArtifactOptions) -> list:
        return fit_to_budget(stride_sample(chunks, 14))

    def generate(self, llm, chunks: list, options: ArtifactOptions):
        count = clamp(options.count, 15, 8, 30)
        topic = f"\nFocus on: {options.topic.strip()[:300]}" if options.topic and options.topic.strip() else ""
        task = (
            f"Extract the {count} most important key terms from the passages.{topic}\n"
            "- term: copied exactly as written in a passage; 1 to 5 words; specific and domain-relevant "
            "(no generic words such as 'system', 'data', 'method').\n"
            "- definition: at most 20 words, based only on the passages.\n"
            "- passage: the number of a passage that contains the term."
        )
        joined = "\n".join(c.content for c in chunks)

        def validate(draft: _Draft):
            problems, seen, rows = [], set(), []
            for item in draft.items:
                term = item.term.strip().strip("\"'`")
                if not 2 <= len(term) <= 40 or len(term.split()) > 5:
                    problems.append(f"'{term}': a term must be 1-5 words")
                    continue
                key = squash(term)
                if key in seen:
                    continue
                if term.casefold() in _GENERIC:
                    problems.append(f"'{term}' is too generic")
                    continue
                cited = passage_at(chunks, item.passage)
                source = cited if cited is not None and contains_term(cited.content, term) else next(
                    (c for c in chunks if contains_term(c.content, term)), None
                )
                if source is None:
                    problems.append(f"'{term}' does not appear in the passages")
                    continue
                seen.add(key)
                rows.append((term, item.definition.strip()[:200], source, count_term(joined, term)))
            return rows, problems

        rows, _ = ask_validated(llm, prompt=build_prompt(chunks, task), draft_model=_Draft,
                                validate=validate, min_valid=max(5, count // 2))
        if len(rows) < 3:
            return Insufficient(MISMATCH_REASON[self.type])

        rows = sorted(rows, key=lambda r: -r[3])[:count]  
        top = max(r[3] for r in rows) or 1
        items = [
            {"term": term, "weight": round(max(freq, 1) / top, 2), "definition": definition or None, "source": src.chunk_id}
            for term, definition, src, freq in rows
        ]
        content = KeywordsContent(items=items, sources=source_refs([r[2] for r in rows]))
        return ArtifactResult(
            type=self.type,
            title=f"Key terms: {options.topic.strip()[:40]}" if options.topic and options.topic.strip() else "Key terms",
            content=content.model_dump(mode="json", exclude_none=True),
        )
