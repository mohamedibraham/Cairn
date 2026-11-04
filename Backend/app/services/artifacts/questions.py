import random
import re
import zlib

from services.artifacts.base import MAX_PROMPT_CHARS, fit_to_budget, passage_at
from services.artifacts.grounding import contains_quote
from services.retrieval_service import stride_sample

MIN_STUDY_WORDS = 25  
_VAGUE_OPTION = re.compile(
    r"^\s*(?:all|none|both)\s+of\s+the\s+above\b|^\s*(?:كل|جميع)\s+ما\s+سبق|^\s*لا\s+شيء\s+مما\s+سبق", re.IGNORECASE
)


def study_chunks(chunks: list, n: int) -> list:
    
    substantial = [c for c in chunks if len(c.content.split()) >= MIN_STUDY_WORDS] or list(chunks)
    return fit_to_budget(stride_sample(substantial, n), MAX_PROMPT_CHARS)


def evidence_chunk(chunks: list, passage: int, quote: str, *, min_chars: int = 12):
    
    cited = passage_at(chunks, passage)
    if cited is not None and contains_quote(cited.content, quote, min_chars=min_chars):
        return cited
    for chunk in chunks:
        if contains_quote(chunk.content, quote, min_chars=min_chars):
            return chunk
    return None


def clean_options(options) -> list:
    
    cleaned = [o.strip() for o in options if o and o.strip()]
    if not 3 <= len(cleaned) <= 5:
        return []
    if len({o.casefold() for o in cleaned}) != len(cleaned):
        return []
    if any(len(o) > 200 or _VAGUE_OPTION.match(o) for o in cleaned):
        return []
    return cleaned


def shuffled(question: str, options: list, correct_index: int):
    
    order = list(range(len(options)))
    random.Random(zlib.crc32(question.encode("utf-8"))).shuffle(order)
    return [options[i] for i in order], order.index(correct_index)


def difficulty_hint(difficulty: str | None) -> str:
    return {
        "easy": "Difficulty: easy — recall of explicit facts.",
        "hard": "Difficulty: hard — require connecting several facts from the passages.",
    }.get(difficulty or "", "Difficulty: medium — understanding, not just word matching.")


def topic_line(topic: str | None) -> str:
    return f"\nFocus on: {topic.strip()[:300]}" if topic and topic.strip() else ""
