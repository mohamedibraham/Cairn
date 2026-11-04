import re
from dataclasses import dataclass

from enums import ArtifactType

LIGHT_TYPES = [ArtifactType.CHART, ArtifactType.DIAGRAM, ArtifactType.KEYWORDS]
HEAVY_TYPES = [ArtifactType.SUMMARY, ArtifactType.QUIZ, ArtifactType.MCQ, ArtifactType.FLASHCARDS]

MISMATCH_REASON = {
    ArtifactType.CHART: "No numeric or comparable values were found in the scoped sources.",
    ArtifactType.DIAGRAM: "The scoped sources describe no structure, flow, or relationships to lay out.",
    ArtifactType.KEYWORDS: "The scoped sources contain too little distinct terminology to extract.",
    ArtifactType.SUMMARY: "The scoped sources contain no continuous prose to condense.",
    ArtifactType.QUIZ: "The scoped sources lack the explanatory prose needed to write reliable questions.",
    ArtifactType.MCQ: "The scoped sources lack the explanatory prose needed to write distinct answer options.",
    ArtifactType.FLASHCARDS: "The scoped sources contain too few distinct terms to make useful cards.",
}

# thresholds 

MIN_DATAPOINTS = 4  # distinct labelled / unit-bearing numbers needed for a chart
MIN_RELATION_CUES = 4  # relation phrases (or code edges) needed for a diagram
MIN_PROSE_WORDS = 60
MIN_PROSE_SENTENCES = 3
MIN_REPEATED_TERMS = 8  # content words appearing at least twice
MIN_UNIQUE_TERMS = 25


@dataclass(frozen=True)
class ContentProfile:
    has_numeric_data: bool
    has_relationships: bool
    has_distinct_terms: bool
    has_prose: bool


SUPPORTED_BY = {
    ArtifactType.CHART: lambda p: p.has_numeric_data,
    ArtifactType.DIAGRAM: lambda p: p.has_relationships,
    ArtifactType.KEYWORDS: lambda p: p.has_distinct_terms,
    ArtifactType.SUMMARY: lambda p: p.has_prose,
    ArtifactType.QUIZ: lambda p: p.has_prose and p.has_distinct_terms,
    ArtifactType.MCQ: lambda p: p.has_prose and p.has_distinct_terms,
    ArtifactType.FLASHCARDS: lambda p: p.has_distinct_terms,
}


def alternatives_for(requested: ArtifactType, profile: ContentProfile) -> list:
    """Same order and rule as the frontend: light types first, then heavy ones."""
    return [t for t in LIGHT_TYPES + HEAVY_TYPES if t != requested and SUPPORTED_BY[t](profile)]


#  text analysis 

_UNITS = r"(?:%|ms|µs|us|ns|s|sec|min|h|hz|khz|mhz|ghz|kb|mb|gb|tb|å|nm|µm|mm|cm|km|m|kg|g|mg|°c|°|x|fold|mv|v|ma|µa|a|mol|um)"
_UNIT_NUMBER = re.compile(rf"(?<![\w.\[])(\d+(?:[.,]\d+)?)\s*({_UNITS})(?![A-Za-z])", re.IGNORECASE)
_LABELED_NUMBER = re.compile(
    r"([^\W\d_][^\n:=|]{1,40}?)\s*(?:[:=|]|\t|\s{2,})\s*([-+]?\d+(?:[.,]\d+)?)(?![\w.]*\])"
)
_TABLE_ROW = re.compile(r"^\s*\S.*?(?:\s{2,}|\t|\|)\s*[-+]?\d+(?:[.,]\d+)?\s*(?:(?:\s{2,}|\t|\|)\s*[-+]?\d+(?:[.,]\d+)?\s*)*$", re.MULTILINE)
_RELATION_CUES = re.compile(
    r"\b(depends? on|consists? of|composed of|made up of|flows? (?:to|from|into|through)|leads? to|results? in|"
    r"feeds?|produces?|followed by|passes? (?:to|through)|interacts? with|connected to|part of|inherits?|"
    r"stage|pipeline|input|output|upstream|downstream)\b|→|->|=>|⟶|"
    r"يعتمد على|يتكون من|يؤدي إلى|ينتج|يتبعه|مرحلة|خطوة|يتصل ب",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"[.!?؟](?:\s|$)")
_WORD = re.compile(r"[^\W\d_]{5,}", re.UNICODE)
_STOPWORDS = frozenset(
    "about above after again against among because before being below between could during each every first from "
    "further having here itself otherould over same should since some such than that their theirs them then there "
    "these they this those through under until upon very were what when where which while whose with within would "
    "your also into only more most many much just like make made using used uses based "
    "الذي التي هذا هذه ذلك تلك عندما حيث بين خلال أيضا كان كانت يكون يمكن على إلى عن من في مع هناك لكن "
    "بعد قبل أثناء ضمن حول فقط كذلك لذلك".split()
)


def _is_code_chunk(chunk) -> bool:
    meta = getattr(chunk, "metadata", None) or {}
    return meta.get("parse_status") == "ok" and meta.get("chunk_type") in {"function", "method", "class", "module", "script", "constants", "imports"}


def compute_profile(chunks) -> ContentProfile:
    
    prose_parts, code_edges = [], 0
    for chunk in chunks:
        meta = getattr(chunk, "metadata", None) or {}
        if _is_code_chunk(chunk):
            docstring = meta.get("docstring")
            if docstring:
                prose_parts.append(docstring)
            code_edges += len(meta.get("calls", [])) + len(meta.get("bases", []))
        else:
            prose_parts.append(chunk.content)
    text = "\n".join(prose_parts)

    return ContentProfile(
        has_numeric_data=_count_datapoints(text) >= MIN_DATAPOINTS,
        has_relationships=(len(_RELATION_CUES.findall(text)) + code_edges) >= MIN_RELATION_CUES,
        has_distinct_terms=_has_distinct_terms(text),
        has_prose=len(text.split()) >= MIN_PROSE_WORDS and len(_SENTENCE_END.findall(text)) >= MIN_PROSE_SENTENCES,
    )


def _count_datapoints(text: str) -> int:
    found = set()
    for match in _LABELED_NUMBER.finditer(text):
        found.add((match.group(1).strip().casefold(), match.group(2)))
    for match in _UNIT_NUMBER.finditer(text):
        found.add(("", match.group(1) + match.group(2).casefold()))
    for match in _TABLE_ROW.finditer(text):
        found.add((match.group(0).strip().casefold(), ""))
    return len(found)


def _has_distinct_terms(text: str) -> bool:
    counts = {}
    for word in _WORD.findall(text.casefold()):
        if word not in _STOPWORDS:
            counts[word] = counts.get(word, 0) + 1
    repeated = sum(1 for n in counts.values() if n >= 2)
    return len(counts) >= MIN_UNIQUE_TERMS and repeated >= MIN_REPEATED_TERMS
