import bisect
import logging
import re
from functools import lru_cache
from typing import Callable, TypedDict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from helpers import settings
from pipeline.embedding import count_tokens, max_seq_length
from pipeline.extraction import ExtractedDocument, ExtractedPage
from pipeline.python_parser import MODULE_LEVEL_KINDS, CodeUnit, PythonParseError, parse_python

logger = logging.getLogger(__name__)


CHUNK_SIZE_TOKENS: int = settings.CHUNK_SIZE_TOKENS
CHUNK_OVERLAP_TOKENS: int = settings.CHUNK_OVERLAP_TOKENS

LINE_TRACKED_TYPES = {"txt", "md", "py"}

# Paragraph -> line -> sentence (Latin + Arabic punctuation) -> clause -> word.
SEPARATORS = ["\n\n", "\n", ". ", "؟ ", "? ", "! ", "؛ ", "; ", " ", ""]


class _RawChunkRequired(TypedDict):
    content: str
    page_number: int | None
    line_start: int | None
    line_end: int | None
    token_count: int


class RawChunk(_RawChunkRequired, total=False):
    
    metadata: dict




def _token_counter() -> Callable[[str], int]:
    
    return count_tokens


@lru_cache(maxsize=1)
def _splitter() -> RecursiveCharacterTextSplitter:
    limit = max_seq_length()
    if limit and CHUNK_SIZE_TOKENS + 2 > limit:  
        logger.warning(
            "CHUNK_SIZE_TOKENS=%d exceeds the embedding model's max_seq_length=%d; "
            "chunk tails will be silently truncated when embedded",
            CHUNK_SIZE_TOKENS,
            limit,
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        length_function=_token_counter(),
        separators=SEPARATORS,
        is_separator_regex=False,
    )




def _line_starts(text: str) -> list[int]:
    
    return [0] + [m.end() for m in re.finditer("\n", text)]


def _line_at(starts: list[int], offset: int) -> int:
    """1-based line number containing the given character offset."""
    return bisect.bisect_right(starts, offset)




def _split_page(page: ExtractedPage, track_lines: bool) -> list[RawChunk]:
    text = page.text
    count_tokens = _token_counter()
    starts = _line_starts(text) if track_lines else None

    chunks: list[RawChunk] = []
    cursor = 0  

    for piece in _splitter().split_text(text):
        piece = piece.strip()
        if not piece:
            continue

        
        offset = text.find(piece, cursor)
        if offset == -1:
            offset = text.find(piece)
        if offset != -1:
            cursor = offset + 1

        line_start = line_end = None
        if starts is not None and offset != -1:
            line_start = _line_at(starts, offset)
            line_end = _line_at(starts, offset + len(piece) - 1)

        chunks.append(
            RawChunk(
                content=piece,
                page_number=page.page_number,
                line_start=line_start,
                line_end=line_end,
                token_count=count_tokens(piece),
            )
        )
    return chunks


def split_into_chunks(document: ExtractedDocument) -> list[RawChunk]:
    if document.file_type == "py":
        return _chunk_python(document)

    track_lines = document.file_type in LINE_TRACKED_TYPES

    chunks: list[RawChunk] = []
    for page in document.pages:
        chunks.extend(_split_page(page, track_lines))
    return chunks




PARSER_VERSION = "python-ast/1"
MAX_USES_IN_HEADER = 8
MIN_SPLIT_FILL = 0.4  


def _chunk_python(document: ExtractedDocument) -> list[RawChunk]:
    source = document.text
    try:
        units = parse_python(source)
        if not units:
            raise PythonParseError("No code structure found")
    except PythonParseError as exc:
        logger.warning("Python structural parsing failed (%s); using plain-text chunking", exc)
        return _python_fallback(document, str(exc))

    chunks: list[RawChunk] = []
    i = 0
    while i < len(units):
        if units[i].kind in MODULE_LEVEL_KINDS:
            group = []
            while i < len(units) and units[i].kind in MODULE_LEVEL_KINDS:
                group.append(units[i])
                i += 1
            chunks.extend(_chunk_module_level(group))
        else:
            chunks.extend(_chunk_symbol(units[i]))
            i += 1
    return chunks


def _python_fallback(document: ExtractedDocument, reason: str) -> list[RawChunk]:
    chunks: list[RawChunk] = []
    for page in document.pages:
        chunks.extend(_split_page(page, track_lines=True))
    for chunk in chunks:
        chunk["metadata"] = {
            "parser": PARSER_VERSION,
            "parse_status": "fallback",
            "parse_error": reason[:300],
            "chunk_type": "text",
        }
    return chunks


def _header(unit: CodeUnit, part: int | None = None, parts: int | None = None) -> str:
    label = f"# {unit.name} [{unit.kind}] L{unit.start_line}-{unit.end_line}"
    if part is not None and parts is not None and parts > 1:
        label += f" (part {part}/{parts})"
    if unit.imports_used:
        shown = ", ".join(unit.imports_used[:MAX_USES_IN_HEADER])
        label += f"\n# uses: {shown}"
    return label


def _metadata(unit: CodeUnit, **extra) -> dict:
    meta: dict = {
        "parser": PARSER_VERSION,
        "parse_status": "ok",
        "chunk_type": unit.kind,
        "symbol": unit.name,
    }
    if unit.signature:
        meta["signature"] = unit.signature
    if unit.parent:
        meta["parent"] = unit.parent
    if unit.decorators:
        meta["decorators"] = list(unit.decorators)
    if unit.bases:
        meta["bases"] = list(unit.bases)
    if unit.is_async:
        meta["async"] = True
    if unit.docstring:
        meta["docstring"] = unit.docstring[:300]
    if unit.imports_used:
        meta["imports_used"] = list(unit.imports_used)
    if unit.calls:
        meta["calls"] = list(unit.calls)[:20]
    meta.update(extra)
    return meta


def _make_chunk(content: str, start: int, end: int, meta: dict) -> RawChunk:
    return RawChunk(
        content=content,
        page_number=None,
        line_start=start,
        line_end=end,
        token_count=_token_counter()(content),
        metadata=meta,
    )


def _chunk_symbol(unit: CodeUnit) -> list[RawChunk]:
    count = _token_counter()
    header = _header(unit)
    if count(header) + count(unit.text) + 2 <= CHUNK_SIZE_TOKENS:
        return [_make_chunk(f"{header}\n{unit.text}", unit.start_line, unit.end_line, _metadata(unit))]

    
    continuation = f"# (continued) {unit.signature}" if unit.signature else ""
    reserved = count(_header(unit, 99, 99)) + count(continuation) + 4
    budget = max(40, CHUNK_SIZE_TOKENS - reserved)
    pieces = _pack_lines(unit.text.split("\n"), unit.start_line, budget)

    chunks = []
    for number, (start, end, body) in enumerate(pieces, start=1):
        head = _header(unit, number, len(pieces))
        if number > 1 and continuation:
            head += "\n" + continuation
        meta = _metadata(unit, part=number, parts=len(pieces))
        chunks.append(_make_chunk(f"{head}\n{body}", start, end, meta))
    return chunks


def _budget_splitter(budget: int) -> RecursiveCharacterTextSplitter:
    
    return RecursiveCharacterTextSplitter(
        chunk_size=budget,
        chunk_overlap=min(CHUNK_OVERLAP_TOKENS, budget // 4),
        length_function=_token_counter(),
        separators=SEPARATORS,
        is_separator_regex=False,
    )


def _pack_lines(lines: list[str], first_line: int, budget: int) -> list[tuple[int, int, str]]:
    
    count = _token_counter()
    parts: list[tuple[int, int, str]] = []
    i, n = 0, len(lines)
    while i < n:
        j, tokens = i, 0
        blank_end, blank_tokens = None, 0
        while j < n:
            cost = count(lines[j]) + 1
            if tokens + cost > budget and j > i:
                break
            tokens += cost
            j += 1
            if not lines[j - 1].strip():
                blank_end, blank_tokens = j, tokens

        if j == i + 1 and tokens > budget:
            number = first_line + i
            for piece in _budget_splitter(budget).split_text(lines[i]):
                if piece.strip():
                    parts.append((number, number, piece))
            i = j
            continue

        if j < n and blank_end is not None and blank_end > i and blank_tokens >= MIN_SPLIT_FILL * budget:
            j = blank_end
        text = "\n".join(lines[i:j]).rstrip()
        if text.strip():
            parts.append((first_line + i, first_line + j - 1, text))
        i = j
    return parts


def _chunk_module_level(group: list[CodeUnit]) -> list[RawChunk]:
    
    count = _token_counter()
    budget = CHUNK_SIZE_TOKENS - 30  
    chunks: list[RawChunk] = []
    current: list[CodeUnit] = []
    used = 0

    def flush() -> None:
        nonlocal current, used
        if not current:
            return
        start, end = current[0].start_line, current[-1].end_line
        sections = [u.kind for u in current]
        declared = [name for u in current for name in u.imports_declared]
        head = f"# <module> [module] L{start}-{end}\n# sections: {', '.join(sections)}"
        meta = {
            "parser": PARSER_VERSION,
            "parse_status": "ok",
            "chunk_type": "module",
            "symbol": "<module>",
            "sections": sections,
        }
        if declared:
            meta["imports"] = list(dict.fromkeys(declared))
        if any(u.name == "__main__" for u in current):
            meta["has_main_guard"] = True
        body = "\n\n".join(u.text for u in current)
        chunks.append(_make_chunk(f"{head}\n{body}", start, end, meta))
        current, used = [], 0

    for unit in group:
        size = count(unit.text) + 2
        if size > budget:
            flush()
            chunks.extend(_chunk_symbol(unit))
            continue
        if current and used + size > budget:
            flush()
        current.append(unit)
        used += size
    flush()
    return chunks
