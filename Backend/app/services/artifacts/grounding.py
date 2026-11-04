import math
import re
import unicodedata
from functools import lru_cache

# Arabic diacritics (tashkeel), Quranic marks and tatweel
_ARABIC_MARKS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


@lru_cache(maxsize=512)
def squash(text: str) -> str:
    
    text = unicodedata.normalize("NFKC", text).translate(_ARABIC_DIGITS)
    text = _ARABIC_MARKS.sub("", text).casefold()
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in "LN")


def contains_quote(passage: str, quote: str, *, min_chars: int = 12) -> bool:
    
    needle = squash(quote)
    return len(needle) >= min_chars and needle in squash(passage)


def contains_term(text: str, term: str, *, min_chars: int = 3) -> bool:
    needle = squash(term)
    return len(needle) >= min_chars and needle in squash(text)


def count_term(text: str, term: str) -> int:
    needle = squash(term)
    return squash(text).count(needle) if len(needle) >= 3 else 0



_NUMBER = re.compile(r"(?<![\w.])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+(?:[.,]\d+)?")


def numbers_in(text: str) -> list:
    text = unicodedata.normalize("NFKC", text).translate(_ARABIC_DIGITS).replace("\u2212", "-").replace("\u066b", ".")
    values = []
    for match in _NUMBER.finditer(text):
        raw = match.group(0)
        if re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", raw):
            raw = raw.replace(",", "")
        else:
            raw = raw.replace(",", ".")
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return values


def number_in_text(value: float, text: str) -> bool:
    return any(math.isclose(n, value, rel_tol=1e-9, abs_tol=1e-9) for n in numbers_in(text))
