import re
import unicodedata
from typing import Literal

from pydantic import BaseModel

from enums import ArtifactType
from schemas.artifact_content import SERIES_COLORS, ChartContent
from services.artifacts.base import (
    ArtifactOptions,
    ArtifactResult,
    ArtifactStrategy,
    Insufficient,
    ask_validated,
    build_prompt,
    fit_to_budget,
    source_refs,
)
from services.artifacts.grounding import number_in_text, numbers_in
from services.artifacts.profile import MISMATCH_REASON
from services.artifacts.questions import evidence_chunk, topic_line
from services.retrieval_service import stride_sample

MIN_POINTS = 3
MAX_X_VALUES = 30
MAX_PIE_SLICES = 7
MAX_SERIES = len(SERIES_COLORS)


class _Point(BaseModel):
    x: str
    series: str
    value: float
    passage: int
    quote: str


class _Draft(BaseModel):
    feasible: bool
    chart_type: Literal["bar", "line", "pie", "scatter"]
    title: str
    x_label: str
    y_label: str
    caption: str
    points: list[_Point]


def _slug(text: str, taken: set) -> str:
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    base = re.sub(r"[^a-z0-9]+", "_", folded.lower()).strip("_") or "series"
    key, n = base, 2
    while key in taken:
        key, n = f"{base}_{n}", n + 1
    taken.add(key)
    return key


def _as_number(text: str):
    found = numbers_in(text)
    return found[0] if len(found) == 1 and re.fullmatch(r"\s*[-+\u2212]?[\d.,\u066b\u066c٠-٩]+\s*%?\s*", text) else None


def _tidy(value: float):
    return int(value) if float(value).is_integer() else value


class ChartStrategy(ArtifactStrategy):
    type = ArtifactType.CHART

    def select(self, chunks: list, options: ArtifactOptions) -> list:
        
        ranked = sorted(range(len(chunks)), key=lambda i: -len(numbers_in(chunks[i].content)))[:10]
        rich = [chunks[i] for i in sorted(ranked) if numbers_in(chunks[i].content)]
        return fit_to_budget(rich or stride_sample(chunks, 6))

    def generate(self, llm, chunks: list, options: ArtifactOptions):
        hint = f"\n- Prefer a {options.chart_type} chart if the data allows it." if options.chart_type else ""
        task = (
            f"Plot the numeric data found in the passages.{topic_line(options.topic)}\n"
            "- Use ONLY numbers written in the passages and copy each value exactly. NEVER estimate, compute, "
            "interpolate or invent a value.\n"
            "- If the passages do not contain enough comparable numbers for a chart, set feasible=false and points=[].\n"
            "- Each point: x (the category / label / x value as written), series (the measured quantity, e.g. 'RMSD (Å)'), "
            "value (the number), passage (its number), quote (a verbatim span of that passage that contains the number).\n"
            "- chart_type: bar for comparisons, line for ordered sequences (time, iterations), pie for parts of one whole "
            "(a single series), scatter for numeric x/y pairs.\n"
            f"- title at most 80 characters; x_label and y_label include units; caption: one sentence saying what is plotted, no new claims.{hint}"
        )
        best = {"draft": None, "n": -1}

        def validate(draft: _Draft):
            problems, rows = [], []
            if draft.feasible:
                for p in draft.points:
                    source = evidence_chunk(chunks, p.passage, p.quote, min_chars=6)
                    if source is None:
                        problems.append(f"point {p.x!r}: the quote is not in the passages")
                    elif not number_in_text(p.value, p.quote):
                        problems.append(f"point {p.x!r}: the value {p.value:g} is not in its quote")
                    elif not p.x.strip() or not p.series.strip():
                        problems.append("a point has an empty x or series")
                    else:
                        rows.append((p, source))
            if len(rows) > best["n"]:
                best["draft"], best["n"] = draft, len(rows)  
            return rows, problems

        rows, _ = ask_validated(llm, prompt=build_prompt(chunks, task), draft_model=_Draft, validate=validate,
                                min_valid=MIN_POINTS, give_up=lambda d: not d.feasible)
        if len(rows) < MIN_POINTS or best["draft"] is None:
            return Insufficient(MISMATCH_REASON[self.type])
        return self._build(best["draft"], rows, options)

    # building the frontend shape 

    def _build(self, draft: _Draft, rows: list, options: ArtifactOptions):
        notes = []

        
        counts = {}
        for p, _ in rows:
            counts[p.series.strip()] = counts.get(p.series.strip(), 0) + 1
        keep = set(sorted(counts, key=lambda s: -counts[s])[:MAX_SERIES])
        seen, points = set(), []
        for p, source in rows:
            name, x = p.series.strip(), p.x.strip()
            if name in keep and (x, name) not in seen:
                seen.add((x, name))
                points.append((x, name, p.value, source))
        if len(points) < MIN_POINTS:
            return Insufficient(MISMATCH_REASON[self.type])

        xs = list(dict.fromkeys(x for x, _, _, _ in points))
        numeric_x = all(_as_number(x) is not None for x in xs)
        names = list(dict.fromkeys(name for _, name, _, _ in points))
        wanted = options.chart_type or draft.chart_type

        chart_type = "bar"
        if wanted == "pie" and len(names) == 1 and all(v > 0 for _, _, v, _ in points) and len(xs) >= 3:
            chart_type = "pie"
        elif wanted == "line" and len(xs) >= 3:
            chart_type = "line"
        elif wanted == "scatter" and numeric_x and len(points) >= 4:
            chart_type = "scatter"

        if chart_type == "pie" and len(xs) > MAX_PIE_SLICES:
            top = {x for x, _, _, _ in sorted(points, key=lambda r: -r[2])[:MAX_PIE_SLICES]}
            points = [r for r in points if r[0] in top]
            xs = [x for x in xs if x in top]
            notes.append(f"Showing the {MAX_PIE_SLICES} largest values.")
        elif chart_type in ("bar", "line") and len(xs) > MAX_X_VALUES:
            xs = xs[:MAX_X_VALUES]
            points = [r for r in points if r[0] in set(xs)]
            notes.append(f"Showing the first {MAX_X_VALUES} values.")
        if chart_type == "line" and numeric_x:
            xs.sort(key=lambda x: _as_number(x))

        x_key = "x" if chart_type == "scatter" else "category"
        taken = {x_key}
        keys = {name: _slug(name, taken) for name in names if any(r[1] == name for r in points)}
        series = [
            {"key": keys[name], "label": name[:40], "color": SERIES_COLORS[i % len(SERIES_COLORS)]}
            for i, name in enumerate(n for n in names if n in keys)
        ]

        if chart_type == "scatter":
            data = [{x_key: _as_number(x), keys[name]: _tidy(value)} for x, name, value, _ in points]
        else:
            by_x = {x: {x_key: x} for x in xs}
            for x, name, value, _ in points:
                by_x[x][keys[name]] = _tidy(value)
            data = [by_x[x] for x in xs]

        caption = " ".join(filter(None, [draft.caption.strip()[:160], *notes])) or None
        content = ChartContent(
            chart_type=chart_type, x_key=x_key,
            x_label=draft.x_label.strip()[:60] or None, y_label=draft.y_label.strip()[:60] or None,
            caption=caption, data=data, series=series, sources=source_refs([r[3] for r in points]),
        )
        return ArtifactResult(
            type=self.type, title=draft.title.strip()[:80] or "Chart",
            content=content.model_dump(mode="json", exclude_none=True), caption=caption,
        )
