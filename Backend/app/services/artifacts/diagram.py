import re
import unicodedata

from pydantic import BaseModel

from enums import ArtifactType
from schemas.artifact_content import DiagramContent
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
from services.artifacts.grounding import contains_term
from services.artifacts.layout import layered_layout
from services.artifacts.profile import _RELATION_CUES, MISMATCH_REASON
from services.artifacts.questions import topic_line
from services.retrieval_service import stride_sample

MAX_NODES = 12
MIN_NODES = 3
MIN_EDGES = 2
MAX_LABEL = 28
MAX_EDGE_LABEL = 24


class _Node(BaseModel):
    id: str
    label: str


class _Edge(BaseModel):
    source: str
    target: str
    label: str


class _Draft(BaseModel):
    title: str
    nodes: list[_Node]
    edges: list[_Edge]


def short(label: str, limit: int = MAX_LABEL) -> str:
    label = " ".join(label.split())
    if len(label) <= limit:
        return label
    cut = label[: limit - 1].rsplit(" ", 1)[0] or label[: limit - 1]
    return cut.rstrip(",;:-") + "…"


def _slug(text: str, taken: set) -> str:
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    base = re.sub(r"[^a-z0-9]+", "_", folded.lower()).strip("_") or "n"
    key, n = base, 2
    while key in taken:
        key, n = f"{base}_{n}", n + 1
    taken.add(key)
    return key


def _is_python_scope(chunks: list) -> bool:
    return bool(chunks) and all(c.file_type == "python" and (c.metadata or {}).get("parse_status") == "ok" for c in chunks)


def compose(title: str, labels: dict, edge_triples: list, chunks_used: list, caption=None):
    
    keys = list(labels)
    coords = layered_layout(keys, [(s, t) for s, t, _ in edge_triples])
    nodes = [{"id": k, "label": labels[k], "x": coords[k][0], "y": coords[k][1]} for k in keys]
    edges = [{"id": f"e{i}", "source": s, "target": t, **({"label": lbl} if lbl else {})}
             for i, (s, t, lbl) in enumerate(edge_triples, 1)]
    content = DiagramContent(nodes=nodes, edges=edges, sources=source_refs(chunks_used))
    return ArtifactResult(type=ArtifactType.DIAGRAM, title=title[:80],
                          content=content.model_dump(mode="json", exclude_none=True), caption=caption)


def _top_by_degree(keys: list, edges: list, limit: int, bonus: dict | None = None) -> set:
    degree = {k: (bonus or {}).get(k, 0) for k in keys}
    for s, t, *_ in edges:
        degree[s] += 1
        degree[t] += 1
    return set(sorted(keys, key=lambda k: (-degree[k], keys.index(k)))[:limit])


class DiagramStrategy(ArtifactStrategy):
    type = ArtifactType.DIAGRAM

    def select(self, chunks: list, options: ArtifactOptions) -> list:
        if _is_python_scope(chunks):
            return list(chunks)  # the structure route needs every chunk's metadata
        ranked = sorted(range(len(chunks)), key=lambda i: -len(_RELATION_CUES.findall(chunks[i].content)))[:10]
        cued = [chunks[i] for i in sorted(ranked) if _RELATION_CUES.search(chunks[i].content)]
        return fit_to_budget(cued or stride_sample(chunks, 8))

    def generate(self, llm, chunks: list, options: ArtifactOptions):
        if _is_python_scope(chunks):
            built = self._python_structure(chunks)
            if built is not None:
                return built
            chunks = fit_to_budget(stride_sample(chunks, 10))
        return self._from_llm(llm, chunks, options)

    # Python: deterministic 

    def _python_structure(self, chunks: list):
        classes = {c.symbol: c for c in chunks if c.chunk_type == "class" and c.symbol}
        edges, externals = [], {}
        for name, chunk in classes.items():
            for base in (chunk.metadata or {}).get("bases", []):
                if "=" in base:  
                    continue
                if base not in classes:
                    externals[base] = chunk
                edges.append((name, base, "inherits from"))
        names = list(classes) + [b for b in externals if b not in classes]
        if len(names) >= MIN_NODES and edges:
            methods = {n: sum(1 for c in chunks if c.chunk_type == "method" and (c.metadata or {}).get("parent") == n) for n in classes}
            keep = _top_by_degree(names, edges, MAX_NODES, methods)
            edges = [e for e in edges if e[0] in keep and e[1] in keep]
            names = [n for n in names if n in keep and any(n in (s, t) for s, t, _ in edges)]
            if len(names) >= MIN_NODES and len(edges) >= MIN_EDGES - 1:
                files = {c.filename for c in chunks}
                title = f"Class diagram — {next(iter(files))}" if len(files) == 1 else "Class diagram"
                taken = set()
                keys = {n: _slug(n, taken) for n in names}
                used = [classes[n] for n in names if n in classes]
                return compose(title, {keys[n]: short(n) for n in names},
                               [(keys[s], keys[t], lbl) for s, t, lbl in edges], used)

        symbols = {c.symbol: c for c in chunks if c.chunk_type in ("function", "method") and c.symbol}
        calls = [(s, t, "calls") for s, c in symbols.items() for t in (c.metadata or {}).get("calls", []) if t in symbols]
        if len(calls) >= MIN_EDGES:
            keep = _top_by_degree(list(symbols), calls, MAX_NODES)
            calls = [e for e in calls if e[0] in keep and e[1] in keep]
            names = [n for n in symbols if n in keep and any(n in (s, t) for s, t, _ in calls)]
            if len(names) >= MIN_NODES and len(calls) >= MIN_EDGES:
                files = {c.filename for c in chunks}
                title = f"Call graph — {next(iter(files))}" if len(files) == 1 else "Call graph"
                taken = set()
                keys = {n: _slug(n, taken) for n in names}
                return compose(title, {keys[n]: short(n) for n in names},
                               [(keys[s], keys[t], lbl) for s, t, lbl in calls], [symbols[n] for n in names])
        return None

    # everything else: LLM extracts, code validates 

    def _from_llm(self, llm, chunks: list, options: ArtifactOptions):
        task = (
            f"Draw a diagram of the structure, flow or relationships described in the passages.{topic_line(options.topic)}\n"
            f"- 4 to 10 nodes. Each node is one entity, component or step named as in the passages; label at most {MAX_LABEL} characters.\n"
            "- edges: source -> target. The viewer draws NO arrowheads, so the edge label must be a short verb phrase that "
            "reads from source to target ('feeds', 'depends on', 'produces').\n"
            "- Every node must be connected. Do not invent components that the passages do not mention.\n"
            "- id: a short unique lowercase identifier; title at most 80 characters."
        )

        best = {"draft": None, "n": -1}

        def validate(draft: _Draft):
            problems, taken, labels, idmap = [], set(), {}, {}
            for node in draft.nodes:
                label = short(node.label)
                words = [w for w in re.findall(r"[^\W\d_]{4,}", label, re.UNICODE)]
                if not label or (words and not any(contains_term(c.content, w) for c in chunks for w in words)):
                    problems.append(f"node '{label}' is not grounded in the passages")
                    continue
                key = _slug(node.id or label, taken)
                idmap[node.id] = key
                labels[key] = label
            edges, seen = [], set()
            for e in draft.edges:
                s, t = idmap.get(e.source), idmap.get(e.target)
                if s is None or t is None or s == t or (s, t) in seen:
                    continue
                seen.add((s, t))
                edges.append((s, t, short(e.label, MAX_EDGE_LABEL) if e.label.strip() else ""))
            connected = {k for s, t, _ in edges for k in (s, t)}
            labels = {k: v for k, v in labels.items() if k in connected}
            if len(labels) > MAX_NODES:
                keep = _top_by_degree(list(labels), edges, MAX_NODES)
                labels = {k: v for k, v in labels.items() if k in keep}
                edges = [e for e in edges if e[0] in keep and e[1] in keep]
            if len(labels) < MIN_NODES or len(edges) < MIN_EDGES:
                problems.append("need at least 3 connected nodes and 2 edges that the passages support")
                return [], problems
            if len(labels) > best["n"]:
                best["draft"], best["n"] = draft, len(labels)
            return [(labels, edges)], problems

        found, _ = ask_validated(llm, prompt=build_prompt(chunks, task), draft_model=_Draft, validate=validate, min_valid=1)
        if not found:
            return Insufficient(MISMATCH_REASON[self.type])
        labels, edges = found[0]
        return compose(best["draft"].title.strip() or "Diagram", labels, edges, chunks)
