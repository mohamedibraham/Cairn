"""Summary / Chart / Diagram strategies and the layout algorithm.
Run:  python -m pytest tests/test_artifact_analysis_types.py"""

import random

import pytest

from Backend.app.llm.providers.fake import FakeLLMClient
from Backend.app.schemas.artifact_content import parse_content
from Backend.app.services.artifacts.base import ArtifactOptions, ArtifactResult, Insufficient
from Backend.app.services.artifacts.chart import ChartStrategy
from Backend.app.services.artifacts.diagram import DiagramStrategy, short
from Backend.app.services.artifacts.layout import layered_layout
from Backend.app.services.artifacts.summary import (
    SummaryStrategy,
    to_supported_markdown,
    unsupported_numbers,
)
from Backend.app.services.retrieval_service import ScopedChunk


def chunk(n, text, file_type="pdf", filename="doc.pdf", **kw):
    return ScopedChunk(chunk_id=f"c{n}", file_id="f1", filename=filename, file_type=file_type,
                       chunk_index=n, content=text, page_number=n if file_type == "pdf" else None, **kw)


def run(strategy, drafts, chunks, options=None):
    fake = FakeLLMClient(drafts)
    return strategy.generate(fake, chunks, options or ArtifactOptions()), fake


def valid(result):
    assert isinstance(result, ArtifactResult)
    parse_content(result.content)
    return result.content


# ================================ summary ============================================

def test_sanitizer_rewrites_everything_the_viewer_cannot_render():
    raw = (
        "#### Deep heading\n"
        "1. First point [1]\n2) Second point\n"
        "* star bullet\n  - nested bullet\n"
        "> quoted text\n"
        "| Name | Value |\n|---|---|\n| RMSD | 6.4 |\n"
        "See [the paper](http://x.org) and ![img](a.png) and *italic* and __bold__ and _under_ and $x^2$.\n"
        "```python\nprint('hi')\n```\n"
        "Keep `arr[1]` and **bold** intact [2]."
    )
    out = to_supported_markdown(raw)
    lines = out.split("\n")
    assert lines[0] == "### Deep heading"  # levels clamp to 3
    assert "- First point" in lines and "- Second point" in lines and "- star bullet" in lines and "- nested bullet" in lines
    assert "quoted text" in lines and not any(l.startswith(">") for l in lines)
    assert "- RMSD — 6.4" in lines and not any("---" in l for l in lines)  # table -> bullet
    assert any("See the paper and img and italic and **bold** and under and x^2." == l for l in lines)
    assert "print('hi')" in lines and not any("```" in l for l in lines)
    assert "Keep `arr[1]` and **bold** intact." in lines  # code span untouched, marker outside removed
    assert "[1]" not in out.replace("`arr[1]`", "") and "[2]" not in out


def test_sanitizer_is_idempotent_and_leaves_supported_markdown_alone():
    supported = "## Overview\n\nA paragraph with **bold** and `code`.\n\n### Details\n\n- one\n- two"
    assert to_supported_markdown(supported) == supported
    messy = "1. a\n> b\n| x | y |\n|--|--|\n| 1 | 2 |"
    once = to_supported_markdown(messy)
    assert to_supported_markdown(once) == once


def test_unsupported_numbers_only_flags_figures_missing_from_the_sources():
    sources = "Accuracy was 92.5 percent on 1,200 targets in 2021."
    assert unsupported_numbers("Accuracy 92.5 on 1200 targets in 2021, three key points, 2 groups.", sources) == []
    assert unsupported_numbers("Accuracy rose by 37 percent to 99.1", sources) == [37.0, 99.1]


SRC_TEXT = ("AlphaFold predicts protein structure using an Evoformer stack. Accuracy reached 92.5 percent on the benchmark. "
            "The structure module refines backbone frames. Recycling improves the prediction over several rounds. ") * 2
SRC = [chunk(1, SRC_TEXT), chunk(2, SRC_TEXT.replace("92.5", "88.1"))]
GOOD_SUMMARY = ("## Overview\n\nAlphaFold predicts protein structure with an Evoformer stack and reaches 92.5 percent accuracy on the "
                "benchmark [1].\n\n## Key points\n\n- The structure module refines backbone frames.\n"
                "- Recycling improves the prediction over several rounds.\n- Accuracy of 88.1 was reported for the second set.")


def test_summary_is_sanitized_and_carries_its_sources():
    result, fake = run(SummaryStrategy(), [GOOD_SUMMARY], SRC)
    content = valid(result)
    assert "[1]" not in content["body"] and "## Overview" in content["body"] and "- Recycling improves" in content["body"]
    assert {s["chunk_id"] for s in content["sources"]} == {"c1", "c2"} and result.title == "Summary"
    assert len(fake.calls) == 1 and "<passages>" in fake.calls[0].messages[0].content
    assert "NO tables" in fake.calls[0].system  # the viewer's limits are part of the instructions too


def test_an_invented_number_triggers_one_retry_naming_it():
    invented = GOOD_SUMMARY + "\n- Accuracy improved by 37 percent over the baseline."
    result, fake = run(SummaryStrategy(), [invented, GOOD_SUMMARY], SRC)
    assert len(fake.calls) == 2 and "37" in fake.calls[1].messages[0].content
    assert "37" not in valid(result)["body"]


def test_if_the_number_survives_the_retry_the_offending_lines_are_removed():
    invented = GOOD_SUMMARY + "\n- Accuracy improved by 37 percent over the baseline."
    result, fake = run(SummaryStrategy(), [invented, invented], SRC)
    body = valid(result)["body"]
    assert len(fake.calls) == 2 and "37" not in body and "Recycling improves" in body


def test_a_useless_reply_is_insufficient():
    result, _ = run(SummaryStrategy(), ["Too short."], SRC)
    assert isinstance(result, Insufficient)


def test_long_scopes_use_a_bounded_map_reduce():
    big = [chunk(i, ("Sentence about the alignment and structure. " * 22)) for i in range(40)]  # ~40k chars
    partial = "- a key fact about the structure and alignment of proteins"
    final = "## Overview\n\n" + " ".join(["The alignment and the structure are summarised here."] * 6)
    result, fake = run(SummaryStrategy(), [partial] * 3 + [final], big)
    assert len(fake.calls) == 4  # 3 map batches + 1 reduce
    assert "<notes>" in fake.calls[-1].messages[0].content and "<passages>" not in fake.calls[-1].messages[0].content
    valid(result)

    huge = [chunk(i, ("Sentence about the alignment and structure. " * 22)) for i in range(300)]
    _, fake2 = run(SummaryStrategy(), [partial] * 6 + [final], huge)
    assert len(fake2.calls) <= 7  # never more than 6 map calls + 1 reduce, however big the scope


# ================================ chart ==============================================

T1 = ("Backbone accuracy improved with recycling. RMSD was 6.4 Å at iteration 0, 3.1 Å at iteration 1, 1.9 Å at iteration 2 "
      "and 1.4 Å at iteration 3. The pLDDT confidence rose from 62 to 78, 86 and finally 91 over the same iterations.")
CH = [chunk(1, T1)]


def pt(x, series, value, quote, passage=1):
    return {"x": x, "series": series, "value": value, "passage": passage, "quote": quote}


RMSD = [pt("0", "RMSD (Å)", 6.4, "RMSD was 6.4 Å at iteration 0"), pt("1", "RMSD (Å)", 3.1, "3.1 Å at iteration 1"),
        pt("2", "RMSD (Å)", 1.9, "1.9 Å at iteration 2"), pt("3", "RMSD (Å)", 1.4, "1.4 Å at iteration 3")]


def chart_draft(points, chart_type="line", feasible=True, **kw):
    base = dict(feasible=feasible, chart_type=chart_type, title="RMSD by iteration", x_label="iteration",
                y_label="RMSD (Å)", caption="Backbone RMSD per recycling iteration.", points=points)
    base.update(kw)
    return base


def test_chart_builds_the_wide_form_the_frontend_reads():
    result, _ = run(ChartStrategy(), [chart_draft(RMSD)], CH)
    content = valid(result)
    assert content["chart_type"] == "line" and content["x_key"] == "category"
    assert content["data"] == [{"category": "0", "rmsd_a": 6.4}, {"category": "1", "rmsd_a": 3.1},
                               {"category": "2", "rmsd_a": 1.9}, {"category": "3", "rmsd_a": 1.4}]
    assert content["series"] == [{"key": "rmsd_a", "label": "RMSD (Å)", "color": "var(--primary)"}]
    assert content["y_label"] == "RMSD (Å)" and result.title == "RMSD by iteration" and content["sources"]


def test_multiple_series_share_rows_and_get_palette_colors():
    plddt = [pt("0", "pLDDT", 62, "pLDDT confidence rose from 62"), pt("1", "pLDDT", 78, "62 to 78"),
             pt("2", "pLDDT", 86, "78, 86 and finally 91"), pt("3", "pLDDT", 91, "86 and finally 91")]
    content = valid(run(ChartStrategy(), [chart_draft(RMSD + plddt)], CH)[0])
    assert content["data"][0] == {"category": "0", "rmsd_a": 6.4, "plddt": 62}
    assert [s["color"] for s in content["series"]] == ["var(--primary)", "var(--trace)"]
    assert isinstance(content["data"][0]["plddt"], int)  # integral values stay integers


def test_invented_values_and_fake_quotes_never_reach_the_chart():
    forged = [pt("4", "RMSD (Å)", 0.9, "RMSD was 0.9 Å at iteration 4"),  # quote not in the passage
              pt("5", "RMSD (Å)", 7.7, "RMSD was 6.4 Å at iteration 0")]  # real quote, but 7.7 isn't in it
    content = valid(run(ChartStrategy(), [chart_draft(RMSD + forged)], CH)[0])
    assert [r["category"] for r in content["data"]] == ["0", "1", "2", "3"]


def test_not_enough_verified_points_is_insufficient_and_gets_one_corrective_retry():
    result, fake = run(ChartStrategy(), [chart_draft(RMSD[:2])] * 2, CH)
    assert isinstance(result, Insufficient) and len(fake.calls) == 2
    assert "REJECTED" in fake.calls[1].messages[0].content  # the retry says why the first answer was refused


def test_a_model_that_says_there_is_no_data_is_believed_without_a_retry():
    result, fake = run(ChartStrategy(), [chart_draft([], feasible=False)], CH)
    assert isinstance(result, Insufficient) and len(fake.calls) == 1


@pytest.mark.parametrize("asked, points, expected", [
    ("pie", RMSD, "pie"),  # 4 positive values, one series
    ("line", RMSD[:2] + [pt("2", "RMSD (Å)", 1.9, "1.9 Å at iteration 2")], "line"),
])
def test_requested_types_are_kept_when_the_data_allows_them(asked, points, expected):
    assert valid(run(ChartStrategy(), [chart_draft(points, asked)], CH)[0])["chart_type"] == expected


def test_a_chart_type_the_data_cannot_support_falls_back_to_bar():
    two_series = RMSD[:3] + [pt("0", "pLDDT", 62, "pLDDT confidence rose from 62"), pt("1", "pLDDT", 78, "62 to 78")]
    assert valid(run(ChartStrategy(), [chart_draft(two_series, "pie")], CH)[0])["chart_type"] == "bar"  # pie = one series
    named = [pt("a", "RMSD (Å)", 6.4, "RMSD was 6.4 Å at iteration 0"), pt("b", "RMSD (Å)", 3.1, "3.1 Å at iteration 1"),
             pt("c", "RMSD (Å)", 1.9, "1.9 Å at iteration 2"), pt("d", "RMSD (Å)", 1.4, "1.4 Å at iteration 3")]
    assert valid(run(ChartStrategy(), [chart_draft(named, "scatter")], CH)[0])["chart_type"] == "bar"  # x not numeric
    two_x = [pt("0", "RMSD (Å)", 6.4, "RMSD was 6.4 Å at iteration 0"), pt("1", "RMSD (Å)", 3.1, "3.1 Å at iteration 1"),
             pt("0", "pLDDT", 62, "pLDDT confidence rose from 62"), pt("1", "pLDDT", 78, "62 to 78")]
    assert valid(run(ChartStrategy(), [chart_draft(two_x, "line")], CH)[0])["chart_type"] == "bar"  # a line needs >= 3 x values
    assert valid(run(ChartStrategy(), [chart_draft(RMSD[:3][:2] + [RMSD[2]], "scatter")], CH)[0])["chart_type"] == "bar"  # < 4 points


def test_scatter_uses_one_row_per_point_with_numeric_x():
    content = valid(run(ChartStrategy(), [chart_draft(RMSD, "scatter")], CH)[0])
    assert content["chart_type"] == "scatter" and content["x_key"] == "x"
    assert content["data"][0] == {"x": 0, "rmsd_a": 6.4} and len(content["data"]) == 4


def test_numeric_x_values_are_sorted_for_line_charts():
    shuffled = [RMSD[2], RMSD[0], RMSD[3], RMSD[1]]
    assert [r["category"] for r in valid(run(ChartStrategy(), [chart_draft(shuffled, "line")], CH)[0])["data"]] == ["0", "1", "2", "3"]


def test_series_keys_are_unique_even_for_non_latin_names_and_x_key_collisions():
    text = "Counts: 12 in group one and 15 in group two and 9 in group three and 21 in group four."
    pts = [pt("one", "عدد", 12, "12 in group one", 1), pt("two", "عدد", 15, "15 in group two", 1),
           pt("three", "category", 9, "9 in group three", 1), pt("four", "category", 21, "21 in group four", 1)]
    content = valid(run(ChartStrategy(), [chart_draft(pts, "bar")], [chunk(1, text)])[0])
    keys = [s["key"] for s in content["series"]]
    assert len(set(keys)) == len(keys) and content["x_key"] not in keys


def test_pie_is_capped_at_seven_slices_and_says_so():
    text = " ".join(f"Item {chr(97 + i)} accounts for {10 + i} units." for i in range(10))
    pts = [pt(chr(97 + i), "units", 10 + i, f"Item {chr(97 + i)} accounts for {10 + i} units", 1) for i in range(10)]
    result, _ = run(ChartStrategy(), [chart_draft(pts, "pie")], [chunk(1, text)])
    content = valid(result)
    assert content["chart_type"] == "pie" and len(content["data"]) == 7 and "7 largest" in content["caption"]


def test_percent_values_are_accepted_when_written_as_in_the_text():
    text = "Adoption reached 45% in 2020, 60% in 2021 and 75% in 2022."
    pts = [pt("2020", "adoption (%)", 45, "reached 45% in 2020", 1), pt("2021", "adoption (%)", 60, "60% in 2021", 1),
           pt("2022", "adoption (%)", 75, "75% in 2022", 1)]
    assert len(valid(run(ChartStrategy(), [chart_draft(pts, "bar")], [chunk(1, text)])[0])["data"]) == 3


def test_chart_selection_prefers_number_rich_chunks_in_document_order():
    chunks = [chunk(1, "No numbers here, just words about proteins."), chunk(2, T1), chunk(3, "Nothing numeric either.")]
    assert [c.chunk_id for c in ChartStrategy().select(chunks, ArtifactOptions())] == ["c2"]


# ================================ layout =============================================

def test_chain_layers_run_left_to_right():
    coords = layered_layout(["a", "b", "c"], [("a", "b"), ("b", "c")])
    assert [coords[n][0] for n in "abc"] == [0, 230, 460] and {coords[n][1] for n in "abc"} == {0}


def test_fan_out_centres_siblings_around_the_parent():
    coords = layered_layout(["a", "b", "c", "d"], [("a", "b"), ("a", "c"), ("a", "d")])
    assert coords["a"] == (0, 0) and sorted(coords[n][1] for n in "bcd") == [-90, 0, 90]
    assert {coords[n][0] for n in "bcd"} == {230}


def test_cycles_do_not_break_layout():
    coords = layered_layout(["a", "b", "c"], [("a", "b"), ("b", "c"), ("c", "a")])
    assert sorted(coords[n][0] for n in "abc") == [0, 230, 460]


def test_disconnected_parts_each_start_at_layer_zero():
    coords = layered_layout(["a", "b", "x", "y"], [("a", "b"), ("x", "y")])
    assert coords["a"][0] == coords["x"][0] == 0 and coords["b"][0] == coords["y"][0] == 230
    assert coords["a"][1] != coords["x"][1]


def test_layout_is_deterministic_and_never_overlaps_nodes():
    rng = random.Random(7)
    for _ in range(50):
        n = rng.randint(3, 12)
        ids = [f"n{i}" for i in range(n)]
        edges = [(rng.choice(ids), rng.choice(ids)) for _ in range(rng.randint(2, 20))]
        first, second = layered_layout(ids, edges), layered_layout(ids, edges)
        assert first == second and set(first) == set(ids)
        assert len(set(first.values())) == n  # no two nodes on the same spot


# ================================ diagram (LLM route) ================================

FLOW = ("Raw files flow to the extractor, which feeds the chunker. The chunker produces chunks that depend on the embedder. "
        "The embedder output is stored in the search index, and the index serves every query.")
DG = [chunk(1, FLOW)]


def node(i, label):
    return {"id": i, "label": label}


def edge(s, t, label="feeds"):
    return {"source": s, "target": t, "label": label}


def diagram_draft(nodes, edges, title="Ingestion flow"):
    return {"title": title, "nodes": nodes, "edges": edges}


GOOD_NODES = [node("raw", "Raw files"), node("ext", "Extractor"), node("chk", "Chunker"), node("emb", "Embedder"), node("idx", "Search index")]
GOOD_EDGES = [edge("raw", "ext"), edge("ext", "chk"), edge("chk", "emb", "depends on"), edge("emb", "idx", "stored in")]


def test_diagram_is_validated_laid_out_and_follows_the_contract():
    result, fake = run(DiagramStrategy(), [diagram_draft(GOOD_NODES, GOOD_EDGES)], DG)
    content = valid(result)
    assert [n["label"] for n in content["nodes"]] == ["Raw files", "Extractor", "Chunker", "Embedder", "Search index"]
    xs = {n["id"]: n["x"] for n in content["nodes"]}
    assert xs["raw"] < xs["ext"] < xs["chk"] < xs["emb"] < xs["idx"]  # flow reads left to right
    assert {e["label"] for e in content["edges"]} == {"feeds", "depends on", "stored in"}
    assert "NO arrowheads" in fake.calls[0].messages[0].content  # direction must live in the label


def test_bad_nodes_and_edges_are_cleaned_up():
    nodes = GOOD_NODES + [node("ghost", "Quantum flux capacitor"), node("lonely", "Index"), node("long", "The extractor component that reads raw files from disk")]
    edges = GOOD_EDGES + [edge("raw", "nowhere"), edge("ext", "ext"), edge("raw", "ext", "duplicate"), edge("ghost", "idx"), edge("long", "chk")]
    content = valid(run(DiagramStrategy(), [diagram_draft(nodes, edges)], DG)[0])
    labels = [n["label"] for n in content["nodes"]]
    assert "Quantum flux capacitor" not in labels  # not grounded in the passages
    assert all(len(l) <= 28 for l in labels) and any(l.endswith("…") for l in labels)
    assert len({(e["source"], e["target"]) for e in content["edges"]}) == len(content["edges"])
    assert all(e["source"] != e["target"] for e in content["edges"])
    assert "Index" not in labels  # the unconnected node was dropped (only "Search index" remains)
    assert "Search index" in labels


def test_oversized_diagrams_keep_the_best_connected_twelve():
    nodes = [node(f"n{i}", f"Extractor {i}") for i in range(16)]
    edges = [edge(f"n{i}", "n0") for i in range(1, 16)] + [edge("n1", "n2")]
    content = valid(run(DiagramStrategy(), [diagram_draft(nodes, edges)], DG)[0])
    assert len(content["nodes"]) == 12 and "n0" in {n["id"] for n in content["nodes"]}


def test_too_small_a_graph_is_insufficient_after_one_retry():
    result, fake = run(DiagramStrategy(), [diagram_draft(GOOD_NODES[:2], GOOD_EDGES[:1])] * 2, DG)
    assert isinstance(result, Insufficient) and len(fake.calls) == 2


def test_labels_are_shortened_at_word_boundaries():
    assert short("short") == "short"
    cut = short("Membrane capacitance and conductance model")
    assert len(cut) <= 28 and cut.endswith("…") and " " not in cut[-3:]


# ================================ diagram (Python route) =============================

def pychunk(n, symbol, chunk_type, **meta):
    return ScopedChunk(chunk_id=f"p{n}", file_id="f9", filename="shapes.py", file_type="python", chunk_index=n, content=f"# {symbol}",
                       metadata={"parse_status": "ok", "chunk_type": chunk_type, "symbol": symbol, **meta})


def test_python_class_diagram_is_built_from_ast_metadata_without_calling_the_llm():
    chunks = [pychunk(1, "Shape", "class", bases=[]), pychunk(2, "Circle", "class", bases=["Shape"]),
              pychunk(3, "Square", "class", bases=["Shape"]), pychunk(4, "Model", "class", bases=["BaseModel"]),
              pychunk(5, "Shape.area", "method", parent="Shape")]
    result, fake = run(DiagramStrategy(), [], DiagramStrategy().select(chunks, ArtifactOptions()))
    content = valid(result)
    assert fake.calls == [] and result.title == "Class diagram — shapes.py"
    labels = {n["id"]: n["label"] for n in content["nodes"]}
    assert set(labels.values()) == {"Shape", "Circle", "Square", "Model", "BaseModel"}  # external base is shown too
    pairs = {(labels[e["source"]], labels[e["target"]]) for e in content["edges"]}
    assert pairs == {("Circle", "Shape"), ("Square", "Shape"), ("Model", "BaseModel")}
    assert {e["label"] for e in content["edges"]} == {"inherits from"}


def test_python_without_classes_gets_a_call_graph():
    chunks = [pychunk(1, "load", "function", calls=["parse", "clean"]), pychunk(2, "parse", "function", calls=["clean"]),
              pychunk(3, "clean", "function", calls=[])]
    result, fake = run(DiagramStrategy(), [], chunks)
    content = valid(result)
    assert fake.calls == [] and result.title == "Call graph — shapes.py" and {e["label"] for e in content["edges"]} == {"calls"}
    assert len(content["edges"]) == 3


def test_python_scope_selection_keeps_every_chunk_for_the_structure_route():
    chunks = [pychunk(i, f"f{i}", "function") for i in range(30)]
    assert len(DiagramStrategy().select(chunks, ArtifactOptions())) == 30
