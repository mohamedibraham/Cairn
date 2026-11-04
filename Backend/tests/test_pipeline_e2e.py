import io
import os
import shutil
import uuid


os.environ.update({
    "DATABASE_URL": os.environ.get("TEST_DB", "postgresql://postgres:postgres@localhost:5432/rag_app_test"),
    "JWT_SECRET_KEY": "test",
    "CHUNK_SIZE_TOKENS": "60",
    "CHUNK_OVERLAP_TOKENS": "10",
    "STORAGE_BASE_PATH": "/tmp/test_uploads",
})
shutil.rmtree("/tmp/test_uploads", ignore_errors=True)

from fastapi.testclient import TestClient  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402
from sqlalchemy import func, select  # noqa: E402

import Backend.app.pipeline.embedding as emb  # noqa: E402
from Backend.app.db.session import SessionLocal  # noqa: E402
from Backend.app.services.answer_service import LLMAnswer  # noqa: E402
from Backend.app.main import app  # noqa: E402
from Backend.app.models.file import Chunk  # noqa: E402


import fake_model  # noqa: E402  (tests/fake_model.py)

fake_model.install()


def ok(msg):
    print(f"  PASS  {msg}")


c = TestClient(app)
email = f"pipe_{uuid.uuid4().hex[:6]}@test.com"
c.post("/api/v1/auth/register", json={"email": email, "full_name": "Pipe", "password": "supersecret1"})
tok = c.post("/api/v1/auth/login", json={"email": email, "password": "supersecret1"}).json()["access_token"]
H = {"Authorization": f"Bearer {tok}"}
ws = c.post("/api/v1/workspaces/", json={"name": "WS"}, headers=H).json()["id"]
proj = c.post(f"/api/v1/workspaces/{ws}/projects/", json={"name": "Pipeline Test"}, headers=H).json()["id"]
F = f"/api/v1/projects/{proj}/files"

PROTEIN = ("AlphaFold predicts protein structure using evoformer attention over multiple sequence alignments. "
           "Protein folding residues distances evoformer structure prediction accuracy. ") * 6
IONS = ("Hodgkin Huxley model describes sodium potassium ion channel gating variables membrane voltage current. "
        "Ion channel kinetics conductance gating activation inactivation. ") * 6
txt_body = (PROTEIN + "\n\n" + IONS).encode()

print("1. Upload TXT -> real extraction, chunking, embedding, indexing")
r = c.post(f"{F}/upload", files={"upload": ("notes.txt", txt_body, "text/plain")}, headers=H)
assert r.status_code == 201, r.text
txt_id = r.json()["id"]
d = c.get(f"{F}/{txt_id}", headers=H).json()
assert d["latest_version"]["status"] == "ready", d["latest_version"]
assert d["latest_version"]["chunk_count"] >= 2, d["latest_version"]["chunk_count"]
ok(f"ready, {d['latest_version']['chunk_count']} chunks")
stages = [(e["stage"], e["status"]) for e in d["latest_version"]["pipeline"]]
assert stages[0] == ("uploaded", "done") and stages[-1] == ("ready", "done"), stages
order = [s for s, _ in stages]
assert order.index("extracting") < order.index("chunking") < order.index("embedding") < order.index("indexing")
ok(f"pipeline events in order: {order}")
ts = [e["created_at"] for e in d["latest_version"]["pipeline"]]
assert ts == sorted(ts) and len(set(ts)) == len(ts), f"timestamps must be distinct & increasing: {ts}"
ok(f"REGRESSION: {len(ts)} pipeline events have {len(set(ts))} distinct, increasing timestamps")
cp = d["chunk_preview"]
assert all(ch["line_start"] and ch["line_end"] >= ch["line_start"] for ch in cp), cp
assert all(ch["page_number"] is None for ch in cp)
ok(f"txt chunks carry line ranges (e.g. L{cp[0]['line_start']}-{cp[0]['line_end']}), no page number")

print("2. Upload PY -> treated as plain text")
py_body = b"def fold(seq):\n    # AlphaFold protein structure evoformer\n    return seq\n" * 12
r = c.post(f"{F}/upload", files={"upload": ("fold_utils.py", py_body, "text/x-python")}, headers=H)
assert r.status_code == 201, r.text
py_id = r.json()["id"]
d = c.get(f"{F}/{py_id}", headers=H).json()
assert d["latest_version"]["status"] == "ready" and d["file_type"] == "python", d
ok("py file ready, file_type=python")

print("3. Upload PDF -> page numbers")
buf = io.BytesIO()
pdf = canvas.Canvas(buf)
for page_text in (IONS, PROTEIN):
    words, y = page_text.split(), 800
    line = []
    for w in words:
        line.append(w)
        if len(line) == 9:
            pdf.drawString(40, y, " ".join(line)); y -= 14; line = []
    if line:
        pdf.drawString(40, y, " ".join(line))
    pdf.showPage()
pdf.save()
r = c.post(f"{F}/upload", files={"upload": ("paper.pdf", buf.getvalue(), "application/pdf")}, headers=H)
assert r.status_code == 201, r.text
pdf_id = r.json()["id"]
d = c.get(f"{F}/{pdf_id}", headers=H).json()
assert d["latest_version"]["status"] == "ready", d["latest_version"]
pages = {ch["page_number"] for ch in d["chunk_preview"]}
assert pages <= {1, 2} and None not in pages, pages
ok(f"pdf ready, chunk page numbers {sorted(pages)}, no line numbers: {all(ch['line_start'] is None for ch in d['chunk_preview'])}")

print("4. Failure handling")
r = c.post(f"{F}/upload", files={"upload": ("fake.txt", b"abc\x00\x01\x02binary", "text/plain")}, headers=H)
assert r.status_code == 201
d = c.get(f"{F}/{r.json()['id']}", headers=H).json()
assert d["latest_version"]["status"] == "failed", d["latest_version"]
assert d["latest_version"]["error_message"] == "File looks binary, not plain text", d["latest_version"]["error_message"]
ok(f"binary-as-txt -> failed with safe message: {d['latest_version']['error_message']!r}")

n_before = len(c.get(f"{F}/", headers=H).json())
r = c.post(f"{F}/upload", files={"upload": ("empty.txt", b"", "text/plain")}, headers=H)
assert r.status_code == 400 and r.json()["error"]["code"] == "EMPTY_FILE", r.text
assert len(c.get(f"{F}/", headers=H).json()) == n_before
ok("empty file -> 400 EMPTY_FILE, no orphan File row left behind")

r = c.post(f"{F}/upload", files={"upload": ("malware.exe", b"MZ....", "application/octet-stream")}, headers=H)
assert r.status_code == 400 and r.json()["error"]["code"] == "UNSUPPORTED_FILE_TYPE", r.text
ok("exe -> 400 UNSUPPORTED_FILE_TYPE")

r = c.post(f"{F}/upload", files={"upload": ("../../etc/evil.txt", b"hello traversal text file content", "text/plain")}, headers=H)
assert r.status_code == 201, r.text
tid = r.json()["id"]
with SessionLocal() as db:
    from Backend.app.models.file import FileVersion
    sp = db.scalar(select(FileVersion.storage_path).where(FileVersion.file_id == uuid.UUID(tid)))
assert os.path.realpath(sp).startswith("/tmp/test_uploads/"), sp
assert c.get(f"{F}/{tid}", headers=H).json()["filename"] == "evil.txt"
ok(f"path traversal neutralised -> stored at {sp.replace('/tmp/test_uploads/', '')}")

print("5. Reprocess replaces chunks atomically; failed retry keeps old chunks")
with SessionLocal() as db:
    vid = db.scalar(select(FileVersion.id).where(FileVersion.file_id == uuid.UUID(txt_id)))
    old_ids = set(db.scalars(select(Chunk.id).where(Chunk.file_version_id == vid)).all())
r = c.post(f"{F}/{txt_id}/reprocess", headers=H)
assert r.status_code == 200, r.text
with SessionLocal() as db:
    new_ids = set(db.scalars(select(Chunk.id).where(Chunk.file_version_id == vid)).all())
assert new_ids and not (old_ids & new_ids) and len(new_ids) == len(old_ids), (len(old_ids), len(new_ids))
assert c.get(f"{F}/{txt_id}", headers=H).json()["latest_version"]["status"] == "ready"
ok(f"reprocess -> {len(new_ids)} fresh chunks, old ones replaced, status ready")

import Backend.app.pipeline.ingestion as ing  # noqa: E402
real_embed = ing.embed_chunks
ing.embed_chunks = lambda texts: (_ for _ in ()).throw(emb.EmbeddingError("simulated GPU failure"))
c.post(f"{F}/{txt_id}/reprocess", headers=H)
ing.embed_chunks = real_embed
d = c.get(f"{F}/{txt_id}", headers=H).json()
with SessionLocal() as db:
    surviving = set(db.scalars(select(Chunk.id).where(Chunk.file_version_id == vid)).all())
assert d["latest_version"]["status"] == "failed" and d["latest_version"]["error_message"] == "simulated GPU failure"
assert surviving == new_ids, "old chunks must survive a failed retry"
ok("embedding failure -> status failed w/ message, previous chunks untouched")
c.post(f"{F}/{txt_id}/reprocess", headers=H)
assert c.get(f"{F}/{txt_id}", headers=H).json()["latest_version"]["status"] == "ready"
ok("subsequent retry recovers to ready")

print("6. New version")
v2 = (PROTEIN + " brand new unique zebra sentence about giraffes").encode()
r = c.post(f"{F}/{txt_id}/versions", files={"upload": ("notes.txt", v2, "text/plain")}, headers=H)
assert r.status_code == 201, r.text
d = c.get(f"{F}/{txt_id}", headers=H).json()
assert d["latest_version"]["version_number"] == 2 and d["latest_version"]["status"] == "ready"
assert len(d["version_history"]) == 2
ok("v2 created & ready, version history has 2 entries")

print("7. Search (real pgvector cosine search over real chunks)")
S = f"/api/v1/workspaces/{ws}/search"
r = c.post(f"{S}/", json={"query": "hodgkin huxley sodium potassium ion channel gating", "scope": "workspace"}, headers=H)
assert r.status_code == 200, r.text
res = r.json()["results"]
assert res and "Hodgkin" in res[0]["excerpt"] or "Ion channel" in res[0]["excerpt"], res[0]
scores = [x["score"] for x in res]
assert scores == sorted(scores, reverse=True) and 0 < scores[0] <= 1, scores
ok(f"top hit relevant ({res[0]['filename']}), scores descending {scores[:3]}")
r = c.post(f"{S}/", json={"query": "giraffes zebra", "scope": "project", "project_id": proj}, headers=H)
assert "zebra" in r.json()["results"][0]["excerpt"], r.json()["results"][0]
ok("v2's unique text is searchable")
r = c.post(f"{S}/", json={"query": "hodgkin huxley sodium potassium ion channel gating", "scope": "project", "project_id": proj}, headers=H)
stale = [x for x in r.json()["results"] if x["filename"] == "notes.txt" and "Hodgkin" in x["excerpt"]]
assert not stale, "v1 chunks (which contained the Hodgkin text) must not be searchable once v2 is current"
ok("v1 chunks of notes.txt are NOT returned -> only the current version is searchable")
r = c.post(f"{S}/", json={"query": "protein", "scope": "workspace"}, headers=H)
hist = c.get(f"{S}/history", headers=H).json()
assert len(hist) == 4 and hist[0]["query_text"] == "protein" and hist[1]["project_name"] == "Pipeline Test"
ok("search history recorded (4 entries, project name resolved)")

print("8. Chat: retrieval + citations (LLM stubbed in-test)")
C = f"/api/v1/projects/{proj}/conversations"
conv = c.post(f"{C}/", json={"title": "t", "scope": "entire_project"}, headers=H).json()["id"]
r = c.post(f"{C}/{conv}/messages", json={"content": "How does AlphaFold use evoformer?"}, headers=H)
assert r.status_code == 503 and r.json()["error"]["code"] == "LLM_NOT_CONFIGURED", r.text
assert "GEMINI" not in r.text and "key" not in r.json()["error"]["message"].lower()  # no config details to clients
ok("first-run experience: real retrieval ran, then 503 LLM_NOT_CONFIGURED (no GEMINI_API_KEY) with a safe message")

import Backend.app.services.conversation_service as cs  # noqa: E402
captured = {}


def fake_llm(question, context):
    captured["ctx"] = context
    return LLMAnswer(text="Evoformer uses MSAs [1] and attention [2].",
                     citations={1: context[0].chunk_id, 2: context[1].chunk_id})


cs.generate_answer = fake_llm
r = c.post(f"{C}/{conv}/messages", json={"content": "How does AlphaFold use evoformer?"}, headers=H)
assert r.status_code == 201, r.text
m = r.json()
assert len(m["citations"]) == 2 and [x["citation_index"] for x in m["citations"]] == [1, 2], m
assert all(x["filename"] and x["excerpt"] for x in m["citations"])
ok(f"assistant reply with {len(m['citations'])} citations -> {m['citations'][0]['filename']}")
assert all(ch.content for ch in captured["ctx"]) and len(captured["ctx"]) <= 6

msgs = c.get(f"{C}/{conv}/messages", headers=H).json()
users_before = sum(1 for x in msgs if x["role"] == "user")
r = c.post(f"{C}/{conv}/messages/{m['id']}/regenerate", headers=H)
assert r.status_code == 200, r.text
msgs = c.get(f"{C}/{conv}/messages", headers=H).json()
assert sum(1 for x in msgs if x["role"] == "user") == users_before, "regenerate must not duplicate the question"
assert sum(1 for x in msgs if x["role"] == "assistant") == 1 and msgs[-1]["id"] != m["id"]
mts = [x["created_at"] for x in msgs]
assert mts == sorted(mts) and len(set(mts)) == len(mts), mts
ok("regenerate replaced the answer, no duplicated user message; message timestamps distinct & ordered")

cs.generate_answer = lambda q, ctx: LLMAnswer(text="no markers here", citations={3: ctx[0].chunk_id})
r = c.post(f"{C}/{conv}/messages", json={"content": "x y z protein"}, headers=H)
assert r.status_code == 500 and r.json()["error"]["code"] == "LLM_CITATION_MISMATCH", r.text
ok("hallucinated citation marker rejected (LLM_CITATION_MISMATCH)")

print("9. File-subset conversation only retrieves from chosen files")
cs.generate_answer = fake_llm
sub = c.post(f"{C}/", json={"title": "sub", "scope": "file_subset", "file_ids": [py_id]}, headers=H)
assert sub.status_code == 201, sub.text
r = c.post(f"{C}/{sub.json()['id']}/messages", json={"content": "protein structure evoformer"}, headers=H)
assert r.status_code == 201, r.text
assert {x.filename for x in captured["ctx"]} == {"fold_utils.py"}, {x.filename for x in captured["ctx"]}
ok("context limited to fold_utils.py even though other files match better")

print("10. Artifact engine through the real API (real chunks in Postgres, scripted LLM)")
import Backend.app.services.artifact_service as arts  # noqa: E402
from Backend.app.llm.providers.fake import FakeLLMClient  # noqa: E402
from Backend.app.enums import ArtifactType as AT  # noqa: E402
from Backend.app.llm import LLMRateLimitError as _RateLimit  # noqa: E402
from Backend.app.schemas.artifact_content import parse_content  # noqa: E402
from test_artifact_engine import DRAFTS  # noqa: E402  (scripted model output for each type)
from test_artifact_study_types import P1, P2, P3  # noqa: E402
from test_artifact_analysis_types import chart_draft  # noqa: E402
from test_artifact_analysis_types import FLOW, T1  # noqa: E402

proj3 = c.post(f"/api/v1/workspaces/{ws}/projects/", json={"name": "Artifact Project"}, headers=H).json()["id"]
F3, A3 = f"/api/v1/projects/{proj3}/files", f"/api/v1/projects/{proj3}/artifacts"
rich = "\n\n".join([P1, P2, P3, T1, FLOW]).encode()
SHAPES = (b"class Shape:\n    def area(self):\n        return 0\n\n"
          b"class Circle(Shape):\n    def area(self):\n        return 3\n    def describe(self):\n        return self.area()\n\n"
          b"class Square(Shape):\n    def area(self):\n        return 4\n    def describe(self):\n        return self.area()\n")
rich_id = c.post(f"{F3}/upload", files={"upload": ("rich.txt", rich, "text/plain")}, headers=H).json()["id"]
shapes_id = c.post(f"{F3}/upload", files={"upload": ("shapes.py", SHAPES, "text/x-python")}, headers=H).json()["id"]
for fid in (rich_id, shapes_id):
    assert c.get(f"{F3}/{fid}", headers=H).json()["latest_version"]["status"] == "ready"


def gen(body, drafts=()):
    fake = FakeLLMClient(list(drafts))
    arts.get_llm_client = lambda: fake
    return c.post(f"{A3}/generate", json=body, headers=H), fake


for t in AT:
    r, fake = gen({"type": t.value, "scope_type": "file", "scope_file_id": rich_id}, DRAFTS[t])
    assert r.status_code == 201, (t, r.status_code, r.text)
    body = r.json()
    parse_content(body["content"])
    assert body["type"] == t.value and body["title"] and body["content"]["sources"]
ok("all 7 artifact types generated from real chunks, persisted, and valid against the frontend contract")

listed = c.get(f"{A3}/", headers=H).json()
assert {a["type"] for a in listed} == {t.value for t in AT} and len(listed) == 7
assert len(c.get(f"{A3}/?type=chart", headers=H).json()) == 1
chart_id = next(a["id"] for a in listed if a["type"] == "chart")
dl = c.get(f"{A3}/{chart_id}/download", headers=H)
assert dl.status_code == 200 and '"chart_type"' in dl.text
ok("list, filter by type and download work on the structured content")

r, fake = gen({"type": "keywords", "scope_type": "file", "scope_file_id": rich_id}, DRAFTS[AT.KEYWORDS])
prompt = fake.calls[0].messages[0].content
assert "rich.txt" in prompt and "shapes.py" not in prompt
r, fake = gen({"type": "keywords", "scope_type": "project"}, DRAFTS[AT.KEYWORDS])
assert "rich.txt" in fake.calls[0].messages[0].content and "shapes.py" in fake.calls[0].messages[0].content
ok("scope isolation: a file-scoped artifact only ever sees that file's passages")

r, fake = gen({"type": "flashcards", "scope_type": "file", "scope_file_id": rich_id, "topic": "recycling structure", "count": 4},
              DRAFTS[AT.FLASHCARDS])
assert r.status_code == 201 and "Focus on: recycling structure" in fake.calls[0].messages[0].content
ok("topic is embedded, used to pick the nearest chunks, and reaches the prompt")

r, fake = gen({"type": "diagram", "scope_type": "file", "scope_file_id": shapes_id})
body = r.json()
assert r.status_code == 201 and fake.calls == [], (r.status_code, r.text, fake.calls)
labels = {n["id"]: n["label"] for n in body["content"]["nodes"]}
assert set(labels.values()) == {"Shape", "Circle", "Square"} and body["title"] == "Class diagram — shapes.py"
assert {(labels[e["source"]], labels[e["target"]], e["label"]) for e in body["content"]["edges"]} == {
    ("Circle", "Shape", "inherits from"), ("Square", "Shape", "inherits from")}
ok("Python class diagram built from chunk_metadata in Postgres with ZERO LLM calls")

r, fake = gen({"type": "summary", "scope_type": "file", "scope_file_id": shapes_id})
err = r.json()["error"]
assert r.status_code == 422 and err["code"] == "ARTIFACT_NOT_SUPPORTED" and fake.calls == []
assert err["details"] == {"requested": "summary", "alternatives": ["diagram"]} and "no continuous prose" in err["message"]
ok("source code has no prose: 422 mismatch with reason + alternatives, and no LLM call was spent")

r, fake = gen({"type": "chart", "scope_type": "file", "scope_file_id": rich_id}, [chart_draft([], feasible=False)])
assert r.status_code == 422 and r.json()["error"]["code"] == "ARTIFACT_NOT_SUPPORTED" and "chart" not in r.json()["error"]["details"]["alternatives"]
r, fake = gen({"type": "chart", "scope_type": "file", "scope_file_id": rich_id, "force": True}, [chart_draft([], feasible=False)])
assert r.status_code == 422 and r.json()["error"]["code"] == "ARTIFACT_INSUFFICIENT_DATA"
ok("grounding failure -> mismatch card data; the same failure with force=true -> a plain error")

empty = c.post(f"/api/v1/workspaces/{ws}/projects/", json={"name": "Empty"}, headers=H).json()["id"]
r = c.post(f"/api/v1/projects/{empty}/artifacts/generate", json={"type": "summary"}, headers=H)
assert r.status_code == 400 and r.json()["error"]["code"] == "NO_CONTENT"
r = c.post(f"{A3}/generate", json={"type": "summary", "scope_type": "file"}, headers=H)
assert r.status_code == 400 and r.json()["error"]["code"] == "SCOPE_FILE_ID_REQUIRED"
r, _ = gen({"type": "keywords", "scope_type": "project"}, [_RateLimit("429", retry_after=5)])
assert r.status_code == 429 and r.headers["retry-after"] == "5"
ok("empty project / missing file id are clean 400s; an LLM rate limit surfaces as 429 + Retry-After")

print("11. Python files: structure-aware chunking end-to-end (separate project)")
proj2 = c.post(f"/api/v1/workspaces/{ws}/projects/", json={"name": "Python Project"}, headers=H).json()["id"]
F2 = f"/api/v1/projects/{proj2}/files"
PY_SRC = (
    '"""Pipeline helpers."""\n'
    "import os\n\n"
    "LIMIT = 10\n\n"
    "@cache_it\n"
    "def normalise(text: str, *, strict: bool = False) -> str:\n"
    '    """Normalise whitespace."""\n'
    "    return os.linesep.join(text.split())\n\n"
    "class Tokenizer(Base):\n"
    '    """Splits text."""\n'
    "    def split(self, text):\n"
    "        return normalise(text).split()\n\n"
    "    def count(self, text):\n"
    "        return len(self.split(text))\n"
).encode()
r = c.post(f"{F2}/upload", files={"upload": ("helpers_mod.py", PY_SRC, "text/x-python")}, headers=H)
assert r.status_code == 201, r.text
py2_id = r.json()["id"]
d = c.get(f"{F2}/{py2_id}", headers=H).json()
assert d["latest_version"]["status"] == "ready", d["latest_version"]
with SessionLocal() as db:
    rows = db.execute(
        select(Chunk.chunk_index, Chunk.line_start, Chunk.line_end, Chunk.chunk_metadata)
        .join(FileVersion, FileVersion.id == Chunk.file_version_id)
        .where(FileVersion.file_id == uuid.UUID(py2_id)).order_by(Chunk.chunk_index)
    ).all()
metas = {m["symbol"]: (a, b, m) for _, a, b, m in rows}
assert {"normalise", "Tokenizer", "Tokenizer.split", "Tokenizer.count", "<module>"} <= set(metas), set(metas)
assert metas["normalise"][2]["signature"].replace(" ", "") == "defnormalise(text:str,*,strict:bool=False)->str"
assert metas["normalise"][2]["decorators"] == ["cache_it"] and metas["normalise"][2]["imports_used"] == ["os"]
_lines = PY_SRC.decode().split("\n")
_a, _b = metas["normalise"][0], metas["normalise"][1]
assert _lines[_a - 1] == "@cache_it" and _lines[_b - 1].strip().startswith("return os.linesep"), (_a, _b)  # decorator..return
assert metas["Tokenizer.split"][2]["calls"] == ["normalise"] and metas["Tokenizer.count"][2]["calls"] == ["Tokenizer.split"]
ok(f"py file ready with {len(rows)} structural chunks: {sorted(metas)}")
assert any(ch["symbol"] for ch in d["chunk_preview"]) and d["chunk_preview"][0]["chunk_type"], d["chunk_preview"]
ok("chunk_preview exposes symbol/chunk_type")

S2 = f"/api/v1/workspaces/{ws}/search"
r = c.post(f"{S2}/", json={"query": "Tokenizer.count len split", "scope": "project", "project_id": proj2}, headers=H)
top = r.json()["results"][0]
assert top["symbol"] == "Tokenizer.count", top
ok(f"search finds the method chunk and reports its symbol: {top['symbol']}")

C2 = f"/api/v1/projects/{proj2}/conversations"
conv2 = c.post(f"{C2}/", json={"title": "py", "scope": "entire_project"}, headers=H).json()["id"]
cs.generate_answer = lambda q, ctx: LLMAnswer(text="It counts tokens [1].", citations={1: ctx[0].chunk_id})
m2 = c.post(f"{C2}/{conv2}/messages", json={"content": "Tokenizer.count len split"}, headers=H).json()
assert m2["citations"][0]["symbol"] == "Tokenizer.count" and m2["citations"][0]["line_start"] is not None, m2
ok("chat citation carries the symbol and a real line number")

BAD = b"def fine():\n    return 1\n\ndef broken(:\n    pass\n"
r = c.post(f"{F2}/upload", files={"upload": ("broken.py", BAD, "text/x-python")}, headers=H)
bad_id = r.json()["id"]
d = c.get(f"{F2}/{bad_id}", headers=H).json()
assert d["latest_version"]["status"] == "ready" and d["latest_version"]["chunk_count"] >= 1, d["latest_version"]
with SessionLocal() as db:
    metas_bad = [m for (m,) in db.execute(
        select(Chunk.chunk_metadata).join(FileVersion, FileVersion.id == Chunk.file_version_id)
        .where(FileVersion.file_id == uuid.UUID(bad_id))).all()]
assert all(m["parse_status"] == "fallback" and "syntax error" in m["parse_error"].lower() for m in metas_bad)
ok("syntax-error .py does not fail the upload: ready, chunked as plain text, flagged parse_status=fallback")

r = c.post(f"{S2}/", json={"query": "def broken fine return", "scope": "project", "project_id": proj2}, headers=H)
assert r.json()["results"] and r.json()["results"][0]["filename"] == "broken.py"
ok("the broken file is still searchable")

# other file types keep chunk_metadata NULL
with SessionLocal() as db:
    n_meta = db.scalar(
        select(func.count()).select_from(Chunk).join(FileVersion, FileVersion.id == Chunk.file_version_id)
        .where(FileVersion.file_id == uuid.UUID(pdf_id), Chunk.chunk_metadata.is_not(None)))
assert n_meta == 0
ok("PDF chunks are untouched (chunk_metadata stays NULL)")

print("12. LLM layer through the real API: provider switch, citations, error mapping")
import Backend.app.llm.factory as lf  # noqa: E402
import Backend.app.services.answer_service as ans  # noqa: E402
from Backend.app.helpers import settings  # noqa: E402
from Backend.app.llm import (  # noqa: E402
    LLMContentBlockedError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from Backend.app.llm.providers.fake import FakeLLMClient  # noqa: E402

cs.generate_answer = ans.generate_answer  # drop the in-test stand-in used by steps 8-10
settings.LLM_PROVIDER = "fake"  # same switch a user flips in .env — no code involved
lf.reset_llm_client()
r = c.post(f"{C2}/{conv2}/messages", json={"content": "Tokenizer.count len split"}, headers=H)
assert r.status_code == 201, r.text
m3 = r.json()
assert m3["content"].startswith("(fake LLM)") and m3["citations"][0]["citation_index"] == 1, m3
assert m3["citations"][0]["filename"] and m3["citations"][0]["line_start"] is not None
ok("LLM_PROVIDER=fake: retrieval -> real RAG prompt -> provider -> citations parsed from [n]")

for error, status_code, code in [
    (LLMRateLimitError("429 from provider, key AIza-SECRET", retry_after=7), 429, "LLM_RATE_LIMITED"),
    (LLMTimeoutError("took 60s"), 504, "LLM_TIMEOUT"),
    (LLMContentBlockedError("safety"), 422, "LLM_CONTENT_BLOCKED"),
    (LLMInvalidResponseError("junk"), 502, "LLM_INVALID_RESPONSE"),
]:
    ans.get_llm_client = lambda e=error: FakeLLMClient([e])
    r = c.post(f"{C2}/{conv2}/messages", json={"content": "Tokenizer.count len split"}, headers=H)
    body = r.json()
    assert r.status_code == status_code and body["error"]["code"] == code, (r.status_code, r.text)
    assert "AIza" not in r.text and "provider" not in body["error"]["message"].lower()
    if status_code == 429:
        assert r.headers["retry-after"] == "7" and body["error"]["details"]["retry_after_seconds"] == 7
    last = c.get(f"{C2}/{conv2}/messages", headers=H).json()[-1]
    assert last["role"] == "user" and last["content"] == "Tokenizer.count len split"  # the question is never lost
ok("LLMError -> 429(+Retry-After) / 504 / 422 / 502, safe messages, user question still saved")
ans.get_llm_client = lf.get_llm_client

print("\nALL CHECKS PASSED")
