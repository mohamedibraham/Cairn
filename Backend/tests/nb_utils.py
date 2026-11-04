"""
nb_utils.py - shared helpers for the RAG test notebooks.

Place this file next to the notebooks, in:  <your-app-folder>/notebooks/
(your-app-folder = the folder you run `uvicorn main:app` from).

The ONLY part you normally need to edit is the CONFIG block below.
"""
from __future__ import annotations

import contextlib
import inspect
import io
import json
import math
import os
import platform
import re
import statistics
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
from IPython.display import HTML, Markdown, display

NB_DIR = Path(__file__).resolve().parent
ROOT = NB_DIR.parent
REPORTS = NB_DIR / "reports"
REPORTS.mkdir(exist_ok=True)

# =============================================================================
# CONFIG  -  adapt to your project (run notebook 01 first: it prints which
# routes were / were not found so you know exactly what to change).
# =============================================================================
CONFIG = {
    # env vars forced BEFORE the app is imported (use the fake LLM in tests!)
    "ENV_OVERRIDES": {"LLM_PROVIDER": "fake"},
    "APP_IMPORT": "main:app",
    # key -> (HTTP method, regex matched against the OpenAPI path)
    "ROUTES": {
        "register": ("POST", r"/auth/register/?$"),
        "login": ("POST", r"/auth/login/?$"),
        "refresh": ("POST", r"/auth/refresh/?$"),
        "logout": ("POST", r"/auth/logout/?$"),
        "me": ("GET", r"/(auth|users)/me/?$"),
        "workspace_create": ("POST", r"/workspaces/?$"),
        "workspace_get": ("GET", r"/workspaces/\{[^/]+\}/?$"),
        "workspace_delete": ("DELETE", r"/workspaces/\{[^/]+\}/?$"),
        "project_create": ("POST", r"/workspaces/\{[^/]+\}/projects/?$|/projects/?$"),
        "project_get": ("GET", r"/projects/\{[^/]+\}/?$"),
        "file_upload": ("POST", r"/files/?$|/files/upload/?$|/upload/?$"),
        "file_get": ("GET", r"/files/\{[^/]+\}/?$"),
        "file_delete": ("DELETE", r"/files/\{[^/]+\}/?$"),
        "file_process": ("POST", r"/files/\{[^/]+\}/(process|ingest|reprocess)/?$"),
        "search": ("POST", r"/search/?$|/projects/\{[^/]+\}/search/?$"),
        "conversation_create": ("POST", r"/conversations/?$|/projects/\{[^/]+\}/conversations/?$"),
        "conversation_get": ("GET", r"/conversations/\{[^/]+\}/?$"),
        "message_send": ("POST", r"/conversations/\{[^/]+\}/messages/?$"),
        "artifact_create": ("POST", r"/artifacts/?$|/projects/\{[^/]+\}/artifacts/?$"),
        "artifact_list": ("GET", r"/artifacts/?$|/projects/\{[^/]+\}/artifacts/?$"),
        "health": ("GET", r"/health/?$|/healthz/?$"),
    },
    # request bodies - edit if your Pydantic schemas use other field names
    "PAYLOADS": {
        "register": lambda email, pw: {"email": email, "password": pw, "full_name": "Test User"},
        "login": lambda email, pw: {"email": email, "password": pw},
        "workspace": lambda name: {"name": name, "description": "created by notebook tests"},
        "project": lambda name: {"name": name, "description": "created by notebook tests"},
        "search": lambda q, project_id, file_ids=None, top_k=5: {
            "query": q, "project_id": project_id, "top_k": top_k,
            **({"file_ids": file_ids} if file_ids else {}),
        },
        "conversation": lambda project_id, title="Test chat": {"title": title, "project_id": project_id},
        "message": lambda text, requested_type=None: {
            "content": text, **({"requested_type": requested_type} if requested_type else {})
        },
        "artifact": lambda t, scope, title, content: {
            "artifact_type": t, "scope": scope, "title": title, "content": content
        },
    },
    "UPLOAD_FIELD": "file",              # multipart field name of the upload endpoint
    "READY_STATUSES": {"READY"},
    "FAILED_STATUSES": {"FAILED", "ERROR"},
    "POLL_TIMEOUT": 240,                 # seconds to wait for ingestion
    "CHUNK_TABLE": None,                 # None = auto-detect (table having a vector column)
    "CHUNK_VECTOR_COL": "embedding",
    "CHUNK_FILE_FK": "file_id",
    "EMBEDDING_MODEL": "intfloat/multilingual-e5-base",
    "EMBED_DIM": 768,
    "MAX_CHUNK_CHARS": 2500,
    "METRICS_PATH": "/metrics",
    "PUBLIC_PATH_REGEX": r"^/(docs|redoc|openapi\.json|health|healthz|metrics)?/?$|/auth/(register|login|refresh)",
    "THRESHOLDS": {
        "hit_at_3": 0.75,                # min retrieval hit@3 on the golden set
        "search_p95_ms": 1500,
        "scrape_p95_ms": 300,
        "max_series_growth_per_new_path": 0,   # label-cardinality guard
    },
}
PAYLOADS = CONFIG["PAYLOADS"]


# =============================================================================
# Small utilities
# =============================================================================
class Skip(Exception):
    """Raise inside rec.case(...) to mark the test as SKIPPED."""


class Warn(Exception):
    """Raise inside rec.case(...) to mark the test as WARNING (not a failure)."""


def need(*xs):
    """Skip the current test when a previous step did not produce its object."""
    if any(x is None for x in xs):
        raise Skip("a previous step failed, so this one cannot run")


def mask(key: str, value):
    if re.search(r"key|secret|password|passwd|token|credential", str(key), re.I):
        return "***" if value else value
    if isinstance(value, str) and "://" in value and "@" in value:
        return re.sub(r"://([^:/]+):[^@]+@", r"://\1:***@", value)
    return value


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100
    f, c = math.floor(k), math.ceil(k)
    return xs[int(k)] if f == c else xs[f] + (xs[c] - xs[f]) * (k - f)


def h(title: str, level: int = 3):
    display(Markdown("#" * level + " " + title))


def safe_json(resp):
    try:
        return resp.json()
    except Exception:
        return {}


def unwrap(js):
    """If the API wraps payloads as {"data": ...}, return the inner value."""
    if isinstance(js, dict) and "data" in js and len(js) <= 4 and js["data"] is not None:
        return js["data"]
    return js


def find_key(obj, key):
    """Depth-first search for the first value stored under `key`."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def flatten_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from flatten_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from flatten_strings(v)
    elif obj is not None and not isinstance(obj, (int, float, bool)):
        yield str(obj)


def find_hits(js):
    """Return the list of result dicts from a search response (shape-agnostic)."""
    js = unwrap(js)
    if isinstance(js, list):
        return [x for x in js if isinstance(x, dict)]
    if isinstance(js, dict):
        for k in ("results", "hits", "items", "chunks", "matches", "sources"):
            if isinstance(js.get(k), list):
                return [x for x in js[k] if isinstance(x, dict)]
        for v in js.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    return []


def hit_blob(hit) -> str:
    return " ".join(flatten_strings(hit)).lower()


def rank_of(hits, keyword: str):
    kw = keyword.lower()
    for i, hit in enumerate(hits, 1):
        if kw in hit_blob(hit):
            return i
    return None


def get_id(resp):
    js = unwrap(safe_json(resp))
    if isinstance(js, dict) and js.get("id") is not None:
        return js["id"]
    return find_key(js, "id")


def fill(path: str, *ids) -> str:
    for i in ids:
        path = re.sub(r"\{[^}]+\}", str(i), path, count=1)
    return path


# =============================================================================
# Test recorder (pretty results table + JSON report)
# =============================================================================
class _Case:
    def __init__(self):
        self.notes = []

    def note(self, *a):
        self.notes.append(" ".join(str(x) for x in a))


_COLORS = {"PASS": "#d4edda", "FAIL": "#f8d7da", "ERROR": "#f5b7b1", "WARN": "#fff3cd", "SKIP": "#e2e3e5"}
_ICONS = {"PASS": "✅", "FAIL": "❌", "ERROR": "💥", "WARN": "⚠️", "SKIP": "⏭️"}


class Recorder:
    def __init__(self, notebook: str):
        self.notebook = notebook
        self.rows: list[dict] = []

    # -- recording -----------------------------------------------------------
    def _add(self, name, status, details, category, ms):
        self.rows.append(dict(category=category, test=name, status=status,
                              ms=round(ms, 1) if ms is not None else None, details=details))
        line = f"{_ICONS[status]} [{status:5}] {name}"
        if details:
            line += f"   ->  {details[:220]}"
        print(line)

    @contextlib.contextmanager
    def case(self, name: str, category: str = ""):
        c, t0, status, err = _Case(), time.perf_counter(), "PASS", ""
        try:
            yield c
        except Skip as e:
            status, err = "SKIP", str(e)
        except Warn as e:
            status, err = "WARN", str(e)
        except AssertionError as e:
            status, err = "FAIL", str(e) or "assertion failed"
        except Exception as e:  # noqa: BLE001
            status, err = "ERROR", f"{type(e).__name__}: {e}"
            tb = traceback.format_exc().strip().splitlines()
            c.note("| " + " | ".join(x.strip() for x in tb[-4:-1]))
        ms = (time.perf_counter() - t0) * 1000
        details = " ; ".join(x for x in [err] + c.notes if x)
        self._add(name, status, details, category, ms)

    def check(self, name, cond, details="", category="", level="fail"):
        status = "PASS" if cond else ("WARN" if level == "warn" else "FAIL")
        self._add(name, status, details, category, None)
        return bool(cond)

    def skip(self, name, reason, category=""):
        self._add(name, "SKIP", reason, category, None)

    # -- reporting -----------------------------------------------------------
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["category", "test", "status", "ms", "details"])

    def counts(self) -> dict:
        d = self.df()
        return {s: int((d.status == s).sum()) for s in _COLORS} if len(d) else {s: 0 for s in _COLORS}

    def table(self, only: str | None = None):
        d = self.df()
        if only:
            d = d[d.status == only]
        if d.empty:
            display(HTML("<i>no rows</i>"))
            return

        def color(v):
            return f"background-color:{_COLORS.get(v, '')}; font-weight:600; text-align:center"

        sty = d.style.hide(axis="index") if hasattr(d.style, "hide") else d.style
        mapper = sty.map if hasattr(sty, "map") else sty.applymap
        sty = mapper(color, subset=["status"])
        sty = sty.set_properties(subset=["details"], **{"white-space": "pre-wrap", "min-width": "340px"})
        display(sty)

    def summary(self):
        c, total = self.counts(), max(1, len(self.rows))
        scored = c["PASS"] + c["FAIL"] + c["ERROR"]
        rate = 100 * c["PASS"] / scored if scored else 0.0
        cards = "".join(
            f"<div style='background:{_COLORS[s]};padding:10px 18px;border-radius:10px;text-align:center;min-width:90px'>"
            f"<div style='font-size:26px;font-weight:700'>{c[s]}</div><div>{_ICONS[s]} {s}</div></div>"
            for s in _COLORS)
        display(HTML(
            f"<h3>{self.notebook} &mdash; {len(self.rows)} checks, pass-rate {rate:.0f}% "
            f"(PASS / (PASS+FAIL+ERROR))</h3><div style='display:flex;gap:12px;flex-wrap:wrap'>{cards}</div>"))
        return rate

    def save(self):
        path = REPORTS / f"{self.notebook}.json"
        path.write_text(json.dumps(
            {"notebook": self.notebook, "finished_at": datetime.now().isoformat(timespec="seconds"),
             "counts": self.counts(), "rows": self.rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report saved -> {path}")


# =============================================================================
# Environment / app / API client
# =============================================================================
def setup():
    for k, v in CONFIG["ENV_OVERRIDES"].items():
        os.environ.setdefault(k, v)
    os.chdir(ROOT)
    for p in (str(ROOT), str(NB_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    pd.set_option("display.max_colwidth", 200)
    pd.set_option("display.width", 200)
    return {"root": str(ROOT), "python": platform.python_version(), "os": platform.platform()}


_APP = None


def get_app():
    global _APP
    if _APP is None:
        import importlib
        mod, attr = CONFIG["APP_IMPORT"].split(":")
        _APP = getattr(importlib.import_module(mod), attr)
    return _APP


def resolve_routes(app) -> dict:
    paths = app.openapi().get("paths", {})
    out = {}
    for key, (method, pat) in CONFIG["ROUTES"].items():
        found = None
        for p, ops in paths.items():
            if method.lower() in ops and re.search(pat, p):
                found = p
                break
        out[key] = (method, found)
    return out


class User:
    def __init__(self, email, password):
        self.email, self.password = email, password
        self.token = self.refresh = self.id = None


class Api:
    def __init__(self, app, routes):
        from fastapi.testclient import TestClient
        self.app = app
        self.client = TestClient(app, raise_server_exceptions=False)
        self.routes = routes
        self.log: list[dict] = []

    def call(self, method, path, token=None, **kw):
        headers = dict(kw.pop("headers", {}) or {})
        if token:
            headers["Authorization"] = f"Bearer {token}"
        t0 = time.perf_counter()
        r = self.client.request(method, path, headers=headers, **kw)
        self.log.append(dict(method=method, path=path, status=r.status_code,
                             ms=round((time.perf_counter() - t0) * 1000, 1)))
        return r

    def path(self, key, *ids):
        m, p = self.routes.get(key, (None, None))
        if not p:
            raise Skip(f"route '{key}' not found in OpenAPI - adjust CONFIG['ROUTES'] in nb_utils.py")
        return m, fill(p, *ids)

    def route(self, key, *ids, token=None, **kw):
        m, p = self.path(key, *ids)
        return self.call(m, p, token=token, **kw)

    # -- users ---------------------------------------------------------------
    def login(self, u: User):
        r = self.route("login", json=PAYLOADS["login"](u.email, u.password))
        if r.status_code == 422:  # OAuth2 form style
            r = self.route("login", data={"username": u.email, "password": u.password})
        assert r.status_code in (200, 201), f"login failed {r.status_code}: {r.text[:200]}"
        js = safe_json(r)
        u.token, u.refresh = find_key(js, "access_token"), find_key(js, "refresh_token")
        assert u.token, "login response has no access_token"
        return r

    def new_user(self, prefix="nbtest") -> User:
        email = f"{prefix}_{uuid.uuid4().hex[:8]}@example.com"
        u = User(email, "Str0ng!Passw0rd#" + uuid.uuid4().hex[:4])
        r = self.route("register", json=PAYLOADS["register"](u.email, u.password))
        assert r.status_code in (200, 201), f"register failed {r.status_code}: {r.text[:200]}"
        u.id = get_id(r)
        self.login(u)
        return u

    # -- domain helpers ------------------------------------------------------
    def make_workspace(self, u, name=None):
        r = self.route("workspace_create", token=u.token, json=PAYLOADS["workspace"](name or f"ws-{uuid.uuid4().hex[:6]}"))
        assert r.status_code in (200, 201), f"workspace create {r.status_code}: {r.text[:200]}"
        return get_id(r)

    def make_project(self, u, ws_id, name=None):
        m, p = self.path("project_create")
        body = PAYLOADS["project"](name or f"proj-{uuid.uuid4().hex[:6]}")
        if "{" in p:
            r = self.call(m, fill(p, ws_id), token=u.token, json=body)
        else:
            r = self.call(m, p, token=u.token, json={**body, "workspace_id": ws_id})
        assert r.status_code in (200, 201), f"project create {r.status_code}: {r.text[:200]}"
        return get_id(r)

    def upload(self, u, project_id, name, data, mime="application/octet-stream", extra=None):
        m, p = self.path("file_upload")
        form = dict(extra or {})
        if "{" in p:
            path = fill(p, project_id)
        else:
            path = p
            form.setdefault("project_id", str(project_id))
        return self.call(m, path, token=u.token, data=form,
                         files={CONFIG["UPLOAD_FIELD"]: (name, data, mime)})

    def wait_ready(self, u, file_id, timeout=None, kick=True):
        timeout, t0, kicked, status = timeout or CONFIG["POLL_TIMEOUT"], time.perf_counter(), False, ""
        while time.perf_counter() - t0 < timeout:
            r = self.route("file_get", file_id, token=u.token)
            status = str(find_key(safe_json(r), "status") or "").upper().split(".")[-1]
            dt = time.perf_counter() - t0
            if status in CONFIG["READY_STATUSES"] or status in CONFIG["FAILED_STATUSES"]:
                return status, dt
            if kick and not kicked and dt > 3 and self.routes.get("file_process", (None, None))[1]:
                try:
                    self.route("file_process", file_id, token=u.token)
                except Skip:
                    pass
                kicked = True
            time.sleep(0.5)
        return status or "TIMEOUT", time.perf_counter() - t0

    def search(self, u, project_id, q, file_ids=None, top_k=5):
        m, p = self.path("search")
        body = PAYLOADS["search"](q, project_id, file_ids, top_k)
        if "{" in p:
            body.pop("project_id", None)
            p = fill(p, project_id)
        if m == "GET":
            return self.call(m, p, token=u.token, params={k: v for k, v in body.items() if v is not None})
        return self.call(m, p, token=u.token, json=body)

    def make_conversation(self, u, project_id, title="Test chat"):
        m, p = self.path("conversation_create")
        body = PAYLOADS["conversation"](project_id, title)
        if "{" in p:
            body.pop("project_id", None)
            p = fill(p, project_id)
        r = self.call(m, p, token=u.token, json=body)
        assert r.status_code in (200, 201), f"conversation create {r.status_code}: {r.text[:200]}"
        return get_id(r)

    def send_message(self, u, conv_id, text, requested_type=None):
        return self.route("message_send", conv_id, token=u.token,
                          json=PAYLOADS["message"](text, requested_type))


def get_api() -> Api:
    app = get_app()
    return Api(app, resolve_routes(app))


# =============================================================================
# Database helpers
# =============================================================================
def get_session():
    try:
        from db.session import SessionLocal
    except Exception as e:  # noqa: BLE001
        raise Skip(f"db.session.SessionLocal not importable: {e}")
    return SessionLocal()


def find_vector_columns(session):
    from sqlalchemy import inspect as sa_inspect
    insp, out = sa_inspect(session.get_bind()), []
    for t in insp.get_table_names():
        for c in insp.get_columns(t):
            if "vector" in str(c["type"]).lower():
                out.append((t, c["name"], str(c["type"])))
    return out


def chunk_table(session):
    if CONFIG["CHUNK_TABLE"]:
        return CONFIG["CHUNK_TABLE"], CONFIG["CHUNK_VECTOR_COL"]
    cols = find_vector_columns(session)
    if not cols:
        raise Skip("no table with a pgvector column found")
    return cols[0][0], cols[0][1]


def text_column(session, table):
    from sqlalchemy import inspect as sa_inspect
    names = [c["name"] for c in sa_inspect(session.get_bind()).get_columns(table)]
    for cand in ("content", "text", "chunk_text", "body"):
        if cand in names:
            return cand
    return None


def chunk_stats(session, file_id):
    from sqlalchemy import text
    t, v = chunk_table(session)
    fk = CONFIG["CHUNK_FILE_FK"]
    row = session.execute(text(
        f'select count(*), count("{v}"), min(vector_dims("{v}")), max(vector_dims("{v}")) '
        f'from "{t}" where "{fk}"::text = :f'), {"f": str(file_id)}).one()
    return dict(total=row[0], embedded=row[1], min_dim=row[2], max_dim=row[3])


# =============================================================================
# Test corpus (fictional facts, so no LLM/embedding can "know" them already)
# =============================================================================
_EN_FILLER = [
    "Greenhouse tomatoes need steady humidity and about six hours of indirect light each day.",
    "The old harbour lighthouse was repainted in spring and its lamp replaced with an LED array.",
    "A sourdough starter should be fed twice daily when kept at room temperature.",
    "Medieval bridges were often built with a central pier to split the force of the river current.",
    "Marathon runners usually taper their weekly mileage during the final two weeks before a race.",
    "The library moved its archive of regional newspapers to a climate-controlled basement.",
    "Basalt columns form when thick lava flows cool slowly and contract into hexagonal prisms.",
    "Urban beekeeping has grown steadily, with rooftop hives producing mild floral honey.",
]


def _en_doc() -> str:
    facts = [
        "The Orion-7 reactor operates at 450 degrees Celsius during normal production runs.",
        "The coolant used in the Orion-7 reactor is liquid helium, circulated by two redundant pumps.",
        "Maintenance of the Orion-7 reactor is required every 90 days, and the inspection takes two shifts.",
        "The chief engineer of the Orion-7 programme is Dr. Selma Hartwell, who joined the team in 2019.",
    ]
    parts = []
    for i, f in enumerate(facts):
        block = " ".join(_EN_FILLER[(i * 2 + j) % len(_EN_FILLER)] for j in range(4))
        parts.append(f"Section {i + 1}\n{block} {f} {_EN_FILLER[(i + 5) % len(_EN_FILLER)]}")
    return "Orion-7 Technical Manual\n\n" + "\n\n".join(parts) + "\n"


_AR_DOC = (
    "الخطة المالية لمشروع النيل الأزرق\n\n"
    "يهدف المشروع إلى تطوير منصة لوجستية حديثة تربط الموانئ الداخلية بالمراكز الصناعية. "
    "ميزانية مشروع النيل الأزرق هي 4.5 مليون جنيه موزعة على ثلاث مراحل رئيسية.\n\n"
    "يبدأ التنفيذ في مارس 2027 وينتهي بعد ثمانية عشر شهرا من تاريخ البدء. "
    "تشمل المرحلة الأولى دراسة الجدوى وتجهيز البنية التحتية للمخازن.\n\n"
    "المدير المسؤول عن المشروع هو أحمد فؤاد، ويقدم تقريرا شهريا إلى لجنة المتابعة. "
    "تعتمد الخطة على التمويل الذاتي بنسبة ستين في المئة والباقي قرض ميسر.\n\n"
    "تشمل المخاطر الرئيسية تأخر توريد المعدات وتقلب أسعار الصرف خلال فترة التنفيذ.\n"
)

_PY_DOC = '''"""Inventory helpers for the demo warehouse."""


class InventoryManager:
    """Keeps track of stock levels for a single warehouse."""

    def __init__(self, safety_stock: int = 20):
        self.safety_stock = safety_stock
        self.items = {}

    def add_item(self, sku: str, quantity: int) -> None:
        self.items[sku] = self.items.get(sku, 0) + quantity

    def calculate_reorder_point(self, daily_demand: float, lead_time_days: int) -> float:
        """Reorder point = demand during lead time + safety stock."""
        return daily_demand * lead_time_days + self.safety_stock


def format_report(manager: InventoryManager) -> str:
    return ", ".join(f"{k}:{v}" for k, v in sorted(manager.items.items()))
'''

_PDF_P1 = ("Regional Operations Report\n\nThis report summarises operations across all regions. "
           "Staffing levels were stable throughout the year and warehouse capacity was sufficient.")
_PDF_P2 = ("Customer Retention\n\nThe quarterly churn rate was 3.7 percent in the Lyon region, "
           "slightly below the company average. Marketing spend remained flat.")


def make_pdf() -> bytes | None:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open()
        for body in (_PDF_P1, _PDF_P2):
            page = doc.new_page()
            page.insert_textbox(fitz.Rect(50, 60, 540, 780), body, fontsize=12)
        return doc.tobytes()
    except Exception:  # noqa: BLE001
        pass
    try:
        from reportlab.pdfgen import canvas
        buf = io.BytesIO()
        c = canvas.Canvas(buf)
        for body in (_PDF_P1, _PDF_P2):
            y = 800
            for line in re.findall(r".{1,90}(?:\s|$)", body.replace("\n", " ")):
                c.drawString(50, y, line.strip())
                y -= 16
            c.showPage()
        c.save()
        return buf.getvalue()
    except Exception:  # noqa: BLE001
        return None


def make_corpus() -> dict:
    """filename -> (bytes, mime)."""
    corpus = {
        "orion_manual.txt": (_en_doc().encode("utf-8"), "text/plain"),
        "blue_nile_plan_ar.txt": (_AR_DOC.encode("utf-8"), "text/plain"),
        "inventory.py": (_PY_DOC.encode("utf-8"), "text/x-python"),
    }
    pdf = make_pdf()
    if pdf:
        corpus["operations_report.pdf"] = (pdf, "application/pdf")
    return corpus


# (question, language, expected keyword inside the retrieved chunk, expected filename)
QA = [
    ("What temperature does the Orion-7 reactor operate at?", "en", "450", "orion_manual.txt"),
    ("How often is maintenance required for the Orion-7 reactor?", "en", "90 days", "orion_manual.txt"),
    ("Who is the chief engineer of the Orion-7 programme?", "en", "hartwell", "orion_manual.txt"),
    ("What coolant does the reactor use?", "en", "liquid helium", "orion_manual.txt"),
    ("ما هي ميزانية مشروع النيل الأزرق؟", "ar", "4.5", "blue_nile_plan_ar.txt"),
    ("من هو المدير المسؤول عن المشروع؟", "ar", "أحمد فؤاد", "blue_nile_plan_ar.txt"),
    ("ما هي درجة حرارة تشغيل مفاعل أوريون؟", "ar->en", "450", "orion_manual.txt"),
    ("Who manages the Blue Nile project?", "en->ar", "أحمد فؤاد", "blue_nile_plan_ar.txt"),
    ("Which function calculates the reorder point?", "code", "calculate_reorder_point", "inventory.py"),
    ("What was the churn rate in Lyon?", "pdf", "3.7", "operations_report.pdf"),
]


# =============================================================================
# Prometheus helpers
# =============================================================================
DEFAULT_PREFIXES = ("python_", "process_", "go_")


def scrape(api: Api, path: str | None = None):
    r = api.call("GET", path or CONFIG["METRICS_PATH"])
    return r


def parse_metrics(text: str):
    from prometheus_client.parser import text_string_to_metric_families
    fams, flat = [], {}
    for fam in text_string_to_metric_families(text):
        if fam.name.endswith("_created"):   # *_created timestamps are exposed as separate gauge families: noise
            continue
        fams.append(fam)
        for s in fam.samples:
            flat[(s.name, tuple(sorted(s.labels.items())))] = s.value
    return fams, flat


def sel(flat: dict, name: str, exclude: dict | None = None, **labels) -> float:
    """Sum of all samples called `name` whose labels contain `labels` (and not `exclude`)."""
    total = 0.0
    for (n, lab), v in flat.items():
        if n != name:
            continue
        d = dict(lab)
        if any(d.get(k) != str(val) for k, val in labels.items()):
            continue
        if exclude and any(d.get(k) == str(val) for k, val in exclude.items()):
            continue
        total += v
    return total


def series_keys(flat: dict, name: str) -> set:
    return {lab for (n, lab) in flat if n == name}


def diff_flat(before: dict, after: dict) -> pd.DataFrame:
    rows = []
    for k, v in after.items():
        b = before.get(k, 0.0)
        if v != b and not k[0].endswith("_created"):
            rows.append(dict(sample=k[0], labels=dict(k[1]), before=b, after=v, delta=v - b))
    return pd.DataFrame(rows).sort_values("sample").reset_index(drop=True) if rows else pd.DataFrame(
        columns=["sample", "labels", "before", "after", "delta"])


_PATH_LABELS = ("path", "handler", "endpoint", "route", "url")
_STATUS_LABELS = ("status", "status_code", "code", "http_status")


def detect_http(fams):
    """Find the HTTP request counter / duration histogram / in-progress gauge, whatever they are called."""
    out = dict(counter=None, hist=None, gauge=None, path_label=None, status_label=None)
    for f in fams:
        labels = set()
        for s in f.samples:
            labels |= set(s.labels)
        has_method = "method" in labels
        st = next((x for x in _STATUS_LABELS if x in labels), None)
        pa = next((x for x in _PATH_LABELS if x in labels), None)
        if f.type == "counter" and has_method and st and not out["counter"]:
            sample_names = [s.name for s in f.samples if not s.name.endswith("_created")]
            out["counter"] = sample_names[0] if sample_names else f.name
            out["path_label"], out["status_label"] = pa, st
        if f.type == "histogram" and has_method and "http" in f.name and not out["hist"]:
            out["hist"] = f.name
            out["path_label"] = out["path_label"] or pa
        if f.type == "gauge" and re.search(r"inprogress|in_progress|inflight|in_flight", f.name) and not out["gauge"]:
            out["gauge"] = f.name
    return out


def hist_quantile(buckets: list, q: float) -> float:
    """buckets: [(le, cumulative_count)] sorted by le; Prometheus-style linear interpolation."""
    buckets = sorted(buckets)
    if not buckets or buckets[-1][1] == 0:
        return float("nan")
    rank, prev_le, prev_c = q * buckets[-1][1], 0.0, 0.0
    for le, c in buckets:
        if c >= rank:
            if math.isinf(le):
                return prev_le
            return prev_le + (le - prev_le) * ((rank - prev_c) / (c - prev_c) if c > prev_c else 0)
        prev_le, prev_c = le, c
    return prev_le


# =============================================================================
# Discovery helper for pipeline components (notebook 02)
# =============================================================================
def find_callable(package: str, *keywords, kinds=("function", "class")):
    import importlib
    import pkgutil
    try:
        pkg = importlib.import_module(package)
    except Exception:  # noqa: BLE001
        return None
    mods = [pkg]
    if hasattr(pkg, "__path__"):
        for m in pkgutil.walk_packages(pkg.__path__, package + "."):
            try:
                mods.append(importlib.import_module(m.name))
            except Exception:  # noqa: BLE001
                continue
    cands = []
    for mod in mods:
        for name, obj in inspect.getmembers(mod):
            if name.startswith("_") or getattr(obj, "__module__", "") != mod.__name__:
                continue
            ok = (inspect.isfunction(obj) and "function" in kinds) or (inspect.isclass(obj) and "class" in kinds)
            if ok and any(k.lower() in name.lower() for k in keywords):
                cands.append(obj)
    return cands[0] if cands else None


def call_flex(fn, *args, **kw):
    """Call fn/class-method passing only the kwargs it accepts."""
    try:
        params = inspect.signature(fn).parameters
        if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
            kw = {k: v for k, v in kw.items() if k in params}
    except (TypeError, ValueError):
        pass
    return fn(*args, **kw)


# =============================================================================
# One-shot flow used by several notebooks
# =============================================================================
def quick_flow(api: Api, files=("orion_manual.txt",), chat=True) -> dict:
    """register -> workspace -> project -> upload -> wait READY -> search -> (chat)."""
    out = {"timings": {}}
    t = time.perf_counter()
    u = api.new_user("flow")
    out["user"] = u
    ws = api.make_workspace(u)
    pid = api.make_project(u, ws)
    out.update(workspace_id=ws, project_id=pid)
    out["timings"]["setup_s"] = time.perf_counter() - t
    corpus, out["files"] = make_corpus(), {}
    for name in files:
        data, mime = corpus[name]
        r = api.upload(u, pid, name, data, mime)
        assert r.status_code in (200, 201, 202), f"upload {name}: {r.status_code} {r.text[:200]}"
        fid = get_id(r)
        status, secs = api.wait_ready(u, fid)
        out["files"][name] = dict(id=fid, status=status, seconds=secs)
        out["timings"][f"ingest_{name}_s"] = secs
    t = time.perf_counter()
    r = api.search(u, pid, "What temperature does the Orion-7 reactor operate at?")
    out["search_status"], out["timings"]["search_s"] = r.status_code, time.perf_counter() - t
    if chat:
        t = time.perf_counter()
        cid = api.make_conversation(u, pid)
        r = api.send_message(u, cid, "What temperature does the Orion-7 reactor operate at?")
        out["chat_status"], out["conversation_id"] = r.status_code, cid
        out["timings"]["chat_s"] = time.perf_counter() - t
    return out
