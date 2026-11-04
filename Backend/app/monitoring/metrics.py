import contextlib
import time

from prometheus_client import Counter, Gauge, Histogram
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector

# ── HTTP 
HTTP_REQUESTS = Counter("http_requests_total", "Total HTTP requests", ["method", "path", "status"])
HTTP_DURATION = Histogram(
    "http_request_duration_seconds", "HTTP latency", ["method", "path"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
HTTP_IN_PROGRESS = Gauge("http_requests_in_progress", "In-flight requests", ["method"])

# ── RAG retrieval: source = chat | search | scope ─
RETRIEVAL_SOURCES = ("chat", "search", "scope")
RAG_RETRIEVAL_SECONDS = Histogram(
    "rag_retrieval_duration_seconds", "Vector search latency (DB query only)", ["source"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
RAG_CHUNKS_RETRIEVED = Histogram(
    "rag_chunks_retrieved", "Chunks returned per retrieval", ["source"], buckets=(0, 1, 3, 5, 10, 20, 50),
)
RAG_EMPTY_RETRIEVALS = Counter("rag_empty_retrievals_total", "Retrievals that returned zero chunks", ["source"])

# ─ Embedding / ingestion 
EMBEDDING_KINDS = ("query", "passage")
EMBEDDING_SECONDS = Histogram(
    "embedding_duration_seconds", "Embedding latency", ["kind"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
# stage: extraction | chunking | embedding | db_write | indexing | total
INGESTION_STAGE_SECONDS = Histogram(
    "ingestion_stage_duration_seconds", "Ingestion stage latency", ["stage"],
    buckets=(0.1, 0.5, 1, 5, 10, 30, 60, 300, 900),
)
FILES_PROCESSED = Counter("files_processed_total", "Ingestion runs by result", ["status"])  # ready | failed
ARTIFACTS_GENERATED = Counter("artifacts_generated_total", "Generated artifacts", ["artifact_type", "status"])  # ok|mismatch|insufficient_data|no_content|error

# Create the known series up front so dashboards show 0 instead of "No data".
for _s in RETRIEVAL_SOURCES:
    RAG_RETRIEVAL_SECONDS.labels(_s)
    RAG_CHUNKS_RETRIEVED.labels(_s)
    RAG_EMPTY_RETRIEVALS.labels(_s)
for _k in EMBEDDING_KINDS:
    EMBEDDING_SECONDS.labels(_k)
for _st in ("ready", "failed"):
    FILES_PROCESSED.labels(_st)


class _Retrieval:
    count = 0


@contextlib.contextmanager
def track_retrieval(source: str):
    
    tracked = _Retrieval()
    started = time.perf_counter()
    try:
        yield tracked
    finally:
        RAG_RETRIEVAL_SECONDS.labels(source).observe(time.perf_counter() - started)
    RAG_CHUNKS_RETRIEVED.labels(source).observe(tracked.count)
    if tracked.count == 0:
        RAG_EMPTY_RETRIEVALS.labels(source).inc()


class DBPoolCollector(Collector):


    def __init__(self, engine):
        self._pool = engine.pool

    def collect(self):
        for name, value, doc in (
            ("db_pool_size", self._pool.size(), "Configured pool size"),
            ("db_pool_checked_out", self._pool.checkedout(), "Connections in use"),
            ("db_pool_overflow", self._pool.overflow(), "Overflow connections"),
        ):
            gauge = GaugeMetricFamily(name, doc)
            gauge.add_metric([], value)
            yield gauge