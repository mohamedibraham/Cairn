import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from monitoring.metrics import track_retrieval
from models.file import Chunk, File, FileVersion

MAX_SCOPE_CHUNKS = 5000


@dataclass(frozen=True)
class ScopedChunk:
    chunk_id: str
    file_id: str
    filename: str
    file_type: str
    chunk_index: int
    content: str
    page_number: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    metadata: dict | None = None

    @property
    def symbol(self) -> str | None:
        return (self.metadata or {}).get("symbol")

    @property
    def chunk_type(self) -> str | None:
        return (self.metadata or {}).get("chunk_type")


def _base_query(project_id: uuid.UUID, file_ids):
    query = (
        select(Chunk, File)
        .join(FileVersion, FileVersion.id == Chunk.file_version_id)
        .join(File, File.id == FileVersion.file_id)
        .where(
            File.project_id == project_id,
            File.deleted_at.is_(None),
            File.current_version_id == FileVersion.id,  
        )
    )
    if file_ids:
        query = query.where(File.id.in_(file_ids))
    return query


def _to_scoped(chunk: Chunk, file: File) -> ScopedChunk:
    return ScopedChunk(
        chunk_id=str(chunk.id), file_id=str(file.id), filename=file.filename, file_type=file.file_type.value,
        chunk_index=chunk.chunk_index, content=chunk.content, page_number=chunk.page_number,
        line_start=chunk.line_start, line_end=chunk.line_end, metadata=chunk.chunk_metadata,
    )


def load_scope_chunks(db: Session, project_id: uuid.UUID, file_ids=None, *, limit: int = MAX_SCOPE_CHUNKS) -> list:
    
    rows = db.execute(
        _base_query(project_id, file_ids).order_by(File.created_at, File.id, Chunk.chunk_index).limit(limit)
    ).all()
    return [_to_scoped(chunk, file) for chunk, file in rows]


def similarity_search(db: Session, project_id: uuid.UUID, file_ids, query_vector, k: int) -> list:
    
    with track_retrieval("scope") as tracked:
        rows = db.execute(
            _base_query(project_id, file_ids).order_by(Chunk.embedding.cosine_distance(query_vector)).limit(k)
        ).all()
        tracked.count = len(rows)
    found = [_to_scoped(chunk, file) for chunk, file in rows]
    return sorted(found, key=lambda c: (c.file_id, c.chunk_index))


def stride_sample(items: list, n: int) -> list:
    
    if n <= 0:
        return []
    if len(items) <= n:
        return list(items)
    step = len(items) / n
    return [items[int(i * step + step / 2)] for i in range(n)]