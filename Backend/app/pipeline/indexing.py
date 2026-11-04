import uuid

"""No-op for pgvector: the HNSW index is maintained on every INSERT, so
    chunks are searchable as soon as they are committed. Kept as an explicit
    stage because the UI shows a distinct 'Search index' step; use this hook
    if a separate index (e.g. a full-text tsvector) is added later."""


def refresh_search_index(file_version_id: uuid.UUID) -> None:

    return None
