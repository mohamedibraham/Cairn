import uuid

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from enums import SearchScope
from models.conversation import Conversation
from models.file import Chunk, File, FileVersion
from models.project import Project
from models.search import SearchQuery
from pipeline.embedding import EmbeddingError, embed_query
from schemas.search import SearchHistoryItemRead, SearchRequest, SearchResponse, SearchResultItem

SEARCH_TOP_K = 10


def _bad_request(code: str, message: str) -> HTTPException:
    return HTTPException(status.HTTP_400_BAD_REQUEST, detail={"error": {"code": code, "message": message}})


def search(db: Session, workspace_id: uuid.UUID, actor_id: uuid.UUID, data: SearchRequest) -> SearchResponse:
    if data.scope == SearchScope.PROJECT:
        if data.project_id is None:
            raise _bad_request("PROJECT_ID_REQUIRED", "project_id is required when scope is project")
        project = db.get(Project, data.project_id)
        if project is None or project.workspace_id != workspace_id or project.deleted_at is not None:
            raise _bad_request("PROJECT_NOT_IN_WORKSPACE", "project_id does not belong to this workspace")

    try:
        query_vector = embed_query(data.query)
    except EmbeddingError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail={
            "error": {"code": "EMBEDDING_FAILED", "message": str(exc)}
        })

    distance = Chunk.embedding.cosine_distance(query_vector).label("distance")
    query = (
        select(Chunk, File, Project, distance)
        .join(FileVersion, FileVersion.id == Chunk.file_version_id)
        .join(File, File.id == FileVersion.file_id)
        .join(Project, Project.id == File.project_id)
        .where(
            Project.workspace_id == workspace_id,
            Project.deleted_at.is_(None),
            File.deleted_at.is_(None),
            File.current_version_id == FileVersion.id,
        )
    )
    if data.scope == SearchScope.PROJECT:
        query = query.where(Project.id == data.project_id)

    query = query.order_by(distance).limit(SEARCH_TOP_K)
    rows = db.execute(query).all()

    results = [
        SearchResultItem(
            chunk_id=chunk.id, file_id=file.id, filename=file.filename,
            project_id=project.id, project_name=project.name,
            excerpt=chunk.content[:280], symbol=chunk.symbol,
            page_number=chunk.page_number, line_start=chunk.line_start,
            score=round(1.0 - float(dist), 4),  
        )
        for chunk, file, project, dist in rows
    ]

    search_query = SearchQuery(
        workspace_id=workspace_id,
        project_id=data.project_id if data.scope == SearchScope.PROJECT else None,
        scope=data.scope, query_text=data.query, result_count=len(results), created_by=actor_id,
    )
    db.add(search_query)
    db.commit()
    db.refresh(search_query)

    return SearchResponse(query_id=search_query.id, results=results, result_count=len(results))


def get_history(db: Session, workspace_id: uuid.UUID) -> list[SearchHistoryItemRead]:
    
    rows = db.execute(
        select(SearchQuery, Project.name, Conversation.project_id)
        .outerjoin(Project, Project.id == SearchQuery.project_id)
        .outerjoin(Conversation, Conversation.id == SearchQuery.conversation_id)
        .where(SearchQuery.workspace_id == workspace_id)
        .order_by(SearchQuery.created_at.desc())
        .limit(50)
    ).all()

    return [
        SearchHistoryItemRead(
            id=r.id, query_text=r.query_text, scope=r.scope,
            project_id=r.project_id, project_name=project_name,
            result_count=r.result_count, conversation_id=r.conversation_id,
            conversation_project_id=conversation_project_id,
            created_at=r.created_at,
        )
        for r, project_name, conversation_project_id in rows
    ]