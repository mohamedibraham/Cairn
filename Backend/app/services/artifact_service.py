import json
import uuid

from fastapi import HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from monitoring.metrics import ARTIFACTS_GENERATED
from enums import ArtifactScope, ArtifactType
from llm import get_llm_client
from monitoring.LLm_metrics import llm_operation
from models.artifact import Artifact
from models.conversation import Message
from models.file import File
from models.project import Project
from pipeline.embedding import EmbeddingError, embed_query
from schemas.artifact import ArtifactFilter, ArtifactFromMessageRequest, ArtifactGenerateRequest, ArtifactRead
from services.artifacts.base import ArtifactGenerationError, ArtifactOptions, Mismatch
from services.artifacts.engine import generate_from_chunks
from services.retrieval_service import load_scope_chunks, similarity_search

TOPIC_CHUNKS = 12  

_ERROR_STATUS = {"NO_CONTENT": 400, "ARTIFACT_INSUFFICIENT_DATA": 422}

_FAILURE_STATUS = {"NO_CONTENT": "no_content", "ARTIFACT_INSUFFICIENT_DATA": "insufficient_data"}


def _not_found(code: str, message: str) -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, detail={"error": {"code": code, "message": message}})


def _http_error(status_code: int, code: str, message: str, details: dict | None = None) -> HTTPException:
    return HTTPException(status_code, detail={"error": {"code": code, "message": message, "details": details}})


def list_artifacts(db: Session, project_id: uuid.UUID, filters: ArtifactFilter) -> list[ArtifactRead]:
    query = select(Artifact).where(Artifact.project_id == project_id)
    if filters.type is not None:
        query = query.where(Artifact.type == filters.type)
    if filters.scope_type is not None:
        query = query.where(Artifact.scope_type == filters.scope_type)
    query = query.order_by(Artifact.created_at.desc())
    artifacts = db.execute(query).scalars().all()
    return [ArtifactRead.model_validate(a) for a in artifacts]


def build_outcome(
    db: Session,
    project: Project,
    *,
    file_ids: list[uuid.UUID] | None,
    artifact_type: ArtifactType,
    topic: str | None = None,
    count: int | None = None,
    difficulty: str | None = None,
    question_types: list[str] | None = None,
    chart_type: str | None = None,
    force: bool = False,
):
    
    chunks = load_scope_chunks(db, project.id, file_ids)
    if chunks and topic and topic.strip():
        try:
            vector = embed_query(topic.strip())
        except EmbeddingError as exc:
            ARTIFACTS_GENERATED.labels(artifact_type.value, "error").inc()
            raise _http_error(503, "EMBEDDING_FAILED", str(exc))
        chunks = similarity_search(db, project.id, file_ids, vector, TOPIC_CHUNKS) or chunks

    options = ArtifactOptions(
        topic=topic, count=count, difficulty=difficulty,
        question_types=tuple(question_types or ()), chart_type=chart_type,
    )
    
    try:
        with llm_operation(f"artifact_{artifact_type.value}"):  # tags every LLM call made by the engine
            outcome = generate_from_chunks(chunks, artifact_type, options, get_llm_client(), force=force)
    except ArtifactGenerationError as exc:
        ARTIFACTS_GENERATED.labels(artifact_type.value, _FAILURE_STATUS.get(exc.code, "error")).inc()
        raise _http_error(_ERROR_STATUS.get(exc.code, 502), exc.code, exc.message)
    except Exception:
        ARTIFACTS_GENERATED.labels(artifact_type.value, "error").inc()
        raise
    ARTIFACTS_GENERATED.labels(artifact_type.value, "mismatch" if isinstance(outcome, Mismatch) else "ok").inc()
    return outcome


def generate_artifact(
    db: Session, project: Project, actor_id: uuid.UUID, data: ArtifactGenerateRequest
) -> ArtifactRead:
    file_ids = None
    if data.scope_type == ArtifactScope.FILE:
        if data.scope_file_id is None:
            raise _http_error(400, "SCOPE_FILE_ID_REQUIRED", "scope_file_id is required when scope_type is file")
        file_ids = [data.scope_file_id]

    outcome = build_outcome(
        db, project, file_ids=file_ids, artifact_type=data.type, topic=data.topic, count=data.count,
        difficulty=data.difficulty, question_types=data.question_types, chart_type=data.chart_type,
        force=data.force,
    )

    if isinstance(outcome, Mismatch):  
        raise _http_error(422, "ARTIFACT_NOT_SUPPORTED", outcome.reason, {
            "requested": outcome.requested.value, "alternatives": [t.value for t in outcome.alternatives],
        })

    artifact = Artifact(
        project_id=project.id, type=data.type, title=outcome.title, scope_type=data.scope_type,
        scope_file_id=data.scope_file_id, content=outcome.content, created_by=actor_id,
    )
    db.add(artifact)
    db.commit()
    db.refresh(artifact)
    return ArtifactRead.model_validate(artifact)


def save_from_message(
    db: Session, project: Project, actor_id: uuid.UUID, message_id: uuid.UUID, data: ArtifactFromMessageRequest
) -> ArtifactRead:
    message = db.get(Message, message_id)
    if message is None or message.conversation.project_id != project.id:
        raise _not_found("MESSAGE_NOT_FOUND", "Message not found in this project")

    
    scope_type = data.scope_type or ArtifactScope.PROJECT
    scope_file_id = None
    if scope_type == ArtifactScope.FILE:
        if data.scope_file_id is None:
            raise _http_error(400, "SCOPE_FILE_ID_REQUIRED", "scope_file_id is required when scope_type is file")
        in_project = db.scalar(
            select(File.id).where(File.id == data.scope_file_id, File.project_id == project.id)
        )
        if in_project is None:
            raise _not_found("FILE_NOT_FOUND", "File not found in this project")
        scope_file_id = data.scope_file_id

    
    if data.content and data.content.get("type") == data.type.value:
        content = data.content
    elif data.type.value == "summary":
        content = {"type": "summary", "body": data.selected_content}
    else:
        content = {"text": data.selected_content}

    artifact = Artifact(
        project_id=project.id, type=data.type, title=data.title,
        scope_type=scope_type, scope_file_id=scope_file_id,
        content=content, source_message_id=message.id, created_by=actor_id,
    )
    db.add(artifact)
    db.commit()
    db.refresh(artifact)
    return ArtifactRead.model_validate(artifact)


def download_artifact(db: Session, project_id: uuid.UUID, artifact_id: uuid.UUID) -> Response:
    artifact = db.get(Artifact, artifact_id)
    if artifact is None or artifact.project_id != project_id:
        raise _not_found("ARTIFACT_NOT_FOUND", "Artifact not found")

    payload = json.dumps(artifact.content, ensure_ascii=False, indent=2).encode("utf-8")
    filename = f"{artifact.title.replace(' ', '_')}.json"
    return Response(
        content=payload,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )