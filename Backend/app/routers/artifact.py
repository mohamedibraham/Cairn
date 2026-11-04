import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from db.session import get_db
from dependencies.project import ProjectContext, require_project_role
from enums import ArtifactScope, ArtifactType, UserRole
from schemas.artifact import ArtifactFilter, ArtifactFromMessageRequest, ArtifactGenerateRequest, ArtifactRead
from services import artifact_service

router = APIRouter(prefix="/projects/{project_id}/artifacts", tags=["artifacts"])


@router.get("/", response_model=list[ArtifactRead])
def list_artifacts(
    type: ArtifactType | None = None,
    scope_type: ArtifactScope | None = None,
    db: Session = Depends(get_db),
    ctx: ProjectContext = Depends(require_project_role(UserRole.VIEWER)),
):
    return artifact_service.list_artifacts(db, ctx.project.id, ArtifactFilter(type=type, scope_type=scope_type))


@router.post("/generate", response_model=ArtifactRead, status_code=201)
def generate_artifact(
    payload: ArtifactGenerateRequest,
    db: Session = Depends(get_db),
    
    ctx: ProjectContext = Depends(require_project_role(UserRole.VIEWER)),
):
    return artifact_service.generate_artifact(db, ctx.project, ctx.member.user_id, payload)


@router.post("/from-message/{message_id}", response_model=ArtifactRead, status_code=201)
def save_from_message(
    message_id: uuid.UUID,
    payload: ArtifactFromMessageRequest,
    db: Session = Depends(get_db),
    ctx: ProjectContext = Depends(require_project_role(UserRole.VIEWER)),
):
    return artifact_service.save_from_message(db, ctx.project, ctx.member.user_id, message_id, payload)


@router.get("/{artifact_id}/download")
def download_artifact(
    artifact_id: uuid.UUID,
    db: Session = Depends(get_db),
    ctx: ProjectContext = Depends(require_project_role(UserRole.VIEWER)),
):
    return artifact_service.download_artifact(db, ctx.project.id, artifact_id)
