import uuid
from datetime import datetime, timezone

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from models.file import File
from models.project import Project
from models.workspace import WorkspaceMember
from schemas.project import ProjectCreate, ProjectRead, ProjectUpdate


def _not_found() -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, detail={
        "error": {"code": "PROJECT_NOT_FOUND", "message": "Project not found"}
    })


def _to_read(db: Session, project: Project) -> ProjectRead:
    file_count = db.scalar(
        select(func.count(File.id)).where(File.project_id == project.id, File.deleted_at.is_(None))
    ) or 0
    return ProjectRead(
        id=project.id, workspace_id=project.workspace_id, name=project.name,
        description=project.description, file_count=file_count,
        created_at=project.created_at, updated_at=project.updated_at,
    )


def list_projects(db: Session, workspace_id: uuid.UUID) -> list[ProjectRead]:
    projects = db.execute(
        select(Project)
        .where(Project.workspace_id == workspace_id, Project.deleted_at.is_(None))
        .order_by(Project.updated_at.desc())
    ).scalars().all()
    return [_to_read(db, p) for p in projects]


def create_project(db: Session, workspace_id: uuid.UUID, actor: WorkspaceMember, data: ProjectCreate) -> ProjectRead:
    project = Project(
        workspace_id=workspace_id, name=data.name, description=data.description, created_by=actor.user_id,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return _to_read(db, project)


def get_project(db: Session, workspace_id: uuid.UUID, project_id: uuid.UUID) -> ProjectRead:
    project = db.get(Project, project_id)
    if project is None or project.workspace_id != workspace_id or project.deleted_at is not None:
        raise _not_found()
    return _to_read(db, project)


def update_project(
    db: Session, workspace_id: uuid.UUID, project_id: uuid.UUID, data: ProjectUpdate
) -> ProjectRead:
    project = db.get(Project, project_id)
    if project is None or project.workspace_id != workspace_id or project.deleted_at is not None:
        raise _not_found()
    if data.name is not None:
        project.name = data.name
    if data.description is not None:
        project.description = data.description
    db.commit()
    db.refresh(project)
    return _to_read(db, project)


def delete_project(db: Session, workspace_id: uuid.UUID, project_id: uuid.UUID) -> None:
    project = db.get(Project, project_id)
    if project is None or project.workspace_id != workspace_id or project.deleted_at is not None:
        raise _not_found()
    project.deleted_at = datetime.now(timezone.utc)
    db.commit()
