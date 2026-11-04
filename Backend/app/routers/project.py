import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from db import get_db
from dependencies.workspace import require_workspace_role
from enums import UserRole
from models.workspace import WorkspaceMember
from schemas.project import ProjectCreate, ProjectRead, ProjectUpdate
from services import project_service

router = APIRouter(prefix="/workspaces/{workspace_id}/projects", tags=["projects"])


@router.get("/", response_model=list[ProjectRead])
def list_projects(
    db: Session = Depends(get_db),
    member: WorkspaceMember = Depends(require_workspace_role(UserRole.VIEWER)),
):
    return project_service.list_projects(db, member.workspace_id)


@router.post("/", response_model=ProjectRead, status_code=201)
def create_project(
    payload: ProjectCreate,
    db: Session = Depends(get_db),
    member: WorkspaceMember = Depends(require_workspace_role(UserRole.EDITOR)),
):
    return project_service.create_project(db, member.workspace_id, member, payload)


@router.get("/{project_id}", response_model=ProjectRead)
def get_project(
    project_id: uuid.UUID,
    db: Session = Depends(get_db),
    member: WorkspaceMember = Depends(require_workspace_role(UserRole.VIEWER)),
):
    return project_service.get_project(db, member.workspace_id, project_id)


@router.patch("/{project_id}", response_model=ProjectRead)
def update_project(
    project_id: uuid.UUID,
    payload: ProjectUpdate,
    db: Session = Depends(get_db),
    member: WorkspaceMember = Depends(require_workspace_role(UserRole.EDITOR)),
):
    return project_service.update_project(db, member.workspace_id, project_id, payload)


@router.delete("/{project_id}", status_code=204)
def delete_project(
    project_id: uuid.UUID,
    db: Session = Depends(get_db),
    member: WorkspaceMember = Depends(require_workspace_role(UserRole.EDITOR)),
):
    project_service.delete_project(db, member.workspace_id, project_id)
