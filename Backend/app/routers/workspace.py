from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from db.session import get_db
from dependencies.auth import get_current_user
from dependencies.workspace import require_workspace_role
from enums import UserRole
from models.user import User
from models.workspace import WorkspaceMember
from schemas.workspace import (
    ActivityRead,
    WorkspaceCreate,
    WorkspaceDashboardRead,
    WorkspaceRead,
    WorkspaceSummary,
    WorkspaceUpdate,
)
from services import workspace_service

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


@router.get("/", response_model=list[WorkspaceSummary])
def list_my_workspaces(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return workspace_service.list_user_workspaces(db, user)


@router.post("/", response_model=WorkspaceRead, status_code=201)
def create_workspace(
    payload: WorkspaceCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)
):
    return workspace_service.create_workspace(db, user, payload)


@router.get("/{workspace_id}", response_model=WorkspaceRead)
def get_workspace(
    db: Session = Depends(get_db),
    member: WorkspaceMember = Depends(require_workspace_role(UserRole.VIEWER)),
):
    return workspace_service.get_workspace(db, member.workspace_id, member)


@router.patch("/{workspace_id}", response_model=WorkspaceRead)
def update_workspace(
    payload: WorkspaceUpdate,
    db: Session = Depends(get_db),
    member: WorkspaceMember = Depends(require_workspace_role(UserRole.OWNER)),
):
    return workspace_service.update_workspace(db, member.workspace_id, member, payload)


@router.delete("/{workspace_id}", status_code=204)
def delete_workspace(
    db: Session = Depends(get_db),
    member: WorkspaceMember = Depends(require_workspace_role(UserRole.OWNER)),
):
    workspace_service.soft_delete_workspace(db, member.workspace_id, member)


@router.get("/{workspace_id}/dashboard", response_model=WorkspaceDashboardRead)
def get_dashboard(
    db: Session = Depends(get_db),
    member: WorkspaceMember = Depends(require_workspace_role(UserRole.VIEWER)),
):
    return workspace_service.get_dashboard(db, member.workspace_id, member)


@router.get("/{workspace_id}/activity", response_model=list[ActivityRead])
def get_activity(
    db: Session = Depends(get_db),
    member: WorkspaceMember = Depends(require_workspace_role(UserRole.VIEWER)),
):
    return workspace_service.list_activity(db, member.workspace_id)