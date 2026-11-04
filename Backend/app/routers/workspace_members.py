import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from db import get_db
from dependencies.workspace import require_workspace_role
from enums import UserRole
from models.workspace import WorkspaceMember
from schemas.workspace import WorkspaceMemberRead, WorkspaceMemberRoleUpdate
from services import membership_service

router = APIRouter(prefix="/workspaces/{workspace_id}/members", tags=["workspace-members"])


@router.get("/", response_model=list[WorkspaceMemberRead])
def list_members(
    db: Session = Depends(get_db),
    member: WorkspaceMember = Depends(require_workspace_role(UserRole.VIEWER)),
):
    return membership_service.list_members(db, member.workspace_id)


@router.patch("/{member_id}", response_model=WorkspaceMemberRead)
def update_member_role(
    member_id: uuid.UUID,
    payload: WorkspaceMemberRoleUpdate,
    db: Session = Depends(get_db),
    actor: WorkspaceMember = Depends(require_workspace_role(UserRole.OWNER)),
):
    return membership_service.update_member_role(db, actor.workspace_id, member_id, payload.role, actor)


@router.delete("/{member_id}", status_code=204)
def remove_member(
    member_id: uuid.UUID,
    db: Session = Depends(get_db),
    actor: WorkspaceMember = Depends(require_workspace_role(UserRole.OWNER)),
):
    membership_service.remove_member(db, actor.workspace_id, member_id, actor)
