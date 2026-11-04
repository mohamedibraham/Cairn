import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from db import get_db
from dependencies.workspace import require_workspace_role
from enums import UserRole
from models.workspace import WorkspaceMember
from schemas.workspace import WorkspaceJoinRequestRead
from services import join_request_service

router = APIRouter(prefix="/workspaces/{workspace_id}/requests", tags=["workspace-join-requests"])


@router.get("/", response_model=list[WorkspaceJoinRequestRead])
def list_requests(
    db: Session = Depends(get_db),
    member: WorkspaceMember = Depends(require_workspace_role(UserRole.OWNER)),
):
    return join_request_service.list_join_requests(db, member.workspace_id)


@router.post("/{request_id}/approve", status_code=204)
def approve_request(
    request_id: uuid.UUID,
    db: Session = Depends(get_db),
    actor: WorkspaceMember = Depends(require_workspace_role(UserRole.OWNER)),
):
    join_request_service.approve_join_request(db, actor.workspace_id, request_id, actor)


@router.post("/{request_id}/reject", status_code=204)
def reject_request(
    request_id: uuid.UUID,
    db: Session = Depends(get_db),
    actor: WorkspaceMember = Depends(require_workspace_role(UserRole.OWNER)),
):
    join_request_service.reject_join_request(db, actor.workspace_id, request_id, actor)
