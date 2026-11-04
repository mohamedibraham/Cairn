from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from db import get_db
from dependencies.auth import get_current_user
from dependencies.workspace import require_workspace_role
from enums import UserRole
from models.user import User
from models.workspace import WorkspaceMember
from schemas.workspace import WorkspaceInviteRead
from services import invite_service

router = APIRouter(prefix="/workspaces/{workspace_id}/invite", tags=["workspace-invite"])
invites_router = APIRouter(prefix="/invites", tags=["invites"])


@router.get("/", response_model=WorkspaceInviteRead)
def get_invite(
    request: Request,
    db: Session = Depends(get_db),
    member: WorkspaceMember = Depends(require_workspace_role(UserRole.OWNER)),
):
    return invite_service.get_or_create_invite(db, member.workspace_id, member, request)


@router.post("/regenerate", response_model=WorkspaceInviteRead)
def regenerate_invite(
    request: Request,
    db: Session = Depends(get_db),
    member: WorkspaceMember = Depends(require_workspace_role(UserRole.OWNER)),
):
    return invite_service.regenerate_invite(db, member.workspace_id, member, request)


@router.post("/revoke", status_code=204)
def revoke_invite(
    db: Session = Depends(get_db),
    member: WorkspaceMember = Depends(require_workspace_role(UserRole.OWNER)),
):
    invite_service.revoke_invite(db, member.workspace_id, member)


@invites_router.post("/join", status_code=201)
def join_via_token(
    token: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)
):
    
    req = invite_service.join_with_token(db, token, user)
    return {"id": req.id, "status": req.status.value}
