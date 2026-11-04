import secrets
import uuid
from datetime import datetime, timezone

from fastapi import HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from enums import JoinRequestStatus, UserRole
from models.user import User
from models.workspace import WorkspaceInvite, WorkspaceJoinRequest, WorkspaceMember
from schemas.workspace import WorkspaceInviteRead
from services.activity_log_service import log_activity


def _invite_url(request: Request, token: str) -> str:
    
    return f"/invite/{token}"


def _to_read(invite: WorkspaceInvite, request: Request) -> WorkspaceInviteRead:
    return WorkspaceInviteRead(
        token=invite.token,
        invite_url=_invite_url(request, invite.token),
        created_at=invite.created_at,
        is_active=invite.revoked_at is None,
    )


def get_or_create_invite(
    db: Session, workspace_id: uuid.UUID, actor: WorkspaceMember, request: Request
) -> WorkspaceInviteRead:
    invite = db.scalar(select(WorkspaceInvite).where(WorkspaceInvite.workspace_id == workspace_id))
    if invite is None:
        invite = WorkspaceInvite(
            workspace_id=workspace_id, token=secrets.token_urlsafe(16), created_by=actor.user_id,
        )
        db.add(invite)
        db.commit()
        db.refresh(invite)
    return _to_read(invite, request)


def regenerate_invite(
    db: Session, workspace_id: uuid.UUID, actor: WorkspaceMember, request: Request
) -> WorkspaceInviteRead:
    invite = db.scalar(select(WorkspaceInvite).where(WorkspaceInvite.workspace_id == workspace_id))
    if invite is None:
        return get_or_create_invite(db, workspace_id, actor, request)

    invite.token = secrets.token_urlsafe(16)
    invite.revoked_at = None
    log_activity(
        db, workspace_id=workspace_id, actor_id=actor.user_id,
        action="invite.regenerated", target_type="workspace_invite", target_id=invite.id,
    )
    db.commit()
    db.refresh(invite)
    return _to_read(invite, request)


def revoke_invite(db: Session, workspace_id: uuid.UUID, actor: WorkspaceMember) -> None:
    invite = db.scalar(select(WorkspaceInvite).where(WorkspaceInvite.workspace_id == workspace_id))
    if invite is None:
        return
    invite.revoked_at = datetime.now(timezone.utc)
    log_activity(
        db, workspace_id=workspace_id, actor_id=actor.user_id,
        action="invite.revoked", target_type="workspace_invite", target_id=invite.id,
    )
    db.commit()


def join_with_token(db: Session, token: str, user: User):
    
    invite = db.scalar(select(WorkspaceInvite).where(WorkspaceInvite.token == token))
    if invite is None or invite.revoked_at is not None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail={
            "error": {"code": "INVITE_INVALID", "message": "This invite link is invalid or has been revoked"}
        })

    already_member = db.scalar(
        select(WorkspaceMember).where(
            WorkspaceMember.workspace_id == invite.workspace_id, WorkspaceMember.user_id == user.id
        )
    )
    if already_member is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, detail={
            "error": {"code": "ALREADY_MEMBER", "message": "You are already a member of this workspace"}
        })

    existing_pending = db.scalar(
        select(WorkspaceJoinRequest).where(
            WorkspaceJoinRequest.workspace_id == invite.workspace_id,
            WorkspaceJoinRequest.user_id == user.id,
            WorkspaceJoinRequest.status == JoinRequestStatus.PENDING,
        )
    )
    if existing_pending is not None:
        return existing_pending

    join_request = WorkspaceJoinRequest(workspace_id=invite.workspace_id, user_id=user.id)
    db.add(join_request)
    db.commit()
    db.refresh(join_request)
    return join_request
