import uuid
from datetime import datetime, timezone

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from enums import JoinRequestStatus, UserRole
from models.workspace import WorkspaceJoinRequest, WorkspaceMember
from schemas.workspace import WorkspaceJoinRequestRead
from services.activity_log_service import log_activity


def list_join_requests(db: Session, workspace_id: uuid.UUID) -> list[WorkspaceJoinRequestRead]:
    requests = db.execute(
        select(WorkspaceJoinRequest)
        .options(joinedload(WorkspaceJoinRequest.user))
        .where(
            WorkspaceJoinRequest.workspace_id == workspace_id,
            WorkspaceJoinRequest.status == JoinRequestStatus.PENDING,
        )
        .order_by(WorkspaceJoinRequest.created_at)
    ).scalars().all()
    return [WorkspaceJoinRequestRead.model_validate(r) for r in requests]


def _get_pending_or_404(db: Session, workspace_id: uuid.UUID, request_id: uuid.UUID) -> WorkspaceJoinRequest:
    req = db.get(WorkspaceJoinRequest, request_id)
    if req is None or req.workspace_id != workspace_id or req.status != JoinRequestStatus.PENDING:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail={
            "error": {"code": "JOIN_REQUEST_NOT_FOUND", "message": "No pending request found"}
        })
    return req


def approve_join_request(
    db: Session, workspace_id: uuid.UUID, request_id: uuid.UUID, actor: WorkspaceMember
) -> None:
    req = _get_pending_or_404(db, workspace_id, request_id)
    now = datetime.now(timezone.utc)

    req.status = JoinRequestStatus.APPROVED
    req.resolved_at = now
    req.resolved_by = actor.user_id

    db.add(WorkspaceMember(workspace_id=workspace_id, user_id=req.user_id, role=UserRole.VIEWER))
    log_activity(
        db, workspace_id=workspace_id, actor_id=actor.user_id,
        action="join_request.approved", target_type="user", target_id=req.user_id,
    )
    db.commit()


def reject_join_request(
    db: Session, workspace_id: uuid.UUID, request_id: uuid.UUID, actor: WorkspaceMember
) -> None:
    req = _get_pending_or_404(db, workspace_id, request_id)
    req.status = JoinRequestStatus.REJECTED
    req.resolved_at = datetime.now(timezone.utc)
    req.resolved_by = actor.user_id

    log_activity(
        db, workspace_id=workspace_id, actor_id=actor.user_id,
        action="join_request.rejected", target_type="user", target_id=req.user_id,
    )
    db.commit()
