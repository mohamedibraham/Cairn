import uuid

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from enums import UserRole
from models.workspace import WorkspaceMember
from schemas.workspace import WorkspaceMemberRead
from services.activity_log_service import log_activity


def _bad_request(code: str, message: str) -> HTTPException:
    return HTTPException(status.HTTP_400_BAD_REQUEST, detail={"error": {"code": code, "message": message}})


def _owner_count(db: Session, workspace_id: uuid.UUID) -> int:
    return db.scalar(
        select(func.count(WorkspaceMember.id)).where(
            WorkspaceMember.workspace_id == workspace_id, WorkspaceMember.role == UserRole.OWNER
        )
    ) or 0


def list_members(db: Session, workspace_id: uuid.UUID) -> list[WorkspaceMemberRead]:
    members = db.execute(
        select(WorkspaceMember)
        .options(joinedload(WorkspaceMember.user))
        .where(WorkspaceMember.workspace_id == workspace_id)
        .order_by(WorkspaceMember.joined_at)
    ).scalars().all()
    return [WorkspaceMemberRead.model_validate(m) for m in members]


def update_member_role(
    db: Session, workspace_id: uuid.UUID, member_id: uuid.UUID, new_role: UserRole, actor: WorkspaceMember
) -> WorkspaceMemberRead:
    target = db.get(WorkspaceMember, member_id)
    if target is None or target.workspace_id != workspace_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail={
            "error": {"code": "MEMBER_NOT_FOUND", "message": "Member not found"}
        })

    if target.role == UserRole.OWNER and new_role != UserRole.OWNER and _owner_count(db, workspace_id) <= 1:
        raise _bad_request("LAST_OWNER", "Workspace must keep at least one Owner")

    old_role = target.role
    target.role = new_role
    log_activity(
        db, workspace_id=workspace_id, actor_id=actor.user_id,
        action="member.role_changed", target_type="workspace_member", target_id=target.id,
        metadata={"from": old_role.value, "to": new_role.value, "user_id": str(target.user_id)},
    )
    db.commit()
    db.refresh(target)
    return WorkspaceMemberRead.model_validate(target)


def remove_member(
    db: Session, workspace_id: uuid.UUID, member_id: uuid.UUID, actor: WorkspaceMember
) -> None:
    target = db.get(WorkspaceMember, member_id)
    if target is None or target.workspace_id != workspace_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail={
            "error": {"code": "MEMBER_NOT_FOUND", "message": "Member not found"}
        })
    if target.role == UserRole.OWNER and _owner_count(db, workspace_id) <= 1:
        raise _bad_request("LAST_OWNER", "Cannot remove the only remaining Owner")

    log_activity(
        db, workspace_id=workspace_id, actor_id=actor.user_id,
        action="member.removed", target_type="workspace_member", target_id=target.id,
        metadata={"user_id": str(target.user_id), "role": target.role.value},
    )
    db.delete(target)
    db.commit()
