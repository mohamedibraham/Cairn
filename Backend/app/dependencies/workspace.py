import uuid
from typing import Callable

from fastapi import Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from db import get_db
from .auth import get_current_user
from enums import ROLE_RANK, UserRole
from models import User
from models import WorkspaceMember


def _forbidden(reason: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={"error": {"code": "FORBIDDEN", "message": reason}},
    )

"""
`workspace_id` is read from the path automatically by FastAPI since the
inner function declares it as a parameter with the same name. This is the
ONLY place membership + role are checked — routers never query
workspace_members themselves, which is what keeps IDOR off the table.
"""


def require_workspace_role(min_role: UserRole) -> Callable[..., WorkspaceMember]:

    def dependency(
        workspace_id: uuid.UUID,
        db: Session = Depends(get_db),
        user: User = Depends(get_current_user),
    ) -> WorkspaceMember:
        member = db.scalar(
            select(WorkspaceMember).where(
                WorkspaceMember.workspace_id == workspace_id,
                WorkspaceMember.user_id == user.id,
            )
        )
        if member is None:
        
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={
                "error": {"code": "WORKSPACE_NOT_FOUND", "message": "Workspace not found"}
            })
        if ROLE_RANK[member.role] < ROLE_RANK[min_role]:
            raise _forbidden(f"Requires {min_role.value} role or higher")
        return member

    return dependency
