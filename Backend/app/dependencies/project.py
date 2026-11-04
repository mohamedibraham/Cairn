import uuid
from typing import Callable

from fastapi import Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.session import get_db
from dependencies.auth import get_current_user
from enums import ROLE_RANK, UserRole
from models import Project
from models import User
from models import WorkspaceMember


class ProjectContext:
    def __init__(self, project: Project, member: WorkspaceMember):
        self.project = project
        self.member = member


def require_project_role(min_role: UserRole) -> Callable[..., ProjectContext]:
    def dependency(
        project_id: uuid.UUID,
        db: Session = Depends(get_db),
        user: User = Depends(get_current_user),
    ) -> ProjectContext:
        project = db.get(Project, project_id)
        if project is None or project.deleted_at is not None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail={
                "error": {"code": "PROJECT_NOT_FOUND", "message": "Project not found"}
            })

        member = db.scalar(
            select(WorkspaceMember).where(
                WorkspaceMember.workspace_id == project.workspace_id,
                WorkspaceMember.user_id == user.id,
            )
        )
        
        if member is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail={
                "error": {"code": "PROJECT_NOT_FOUND", "message": "Project not found"}
            })
        if ROLE_RANK[member.role] < ROLE_RANK[min_role]:
            raise HTTPException(status.HTTP_403_FORBIDDEN, detail={
                "error": {"code": "FORBIDDEN", "message": f"Requires {min_role.value} role or higher"}
            })

        return ProjectContext(project=project, member=member)

    return dependency
