import uuid

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from models.conversation import Conversation
from enums import ProcessingStage, UserRole, WorkspaceVisibility
from models.file import File, FileVersion
from models.project import Project
from models.user import User
from models.workspace import Workspace, WorkspaceActivityLog, WorkspaceMember
from schemas.auth import UserRead
from schemas.workspace import (
    ActivityRead,
    ConversationSummary,
    FileProcessingSummary,
    FileSummary,
    WorkspaceCreate,
    WorkspaceDashboardRead,
    WorkspaceRead,
    WorkspaceSummary,
    WorkspaceUpdate,
)
from services.activity_log_service import log_activity
from services.activity_text import describe_activity


def create_workspace(db: Session, user: User, data: WorkspaceCreate) -> WorkspaceRead:
    workspace = Workspace(name=data.name, visibility=data.visibility, owner_id=user.id)
    db.add(workspace)
    db.flush()  

    db.add(WorkspaceMember(workspace_id=workspace.id, user_id=user.id, role=UserRole.OWNER))
    db.commit()
    db.refresh(workspace)

    return WorkspaceRead(
        id=workspace.id, name=workspace.name, visibility=workspace.visibility,
        owner_id=workspace.owner_id, my_role=UserRole.OWNER, created_at=workspace.created_at,
    )


def list_user_workspaces(db: Session, user: User) -> list[WorkspaceSummary]:
    rows = db.execute(
        select(
            Workspace,
            WorkspaceMember.role,
            func.count(func.distinct(Project.id)).label("project_count"),
        )
        .join(WorkspaceMember, WorkspaceMember.workspace_id == Workspace.id)
        .outerjoin(Project, (Project.workspace_id == Workspace.id) & Project.deleted_at.is_(None))
        .where(WorkspaceMember.user_id == user.id, Workspace.deleted_at.is_(None))
        .group_by(Workspace.id, WorkspaceMember.role)
        .order_by(Workspace.created_at.desc())
    ).all()

    return [
        WorkspaceSummary(
            id=ws.id, name=ws.name, visibility=ws.visibility,
            my_role=role, project_count=project_count,
        )
        for ws, role, project_count in rows
    ]


def get_workspace(db: Session, workspace_id: uuid.UUID, member: WorkspaceMember) -> WorkspaceRead:
    workspace = db.get(Workspace, workspace_id)
    if workspace is None or workspace.deleted_at is not None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail={
            "error": {"code": "WORKSPACE_NOT_FOUND", "message": "Workspace not found"}
        })
    return WorkspaceRead(
        id=workspace.id, name=workspace.name, visibility=workspace.visibility,
        owner_id=workspace.owner_id, my_role=member.role, created_at=workspace.created_at,
    )


def update_workspace(
    db: Session, workspace_id: uuid.UUID, member: WorkspaceMember, data: WorkspaceUpdate
) -> WorkspaceRead:
    workspace = db.get(Workspace, workspace_id)
    if data.name is not None:
        workspace.name = data.name
    if data.visibility is not None:
        workspace.visibility = data.visibility

    log_activity(
        db, workspace_id=workspace_id, actor_id=member.user_id,
        action="workspace.updated", target_type="workspace", target_id=workspace_id,
        metadata={"name": data.name, "visibility": data.visibility and data.visibility.value},
    )
    db.commit()
    db.refresh(workspace)
    return get_workspace(db, workspace_id, member)


def soft_delete_workspace(db: Session, workspace_id: uuid.UUID, member: WorkspaceMember) -> None:
    workspace = db.get(Workspace, workspace_id)
    workspace.deleted_at = func.now()
    log_activity(
        db, workspace_id=workspace_id, actor_id=member.user_id,
        action="workspace.deleted", target_type="workspace", target_id=workspace_id,
    )
    db.commit()


def get_dashboard(db: Session, workspace_id: uuid.UUID, member: WorkspaceMember) -> WorkspaceDashboardRead:
    workspace_read = get_workspace(db, workspace_id, member)

    project_count = db.scalar(
        select(func.count(Project.id)).where(
            Project.workspace_id == workspace_id, Project.deleted_at.is_(None)
        )
    ) or 0

    file_count = db.scalar(
        select(func.count(File.id))
        .join(Project, Project.id == File.project_id)
        .where(Project.workspace_id == workspace_id, File.deleted_at.is_(None))
    ) or 0

    conversation_count = db.scalar(
        select(func.count(Conversation.id))
        .join(Project, Project.id == Conversation.project_id)
        .where(Project.workspace_id == workspace_id)
    ) or 0

    processing_rows = db.execute(
        select(File.id, File.filename, FileVersion.status)
        .join(Project, Project.id == File.project_id)
        .join(FileVersion, FileVersion.id == File.current_version_id)
        .where(
            Project.workspace_id == workspace_id,
            File.deleted_at.is_(None),
            FileVersion.status.notin_([ProcessingStage.READY, ProcessingStage.FAILED]),
        )
    ).all()
    files_processing = [
        FileProcessingSummary(file_id=fid, filename=fname, status=status_.value)
        for fid, fname, status_ in processing_rows
    ]

    recent_file_rows = db.execute(
    select(File, User)
    .join(Project, Project.id == File.project_id)
    .join(User, User.id == File.uploaded_by)
    .options(joinedload(File.current_version))
    .where(
        Project.workspace_id == workspace_id,
        Project.deleted_at.is_(None),
        File.deleted_at.is_(None),
    )
    .order_by(File.created_at.desc())
    .limit(5)
).unique().all()

    recent_files = [
    FileSummary(
        id=f.id,
        filename=f.filename,
        uploaded_by=UserRead.model_validate(uploader),
        status=(f.current_version.status.value if f.current_version else "uploaded"),
    )
    for f, uploader in recent_file_rows
]

    recent_conv_rows = db.execute(
        select(Conversation)
        .join(Project, Project.id == Conversation.project_id)
        .options(joinedload(Conversation.messages))
        .where(Project.workspace_id == workspace_id)
        .order_by(Conversation.created_at.desc())
        .limit(5)
    ).unique().scalars().all()
    recent_conversations = [
        ConversationSummary(
            id=conv.id, title=conv.title,
            context_label="Entire project" if conv.scope.value == "entire_project" else "File subset",
            message_count=len(conv.messages),
        )
        for conv in recent_conv_rows
    ]

    return WorkspaceDashboardRead(
        workspace=workspace_read,
        project_count=project_count,
        file_count=file_count,
        conversation_count=conversation_count,
        files_processing=files_processing,
        recent_files=recent_files,
        recent_conversations=recent_conversations,
    )


ACTIVITY_LIMIT = 50


def _metadata_of(row: WorkspaceActivityLog) -> dict | None:
    return row.event_metadata if isinstance(row.event_metadata, dict) else None


def list_activity(db: Session, workspace_id: uuid.UUID) -> list[ActivityRead]:
    """Latest activity of the workspace, newest first, with the actor's name and a readable sentence."""
    rows = db.execute(
        select(WorkspaceActivityLog, User.full_name)
        .outerjoin(User, User.id == WorkspaceActivityLog.actor_id)
        .where(WorkspaceActivityLog.workspace_id == workspace_id)
        .order_by(WorkspaceActivityLog.created_at.desc())
        .limit(ACTIVITY_LIMIT)
    ).all()
    return [
        ActivityRead(
            id=row.id,
            actor_name=actor_name or "Someone",
            action=describe_activity(row.action, _metadata_of(row)),
            created_at=row.created_at,
        )
        for row, actor_name in rows
    ]