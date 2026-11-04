import uuid

from sqlalchemy.orm import Session

from models.workspace import WorkspaceActivityLog


def log_activity(
    db: Session,
    *,
    workspace_id: uuid.UUID,
    actor_id: uuid.UUID,
    action: str,
    target_type: str,
    target_id: uuid.UUID | None = None,
    metadata: dict | None = None,
) -> None:
    
    db.add(WorkspaceActivityLog(
        workspace_id=workspace_id,
        actor_id=actor_id,
        action=action,
        target_type=target_type,
        target_id=target_id,
        event_metadata=metadata or {},
    ))
