import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from enums import JoinRequestStatus, UserRole, WorkspaceVisibility
from .auth import UserRead


class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    visibility: WorkspaceVisibility = WorkspaceVisibility.PRIVATE


class WorkspaceUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    visibility: WorkspaceVisibility | None = None


class WorkspaceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    visibility: WorkspaceVisibility
    owner_id: uuid.UUID
    my_role: UserRole
    created_at: datetime

class ActivityRead(BaseModel):
    id: uuid.UUID
    actor_name: str
    action: str            
    created_at: datetime


class WorkspaceSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    visibility: WorkspaceVisibility
    my_role: UserRole
    project_count: int


class WorkspaceDashboardRead(BaseModel):
    workspace: WorkspaceRead
    project_count: int
    file_count: int
    conversation_count: int
    files_processing: list["FileProcessingSummary"]
    recent_files: list["FileSummary"]
    recent_conversations: list["ConversationSummary"]


class FileProcessingSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    file_id: uuid.UUID
    filename: str
    status: str


class FileSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    filename: str
    status: str
    uploaded_by: UserRead


class ConversationSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    title: str
    context_label: str  
    message_count: int



class WorkspaceMemberRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user: UserRead
    role: UserRole
    joined_at: datetime


class WorkspaceMemberRoleUpdate(BaseModel):
    role: UserRole




class WorkspaceInviteRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    token: str
    invite_url: str  
    created_at: datetime
    is_active: bool  


# Join requests 

class WorkspaceJoinRequestRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user: UserRead
    status: JoinRequestStatus
    created_at: datetime


#  Activity log 

class WorkspaceActivityLogRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    actor: UserRead
    action: str
    target_type: str
    target_id: uuid.UUID | None
    event_metadata: dict
    created_at: datetime
