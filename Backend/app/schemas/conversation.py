import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field 

from typing import Any

from enums import ConversationScope, MessageRole ,ArtifactType

class AuthorRead(BaseModel):                      
    id: uuid.UUID
    full_name: str


class ConversationCreate(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    scope: ConversationScope = ConversationScope.ENTIRE_PROJECT
    
    file_ids: list[uuid.UUID] | None = None


class ConversationUpdate(BaseModel):               
    title: str | None = None
    scope: ConversationScope | None = None
    file_ids: list[uuid.UUID] | None = None


class ConversationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    scope: ConversationScope
    created_by: uuid.UUID
    created_at: datetime
    file_ids: list[uuid.UUID] = []


class MessageCreate(BaseModel):
    content: str = Field(min_length=1)
    requested_artifact_type: ArtifactType | None = None


class MessageArtifactRequest(BaseModel):           
    type: ArtifactType


class CitationRead(BaseModel):
    
    model_config = ConfigDict(from_attributes=True)

    citation_index: int
    chunk_id: uuid.UUID
    file_id: uuid.UUID
    filename: str
    page_number: int | None
    line_start: int | None
    excerpt: str
    symbol: str | None = None


class MessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: MessageRole
    content: str
    created_at: datetime
    citations: list[CitationRead] = []
    blocks: list[dict[str, Any]] | None = None
    author: AuthorRead | None = None
