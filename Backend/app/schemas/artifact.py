import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from enums import ArtifactScope, ArtifactType


class ArtifactGenerateRequest(BaseModel):
    
    type: ArtifactType
    scope_type: ArtifactScope = ArtifactScope.PROJECT
    scope_file_id: uuid.UUID | None = None  
    topic: str | None = Field(default=None, max_length=500)  
    count: int | None = Field(default=None, ge=1, le=50)  
    difficulty: Literal["easy", "medium", "hard"] | None = None
    question_types: list[Literal["true_false", "mcq", "fill_blank"]] | None = None 
    chart_type: Literal["bar", "line", "pie", "scatter"] | None = None  
    force: bool = False  


class ArtifactFromMessageRequest(BaseModel):
    """'Save as artifact' from a chat reply (design decision, section 13 #4) —
    frontend sends exactly what it wants saved, Backend does not re-derive it
    from the message."""
    type: ArtifactType
    title: str = Field(min_length=1, max_length=255)
    selected_content: str = Field(min_length=1)
    scope_type: ArtifactScope | None = None 
    scope_file_id: uuid.UUID | None = None        
    content: dict[str, Any] | None = None  


class ArtifactRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    type: ArtifactType
    title: str
    scope_type: ArtifactScope
    scope_file_id: uuid.UUID | None
    content: dict[str, Any]
    source_message_id: uuid.UUID | None
    created_at: datetime


class ArtifactFilter(BaseModel):
    
    type: ArtifactType | None = None
    scope_type: ArtifactScope | None = None
