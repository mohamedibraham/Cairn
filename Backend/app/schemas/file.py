import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from enums import FileType, ProcessingEventStatus, ProcessingStage


class ProcessingEventRead(BaseModel):

    model_config = ConfigDict(from_attributes=True)

    stage: ProcessingStage
    status: ProcessingEventStatus
    error_message: str | None
    created_at: datetime


class ChunkPreviewRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    chunk_index: int
    content: str
    page_number: int | None
    line_start: int | None
    line_end: int | None
    symbol: str | None = None
    chunk_type: str | None = None


class FileVersionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    version_number: int
    status: ProcessingStage
    chunk_count: int
    error_message: str | None
    uploaded_by_id: uuid.UUID
    created_at: datetime
    pipeline: list[ProcessingEventRead] = []


class FileRead(BaseModel):
    
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    file_type: FileType
    uploaded_by_id: uuid.UUID
    latest_version: FileVersionRead


class FileDetailRead(FileRead):
    
    version_history: list[FileVersionRead]
    chunk_preview: list[ChunkPreviewRead]
