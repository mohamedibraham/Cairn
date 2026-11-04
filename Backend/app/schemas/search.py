import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from enums import SearchScope


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    scope: SearchScope = SearchScope.WORKSPACE
    project_id: uuid.UUID | None = None  


class SearchResultItem(BaseModel):
    chunk_id: uuid.UUID
    file_id: uuid.UUID
    filename: str
    project_id: uuid.UUID
    project_name: str
    excerpt: str
    score: float
    symbol: str | None = None
    page_number: int | None = None     
    line_start: int | None = None 


class SearchResponse(BaseModel):
    query_id: uuid.UUID  
    results: list[SearchResultItem]
    result_count: int


class SearchHistoryItemRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    query_text: str
    scope: SearchScope
    project_id: uuid.UUID | None
    project_name: str | None
    result_count: int
    conversation_id: uuid.UUID | None
    created_at: datetime
    conversation_project_id: uuid.UUID | None = None
