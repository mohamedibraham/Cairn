import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db import Base, UUIDPKMixin
from enums import SearchScope


class SearchQuery(UUIDPKMixin, Base):

    __tablename__ = "search_queries"
    __table_args__ = (Index("ix_search_queries_workspace_created", "workspace_id", "created_at"),)

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=True
    )
    scope: Mapped[SearchScope] = mapped_column(nullable=False)
    query_text: Mapped[str] = mapped_column(String(500), nullable=False)
    result_count: Mapped[int] = mapped_column(Integer, nullable=False)
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True
    )
    created_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(server_default="now()", nullable=False)

    conversation: Mapped["Conversation | None"] = relationship(back_populates="search_queries")  

    def __repr__(self) -> str:
        return f"<SearchQuery '{self.query_text}'>"
