import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db import Base, UUIDPKMixin
from enums import ArtifactScope, ArtifactType


class Artifact(UUIDPKMixin, Base):
    
    __tablename__ = "artifacts"
    __table_args__ = (
        Index("ix_artifacts_project_type", "project_id", "type"),
        Index("ix_artifacts_project_scope", "project_id", "scope_type"),
    )

    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    type: Mapped[ArtifactType] = mapped_column(nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    scope_type: Mapped[ArtifactScope] = mapped_column(nullable=False)
    scope_file_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("files.id", ondelete="SET NULL"), nullable=True
    )
    content: Mapped[dict] = mapped_column(JSONB, nullable=False)  # shape depends on `type`
    source_message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("messages.id", ondelete="SET NULL"), nullable=True
    )
    created_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(server_default="now()", nullable=False)

    project: Mapped["Project"] = relationship(back_populates="artifacts")  # noqa: F821
    source_message: Mapped["Message | None"] = relationship(back_populates="artifacts")  # noqa: F821

    def __repr__(self) -> str:
        return f"<Artifact {self.type} '{self.title}'>"
