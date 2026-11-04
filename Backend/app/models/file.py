import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from helpers import settings
from db.base import Base, SoftDeleteMixin, TimestampMixin, UUIDPKMixin
from enums import FileType, ProcessingEventStatus, ProcessingStage


class File(UUIDPKMixin, TimestampMixin, SoftDeleteMixin, Base):
   
    __tablename__ = "files"
    __table_args__ = (Index("ix_files_project", "project_id"),)

    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[FileType] = mapped_column(nullable=False)
    uploaded_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    current_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("file_versions.id", ondelete="SET NULL", use_alter=True, name="fk_file_current_version"),
        nullable=True,
    )

    project: Mapped["Project"] = relationship(back_populates="files")  # noqa: F821
    versions: Mapped[list["FileVersion"]] = relationship(
        back_populates="file",
        cascade="all, delete-orphan",
        foreign_keys="FileVersion.file_id",
        order_by="FileVersion.version_number.desc()",
    )
    current_version: Mapped["FileVersion | None"] = relationship(
        foreign_keys=[current_version_id], post_update=True, viewonly=True
    )

    def __repr__(self) -> str:
        return f"<File {self.filename}>"


class FileVersion(UUIDPKMixin, Base):
    __tablename__ = "file_versions"
    __table_args__ = (
        UniqueConstraint("file_id", "version_number", name="uq_file_version_number"),
        Index("ix_file_versions_file", "file_id", "version_number"),
    )

    file_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("files.id", ondelete="CASCADE"), nullable=False
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    storage_path: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[ProcessingStage] = mapped_column(
        default=ProcessingStage.UPLOADED, nullable=False
    )
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)  # denormalized cache
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    uploaded_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    file: Mapped["File"] = relationship(back_populates="versions", foreign_keys=[file_id])
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="file_version", cascade="all, delete-orphan"
    )
    processing_events: Mapped[list["FileProcessingEvent"]] = relationship(
        back_populates="file_version",
        cascade="all, delete-orphan",
        order_by="FileProcessingEvent.created_at",
    )

    def __repr__(self) -> str:
        return f"<FileVersion {self.file_id} v{self.version_number}>"


class FileProcessingEvent(UUIDPKMixin, Base):
    """Append-only pipeline log for one FileVersion. FileVersion.status is the
    fast-path "current stage" column; this table is what the pipeline popovers
    (Dashboard/Knowledge Base/File Detail) actually render, since it has one
    row per stage with its own status, not just a single current value."""
    __tablename__ = "file_processing_events"
    __table_args__ = (Index("ix_processing_events_version", "file_version_id", "created_at"),)

    file_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("file_versions.id", ondelete="CASCADE"), nullable=False
    )
    stage: Mapped[ProcessingStage] = mapped_column(nullable=False)
    status: Mapped[ProcessingEventStatus] = mapped_column(nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    file_version: Mapped["FileVersion"] = relationship(back_populates="processing_events")


class Chunk(UUIDPKMixin, Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("file_version_id", "chunk_index", name="uq_chunk_index"),
        Index("ix_chunks_file_version", "file_version_id"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    file_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("file_versions.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)  # PDFs
    line_start: Mapped[int | None] = mapped_column(Integer, nullable=True)   # TXT/PY
    line_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    # Structure info from structure-aware chunkers (Python files today): symbol,
    # signature, chunk_type, parent, bases, imports_used, calls ... NULL for
    # PDF/TXT chunks. (Not named `metadata`: that is reserved by SQLAlchemy.)
    # none_as_null: store Python None as SQL NULL, not as the JSON value `null`.
    chunk_metadata: Mapped[dict | None] = mapped_column(JSONB(none_as_null=True), nullable=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.EMBEDDING_DIM), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    file_version: Mapped["FileVersion"] = relationship(back_populates="chunks")

    @property
    def symbol(self) -> str | None:
        """e.g. "Folder.run" for a Python method chunk."""
        return (self.chunk_metadata or {}).get("symbol")

    @property
    def chunk_type(self) -> str | None:
        return (self.chunk_metadata or {}).get("chunk_type")

    def __repr__(self) -> str:
        return f"<Chunk {self.file_version_id}#{self.chunk_index}>"
