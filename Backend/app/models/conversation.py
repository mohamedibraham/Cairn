import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID ,JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db import Base, TimestampMixin, UUIDPKMixin
from enums import ConversationScope, MessageRole


class Conversation(UUIDPKMixin, TimestampMixin, Base):
    __tablename__ = "conversations"
    __table_args__ = (Index("ix_conversations_project", "project_id"),)

    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    scope: Mapped[ConversationScope] = mapped_column(
        default=ConversationScope.ENTIRE_PROJECT, nullable=False
    )
    created_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )

    project: Mapped["Project"] = relationship(back_populates="conversations")  # noqa: F821
    file_scopes: Mapped[list["ConversationFileScope"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )
    search_queries: Mapped[list["SearchQuery"]] = relationship(back_populates="conversation")

    def __repr__(self) -> str:
        return f"<Conversation {self.title}>"


class ConversationFileScope(UUIDPKMixin, Base):
    """Only populated when Conversation.scope == FILE_SUBSET — the set of files
    the retrieval step is restricted to for this conversation."""
    __tablename__ = "conversation_file_scopes"
    __table_args__ = (UniqueConstraint("conversation_id", "file_id", name="uq_conv_file_scope"),)

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    file_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("files.id", ondelete="CASCADE"), nullable=False
    )

    conversation: Mapped["Conversation"] = relationship(back_populates="file_scopes")


class Message(UUIDPKMixin, Base):
    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_conversation_created", "conversation_id", "created_at"),)

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[MessageRole] = mapped_column(nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # NULL for assistant messages — only user-authored messages have an author.
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(server_default="now()", nullable=False)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")
    citations: Mapped[list["MessageCitation"]] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
        order_by="MessageCitation.citation_index",
    )
    blocks: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    artifacts: Mapped[list["Artifact"]] = relationship(back_populates="source_message")  # noqa: F821

    def __repr__(self) -> str:
        return f"<Message {self.role} in {self.conversation_id}>"


class MessageCitation(UUIDPKMixin, Base):
    __tablename__ = "message_citations"
    
    __table_args__ = (UniqueConstraint("message_id", "citation_index", name="uq_message_citation_index"),)

    message_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="RESTRICT"), nullable=False
    )
    citation_index: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default="now()", nullable=False)

    message: Mapped["Message"] = relationship(back_populates="citations")

    chunk: Mapped["Chunk"] = relationship()
