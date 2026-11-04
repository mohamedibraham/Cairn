import re
import uuid

from fastapi import HTTPException, status
from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session, joinedload

from enums import ArtifactType, ConversationScope, MessageRole
from services.retrieval_service import load_scope_chunks
from services import artifact_service
from services.answer_service import LLMAnswer, RetrievedChunk, generate_answer, parse_answer
from .retrieval_service import load_scope_chunks, similarity_search
from services.artifacts.base import Mismatch
from models.artifact import Artifact
from models.conversation import Conversation, ConversationFileScope, Message, MessageCitation
from models.file import Chunk, File, FileVersion
from models.project import Project
from models.user import User 
from pipeline.embedding import EmbeddingError, embed_query
from schemas.conversation import (
    AuthorRead,
    CitationRead,
    ConversationCreate,
    ConversationRead,
    ConversationUpdate,
    MessageRead,
)

RETRIEVAL_TOP_K = 6

_FILE_SUBSET = ConversationScope.FILE_SUBSET
_ENTIRE_PROJECT = ConversationScope("entire_project")


def _not_found(code: str, message: str) -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, detail={"error": {"code": code, "message": message}})


def _bad_request(code: str, message: str) -> HTTPException:
    return HTTPException(status.HTTP_400_BAD_REQUEST, detail={"error": {"code": code, "message": message}})



def _validate_file_ids(db: Session, project_id: uuid.UUID, file_ids: list[uuid.UUID]) -> list[uuid.UUID]:
    unique = list(dict.fromkeys(file_ids))
    valid = set(db.scalars(select(File.id).where(File.project_id == project_id, File.id.in_(unique))).all())
    missing = set(unique) - valid
    if missing:
        raise _bad_request("INVALID_FILE_IDS", f"Files not in this project: {sorted(str(m) for m in missing)}")
    return unique


def _conversation_read(conversation: Conversation, file_ids: list[uuid.UUID]) -> ConversationRead:
    read = ConversationRead.model_validate(conversation)
    read.file_ids = file_ids  
    return read


def create_conversation(
    db: Session, project: Project, creator_id: uuid.UUID, data: ConversationCreate
) -> ConversationRead:
    if data.scope == _FILE_SUBSET and not data.file_ids:
        raise _bad_request("FILE_IDS_REQUIRED", "file_ids is required when scope is file_subset")

    file_ids: list[uuid.UUID] = []
    if data.scope == _FILE_SUBSET:
        file_ids = _validate_file_ids(db, project.id, data.file_ids)  # validate BEFORE writing anything

    conversation = Conversation(
        project_id=project.id, title=data.title, scope=data.scope, created_by=creator_id,
    )
    db.add(conversation)
    db.flush()
    for file_id in file_ids:
        db.add(ConversationFileScope(conversation_id=conversation.id, file_id=file_id))

    db.commit()
    db.refresh(conversation)
    return _conversation_read(conversation, file_ids)


def list_conversations(db: Session, project_id: uuid.UUID) -> list[ConversationRead]:
    conversations = db.execute(
        select(Conversation)
        .where(Conversation.project_id == project_id)
        .order_by(Conversation.updated_at.desc())
    ).scalars().all()

    scoped: dict[uuid.UUID, list[uuid.UUID]] = {}
    if conversations:
        rows = db.execute(
            select(ConversationFileScope.conversation_id, ConversationFileScope.file_id)
            .where(ConversationFileScope.conversation_id.in_([c.id for c in conversations]))
        ).all()
        for conversation_id, file_id in rows:
            scoped.setdefault(conversation_id, []).append(file_id)
    return [_conversation_read(c, scoped.get(c.id, [])) for c in conversations]


def get_conversation_or_404(db: Session, project_id: uuid.UUID, conversation_id: uuid.UUID) -> Conversation:
    conv = db.get(Conversation, conversation_id)
    if conv is None or conv.project_id != project_id:
        raise _not_found("CONVERSATION_NOT_FOUND", "Conversation not found")
    return conv


def update_conversation(
    db: Session, project: Project, conversation: Conversation, data: ConversationUpdate
) -> ConversationRead:
    """Rename and/or change the context. Only the fields that are sent are changed."""
    if data.title is not None:
        title = data.title.strip()
        if not title:
            raise _bad_request("EMPTY_TITLE", "The title cannot be empty")
        conversation.title = title

    if data.scope is not None or data.file_ids is not None:
        scope = data.scope or (_FILE_SUBSET if data.file_ids else _ENTIRE_PROJECT)
        file_ids: list[uuid.UUID] = []
        if scope == _FILE_SUBSET:
            requested = data.file_ids if data.file_ids is not None else (_scoped_file_ids(db, conversation) or [])
            if not requested:
                raise _bad_request("FILE_IDS_REQUIRED", "file_ids is required when scope is file_subset")
            file_ids = _validate_file_ids(db, project.id, requested)

        db.execute(delete(ConversationFileScope).where(ConversationFileScope.conversation_id == conversation.id))
        for file_id in file_ids:
            db.add(ConversationFileScope(conversation_id=conversation.id, file_id=file_id))
        conversation.scope = scope

    db.commit()
    db.refresh(conversation)
    return _conversation_read(conversation, _scoped_file_ids(db, conversation) or [])


def delete_conversation(db: Session, conversation: Conversation) -> None:
    """Removes the conversation, its messages, citations and file scope.
    Artifacts that were saved from one of its messages are kept (they just lose the link)."""
    message_ids = select(Message.id).where(Message.conversation_id == conversation.id)
    db.execute(update(Artifact).where(Artifact.source_message_id.in_(message_ids)).values(source_message_id=None))
    db.execute(delete(MessageCitation).where(MessageCitation.message_id.in_(message_ids)))
    db.execute(delete(Message).where(Message.conversation_id == conversation.id))
    db.execute(delete(ConversationFileScope).where(ConversationFileScope.conversation_id == conversation.id))
    db.expire(conversation)
    db.delete(conversation)
    db.commit()


def list_messages(db: Session, project_id: uuid.UUID, conversation_id: uuid.UUID) -> list[MessageRead]:
    conv = get_conversation_or_404(db, project_id, conversation_id)
    messages = db.execute(
        select(Message)
        .options(_MESSAGE_EAGER)
        .where(Message.conversation_id == conv.id)
        .order_by(Message.created_at)
    ).unique().scalars().all()
    authors = _authors(db, messages)
    return [_to_message_read(m, authors) for m in messages]


_MESSAGE_EAGER = joinedload(Message.citations).joinedload(MessageCitation.chunk).joinedload(Chunk.file_version)


def _load_message_read(db: Session, message_id: uuid.UUID) -> MessageRead:
    """Loads exactly one message (with citations) by id — never 'the latest
    one', which would depend on timestamp ordering."""
    message = db.execute(
        select(Message).options(_MESSAGE_EAGER).where(Message.id == message_id)
        .execution_options(populate_existing=True)
    ).unique().scalar_one()
    return _to_message_read(message, _authors(db, [message]))


def _authors(db: Session, messages: list[Message]) -> dict[uuid.UUID, AuthorRead]:
    """Who wrote each user message (assistant messages have no author)."""
    ids = {m.created_by for m in messages if m.role == MessageRole.USER and m.created_by is not None}
    if not ids:
        return {}
    users = db.scalars(select(User).where(User.id.in_(ids))).all()
    return {u.id: AuthorRead(id=u.id, full_name=u.full_name) for u in users}


def _to_message_read(message: Message, authors: dict[uuid.UUID, AuthorRead] | None = None) -> MessageRead:
    citations = []
    for c in sorted(message.citations, key=lambda c: c.citation_index):
        chunk = c.chunk
        citations.append(CitationRead(
            citation_index=c.citation_index, chunk_id=chunk.id,
            file_id=chunk.file_version.file_id, filename=chunk.file_version.file.filename,
            page_number=chunk.page_number, line_start=chunk.line_start,
            excerpt=chunk.content[:280], symbol=chunk.symbol,
        ))
    return MessageRead(
        id=message.id, role=message.role, content=message.content,
        created_at=message.created_at, citations=citations,
        blocks=message.blocks or None,  # None = plain text answer
        author=(authors or {}).get(message.created_by) if message.created_by is not None else None,
    )


#  retrieval 

def _retrieve_chunks(db: Session, project_id: uuid.UUID, file_ids: list[uuid.UUID] | None, query_vector: list[float]) -> list[Chunk]:
    query = (
        select(Chunk)
        .join(FileVersion, FileVersion.id == Chunk.file_version_id)
        .join(File, File.id == FileVersion.file_id)
        .where(File.project_id == project_id, File.deleted_at.is_(None), File.current_version_id == FileVersion.id)
        .options(joinedload(Chunk.file_version).joinedload(FileVersion.file))
    )
    if file_ids:
        query = query.where(File.id.in_(file_ids))
    query = query.order_by(Chunk.embedding.cosine_distance(query_vector)).limit(RETRIEVAL_TOP_K)
    return db.execute(query).unique().scalars().all()


def _scoped_file_ids(db: Session, conversation: Conversation) -> list[uuid.UUID] | None:
    if conversation.scope != ConversationScope.FILE_SUBSET:
        return None
    return list(db.scalars(
        select(ConversationFileScope.file_id).where(ConversationFileScope.conversation_id == conversation.id)
    ).all())


#  ask 

def ask_question(
    db: Session,
    project: Project,
    conversation: Conversation,
    asker_id: uuid.UUID,
    content: str,
    requested_type: ArtifactType | None = None,
) -> MessageRead:
    db.add(Message(conversation_id=conversation.id, role=MessageRole.USER, content=content, created_by=asker_id))
    db.commit()
    if requested_type is not None:  # "+" menu in the composer: build a structured artifact instead of a text answer
        return _generate_artifact_reply(db, project, conversation, content, requested_type)
    return _generate_assistant_reply(db, project, conversation, content)


#  artifact replies 


_DEFAULT_PROMPT = re.compile(r"^\s*generate an?\s+\w+\s+from\s+(entire project|\d+ files?)\.?\s*$", re.IGNORECASE)


def _topic_from(question: str) -> str | None:
    """The user's own words become the topic; the composer's default sentence does not."""
    text = question.strip()
    return None if not text or _DEFAULT_PROMPT.match(text) else text


def _artifact_blocks(
    db: Session,
    project: Project,
    conversation: Conversation,
    question: str,
    artifact_type: ArtifactType,
    *,
    force: bool = False,
) -> tuple[str, list[dict], dict[int, str]]:
    
    file_ids = _scoped_file_ids(db, conversation)
    outcome = artifact_service.build_outcome(
        db, project, file_ids=file_ids, artifact_type=artifact_type,
        topic=_topic_from(question), force=force,
    )
    if isinstance(outcome, Mismatch):
        return outcome.reason, [{
            "type": "mismatch",
            "requested": outcome.requested.value,
            "reason": outcome.reason,
            "alternatives": [t.value for t in outcome.alternatives],
        }], {}

    
    citations_map: dict[int, str] = {}
    if artifact_type == ArtifactType.SUMMARY:
        body: str = outcome.content.get("body", "")
        
        scope_chunks = load_scope_chunks(db, project.id, file_ids)
        if scope_chunks:
            llm_answer = parse_answer(body, scope_chunks)
            citations_map = llm_answer.citations  


    return f"Here is your {artifact_type.value}: {outcome.title}.", [{
        "type": "artifact",
        "title": outcome.title,
        "content": {**outcome.content, "type": artifact_type.value},
    }], citations_map


def _generate_artifact_reply(
    db: Session, project: Project, conversation: Conversation, question: str, artifact_type: ArtifactType
) -> MessageRead:
    text, blocks, citations_map = _artifact_blocks(db, project, conversation, question, artifact_type)
    message = Message(conversation_id=conversation.id, role=MessageRole.ASSISTANT, content=text, blocks=blocks)
    db.add(message)
    db.flush()  

    for citation_index, chunk_id_str in citations_map.items():
        try:
            db.add(MessageCitation(
                message_id=message.id,
                chunk_id=uuid.UUID(chunk_id_str),
                citation_index=citation_index,
            ))
        except (ValueError, Exception):
           
            pass

    db.commit()
    return _load_message_read(db, message.id)


def _get_assistant_message(db: Session, conversation: Conversation, message_id: uuid.UUID) -> Message:
    target = db.get(Message, message_id)
    if target is None or target.conversation_id != conversation.id or target.role != MessageRole.ASSISTANT:
        raise _not_found("MESSAGE_NOT_FOUND", "No assistant message found")
    return target


def _preceding_question(db: Session, conversation: Conversation, target: Message) -> Message:
    question = db.scalar(
        select(Message)
        .where(
            Message.conversation_id == conversation.id,
            Message.role == MessageRole.USER,
            Message.created_at < target.created_at,
        )
        .order_by(Message.created_at.desc())
        .limit(1)
    )
    if question is None:
        raise _not_found("NO_PRIOR_QUESTION", "No preceding question found")
    return question


def generate_artifact_for_message(
    db: Session, project: Project, conversation: Conversation, message_id: uuid.UUID, artifact_type: ArtifactType
) -> MessageRead:
   
    target = _get_assistant_message(db, conversation, message_id)
    question = _preceding_question(db, conversation, target)

    existing = list(target.blocks or [{"type": "text", "text": target.content}])
    forced = any(b.get("type") == "mismatch" and b.get("requested") == artifact_type.value for b in existing)

    _text, new_blocks, _cmap = _artifact_blocks(db, project, conversation, question.content, artifact_type, force=forced)
    

    kept = [
        b for b in existing
        if b.get("type") != "mismatch"
        and not (b.get("type") == "artifact" and (b.get("content") or {}).get("type") == artifact_type.value)
    ]
    target.blocks = kept + new_blocks  
    db.commit()
    return _load_message_read(db, target.id)


def _requested_type_from_blocks(blocks: list[dict] | None) -> ArtifactType | None:
    """If an answer was an artifact (or a refused artifact), regenerating must build the same kind again."""
    for block in blocks or []:
        try:
            if block.get("type") == "artifact":
                return ArtifactType((block.get("content") or {}).get("type"))
            if block.get("type") == "mismatch":
                return ArtifactType(block.get("requested"))
        except ValueError:
            continue
    return None




def _generate_assistant_reply(
    db: Session, project: Project, conversation: Conversation, question: str
) -> MessageRead:
    file_ids = _scoped_file_ids(db, conversation)

    try:
        query_vector = embed_query(question)
    except EmbeddingError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail={
            "error": {"code": "EMBEDDING_FAILED", "message": str(exc)}
        })

    chunks = _retrieve_chunks(db, project.id, file_ids, query_vector)
    if not chunks:
        raise _bad_request("NO_CONTENT", "No ready (embedded) content found in scope to answer from")

    context = [
        RetrievedChunk(
            chunk_id=str(c.id), file_id=str(c.file_version.file_id), filename=c.file_version.file.filename,
            content=c.content, page_number=c.page_number, line_start=c.line_start, symbol=c.symbol,
        )
        for c in chunks
    ]

    
    answer: LLMAnswer = generate_answer(question, context)

    
    markers_in_text = {int(m) for m in re.findall(r"\[(\d+)\]", answer.text)}
    stray = set(answer.citations) - markers_in_text
    if stray:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail={
            "error": {"code": "LLM_CITATION_MISMATCH", "message": f"Citations {stray} have no matching marker in the answer text"}
        })

    chunk_by_id = {c.chunk_id: c for c in context}
    assistant_message = Message(conversation_id=conversation.id, role=MessageRole.ASSISTANT, content=answer.text)
    db.add(assistant_message)
    db.flush()

    for citation_index, chunk_id in answer.citations.items():
        if chunk_id not in chunk_by_id:
            continue  
        db.add(MessageCitation(
            message_id=assistant_message.id, chunk_id=uuid.UUID(chunk_id), citation_index=citation_index,
        ))
    db.commit()

    return _load_message_read(db, assistant_message.id)


def regenerate_message(
    db: Session, project: Project, conversation: Conversation, message_id: uuid.UUID, asker_id: uuid.UUID
) -> MessageRead:
    target = db.get(Message, message_id)
    if target is None or target.conversation_id != conversation.id or target.role != MessageRole.ASSISTANT:
        raise _not_found("MESSAGE_NOT_FOUND", "No assistant message found to regenerate")

    preceding_user_message = _preceding_question(db, conversation, target)
    requested_type = _requested_type_from_blocks(target.blocks)

    
    db.execute(update(Artifact).where(Artifact.source_message_id == target.id).values(source_message_id=None))
    db.delete(target)  
    db.commit()

    if requested_type is not None:
        return _generate_artifact_reply(db, project, conversation, preceding_user_message.content, requested_type)
    return _generate_assistant_reply(db, project, conversation, preceding_user_message.content)