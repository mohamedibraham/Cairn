import uuid

from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session

from db.session import get_db
from dependencies.project import ProjectContext, require_project_role
from enums import UserRole
from schemas.conversation import (
    ConversationCreate,
    ConversationRead,
    ConversationUpdate,
    MessageArtifactRequest,
    MessageCreate,
    MessageRead,
)
from services import conversation_service

router = APIRouter(prefix="/projects/{project_id}/conversations", tags=["conversations"])

MIN_ROLE = UserRole.VIEWER
EDIT_ROLE = UserRole.EDITOR


@router.get("/", response_model=list[ConversationRead])
def list_conversations(
    db: Session = Depends(get_db), ctx: ProjectContext = Depends(require_project_role(MIN_ROLE)),
):
    return conversation_service.list_conversations(db, ctx.project.id)


@router.post("/", response_model=ConversationRead, status_code=201)
def create_conversation(
    payload: ConversationCreate,
    db: Session = Depends(get_db),
    ctx: ProjectContext = Depends(require_project_role(MIN_ROLE)),
):
    return conversation_service.create_conversation(db, ctx.project, ctx.member.user_id, payload)


@router.patch("/{conversation_id}", response_model=ConversationRead)
def update_conversation(
    conversation_id: uuid.UUID,
    payload: ConversationUpdate,
    db: Session = Depends(get_db),
    ctx: ProjectContext = Depends(require_project_role(EDIT_ROLE)),
):
    conversation = conversation_service.get_conversation_or_404(db, ctx.project.id, conversation_id)
    return conversation_service.update_conversation(db, ctx.project, conversation, payload)


@router.delete("/{conversation_id}", status_code=204)
def delete_conversation(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
    ctx: ProjectContext = Depends(require_project_role(EDIT_ROLE)),
):
    conversation = conversation_service.get_conversation_or_404(db, ctx.project.id, conversation_id)
    conversation_service.delete_conversation(db, conversation)
    return Response(status_code=204)


@router.get("/{conversation_id}/messages", response_model=list[MessageRead])
def list_messages(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
    ctx: ProjectContext = Depends(require_project_role(MIN_ROLE)),
):
    return conversation_service.list_messages(db, ctx.project.id, conversation_id)


@router.post("/{conversation_id}/messages", response_model=MessageRead, status_code=201)
def ask(
    conversation_id: uuid.UUID,
    payload: MessageCreate,
    db: Session = Depends(get_db),
    ctx: ProjectContext = Depends(require_project_role(MIN_ROLE)),
):
    conversation = conversation_service.get_conversation_or_404(db, ctx.project.id, conversation_id)
    return conversation_service.ask_question(
        db, ctx.project, conversation, ctx.member.user_id, payload.content,
        requested_type=payload.requested_artifact_type,
    )


@router.post("/{conversation_id}/messages/{message_id}/regenerate", response_model=MessageRead)
def regenerate(
    conversation_id: uuid.UUID,
    message_id: uuid.UUID,
    db: Session = Depends(get_db),
    ctx: ProjectContext = Depends(require_project_role(MIN_ROLE)),
):
    conversation = conversation_service.get_conversation_or_404(db, ctx.project.id, conversation_id)
    return conversation_service.regenerate_message(db, ctx.project, conversation, message_id, ctx.member.user_id)


@router.post("/{conversation_id}/messages/{message_id}/generate-artifact", response_model=MessageRead)
def generate_artifact_for_message(
    conversation_id: uuid.UUID,
    message_id: uuid.UUID,
    payload: MessageArtifactRequest,
    db: Session = Depends(get_db),
    ctx: ProjectContext = Depends(require_project_role(MIN_ROLE)),
):
    
    conversation = conversation_service.get_conversation_or_404(db, ctx.project.id, conversation_id)
    return conversation_service.generate_artifact_for_message(db, ctx.project, conversation, message_id, payload.type)