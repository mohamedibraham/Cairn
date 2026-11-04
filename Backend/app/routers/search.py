from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from db.session import get_db
from dependencies.workspace import require_workspace_role
from enums import UserRole
from models.workspace import WorkspaceMember
from schemas.search import SearchHistoryItemRead, SearchRequest, SearchResponse
from services import search_service

router = APIRouter(prefix="/workspaces/{workspace_id}/search", tags=["search"])


@router.post("/", response_model=SearchResponse)
def run_search(
    payload: SearchRequest,
    db: Session = Depends(get_db),
    member: WorkspaceMember = Depends(require_workspace_role(UserRole.VIEWER)),
):
    return search_service.search(db, member.workspace_id, member.user_id, payload)


@router.get("/history", response_model=list[SearchHistoryItemRead])
def search_history(
    db: Session = Depends(get_db),
    member: WorkspaceMember = Depends(require_workspace_role(UserRole.VIEWER)),
):
    return search_service.get_history(db, member.workspace_id)
