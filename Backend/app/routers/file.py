import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, File as FastAPIFile, UploadFile
from sqlalchemy.orm import Session

from db import get_db
from dependencies.project import ProjectContext, require_project_role
from enums import UserRole
from schemas.file import FileDetailRead, FileRead
from services import file_service

router = APIRouter(prefix="/projects/{project_id}/files", tags=["files"])


@router.get("/", response_model=list[FileRead])
def list_files(
    db: Session = Depends(get_db),
    ctx: ProjectContext = Depends(require_project_role(UserRole.VIEWER)),
):
    return file_service.list_files(db, ctx.project.id)


@router.post("/upload", response_model=FileRead, status_code=201)
def upload_file(
    background_tasks: BackgroundTasks,
    upload: UploadFile = FastAPIFile(...),
    db: Session = Depends(get_db),
    ctx: ProjectContext = Depends(require_project_role(UserRole.EDITOR)),
):
    return file_service.upload_file(db, ctx.project, ctx.member.user_id, upload, background_tasks)


@router.get("/{file_id}", response_model=FileDetailRead)
def get_file(
    file_id: uuid.UUID,
    db: Session = Depends(get_db),
    ctx: ProjectContext = Depends(require_project_role(UserRole.VIEWER)),
):
    return file_service.get_file_detail(db, ctx.project.id, file_id)


@router.post("/{file_id}/versions", response_model=FileRead, status_code=201)
def upload_new_version(
    file_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    upload: UploadFile = FastAPIFile(...),
    db: Session = Depends(get_db),
    ctx: ProjectContext = Depends(require_project_role(UserRole.EDITOR)),
):
    return file_service.upload_new_version(
        db, ctx.project.id, file_id, ctx.member.user_id, upload, background_tasks
    )


@router.post("/{file_id}/reprocess", response_model=FileRead)
def reprocess_file(
    file_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    ctx: ProjectContext = Depends(require_project_role(UserRole.EDITOR)),
):
    return file_service.reprocess_file(db, ctx.project.id, file_id, background_tasks)


@router.delete("/{file_id}", status_code=204)
def delete_file(
    file_id: uuid.UUID,
    db: Session = Depends(get_db),
    ctx: ProjectContext = Depends(require_project_role(UserRole.EDITOR)),
):
    file_service.delete_file(db, ctx.project.id, file_id)
