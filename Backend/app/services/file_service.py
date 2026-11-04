import uuid
from datetime import datetime, timezone

from fastapi import BackgroundTasks, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from core import storage
from helpers import settings
from enums import FileType, ProcessingEventStatus, ProcessingStage
from models.file import Chunk, File, FileProcessingEvent, FileVersion
from models.project import Project
from pipeline.ingestion import run_ingestion_in_background
from schemas.file import ChunkPreviewRead, FileDetailRead, FileRead, FileVersionRead, ProcessingEventRead

_EXTENSION_TO_TYPE = {"pdf": FileType.PDF, "txt": FileType.TXT, "py": FileType.PYTHON}


def _not_found() -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, detail={
        "error": {"code": "FILE_NOT_FOUND", "message": "File not found"}
    })


def _to_version_read(version: FileVersion) -> FileVersionRead:
    return FileVersionRead(
        id=version.id, version_number=version.version_number, status=version.status,
        chunk_count=version.chunk_count, error_message=version.error_message,
        uploaded_by_id=version.uploaded_by, created_at=version.created_at,
        pipeline=[ProcessingEventRead.model_validate(e) for e in version.processing_events],
    )


def list_files(db: Session, project_id: uuid.UUID) -> list[FileRead]:
    files = db.execute(
        select(File)
        .options(joinedload(File.current_version).joinedload(FileVersion.processing_events))
        .where(File.project_id == project_id, File.deleted_at.is_(None))
        .order_by(File.created_at.desc())
    ).unique().scalars().all()

    return [
        FileRead(
            id=f.id, filename=f.filename, file_type=f.file_type, uploaded_by_id=f.uploaded_by,
            latest_version=_to_version_read(f.current_version),
        )
        for f in files if f.current_version is not None
    ]


def get_file_detail(db: Session, project_id: uuid.UUID, file_id: uuid.UUID) -> FileDetailRead:
    file = db.get(File, file_id)
    if file is None or file.project_id != project_id or file.deleted_at is not None:
        raise _not_found()
    if file.current_version is None:
        # Should not happen in practice — every File gets a version on creation.
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail={
            "error": {"code": "FILE_MISSING_VERSION", "message": "File has no current version"}
        })

    version_history = [_to_version_read(v) for v in file.versions]
    chunk_preview: list[ChunkPreviewRead] = []
    if file.current_version is not None:
        chunks = db.execute(
            select(Chunk)
            .where(Chunk.file_version_id == file.current_version_id)
            .order_by(Chunk.chunk_index)
            .limit(3)
        ).scalars().all()
        chunk_preview = [ChunkPreviewRead.model_validate(c) for c in chunks]

    return FileDetailRead(
        id=file.id, filename=file.filename, file_type=file.file_type, uploaded_by_id=file.uploaded_by,
        latest_version=_to_version_read(file.current_version),
        version_history=version_history,
        chunk_preview=chunk_preview,
    )


def upload_file(
    db: Session, project: Project, uploaded_by: uuid.UUID, upload: UploadFile, background_tasks: BackgroundTasks,
) -> FileRead:
    extension = storage.validate_upload(upload, settings.ALLOWED_FILE_TYPES)
    file_type = _EXTENSION_TO_TYPE[extension]

    file = File(project_id=project.id, filename=upload.filename, file_type=file_type, uploaded_by=uploaded_by)
    db.add(file)
    db.flush()  # need file.id for the storage path and the version FK

    version = FileVersion(file_id=file.id, version_number=1, storage_path="", uploaded_by=uploaded_by)
    db.add(version)
    db.flush()

    stored = storage.save_upload(upload, project.id, file.id, version.version_number)
    version.storage_path = stored.storage_path
    file.current_version_id = version.id

    db.add(FileProcessingEvent(
        file_version_id=version.id, stage=ProcessingStage.UPLOADED, status=ProcessingEventStatus.DONE,
    ))
    db.commit()
    db.refresh(file)
    db.refresh(version)

    background_tasks.add_task(run_ingestion_in_background, version.id)

    return FileRead(
        id=file.id, filename=file.filename, file_type=file.file_type, uploaded_by_id=file.uploaded_by,
        latest_version=_to_version_read(version),
    )


def upload_new_version(
    db: Session, project_id: uuid.UUID, file_id: uuid.UUID, uploaded_by: uuid.UUID,
    upload: UploadFile, background_tasks: BackgroundTasks,
) -> FileRead:
    file = db.get(File, file_id)
    if file is None or file.project_id != project_id or file.deleted_at is not None:
        raise _not_found()

    extension = storage.validate_upload(upload, settings.ALLOWED_FILE_TYPES)
    next_version_number = (
        db.scalar(select(FileVersion.version_number).where(FileVersion.file_id == file_id).order_by(
            FileVersion.version_number.desc()
        ).limit(1)) or 0
    ) + 1

    version = FileVersion(
        file_id=file.id, version_number=next_version_number, storage_path="", uploaded_by=uploaded_by,
    )
    db.add(version)
    db.flush()

    stored = storage.save_upload(upload, file.project_id, file.id, next_version_number)
    version.storage_path = stored.storage_path
    file.current_version_id = version.id

    db.add(FileProcessingEvent(
        file_version_id=version.id, stage=ProcessingStage.UPLOADED, status=ProcessingEventStatus.DONE,
    ))
    db.commit()
    db.refresh(version)

    background_tasks.add_task(run_ingestion_in_background, version.id)

    return FileRead(
        id=file.id, filename=file.filename, file_type=file.file_type, uploaded_by_id=file.uploaded_by,
        latest_version=_to_version_read(version),
    )


def reprocess_file(
    db: Session, project_id: uuid.UUID, file_id: uuid.UUID, background_tasks: BackgroundTasks,
) -> FileRead:
    file = db.get(File, file_id)
    if file is None or file.project_id != project_id or file.deleted_at is not None or file.current_version is None:
        raise _not_found()

    version = file.current_version
    # Clear previously generated chunks — reprocessing starts the pipeline over.
    db.query(Chunk).filter(Chunk.file_version_id == version.id).delete()
    version.chunk_count = 0
    version.error_message = None
    version.status = ProcessingStage.UPLOADED
    db.commit()

    background_tasks.add_task(run_ingestion_in_background, version.id)
    return FileRead(
        id=file.id, filename=file.filename, file_type=file.file_type, uploaded_by_id=file.uploaded_by,
        latest_version=_to_version_read(version),
    )


def delete_file(db: Session, project_id: uuid.UUID, file_id: uuid.UUID) -> None:
    file = db.get(File, file_id)
    if file is None or file.project_id != project_id or file.deleted_at is not None:
        raise _not_found()

    file.deleted_at = datetime.now(timezone.utc)
    db.commit()
    storage.delete_file_tree(project_id, file_id)