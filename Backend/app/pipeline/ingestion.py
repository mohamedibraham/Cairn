import logging
import time
import uuid
from pathlib import Path

from sqlalchemy import delete
from sqlalchemy.orm import Session

from monitoring.metrics import FILES_PROCESSED, INGESTION_STAGE_SECONDS
from db.session import SessionLocal
from enums import ProcessingEventStatus, ProcessingStage
from models.file import Chunk, FileProcessingEvent, FileVersion
from pipeline.chunking import split_into_chunks
from pipeline.embedding import EmbeddingError, embed_chunks
from pipeline.extraction import EmptyDocumentError, ExtractionError, extract_document
from pipeline.indexing import refresh_search_index

logger = logging.getLogger(__name__)


_USER_FACING_ERRORS = (ExtractionError, EmbeddingError)
_UNEXPECTED_ERROR_MESSAGE = "Unexpected error while processing the file"



def _set_stage(
    db: Session,
    version: FileVersion,
    stage: ProcessingStage,
    event_status: ProcessingEventStatus,
    error_message: str | None = None,
) -> None:
    version.status = stage
    if error_message is not None:
        version.error_message = error_message
    db.add(FileProcessingEvent(
        file_version_id=version.id, stage=stage, status=event_status, error_message=error_message,
    ))
    db.commit()


def _record_failure(db: Session, version: FileVersion, message: str) -> None:
    """Rolls back the broken transaction first; otherwise the session is
    unusable and the FAILED status could not be saved."""
    FILES_PROCESSED.labels("failed").inc()
    try:
        db.rollback()
        _set_stage(db, version, ProcessingStage.FAILED, ProcessingEventStatus.FAILED, message)
    except Exception:  # noqa: BLE001
        logger.exception("Could not record failure for file version %s", version.id)
        db.rollback()



def run_ingestion_pipeline(
    db: Session, file_version: FileVersion, storage_path: str, file_type: str
) -> None:
    
    started = time.perf_counter()
    try:
        file_version.error_message = None

        _set_stage(db, file_version, ProcessingStage.EXTRACTING, ProcessingEventStatus.IN_PROGRESS)
        with INGESTION_STAGE_SECONDS.labels("extraction").time():
            document = extract_document(storage_path, file_type)
        _set_stage(db, file_version, ProcessingStage.EXTRACTED, ProcessingEventStatus.DONE)

        _set_stage(db, file_version, ProcessingStage.CHUNKING, ProcessingEventStatus.IN_PROGRESS)
        with INGESTION_STAGE_SECONDS.labels("chunking").time():
            raw_chunks = split_into_chunks(document)
        if not raw_chunks:
            raise EmptyDocumentError("No chunks could be produced from this file")
        _set_stage(db, file_version, ProcessingStage.CHUNKED, ProcessingEventStatus.DONE)

        _set_stage(db, file_version, ProcessingStage.EMBEDDING, ProcessingEventStatus.IN_PROGRESS)
        with INGESTION_STAGE_SECONDS.labels("embedding").time():
            vectors = embed_chunks([c["content"] for c in raw_chunks])

        # Embed first, replace after: if embedding fails, old chunks survive.
        # Delete + insert + stage change commit together (one transaction).
        with INGESTION_STAGE_SECONDS.labels("db_write").time():
            db.execute(delete(Chunk).where(Chunk.file_version_id == file_version.id))
            db.add_all([
                Chunk(
                    file_version_id=file_version.id,
                    chunk_index=i,
                    content=chunk["content"],
                    page_number=chunk["page_number"],
                    line_start=chunk["line_start"],
                    line_end=chunk["line_end"],
                    token_count=chunk["token_count"],
                    chunk_metadata=chunk.get("metadata"),
                    embedding=vector,
                )
                for i, (chunk, vector) in enumerate(zip(raw_chunks, vectors, strict=True))
            ])
            file_version.chunk_count = len(raw_chunks)
            _set_stage(db, file_version, ProcessingStage.EMBEDDED, ProcessingEventStatus.DONE)

        _set_stage(db, file_version, ProcessingStage.INDEXING, ProcessingEventStatus.IN_PROGRESS)
        with INGESTION_STAGE_SECONDS.labels("indexing").time():
            refresh_search_index(file_version.id)
        _set_stage(db, file_version, ProcessingStage.READY, ProcessingEventStatus.DONE)
        FILES_PROCESSED.labels("ready").inc()

    except _USER_FACING_ERRORS as exc:
        logger.warning("Ingestion failed for version %s: %s", file_version.id, exc)
        _record_failure(db, file_version, str(exc))
    except Exception:  # noqa: BLE001 — pipeline failures must never raise past here
        logger.exception("Unexpected ingestion failure for version %s", file_version.id)
        _record_failure(db, file_version, _UNEXPECTED_ERROR_MESSAGE)
    finally:
        INGESTION_STAGE_SECONDS.labels("total").observe(time.perf_counter() - started)


def run_ingestion_in_background(file_version_id: uuid.UUID) -> None:
    
    with SessionLocal() as db:
        version = db.get(FileVersion, file_version_id)
        if version is None:
            logger.error("Ingestion skipped: file version %s not found", file_version_id)
            return

        file_type = Path(version.storage_path).suffix.lstrip(".").lower()
        run_ingestion_pipeline(db, version, version.storage_path, file_type)