import hashlib
import logging
import re
import shutil
import uuid
from pathlib import Path
from typing import NamedTuple

from fastapi import HTTPException, UploadFile, status

from helpers import settings

logger = logging.getLogger(__name__)


CHUNK_SIZE = 1024 * 1024  # 1MB
MAX_FILENAME_LENGTH = 200
_UNSAFE_CHARS = re.compile(r"[^\w.\- ]", flags=re.UNICODE)


class StoredFile(NamedTuple):
    storage_path: str
    size_bytes: int
    sha256: str
    filename: str  # sanitized name actually used on disk


def _error(code: str, message: str, http_status: int = status.HTTP_400_BAD_REQUEST) -> HTTPException:
    return HTTPException(http_status, detail={"error": {"code": code, "message": message}})


# --------------------------------------------------------------------------
# Paths (pure, no I/O)
# --------------------------------------------------------------------------

def project_dir(project_id: uuid.UUID) -> Path:
    return Path(settings.STORAGE_BASE_PATH) / str(project_id)


def version_dir(project_id: uuid.UUID, file_id: uuid.UUID, version_number: int) -> Path:
    return project_dir(project_id) / str(file_id) / f"v{version_number}"


def ensure_storage_root() -> None:
    Path(settings.STORAGE_BASE_PATH).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def sanitize_filename(filename: str | None) -> str:
    """Keeps only the base name (kills '../' and absolute paths), strips
    unsafe characters, and caps the length while preserving the extension."""
    raw = (filename or "").replace("\\", "/")
    name = Path(raw).name.strip().strip(".")
    name = _UNSAFE_CHARS.sub("_", name)

    if not name:
        return "upload.bin"

    if len(name) > MAX_FILENAME_LENGTH:
        p = Path(name)
        name = p.stem[: MAX_FILENAME_LENGTH - len(p.suffix)] + p.suffix
    return name


def validate_upload(
    upload: UploadFile,
    allowed_extensions: list[str],
    allowed_content_types: list[str] | None = None,
) -> str:
    """Cheap, early checks: extension, optional client MIME type, and the
    declared size (a hint only; the real limit is enforced while writing).
    Magic-bytes validation happens in pipeline/extraction.py.
    Returns the lowercase extension without the dot."""
    suffix = Path(sanitize_filename(upload.filename)).suffix.lstrip(".").lower()

    if suffix not in allowed_extensions:
        raise _error(
            "UNSUPPORTED_FILE_TYPE",
            f"'.{suffix}' is not allowed. Allowed: {', '.join(allowed_extensions)}",
        )

    if allowed_content_types and upload.content_type not in allowed_content_types:
        raise _error("UNSUPPORTED_CONTENT_TYPE", f"'{upload.content_type}' is not allowed")

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if upload.size and upload.size > max_bytes:
        raise _error("FILE_TOO_LARGE", f"File exceeds the {settings.MAX_UPLOAD_SIZE_MB}MB limit")

    return suffix


# --------------------------------------------------------------------------
# Save / delete
# --------------------------------------------------------------------------

def save_upload(
    upload: UploadFile, project_id: uuid.UUID, file_id: uuid.UUID, version_number: int
) -> StoredFile:
    """Streams the upload to disk in chunks (never loads the whole file in
    memory), enforces MAX_UPLOAD_SIZE_MB while writing, computes a SHA-256
    checksum, and removes partial files on any failure."""
    target_dir = version_dir(project_id, file_id, version_number)
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = sanitize_filename(upload.filename)
    target_path = (target_dir / filename).resolve()

    # Defense in depth: the final path must stay inside the version dir.
    if target_dir.resolve() not in target_path.parents:
        raise _error("INVALID_FILENAME", "Invalid file name")

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    written = 0
    digest = hashlib.sha256()

    try:
        with target_path.open("wb") as out:
            while chunk := upload.file.read(CHUNK_SIZE):
                written += len(chunk)
                if written > max_bytes:
                    raise _error(
                        "FILE_TOO_LARGE",
                        f"File exceeds the {settings.MAX_UPLOAD_SIZE_MB}MB limit",
                    )
                digest.update(chunk)
                out.write(chunk)

        if written == 0:
            raise _error("EMPTY_FILE", "Uploaded file is empty")

    except HTTPException:
        target_path.unlink(missing_ok=True)
        raise
    except Exception:
        target_path.unlink(missing_ok=True)
        logger.exception("Failed to save upload '%s' (file_id=%s)", filename, file_id)
        raise _error(
            "STORAGE_WRITE_FAILED",
            "Could not store the file",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    return StoredFile(str(target_path), written, digest.hexdigest(), filename)


def delete_file_tree(project_id: uuid.UUID, file_id: uuid.UUID) -> None:
    """Best-effort cleanup. The DB soft-delete is the source of truth, so a
    failure here must not block the API response."""
    shutil.rmtree(project_dir(project_id) / str(file_id), ignore_errors=True)