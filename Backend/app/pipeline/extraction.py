import codecs
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from docx import Document as load_docx
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader
from pypdf.errors import PyPdfError

logger = logging.getLogger(__name__)

MAX_DOCX_UNCOMPRESSED_BYTES = 200 * 1024 * 1024  
SNIFF_BYTES = 8192




class ExtractionError(Exception):
    pass

class UnsupportedFileTypeError(ExtractionError):
    pass


class FileSignatureError(ExtractionError):
    pass

class EmptyDocumentError(ExtractionError):
    pass




@dataclass(frozen=True)
class ExtractedPage:
    text: str
    page_number: int | None  


@dataclass(frozen=True)
class ExtractedDocument:
    pages: list[ExtractedPage]
    file_type: str

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    @property
    def page_count(self) -> int:
        return len(self.pages)




_MULTI_NEWLINES = re.compile(r"\n{3,}")


def _normalize_newlines(text: str) -> str:
    
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")


def _clean(text: str) -> str:
    """Aggressive cleanup for formats where line numbers are meaningless."""
    return _MULTI_NEWLINES.sub("\n\n", _normalize_newlines(text)).strip()


def _decode_text(raw: bytes) -> str:
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        logger.warning("File is not valid UTF-8; falling back to cp1256")
        return raw.decode("cp1256", errors="replace")




def verify_signature(path: Path, file_type: str) -> None:
    """Checks that the file's real content matches its extension.
    Raises FileSignatureError on mismatch."""
    with path.open("rb") as f:
        head = f.read(SNIFF_BYTES)

    if file_type == "pdf":
        
        if b"%PDF-" not in head[:1024]:
            raise FileSignatureError("File is not a valid PDF")

    elif file_type == "docx":
        if not head.startswith(b"PK\x03\x04") or not zipfile.is_zipfile(path):
            raise FileSignatureError("File is not a valid DOCX")
        try:
            with zipfile.ZipFile(path) as zf:
                if "word/document.xml" not in zf.namelist():
                    raise FileSignatureError("File is not a valid DOCX")
                total = sum(i.file_size for i in zf.infolist())
                if total > MAX_DOCX_UNCOMPRESSED_BYTES:
                    raise FileSignatureError("DOCX uncompressed size is too large")
        except zipfile.BadZipFile as exc:
            raise FileSignatureError("File is not a valid DOCX") from exc

    elif file_type in ("txt", "md", "py"):
        is_utf16 = head.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE))
        if not is_utf16 and b"\x00" in head:
            raise FileSignatureError("File looks binary, not plain text")




def _extract_pdf(path: Path) -> list[ExtractedPage]:
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ExtractionError("PDF is password-protected")

        pages = []
        for index, page in enumerate(reader.pages, start=1):
            text = _clean(page.extract_text() or "")
            if text:
                pages.append(ExtractedPage(text=text, page_number=index))
        return pages
    except PyPdfError as exc:
        raise ExtractionError(f"Could not read PDF: {exc}") from exc


def _extract_docx(path: Path) -> list[ExtractedPage]:
    try:
        doc = load_docx(str(path))
    except Exception as exc:  
        raise ExtractionError(f"Could not read DOCX: {exc}") from exc

    
    blocks: list[str] = []
    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            blocks.append(Paragraph(child, doc).text)
        elif child.tag.endswith("}tbl"):
            for row in Table(child, doc).rows:
                cells = [c.text.strip() for c in row.cells]
                blocks.append(" | ".join(cells))

    text = _clean("\n".join(blocks))
    return [ExtractedPage(text=text, page_number=None)] if text else []


def _extract_plain_text(path: Path) -> list[ExtractedPage]:
    
    text = _normalize_newlines(_decode_text(path.read_bytes()))
    return [ExtractedPage(text=text, page_number=None)] if text.strip() else []


_EXTRACTORS = {
    "pdf": _extract_pdf,
    "docx": _extract_docx,
    "txt": _extract_plain_text,
    "md": _extract_plain_text,
    "py": _extract_plain_text,
}




def extract_document(storage_path: str, file_type: str) -> ExtractedDocument:
    file_type = file_type.lower().lstrip(".")
    extractor = _EXTRACTORS.get(file_type)
    if extractor is None:
        raise UnsupportedFileTypeError(f"No extractor for '.{file_type}'")

    path = Path(storage_path)
    if not path.is_file():
        raise ExtractionError("Stored file not found on disk")

    verify_signature(path, file_type)
    pages = extractor(path)

    if not pages:
        raise EmptyDocumentError(
            "No extractable text found (the file may be empty or a scanned document)"
        )

    return ExtractedDocument(pages=pages, file_type=file_type)