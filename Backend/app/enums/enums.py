import enum


class UserRole(str, enum.Enum):
    OWNER = "owner"
    EDITOR = "editor"
    VIEWER = "viewer"



ROLE_RANK: dict[UserRole, int] = {
    UserRole.VIEWER: 0,
    UserRole.EDITOR: 1,
    UserRole.OWNER: 2,
}


class WorkspaceVisibility(str, enum.Enum):
    PRIVATE = "private"
    SHARED = "shared"


class JoinRequestStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class FileType(str, enum.Enum):
    PDF = "pdf"
    TXT = "txt"
    PYTHON = "python"


class ProcessingStage(str, enum.Enum):

    UPLOADED = "uploaded"
    EXTRACTING = "extracting"
    EXTRACTED = "extracted"
    CHUNKING = "chunking"
    CHUNKED = "chunked"
    EMBEDDING = "embedding"
    EMBEDDED = "embedded"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


class ProcessingEventStatus(str, enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"


class ConversationScope(str, enum.Enum):
    ENTIRE_PROJECT = "entire_project"
    FILE_SUBSET = "file_subset"


class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ArtifactType(str, enum.Enum):
    SUMMARY = "summary"
    CHART = "chart"
    DIAGRAM = "diagram"
    QUIZ = "quiz"
    MCQ = "mcq"
    FLASHCARDS = "flashcards"
    KEYWORDS = "keywords"


class ArtifactScope(str, enum.Enum):
    FILE = "file"
    PROJECT = "project"


class SearchScope(str, enum.Enum):
    WORKSPACE = "workspace"
    PROJECT = "project"
