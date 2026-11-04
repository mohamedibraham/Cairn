import enum

class WorkspaceVisibility(str, enum.Enum):
    private = "private"
    shared = "shared"

class WorkspaceRole(str, enum.Enum):
    owner = "owner"
    editor = "editor"
    viewer = "viewer"


class FileType(str, enum.Enum):
    pdf = "pdf"
    txt = "txt"
    python = "python"

"""It will be added in future updates."""
    # docx = "docx" 
    # csv = "csv"

class ProcessingStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    chunked = "chunked"
    failed = "failed"          

class EmbeddingStatus(str, enum.Enum):
    pending = "pending"
    embedding = "embedding"
    completed = "completed"
    failed = "failed"

class ConversationScope(str, enum.Enum):
    file = "file"
    project = "project"
    workspace = "workspace"

class MessageRole(str, enum.Enum):
    user = "user"
    assistant = "assistant"

class ArtifactType(str, enum.Enum):
    summary = "summary"
    chart = "chart"
    diagram = "diagram"
    quiz = "quiz"
    mcq = "mcq"
    flashcards = "flashcards"
    keywords = "keywords"

class ArtifactSourceScope(str, enum.Enum):
    file = "file"
    project = "project"