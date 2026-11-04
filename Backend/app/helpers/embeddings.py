from sentence_transformers import SentenceTransformer

from .config import get_settings
from .logger import get_logger

settings = get_settings()
logger = get_logger("embeddings")

_model = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model