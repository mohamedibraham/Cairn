import logging
import threading
from typing import Sequence

import numpy as np

from monitoring.metrics import EMBEDDING_SECONDS
from helpers import settings

logger = logging.getLogger(__name__)

MODEL_NAME: str = settings.EMBEDDING_MODEL_NAME
EMBEDDING_DIM: int = settings.EMBEDDING_DIM
BATCH_SIZE: int = getattr(settings, "EMBEDDING_BATCH_SIZE", 32)
DEVICE: str | None = getattr(settings, "EMBEDDING_DEVICE", None)
DOC_PREFIX: str = getattr(settings, "EMBEDDING_DOC_PREFIX", "")
QUERY_PREFIX: str = getattr(settings, "EMBEDDING_QUERY_PREFIX", "")

CACHE_DIR: str | None = getattr(settings, "EMBEDDING_CACHE_DIR", None) or None


class EmbeddingError(Exception):
    pass


_model = None
_load_lock = threading.Lock()

_encode_lock = threading.Lock()




def get_model():
    
    global _model
    if _model is None:
        with _load_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer  # lazy: heavy import

                logger.info("Loading embedding model '%s'", MODEL_NAME)
                try:
                    model = SentenceTransformer(MODEL_NAME, device=DEVICE, cache_folder=CACHE_DIR)
                except Exception as exc:  
                    raise EmbeddingError(f"Could not load embedding model: {exc}") from exc

                actual_dim = model.get_sentence_embedding_dimension()
                if actual_dim != EMBEDDING_DIM:
                    raise EmbeddingError(
                        f"Model outputs {actual_dim} dimensions but EMBEDDING_DIM={EMBEDDING_DIM}"
                    )
                _model = model
    return _model


def max_seq_length() -> int | None:
    
    return get_model().max_seq_length


def count_tokens(text: str) -> int:
    """Token count using the model's own tokenizer (no special tokens)."""
    tokenizer = get_model().tokenizer
    with _encode_lock:
        return len(tokenizer.encode(text, add_special_tokens=False))



# Encoding

def _free_gpu_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  
        pass


def _encode(texts: list[str]) -> np.ndarray:
    model = get_model()
    batch_size = BATCH_SIZE

    while True:
        try:
            with _encode_lock:
                vectors = model.encode(
                    texts,
                    batch_size=batch_size,
                    normalize_embeddings=True,  
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            break
        except RuntimeError as exc:
            
            if "out of memory" in str(exc).lower() and batch_size > 1:
                batch_size = max(1, batch_size // 2)
                logger.warning("Embedding OOM; retrying with batch_size=%d", batch_size)
                _free_gpu_cache()
                continue
            raise EmbeddingError(f"Embedding failed: {exc}") from exc
        except Exception as exc:  
            raise EmbeddingError(f"Embedding failed: {exc}") from exc

    if vectors.shape != (len(texts), EMBEDDING_DIM):
        raise EmbeddingError(
            f"Unexpected embedding shape {vectors.shape}, expected ({len(texts)}, {EMBEDDING_DIM})"
        )
    if not np.isfinite(vectors).all():
        raise EmbeddingError("Embedding produced NaN/inf values")
    return vectors



def embed_chunks(chunk_texts: Sequence[str]) -> list[list[float]]:
    
    if not chunk_texts:
        return []
    texts = [DOC_PREFIX + t for t in chunk_texts]
    with EMBEDDING_SECONDS.labels("passage").time():
        return _encode(texts).tolist()


def embed_query(query: str) -> list[float]:
    
        return _encode([QUERY_PREFIX + query])[0].tolist()