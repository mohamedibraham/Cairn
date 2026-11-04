from functools import lru_cache

from pydantic import PostgresDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App 
    PROJECT_NAME: str = "Cairn"
    API_V1_PREFIX: str = "/api/v1"
    ENV: str = "development"

    # ── Database
    DATABASE_URL: PostgresDsn

    # ── JWT 
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # ── CORS 
    CORS_ALLOWED_ORIGINS: list[str] = []

    # ── File storage 
    STORAGE_BASE_PATH: str = "storage/uploads"   
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_FILE_TYPES: list[str] = ["pdf", "txt", "py"]

    # ── Embeddings 
    EMBEDDING_DIM: int = 768
    EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-base"  
    EMBEDDING_CACHE_DIR: str
    EMBEDDING_BATCH_SIZE: int = 32          
    EMBEDDING_DOC_PREFIX: str = "passage: " 
    EMBEDDING_QUERY_PREFIX: str = "query: " 

    CHUNK_SIZE_TOKENS: int = 400
    CHUNK_OVERLAP_TOKENS: int

    LOG_LEVEL: str = "INFO"

    # ── LLM 
    LLM_PROVIDER: str = "cohere"                
    LLM_MODEL: str = "command-r-plus-08-2024"    
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_OUTPUT_TOKENS: int = 2048
    LLM_TIMEOUT_SECONDS: float = 45.0           
    LLM_MAX_RETRIES: int = 2                     
    LLM_RETRY_BASE_DELAY: float = 2.0
    LLM_MAX_RETRY_WAIT: float = 60.0
    LLM_MAX_CONCURRENCY: int = 4                 
    LLM_REQUESTS_PER_MINUTE: int = 0             

    LLM_FALLBACK_PROVIDER: str
    LLM_FALLBACK_MODEL: str

    # ── Provider credentials 
    
    COHERE_API_KEY: SecretStr | None = None
    GEMINI_API_KEY: SecretStr | None = None
    GEMINI_THINKING_BUDGET: int | None = None   
    GEMINI_BASE_URL: str | None = None         


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()