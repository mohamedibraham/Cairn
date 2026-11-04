from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from helpers import settings
from helpers.exception_handlers import register_exception_handlers
from helpers.llm_exception_handler import register_llm_exception_handlers
from helpers.logger import get_logger
from core.storage import ensure_storage_root
from pipeline.embedding import EmbeddingError, get_model
from routers import artifact, auth_router, conversation, file, project, search, workspace, workspace_invite, workspace_join_requests
from routers import (
    workspace_members,
)

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, REGISTRY
from fastapi import Depends, FastAPI, Response
from sqlalchemy import text

from monitoring.metrics import DBPoolCollector
from monitoring.middleware import PrometheusMiddleware
from db.session import engine, get_db   

logger = get_logger("app")

app = FastAPI(title=settings.PROJECT_NAME)

app.add_middleware(PrometheusMiddleware)
REGISTRY.register(DBPoolCollector(engine))

@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health/live", include_in_schema=False)
def live():
    return {"status": "ok"}

@app.get("/health/ready", include_in_schema=False)
def ready(db=Depends(get_db)):
    db.execute(text("SELECT 1"))
    return {"status": "ready"}

register_exception_handlers(app)
register_llm_exception_handlers(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOWED_ORIGINS,  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    ensure_storage_root()  
    try:
        get_model()
    except EmbeddingError as exc:
        logger.warning("Embedding model not preloaded: %s", exc)


@app.get("/health", tags=["health"])
def health() -> dict:
    return {"status": "ok"}



app.include_router(auth_router.router, prefix=settings.API_V1_PREFIX)
app.include_router(workspace.router, prefix=settings.API_V1_PREFIX)
app.include_router(workspace_members.router, prefix=settings.API_V1_PREFIX)
app.include_router(workspace_invite.router, prefix=settings.API_V1_PREFIX)
app.include_router(workspace_invite.invites_router, prefix=settings.API_V1_PREFIX)
app.include_router(workspace_join_requests.router, prefix=settings.API_V1_PREFIX)
app.include_router(project.router, prefix=settings.API_V1_PREFIX)
app.include_router(file.router, prefix=settings.API_V1_PREFIX)
app.include_router(conversation.router, prefix=settings.API_V1_PREFIX)
app.include_router(artifact.router, prefix=settings.API_V1_PREFIX)
app.include_router(search.router, prefix=settings.API_V1_PREFIX)
