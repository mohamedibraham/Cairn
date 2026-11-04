import math

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from helpers.logger import get_logger
from llm import LLMError


"""Maps llm.LLMError to HTTP responses, in one place."""

logger = get_logger("llm")

_STATUS = {
    "LLM_NOT_CONFIGURED": 503,
    "LLM_AUTH_FAILED": 503,
    "LLM_RATE_LIMITED": 429,
    "LLM_TIMEOUT": 504,
    "LLM_UNAVAILABLE": 503,
    "LLM_CONTENT_BLOCKED": 422,
    "LLM_INVALID_RESPONSE": 502,
    "LLM_INVALID_REQUEST": 502,
}


def register_llm_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(LLMError)
    async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
        if exc.code in ("LLM_NOT_CONFIGURED", "LLM_AUTH_FAILED"):
            logger.error("LLM configuration problem (%s): %s", exc.code, exc)
        status_code = _STATUS.get(exc.code, 502)
        details = {}
        headers = {}
        if exc.code == "LLM_RATE_LIMITED" and exc.retry_after:
            seconds = max(1, math.ceil(exc.retry_after))
            details["retry_after_seconds"] = seconds
            headers["Retry-After"] = str(seconds)
        return JSONResponse(
            status_code=status_code,
            content={"error": {"code": exc.code, "message": exc.public_message, "details": details or None}},
            headers=headers,
        )
