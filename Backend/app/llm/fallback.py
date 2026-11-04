
from helpers import get_logger
from .base import LLMClient, LLMRequest, LLMResponse, StructuredResponse
from .errors import (
    LLMConnectionError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)

logger = get_logger("llm")

FALLBACK_ON = (
    LLMUnavailableError,
    LLMConnectionError,
    LLMTimeoutError,
    LLMRateLimitError,
)


class FallbackLLMClient(LLMClient):
    

    def __init__(self, primary: LLMClient, secondary: LLMClient):
        self._primary = primary
        self._secondary = secondary
        
        self.provider = primary.provider
        self.model = primary.model

    
    def generate(self, request: LLMRequest) -> LLMResponse:
        return self._run("generate", lambda c: c.generate(request))

    def generate_structured(self, request: LLMRequest, schema: type) -> StructuredResponse:
        return self._run("generate_structured", lambda c: c.generate_structured(request, schema))

    

    def _run(self, op: str, call):
        try:
            return call(self._primary)
        except FALLBACK_ON as exc:
            logger.warning(
                "llm.%s primary=%s/%s failed with %s (code=%s) — "
                "switching to fallback=%s/%s",
                op,
                self._primary.provider, self._primary.model,
                type(exc).__name__, exc.code,
                self._secondary.provider, self._secondary.model,
            )
            return call(self._secondary)
