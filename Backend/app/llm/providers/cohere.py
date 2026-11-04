import json
import re

import httpx
from pydantic import ValidationError

from llm import (
    FinishReason,
    LLMClient,
    LLMRequest,
    LLMResponse,
    StructuredResponse,
    TokenUsage,
)
from llm import LLMConfig
from llm.errors import (
    LLMAuthError,
    LLMConfigError,
    LLMConnectionError,
    LLMContentBlockedError,
    LLMError,
    LLMInvalidRequestError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)

PROVIDER = "cohere"

_BLOCKED_FINISH = {"ERROR_TOXIC"}

_ERROR_FINISH = {"ERROR", "TIMEOUT"}


_SCHEMA_CAPABLE = {
    "command-r-plus-08-2024",
    "command-r-08-2024",
    "command-a-03-2025",
    "command-r-plus",   
    "command-r",        
    "command-a",        
}


_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


class CohereClient(LLMClient):
    

    provider = PROVIDER

    def __init__(self, config: LLMConfig, *, sdk_client=None):
        try:
            import cohere  
        except ImportError as exc:
            raise LLMConfigError(
                "The 'cohere' package is not installed. Run: pip install cohere>=7.0.0",
                provider=PROVIDER,
            ) from exc

        if sdk_client is None and not config.api_key:
            raise LLMConfigError(

                provider=PROVIDER,
            )

        self.model = config.model
        self._config = config

        if sdk_client is not None:
            self._client = sdk_client
        else:
            import cohere
            self._client = cohere.ClientV2(
                api_key=config.api_key,
                timeout=config.timeout_seconds,
                
                client_name=config.extra.get("client_name", "rag-app"),
            )

    # LLMClient interface 

    def generate(self, request: LLMRequest) -> LLMResponse:
        return self._to_response(self._call(request, schema=None))

    def generate_structured(self, request: LLMRequest, schema: type) -> StructuredResponse:
        response = self._to_response(self._call(request, schema=schema))
        return StructuredResponse(self._validate(response.text, schema), response)

    # request

    def _call(self, request: LLMRequest, *, schema):
        
        try:
            messages = self._build_messages(request)
            kwargs = self._build_kwargs(request, messages, schema)
        except (ValueError, TypeError) as exc:
            raise LLMInvalidRequestError(
                f"Could not build the Cohere request: {exc}", provider=PROVIDER
            ) from exc

        try:
            return self._client.chat(**kwargs)
        except Exception as exc:          
            raise self._translate(exc) from exc

    def _build_messages(self, request: LLMRequest) -> list:
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        for m in request.messages:
            
            if isinstance(m, dict):
                messages.append({"role": m.get("role"), "content": m.get("content")})
            else:
                messages.append({"role": m.role, "content": m.content})
        
        return messages 
    def _build_kwargs(self, request: LLMRequest, messages: list, schema) -> dict:
        p, c = request.params, self._config

        kwargs: dict = {
            "model":        self.model,
            "messages":     messages, 
            "temperature":  p.temperature if p.temperature is not None else c.temperature,
            "max_tokens":   p.max_output_tokens or c.max_output_tokens,
        }

        if p.top_p is not None:
            kwargs["p"] = p.top_p

        if p.stop:
            kwargs["stop_sequences"] = list(p.stop)

        if schema is not None:
            kwargs["response_format"] = self._build_response_format(schema)

       
        return {k: v for k, v in kwargs.items() if v is not None}

    def _build_response_format(self, schema: type) -> "object":
        
        from cohere.types import JsonObjectResponseFormatV2

        base_model = self.model.split(":")[0]   
        if base_model in _SCHEMA_CAPABLE:
            return JsonObjectResponseFormatV2(
                type="json_object",
                json_schema=schema.model_json_schema(),
            )
        
        return JsonObjectResponseFormatV2(type="json_object")

    # response 

    def _to_response(self, resp) -> LLMResponse:
        
        finish = getattr(resp, "finish_reason", None) or ""

        
        if finish in _BLOCKED_FINISH:
            raise LLMContentBlockedError(
                f"Cohere blocked the response (finish_reason={finish!r})",
                provider=PROVIDER,
            )

        
        text = _extract_text(resp)

        if not text:
            if finish in _ERROR_FINISH:
                raise LLMInvalidResponseError(
                    f"Cohere returned an error response (finish_reason={finish!r})",
                    provider=PROVIDER,
                )
            if finish == "MAX_TOKENS":
                raise LLMInvalidResponseError(
                    "Cohere exhausted max_tokens before producing any text. "
                    "Raise LLM_MAX_OUTPUT_TOKENS.",
                    provider=PROVIDER,
                )
            raise LLMInvalidResponseError(
                "Cohere returned an empty response", provider=PROVIDER
            )

       
        usage_obj  = getattr(resp, "usage", None)
        tokens_obj = getattr(usage_obj, "tokens", None)

        return LLMResponse(
            text=text,
            provider=PROVIDER,
            model=self.model,
            usage=TokenUsage(
                input_tokens=getattr(tokens_obj, "input_tokens", None),
                output_tokens=getattr(tokens_obj, "output_tokens", None),
                total_tokens=(
                    (getattr(tokens_obj, "input_tokens", 0) or 0)
                    + (getattr(tokens_obj, "output_tokens", 0) or 0)
                ) or None,
            ),
            finish_reason=_finish_reason(finish),
        )

    @staticmethod
    def _validate(text: str, schema: type):
        
        stripped = _FENCE.match(text)
        payload = stripped.group(1) if stripped else text
        try:
            return schema.model_validate_json(payload)
        except (ValidationError, ValueError) as exc:
            raise LLMInvalidResponseError(
                "Structured output failed Pydantic validation",
                detail=str(exc)[:800],
                provider=PROVIDER,
            ) from exc

   
    @staticmethod
    def _translate(exc: Exception) -> LLMError:
       
        if isinstance(exc, LLMError):
            return exc

        
        if isinstance(exc, httpx.TimeoutException):
            return LLMTimeoutError(
                f"Cohere request timed out: {exc}", provider=PROVIDER
            )
        if isinstance(exc, httpx.TransportError):
            return LLMConnectionError(
                f"Could not reach Cohere: {exc}", provider=PROVIDER
            )

        
        from cohere.core.api_error import ApiError

        if isinstance(exc, ApiError):
            code    = exc.status_code or 0
            body    = exc.body or {}
            message = (
                body.get("message", "") if isinstance(body, dict) else str(body)
            )[:500]
            detail = f"HTTP {code}: {message}".strip() if message else f"HTTP {code}"

            if code in (401, 403):
                return LLMAuthError(detail, provider=PROVIDER)

            if code == 404:
                return LLMConfigError(
                    f"{detail} (check LLM_MODEL — is the model name correct?)",
                    provider=PROVIDER,
                )

            if code == 429:
                return LLMRateLimitError(
                    detail,
                    provider=PROVIDER,
                    retry_after=_retry_after(exc),
                    retryable=True,
                )

            if code in (408, 504):
                return LLMTimeoutError(detail, provider=PROVIDER)

            if code in (500, 503):
                return LLMUnavailableError(detail, provider=PROVIDER)

            if code >= 500:
                return LLMUnavailableError(detail, provider=PROVIDER)

            if 400 <= code < 500:
                return LLMInvalidRequestError(detail, provider=PROVIDER)

        # Unknown / unexpected
        return LLMError(
            f"Unexpected {type(exc).__name__} from Cohere: {str(exc)[:300]}",
            provider=PROVIDER,
        )




def _extract_text(resp) -> str:
    
    try:
        content = resp.message.content  
        if not content:
            return ""
        parts = []
        for item in content:
            if getattr(item, "type", None) == "text":
                parts.append(getattr(item, "text", "") or "")
        return "".join(parts).strip()
    except (AttributeError, TypeError):
        return ""


def _finish_reason(name: str) -> FinishReason:
    name = (name or "").upper()
    if name in ("COMPLETE", "STOP_SEQUENCE", ""):
        return "stop"
    if name == "MAX_TOKENS":
        return "length"
    if name in _BLOCKED_FINISH:
        return "safety"
    return "other"


def _retry_after(exc) -> float | None:
   
    headers = getattr(exc, "headers", None) or {}
    for key in ("retry-after", "x-ratelimit-reset-requests", "ratelimit-reset"):
        val = headers.get(key)
        if val:
            try:
                return float(val)
            except ValueError:
                pass
    return None
