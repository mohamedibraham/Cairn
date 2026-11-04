import re

import httpx
from google import genai
from google.genai import errors, types
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
from llm import (
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


"""Google Gemini provider (Google AI Studio / Gemini API) via the official
`google-genai` SDK.

Everything vendor-specific lives here and nowhere else: the SDK client, request
and config construction, response parsing, and translation of SDK/HTTP errors
into the LLMError hierarchy. Retries are NOT done here (the SDK's own retry is
left off) — `llm/resilience.py` does them uniformly for every provider.
"""


PROVIDER = "gemini"


_BLOCKED_FINISH = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"}
_RETRY_DELAY = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*s\s*$")
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


class GeminiClient(LLMClient):
    provider = PROVIDER

    def __init__(self, config: LLMConfig, *, sdk_client=None):
        if sdk_client is None and not config.api_key:
            raise LLMConfigError(
                "GEMINI_API_KEY is not set (create a key in Google AI Studio)", provider=PROVIDER
            )
        self.model = config.model
        self._config = config
        self._client = sdk_client or genai.Client(
            api_key=config.api_key,
            http_options=types.HttpOptions(
                timeout=int(config.timeout_seconds * 1000),  # the SDK takes milliseconds
                base_url=config.extra.get("base_url") or None,
            ),
        )

    # -- LLMClient 

    def generate(self, request: LLMRequest) -> LLMResponse:
        return self._to_response(self._call(request, schema=None))

    def generate_structured(self, request: LLMRequest, schema: type) -> StructuredResponse:
        response = self._to_response(self._call(request, schema=schema))
        return StructuredResponse(self._validate(response.text, schema), response)

    # -- request 

    def _call(self, request: LLMRequest, *, schema):
        try:
            contents = [
                types.Content(
                    role="user" if m.role == "user" else "model",
                    parts=[types.Part(text=m.content)],
                )
                for m in request.messages
            ]
            config = self._build_config(request, schema)
        except (ValueError, TypeError, ValidationError) as exc:
            raise LLMInvalidRequestError(f"could not build the Gemini request: {exc}", provider=PROVIDER) from exc

        try:
            return self._client.models.generate_content(model=self.model, contents=contents, config=config)
        except Exception as exc:  
            raise self._translate(exc) from exc

    def _build_config(self, request: LLMRequest, schema) -> "types.GenerateContentConfig":
        p, c = request.params, self._config
        kwargs = {
            "system_instruction": request.system,
            "temperature": p.temperature if p.temperature is not None else c.temperature,
            "max_output_tokens": p.max_output_tokens or c.max_output_tokens,
            "top_p": p.top_p,
            "stop_sequences": list(p.stop) or None,
        }
        budget = c.extra.get("thinking_budget")
        if budget is not None:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=int(budget))
        if schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = schema
        return types.GenerateContentConfig(**{k: v for k, v in kwargs.items() if v is not None})

    # -- response 

    def _to_response(self, resp) -> LLMResponse:
        feedback = getattr(resp, "prompt_feedback", None)
        if feedback is not None and getattr(feedback, "block_reason", None):
            raise LLMContentBlockedError(f"prompt blocked: {feedback.block_reason.name}", provider=PROVIDER)
        if not resp.candidates:
            raise LLMInvalidResponseError("the model returned no candidates", provider=PROVIDER)

        finish = resp.candidates[0].finish_reason
        finish_name = finish.name if finish is not None else None
        text = _text_of(resp)

        if not text:
            if finish_name in _BLOCKED_FINISH:
                raise LLMContentBlockedError(f"answer blocked: {finish_name}", provider=PROVIDER)
            if finish_name == "MAX_TOKENS":
                raise LLMInvalidResponseError(
                    "output budget exhausted before any text; thinking tokens count toward "
                    "max_output_tokens — raise LLM_MAX_OUTPUT_TOKENS or lower GEMINI_THINKING_BUDGET",
                    provider=PROVIDER,
                )
            raise LLMInvalidResponseError("the model returned an empty response", provider=PROVIDER)

        usage = getattr(resp, "usage_metadata", None)
        return LLMResponse(
            text=text,
            provider=PROVIDER,
            model=self.model,
            usage=TokenUsage(
                getattr(usage, "prompt_token_count", None),
                getattr(usage, "candidates_token_count", None),
                getattr(usage, "total_token_count", None),
            ),
            finish_reason=_finish_reason(finish_name),
        )

    @staticmethod
    def _validate(text: str, schema: type):
        stripped = _FENCE.match(text)
        payload = stripped.group(1) if stripped else text
        try:
            return schema.model_validate_json(payload)
        except (ValidationError, ValueError) as exc:
            raise LLMInvalidResponseError(
                "structured output failed validation", detail=str(exc)[:800], provider=PROVIDER
            ) from exc

    # -- errors 

    @staticmethod
    def _translate(exc: Exception) -> LLMError:
        if isinstance(exc, LLMError):
            return exc
        if isinstance(exc, httpx.TimeoutException):
            return LLMTimeoutError(f"request timed out: {exc}", provider=PROVIDER)
        if isinstance(exc, httpx.TransportError):
            return LLMConnectionError(f"could not reach Gemini: {exc}", provider=PROVIDER)
        if not isinstance(exc, errors.APIError):
            return LLMError(f"unexpected {type(exc).__name__}: {exc}", provider=PROVIDER)

        code = exc.code or 0
        message = (exc.message or "")[:500]
        detail = f"HTTP {code} {exc.status or ''}: {message}".strip()
        low = message.lower()

        if code in (401, 403) or (code == 400 and "api key" in low):
            return LLMAuthError(detail, provider=PROVIDER)
        if code == 404:
            return LLMConfigError(f"{detail} (check LLM_MODEL)", provider=PROVIDER)
        if code == 429:
            return LLMRateLimitError(
                detail, provider=PROVIDER,
                retry_after=_retry_after(exc),
                retryable=not _is_daily_quota(exc, low),  
            )
        if code in (408, 504):
            return LLMTimeoutError(detail, provider=PROVIDER)
        if code >= 500:
            return LLMUnavailableError(detail, provider=PROVIDER)
        if 400 <= code < 500:
            return LLMInvalidRequestError(detail, provider=PROVIDER)
        return LLMError(detail, provider=PROVIDER)


# -- helpers 

def _text_of(resp) -> str:
    try:
        return (resp.text or "").strip()
    except Exception:  
        return ""


def _finish_reason(name: str | None) -> FinishReason:
    if name in (None, "STOP", "FINISH_REASON_UNSPECIFIED"):
        return "stop"
    if name == "MAX_TOKENS":
        return "length"
    if name in _BLOCKED_FINISH:
        return "safety"
    return "other"


def _detail_items(exc) -> list:
    
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        inner = details.get("error", details)
        details = inner.get("details") if isinstance(inner, dict) else None
    return [item for item in (details or []) if isinstance(item, dict)]


def _retry_after(exc) -> float | None:
    for item in _detail_items(exc):
        if str(item.get("@type", "")).endswith("RetryInfo"):
            match = _RETRY_DELAY.match(str(item.get("retryDelay", "")))
            if match:
                return float(match.group(1))
    return None


def _is_daily_quota(exc, low_message: str) -> bool:
    for item in _detail_items(exc):
        if str(item.get("@type", "")).endswith("QuotaFailure"):
            for violation in item.get("violations", []) or []:
                if "perday" in str(violation.get("quotaId", "")).lower():
                    return True
    return "per day" in low_message or "perday" in low_message.replace(" ", "")
