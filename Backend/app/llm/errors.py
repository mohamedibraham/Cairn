
"""Unified error hierarchy for the LLM layer.

Everything a provider can throw is translated into one of these inside the
provider's own module, so no SDK exception type ever leaks to the rest of the
app. The original exception is chained (`__cause__`) for logs only.

"""

class LLMError(Exception):
    code = "LLM_ERROR"
    retryable = False
    public_message = "The language model request failed."

    def __init__(
        self,
        message: str = "",
        *,
        provider: str | None = None,
        retry_after: float | None = None,
        retryable: bool | None = None,
    ):
        super().__init__(message or self.public_message)
        self.provider = provider
        self.retry_after = retry_after  
        if retryable is not None:
            self.retryable = retryable


class LLMConfigError(LLMError):
    """Missing/invalid configuration"""

    code = "LLM_NOT_CONFIGURED"
    public_message = "The language model is not configured on the server."


class LLMAuthError(LLMError):
    code = "LLM_AUTH_FAILED"
    public_message = "The server could not authenticate with the language model provider."


class LLMRateLimitError(LLMError):
    code = "LLM_RATE_LIMITED"
    retryable = True
    public_message = "The language model is busy right now. Please try again in a moment."


class LLMTimeoutError(LLMError):
    code = "LLM_TIMEOUT"
    retryable = True
    public_message = "The language model took too long to respond."


class LLMConnectionError(LLMError):
    code = "LLM_UNAVAILABLE"
    retryable = True
    public_message = "The language model could not be reached."


class LLMUnavailableError(LLMError):
    """Provider-side 5xx."""
    code = "LLM_UNAVAILABLE"
    retryable = True
    public_message = "The language model service is temporarily unavailable."


class LLMInvalidRequestError(LLMError):
    """The provider rejected our request (4xx other than auth/rate limit)"""

    code = "LLM_INVALID_REQUEST"
    public_message = "The language model rejected the request."


class LLMContentBlockedError(LLMError):
    """The provider's safety filters blocked the prompt or the answer."""

    code = "LLM_CONTENT_BLOCKED"
    public_message = "The request was blocked by the language model's safety filters."


class LLMInvalidResponseError(LLMError):
    """Empty / truncated / unparseable output"""

    code = "LLM_INVALID_RESPONSE"
    public_message = "The language model returned an unusable response."

    def __init__(self, message: str = "", *, detail: str | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.detail = detail
