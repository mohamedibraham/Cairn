from dataclasses import dataclass, field
from helpers import settings

from .errors import LLMConfigError



"""Provider-agnostic LLM configuration.

`LLMConfig` is a plain, immutable value object. Providers receive it, never the
app-wide settings object, so they can be built and tested without the app.
`LLMConfig.from_settings` is the single place that knows how app settings map
onto it.

"""



@dataclass(frozen=True)
class LLMConfig:
    provider: str = settings.LLM_PROVIDER 
    model: str = settings.LLM_MODEL
    api_key: str | None = field(default=None, repr=False)  
    temperature: float = settings.LLM_TEMPERATURE
    max_output_tokens: int = settings.LLM_MAX_OUTPUT_TOKENS
    timeout_seconds: float = settings.LLM_TIMEOUT_SECONDS
    max_retries: int = settings.LLM_MAX_RETRIES
    retry_base_delay: float = 2.0
    max_retry_wait: float = 60.0  
    max_concurrency: int = settings.LLM_MAX_CONCURRENCY
    requests_per_minute: int = settings.LLM_REQUESTS_PER_MINUTE
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.provider:
            raise LLMConfigError("LLM_PROVIDER is empty")
        if self.timeout_seconds <= 0:
            raise LLMConfigError("LLM_TIMEOUT_SECONDS must be > 0")
        if self.max_concurrency < 1:
            raise LLMConfigError("LLM_MAX_CONCURRENCY must be >= 1")
        if self.max_retries < 0 or self.requests_per_minute < 0:
            raise LLMConfigError("LLM_MAX_RETRIES / LLM_REQUESTS_PER_MINUTE must be >= 0")

    @classmethod
    def from_settings(cls, settings) -> "LLMConfig":

        values = _settings_items(settings)
        defaults = cls()

        def get(name, default):
            value = values.get(name)
            return default if value is None else value

        provider = str(get("LLM_PROVIDER", defaults.provider)).strip().lower()
        prefix = provider.upper() + "_"

        extra = {}
        for key, value in values.items():
            if key.startswith(prefix) and key != prefix + "API_KEY" and value is not None:
                extra[key[len(prefix):].lower()] = _unwrap(value)

        return cls(
            provider=provider,
            model=str(get("LLM_MODEL", defaults.model)),
            api_key=_unwrap(values.get(prefix + "API_KEY")) or None,
            temperature=float(get("LLM_TEMPERATURE", defaults.temperature)),
            max_output_tokens=int(get("LLM_MAX_OUTPUT_TOKENS", defaults.max_output_tokens)),
            timeout_seconds=float(get("LLM_TIMEOUT_SECONDS", defaults.timeout_seconds)),
            max_retries=int(get("LLM_MAX_RETRIES", defaults.max_retries)),
            retry_base_delay=float(get("LLM_RETRY_BASE_DELAY", defaults.retry_base_delay)),
            max_retry_wait=float(get("LLM_MAX_RETRY_WAIT", defaults.max_retry_wait)),
            max_concurrency=int(get("LLM_MAX_CONCURRENCY", defaults.max_concurrency)),
            requests_per_minute=int(get("LLM_REQUESTS_PER_MINUTE", defaults.requests_per_minute)),
            extra=extra,
        )


def _settings_items(settings) -> dict:
    fields = getattr(type(settings), "model_fields", None)  
    if fields is not None:
        return {name: getattr(settings, name, None) for name in fields}
    return dict(vars(settings))


def _unwrap(value):
    
    getter = getattr(value, "get_secret_value", None)
    return getter() if callable(getter) else value
