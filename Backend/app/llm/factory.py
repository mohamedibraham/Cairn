import threading
from functools import lru_cache
from typing import Callable

from .base import LLMClient
from .config import LLMConfig
from .errors import LLMConfigError
from .resilience import ResilientLLMClient



"""Creates the right LLMClient from configuration (the "LLM Factory").

Adding a provider:
  1. write `llm/providers/<name>.py` with a class implementing `LLMClient`
  2. register it:  register_provider("<name>", lambda config: MyClient(config))
     (built-in providers are registered below, lazily, so a provider's SDK is
     only imported when that provider is actually selected)
  3. set LLM_PROVIDER=<name> and <NAME>_API_KEY in the environment.
No caller of the LLM layer changes.

App code uses `get_llm_client()` only.
"""

ProviderBuilder = Callable[[LLMConfig], LLMClient]

_builders: dict = {}
_lock = threading.Lock()


def register_provider(name: str, builder: ProviderBuilder, *, replace: bool = False) -> None:
    name = name.strip().lower()
    with _lock:
        if name in _builders and not replace:
            raise ValueError(f"LLM provider {name!r} is already registered")
        _builders[name] = builder


def registered_providers() -> list:
    _register_builtin()
    return sorted(_builders)


def create_llm_client(config: LLMConfig, *, resilient: bool = True) -> LLMClient:

    
    _register_builtin()
    builder = _builders.get(config.provider)
    if builder is None:
        raise LLMConfigError(
            f"Unknown LLM provider {config.provider!r}. Available: {', '.join(sorted(_builders))}"
        )
    client = builder(config)
    if not resilient:
        return client
    return ResilientLLMClient(
        client,
        max_retries=config.max_retries,
        base_delay=config.retry_base_delay,
        max_wait=config.max_retry_wait,
        max_concurrency=config.max_concurrency,
        requests_per_minute=config.requests_per_minute,
    )



def reset_llm_client() -> None:
   
    get_llm_client.cache_clear()


# -- built-in providers 

def _build_gemini(config: LLMConfig) -> LLMClient:
    try:
        from .providers.gemini import GeminiClient
    except ImportError as exc:
        raise LLMConfigError("The 'google-genai' package is not installed (pip install google-genai)",
                             provider="gemini") from exc
    return GeminiClient(config)



def _build_cohere(config: LLMConfig) -> LLMClient:
    try:
        from .providers.cohere import CohereClient
    except ImportError as exc:
        raise LLMConfigError(
            "The 'cohere' package is not installed (pip install cohere>=7.0.0)",
            provider="cohere",
        ) from exc
    return CohereClient(config)
 
 
def _register_builtin() -> None:
    with _lock:
        _builders.setdefault("gemini", _build_gemini)
        _builders.setdefault("cohere", _build_cohere)   
 
@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    from helpers import settings
 
    config = LLMConfig.from_settings(settings)
    primary = create_llm_client(config)
 
    
    fb_provider = str(getattr(settings, "LLM_FALLBACK_PROVIDER", None) or "").lower().strip()
    fb_model    = str(getattr(settings, "LLM_FALLBACK_MODEL",    None) or "").strip()
 
    if fb_provider and fb_model and fb_provider != config.provider:
        _register_builtin()
        if fb_provider not in _builders:
            raise LLMConfigError(
                f"LLM_FALLBACK_PROVIDER={fb_provider!r} is unknown. "
                f"Available: {', '.join(sorted(_builders))}"
            )
        # Build a separate LLMConfig for the fallback provider
        fb_config = _fallback_config(settings, config, fb_provider, fb_model)
        secondary = create_llm_client(fb_config)
 
        from .fallback import FallbackLLMClient
        return FallbackLLMClient(primary, secondary)
 
    return primary
 
 
def _fallback_config(settings, base: LLMConfig, provider: str, model: str) -> LLMConfig:
    """
    Build an LLMConfig for the fallback provider by reading <PROVIDER>_API_KEY
    directly from settings, and inheriting the rest from the base config.
    We intentionally keep max_retries=1 for the fallback — the primary already
    exhausted retries before we got here.
    """
    key_attr = f"{provider.upper()}_API_KEY"
    api_key_raw = getattr(settings, key_attr, None)
    if api_key_raw is None:
        raise LLMConfigError(
            f"{key_attr} is not set — required for the {provider!r} fallback"
        )
    # Unwrap SecretStr if used
    getter = getattr(api_key_raw, "get_secret_value", None)
    api_key = getter() if callable(getter) else api_key_raw
 
    return LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=base.temperature,
        max_output_tokens=base.max_output_tokens,
        timeout_seconds=base.timeout_seconds,
        max_retries=1,                      
        retry_base_delay=base.retry_base_delay,
        max_retry_wait=base.max_retry_wait,
        max_concurrency=base.max_concurrency,
        requests_per_minute=base.requests_per_minute,
    )