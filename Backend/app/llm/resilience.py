import random
import threading
import time

from .base import LLMClient, LLMRequest, LLMResponse, StructuredResponse
from .errors import LLMError
from monitoring import LLm_metrics

try:  
    from helpers import get_logger 

    logger = get_logger("llm")
except Exception:  
    import logging

    logger = logging.getLogger("llm")


"""Cross-cutting behaviour for ANY LLM client, as a decorator.

Providers stay thin (translate calls and errors); this wrapper adds, once and
identically for every provider:
  * retries with exponential backoff + jitter, only for errors marked retryable
  * honouring the provider's "retry after N seconds" hint, and refusing to
    sleep longer than `max_wait` (e.g. a daily quota that resets in hours)
  * a concurrency cap and optional requests-per-minute pacing (free tiers
    have tiny RPM limits; pacing avoids burning retries on predictable 429s)
  * one structured log line per call: provider, model, latency, token usage,
    attempt. Prompts and answers are NEVER logged (they contain user documents).

`attempts` below = 1 + max_retries.
"""

JITTER = 0.25  


class ResilientLLMClient(LLMClient):
    def __init__(
        self,
        inner: LLMClient,
        *,
        max_retries: int = 3,
        base_delay: float = 2.0,
        max_wait: float = 60.0,
        max_concurrency: int = 2,
        requests_per_minute: int = 0,
        sleep=time.sleep,
        monotonic=time.monotonic,
        random_fn=random.random,
    ):
        self._inner = inner
        self.provider = inner.provider
        self.model = inner.model
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_wait = max_wait
        self._slots = threading.BoundedSemaphore(max(1, max_concurrency))
        self._interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._pace_lock = threading.Lock()
        self._next_slot = 0.0
        self._sleep = sleep
        self._monotonic = monotonic
        self._random = random_fn

    # -- LLMClient 

    def generate(self, request: LLMRequest) -> LLMResponse:
        return self._run("generate", lambda: self._inner.generate(request))

    def generate_structured(self, request: LLMRequest, schema: type) -> StructuredResponse:
        return self._run("generate_structured", lambda: self._inner.generate_structured(request, schema))

    # -- internals 

    def _attempt(self, call):
        
        p, m = self.provider, self.model
        queued = self._monotonic()
        LLm_metrics.LLM_WAITING.labels(p, m).inc()
        try:
            self._slots.acquire()
        finally:
            LLm_metrics.LLM_WAITING.labels(p, m).dec()
        try:
            self._pace()
            LLm_metrics.LLM_QUEUE_SECONDS.labels(p, m).observe(self._monotonic() - queued)
            LLm_metrics.LLM_IN_FLIGHT.labels(p, m).inc()
            t0 = self._monotonic()
            try:
                return call()
            finally:
                LLm_metrics.LLM_ATTEMPT_SECONDS.labels(p, m).observe(self._monotonic() - t0)
                LLm_metrics.LLM_IN_FLIGHT.labels(p, m).dec()
        finally:
            self._slots.release()

    def _run(self, op: str, call):
        p, m = self.provider, self.model
        operation = LLm_metrics.current_operation(op)
        call_started = self._monotonic()
        outcome = "ok"
        attempt = 0
        try:
            while True:
                attempt += 1
                started = self._monotonic()
                try:
                    result = self._attempt(call)
                except LLMError as exc:
                    LLm_metrics.LLM_ATTEMPTS.labels(p, m, exc.code.lower()).inc()
                    delay = self._retry_delay(exc, attempt)
                    latency_ms = int((self._monotonic() - started) * 1000)
                    if delay is None:
                        outcome = exc.code.lower()
                        logger.warning(
                            "llm.%s failed provider=%s model=%s code=%s attempt=%d latency_ms=%d retryable=%s: %s",
                            op, p, m, exc.code, attempt, latency_ms, exc.retryable, exc,
                        )
                        raise
                    LLm_metrics.LLM_RETRIES.labels(p, m, exc.code.lower()).inc()
                    logger.warning(
                        "llm.%s retrying provider=%s model=%s code=%s attempt=%d wait_s=%.1f",
                        op, p, m, exc.code, attempt, delay,
                    )
                    self._sleep(delay)  
                    continue
                except Exception as exc:  
                    outcome = "unexpected"
                    LLm_metrics.LLM_ATTEMPTS.labels(p, m, "unexpected").inc()
                    logger.exception("llm.%s unexpected provider error provider=%s", op, p)
                    raise LLMError(f"Unexpected {type(exc).__name__} from provider", provider=p) from exc

                # --- التعديل تم هنا (مسار النجاح) ---
                LLm_metrics.LLM_ATTEMPTS.labels(p, m, "ok").inc()
                LLm_metrics.record_usage(p, m, operation, result)
                usage = _usage_of(result)
                
                # يمكنك ترك رسالة نجاح بسيطة أو إزالة الـ logger من هنا تماماً
                latency_ms = int((self._monotonic() - started) * 1000)
                logger.info(
                    "llm.%s success provider=%s model=%s attempt=%d latency_ms=%d",
                    op, p, m, attempt, latency_ms
                )
                
                return result
        finally:
            LLm_metrics.LLM_CALLS.labels(p, m, operation, outcome).inc()
            LLm_metrics.LLM_CALL_SECONDS.labels(p, m, operation).observe(self._monotonic() - call_started)

    def _retry_delay(self, exc: LLMError, attempt: int) -> float | None:
        """Seconds to wait before the next attempt, or None to give up."""
        if not exc.retryable or attempt > self._max_retries:
            return None
        wait = exc.retry_after if exc.retry_after is not None else self._base_delay * (2 ** (attempt - 1))
        wait += wait * JITTER * self._random()
        if wait > self._max_wait:
            return None  
        return wait

    def _pace(self) -> None:
        if not self._interval:
            return
        with self._pace_lock:
            now = self._monotonic()
            wait = self._next_slot - now
            if wait > 0:
                self._sleep(wait)
                self._next_slot += self._interval
            else:
                self._next_slot = now + self._interval


def _usage_of(result):
    response = result.response if isinstance(result, StructuredResponse) else result
    return getattr(response, "usage", None)
