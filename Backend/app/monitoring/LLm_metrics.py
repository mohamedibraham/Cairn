import contextlib
import contextvars

from prometheus_client import Counter, Gauge, Histogram

_operation = contextvars.ContextVar("llm_operation", default=None)


@contextlib.contextmanager
def llm_operation(name: str):

    token = _operation.set(name)
    try:
        yield
    finally:
        _operation.reset(token)


def current_operation(default: str) -> str:
    return _operation.get() or default


_PM = ("provider", "model")
_PMO = ("provider", "model", "operation")

LLM_CALLS = Counter("llm_calls_total", "Logical LLM calls by FINAL outcome (after retries)", (*_PMO, "outcome"))
LLM_ATTEMPTS = Counter("llm_attempts_total", "Individual provider attempts by outcome", (*_PM, "outcome"))
LLM_RETRIES = Counter("llm_retries_total", "Retries triggered, by reason", (*_PM, "reason"))

LLM_ATTEMPT_SECONDS = Histogram(
    "llm_attempt_duration_seconds", "Pure provider latency per attempt", _PM,
    buckets=(0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120),
)
LLM_CALL_SECONDS = Histogram(
    "llm_call_duration_seconds", "End-to-end latency incl. queueing and retries", _PMO,
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300),
)
LLM_QUEUE_SECONDS = Histogram(
    "llm_queue_wait_seconds", "Wait for a concurrency slot + pacing", _PM,
    buckets=(0.01, 0.1, 0.5, 1, 2, 5, 10, 30, 60),
)
LLM_IN_FLIGHT = Gauge("llm_in_flight", "Calls currently executing at the provider", _PM)
LLM_WAITING = Gauge("llm_waiting", "Calls waiting for a concurrency slot", _PM)

LLM_TOKENS = Counter("llm_tokens_total", "Tokens used", (*_PMO, "type"))  
LLM_COST = Counter("llm_cost_usd_total", "Estimated cost in USD", _PMO)
LLM_FINISH = Counter("llm_finish_reason_total", "Finish reasons (watch 'length' = truncated)", (*_PM, "reason"))
LLM_ACCOUNTING_GAPS = Counter(
    "llm_accounting_gaps_total", "Calls whose tokens/cost could not be accounted", (*_PM, "reason"),
)  

def record_usage(provider, model, operation, result) -> None:
    try:
        from ..llm.pricing import price_for
        response = getattr(result, "response", result)
        LLM_FINISH.labels(provider, model, str(response.finish_reason)).inc()
        usage = getattr(response, "usage", None)
        if usage is None or (usage.input_tokens is None and usage.output_tokens is None):
            LLM_ACCOUNTING_GAPS.labels(provider, model, "usage_missing").inc()
            return

        in_tok  = usage.input_tokens  or 0
        out_tok = usage.output_tokens or 0
        LLM_TOKENS.labels(provider, model, operation, "input").inc(in_tok)
        LLM_TOKENS.labels(provider, model, operation, "output").inc(out_tok)

        prices = price_for(provider, model)
        if prices:
            cost = (in_tok * prices[0] + out_tok * prices[1]) / 1_000_000
            LLM_COST.labels(provider, model, operation).inc(cost)
        else:
            LLM_ACCOUNTING_GAPS.labels(provider, model, "no_price").inc()

    except Exception:
        LLM_ACCOUNTING_GAPS.labels(provider, model, "exception").inc()


def init_series(provider: str, model: str, error_codes, operations) -> None:

    for code in (*error_codes, "ok", "unexpected"):
        LLM_ATTEMPTS.labels(provider, model, code)
        for op in operations:
            LLM_CALLS.labels(provider, model, op, code)
    for code in error_codes:
        LLM_RETRIES.labels(provider, model, code)