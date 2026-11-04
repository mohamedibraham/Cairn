import time
from .metrics import HTTP_REQUESTS, HTTP_DURATION, HTTP_IN_PROGRESS

EXCLUDED = {"/metrics", "/health/live", "/health/ready"}


class PrometheusMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] in EXCLUDED:
            return await self.app(scope, receive, send)

        method, status = scope["method"], 500
        start = time.perf_counter()
        finished = False
        HTTP_IN_PROGRESS.labels(method).inc()

        def finish():
            nonlocal finished
            if finished:
                return
            finished = True
            route = scope.get("route")
            path = getattr(route, "path", None) or "unmatched"
            HTTP_DURATION.labels(method, path).observe(time.perf_counter() - start)
            HTTP_REQUESTS.labels(method, path, str(status)).inc()
            HTTP_IN_PROGRESS.labels(method).dec()

        async def send_wrapper(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)
            
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                finish()

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            finish()