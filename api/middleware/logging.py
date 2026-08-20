# api/middleware/logging.py
"""Request logging middleware: method, path, status code, and latency in ms."""

from __future__ import annotations

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp

logger = logging.getLogger("ppe_detector.api.access")


class LoggingMiddleware(BaseHTTPMiddleware):
    """Logs one line per request: METHOD PATH -> STATUS (latency_ms)."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "%s %s -> %s (%.2fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response
