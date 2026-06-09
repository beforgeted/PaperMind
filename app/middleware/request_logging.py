"""HTTP 请求日志中间件：request_id 生成与请求生命周期记录。"""

from __future__ import annotations

import logging
import time
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.log_context import clear, generate_request_id, log_bind
from app.observability.events import (
    HTTP_REQUEST_COMPLETED,
    HTTP_REQUEST_FAILED,
    HTTP_REQUEST_STARTED,
)

logger = logging.getLogger(__name__)

_REQUEST_ID_HEADER = "X-Request-ID"


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """为每个 HTTP 请求注入 request_id 并记录耗时。"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = (request.headers.get(_REQUEST_ID_HEADER) or "").strip()
        if not request_id:
            request_id = generate_request_id()

        start = time.perf_counter()
        status_code = 500
        clear()
        try:
            with log_bind(request_id=request_id):
                logger.info(
                    "HTTP request started",
                    extra={
                        "event": HTTP_REQUEST_STARTED,
                        "method": request.method,
                        "path": request.url.path,
                    },
                )
                response = await call_next(request)
                status_code = response.status_code
                latency_ms = round((time.perf_counter() - start) * 1000)
                logger.info(
                    "HTTP request completed",
                    extra={
                        "event": HTTP_REQUEST_COMPLETED,
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": status_code,
                        "latency_ms": latency_ms,
                    },
                )
                response.headers[_REQUEST_ID_HEADER] = request_id
                return response
        except Exception:
            latency_ms = round((time.perf_counter() - start) * 1000)
            logger.exception(
                "HTTP request failed",
                extra={
                    "event": HTTP_REQUEST_FAILED,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "latency_ms": latency_ms,
                },
            )
            raise
        finally:
            clear()
