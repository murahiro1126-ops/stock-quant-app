"""Retry transient Gemini API failures with a small, bounded backoff."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")
MAX_ATTEMPTS = 3
RETRY_DELAYS_SECONDS = (2, 4)
RETRYABLE_STATUS_CODES = {502, 503, 504}


def generate_with_retries(
    request: Callable[[], T],
    *,
    api_error_type: type[BaseException],
    timeout_error_types: tuple[type[BaseException], ...],
) -> T:
    """Run a Gemini request up to three times for transient failures only."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            return request()
        except Exception as exc:
            status_code = getattr(exc, "code", None)
            retryable = isinstance(exc, timeout_error_types) or (
                isinstance(exc, api_error_type)
                and status_code in RETRYABLE_STATUS_CODES
            )
            if not retryable or attempt == MAX_ATTEMPTS - 1:
                raise

            delay = RETRY_DELAYS_SECONDS[attempt]
            LOGGER.warning(
                "Gemini APIの一時エラーです。%d秒後に再試行します（%d/%d）: %s",
                delay,
                attempt + 1,
                MAX_ATTEMPTS - 1,
                exc,
            )
            time.sleep(delay)

    raise RuntimeError("Gemini APIの再試行処理が予期せず終了しました。")
