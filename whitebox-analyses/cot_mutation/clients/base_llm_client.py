"""Base class for LLM clients."""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypedDict

try:  # Python 3.11+
    from typing import NotRequired
except ImportError:  # pragma: no cover - for very old Python versions
    from typing_extensions import NotRequired  # type: ignore

logger = logging.getLogger(__name__)


def retry_on_error(max_retries: int = 3, initial_wait: float = 30.0):
    """Decorator to retry on 429/503 errors with exponential backoff."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            wait_time = initial_wait
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    status_code = None
                    
                    # Try to extract status code from common error formats
                    if hasattr(e, 'status_code'):
                        status_code = e.status_code
                    elif hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                        status_code = e.response.status_code
                    elif '429' in error_str or 'rate limit' in error_str or 'too many requests' in error_str:
                        status_code = 429
                    elif '503' in error_str or 'service unavailable' in error_str:
                        status_code = 503
                    
                    # Only retry on 429 or 503
                    if status_code in [429, 503]:
                        if attempt < max_retries:
                            logger.warning(
                                f"Got {status_code} error (attempt {attempt + 1}/{max_retries + 1}). "
                                f"Retrying in {wait_time:.0f}s... Error: {e}"
                            )
                            time.sleep(wait_time)
                            wait_time *= 2  # Exponential backoff
                            continue
                        else:
                            logger.error(
                                f"Max retries ({max_retries}) reached for {status_code} error. Giving up."
                            )
                    
                    # Re-raise the exception if not retryable or max retries reached
                    raise
            
            return None  # Should never reach here
        return wrapper
    return decorator


class ChatResult(TypedDict):
    """Standardized response payload returned by :class:`BaseLLMClient`."""

    text: str
    usage: Dict[str, Any]
    raw: Dict[str, Any]
    reasoning_text: NotRequired[Optional[str]]
    process_tokens: NotRequired[Optional[int]]
    flags: NotRequired[Dict[str, Any]]


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def send_chat_request(
        self, model_name: str, request: Dict[str, Any]
    ) -> ChatResult:
        """Send a chat completion request."""
        pass

    def close(self) -> None:
        """Optional cleanup method."""
        pass


__all__ = ["BaseLLMClient", "ChatResult", "retry_on_error"]
