"""Common interface implemented by the real Gemini client and the mock client."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

from ..models import UsageStats

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when the language model cannot produce a valid response."""


class LLMClient(ABC):
    is_mock: bool = False

    def __init__(self) -> None:
        self.usage = UsageStats()
        self._usage_lock = threading.Lock()

    def _record(self, inp: int = 0, out: int = 0, cache_read: int = 0, cache_write: int = 0, cached: bool = False) -> None:
        with self._usage_lock:
            if cached:
                self.usage.cached_responses += 1
                return
            self.usage.calls += 1
            self.usage.input_tokens += inp
            self.usage.output_tokens += out
            self.usage.cache_read_tokens += cache_read
            self.usage.cache_write_tokens += cache_write

    @abstractmethod
    def structured(
        self,
        *,
        system: str,
        prompt: str,
        output_model: type[T],
        context: str | None = None,
        max_tokens: int | None = None,
        task: str = "",
    ) -> T:
        """Ask the model for an answer that validates against `output_model`.

        `context` is a large, reusable block (document text, excerpts). It is sent
        first so repeated calls on the same document benefit from prompt caching.
        `task` is a short label used for logging and by the mock.
        """
