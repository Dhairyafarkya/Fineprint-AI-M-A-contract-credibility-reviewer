"""Pick the model client from utils.py: Gemini (API), Ollama (local GPU), or the offline mock.

Set the environment variable MA_DEMO=1 to force the offline mock (used by tests
and for working without any model).
"""

from __future__ import annotations

import os

from ..config import LLMConfig
from .base import LLMClient
from .gemini_client import GeminiClient
from .mock_client import MockLLMClient
from .ollama_client import OllamaClient


def demo_forced() -> bool:
    return os.getenv("MA_DEMO", "").strip().lower() in ("1", "true", "yes")


def get_llm_client(cfg: LLMConfig | None = None, demo: bool = False) -> LLMClient:
    cfg = cfg or LLMConfig()
    if demo or demo_forced():
        return MockLLMClient()
    if cfg.is_local:
        return OllamaClient(cfg)
    if not cfg.api_key:
        return MockLLMClient()
    return GeminiClient(cfg)
