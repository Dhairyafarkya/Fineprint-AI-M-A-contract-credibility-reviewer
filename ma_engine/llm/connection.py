"""Check, before any analysis, whether the configured model provider actually works.

Gemini: asks for the model's metadata (costs no tokens).
Ollama: asks the local server which models are installed (/api/tags).
Results:
    live     - provider reachable and model available
    offline  - Gemini without a key (or MA_DEMO=1); the app runs with placeholder AI output
    error    - utils.py is broken, the key is rejected, a model is missing, or the server is unreachable
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import LLMConfig, load_user_settings
from .factory import demo_forced


@dataclass(frozen=True)
class ConnectionStatus:
    state: str  # "live" | "offline" | "error"
    message: str
    model: str = ""
    detail: str = ""  # optional advice shown below the status, e.g. a missing embedding model

    @property
    def ok(self) -> bool:
        return self.state == "live"


def check_connection(cfg: LLMConfig | None = None, sdk_client: Any | None = None) -> ConnectionStatus:
    cfg = cfg or LLMConfig()
    _settings, settings_error = load_user_settings()
    if demo_forced():
        return ConnectionStatus("offline", "Offline mode is switched on (MA_DEMO=1).", cfg.model)
    if cfg.is_local:
        if settings_error:
            return ConnectionStatus("error", settings_error, cfg.ollama_model)
        return check_ollama(cfg, sdk_client)
    if settings_error and not cfg.api_key:
        return ConnectionStatus("error", settings_error, cfg.model)
    if not cfg.api_key:
        return ConnectionStatus("offline", "No Gemini API key in utils.py. AI sections will show placeholders.", cfg.model)
    if settings_error:
        return ConnectionStatus("error", settings_error, cfg.model)

    try:
        if sdk_client is None:
            from google import genai
            from google.genai import types

            sdk_client = genai.Client(api_key=cfg.api_key, http_options=types.HttpOptions(timeout=15_000))
        sdk_client.models.get(model=cfg.model)
    except Exception as exc:  # noqa: BLE001 - every failure is reported to the operator
        from .gemini_client import _friendly_error

        return ConnectionStatus("error", _friendly_error(getattr(exc, "code", None), str(exc), cfg.model), cfg.model)
    return ConnectionStatus("live", f"Connected to Gemini ({cfg.model}).", cfg.model)


def _has_model(installed: set[str], name: str) -> bool:
    return name in installed or f"{name}:latest" in installed or (":" not in name and any(
        n.split(":")[0] == name for n in installed))


def check_ollama(cfg: LLMConfig, http: Any | None = None) -> ConnectionStatus:
    host = cfg.ollama_host.rstrip("/")
    try:
        if http is None:
            import httpx

            http = httpx.Client(timeout=10.0)
        r = http.get(f"{host}/api/tags")
        r.raise_for_status()
        installed = {m.get("name", "") for m in r.json().get("models", [])}
    except Exception as exc:  # noqa: BLE001
        return ConnectionStatus("error", f"Ollama is not reachable at {host} ({exc.__class__.__name__}). Start it "
                                "with `ollama serve` or scripts/setup_t4.sh, and check OLLAMA_HOST in utils.py.",
                                cfg.ollama_model)
    missing = [m for m in (cfg.ollama_model,) if not _has_model(installed, m)]
    if missing:
        return ConnectionStatus("error", f"Ollama is running but '{missing[0]}' is not installed. "
                                f"Run: ollama pull {missing[0]}", cfg.ollama_model)
    note = "" if _has_model(installed, cfg.ollama_embed_model) else (
        f"Embedding model '{cfg.ollama_embed_model}' is not installed, so a simpler local search is used. "
        f"For better retrieval run: ollama pull {cfg.ollama_embed_model}")
    return ConnectionStatus("live", f"Running locally on Ollama ({cfg.ollama_model})", cfg.ollama_model, note)
