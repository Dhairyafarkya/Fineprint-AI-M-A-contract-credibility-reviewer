"""Session state: the engine (rebuilt when utils.py changes), connection status and results."""

from __future__ import annotations

import hashlib

import streamlit as st

from ..config import LLMConfig, get_settings
from ..llm import get_llm_client
from ..llm.connection import ConnectionStatus, check_connection
from ..pipeline import MAEngine

KEYS = ("workspace", "report", "qa_history", "engine", "engine_sig", "error", "advice_stale")


def init_state() -> None:
    for k in KEYS:
        st.session_state.setdefault(k, [] if k == "qa_history" else None)


def _signature(cfg: LLMConfig) -> str:
    parts = (cfg.provider, cfg.api_key, cfg.model, cfg.thinking_level, cfg.ollama_host, cfg.ollama_model,
             cfg.ollama_embed_model, str(cfg.num_ctx))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


@st.cache_data(ttl=600, show_spinner=False)
def _cached_status(signature: str) -> ConnectionStatus:  # noqa: ARG001 - signature is the cache key
    return check_connection(LLMConfig())


def connection_status() -> ConnectionStatus:
    """Re-checked whenever utils.py changes, otherwise at most every 10 minutes."""
    return _cached_status(_signature(LLMConfig()))


def get_engine() -> MAEngine:
    """Key and model come from utils.py; the engine is rebuilt as soon as they change."""
    settings = get_settings()
    sig = _signature(settings.llm)
    if st.session_state.engine is None or st.session_state.engine_sig != sig:
        st.session_state.engine = MAEngine(get_llm_client(settings.llm), settings)
        st.session_state.engine_sig = sig
    return st.session_state.engine


def reset_results() -> None:
    st.session_state.report = None
    st.session_state.workspace = None
    st.session_state.qa_history = []
    st.session_state.error = None
    st.session_state.advice_stale = False
