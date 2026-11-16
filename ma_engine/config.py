"""Central configuration for the M&A Fine Print Engine.

Every tunable number lives here so the rest of the code stays free of magic
constants. The Gemini API key and model are set in utils.py at the project root. Other
values can be overridden with environment variables (or a .env file).
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


USER_SETTINGS_FILE = "utils.py"
_SETTING_NAMES = (
    "LLM_PROVIDER", "GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_THINKING_LEVEL",
    "OLLAMA_HOST", "OLLAMA_MODEL", "OLLAMA_EMBED_MODEL", "OLLAMA_CONTEXT", "OLLAMA_PARALLEL",
)


def load_user_settings() -> tuple[dict[str, str], str]:
    """Read the provider, Gemini and Ollama settings from utils.py in the project root.

    Returns (settings, error). `error` explains what is wrong with utils.py (for example a
    missing quote around the key) so the app can say so instead of silently going offline.
    """
    path = Path(os.getenv("MA_SETTINGS_FILE") or PROJECT_ROOT / USER_SETTINGS_FILE)
    if not path.is_file():
        return {}, f"{USER_SETTINGS_FILE} was not found in {PROJECT_ROOT}."
    spec = importlib.util.spec_from_file_location("_ma_user_settings", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except SyntaxError as exc:
        return {}, (f"{USER_SETTINGS_FILE} line {exc.lineno} has a syntax error. The key must be inside quotes, "
                    'for example GEMINI_API_KEY = "AIza...".')
    except Exception as exc:  # noqa: BLE001 - any error in a user-edited file
        return {}, f"{USER_SETTINGS_FILE} could not be read: {exc}. The key must be inside quotes."
    values = {k: str(getattr(module, k)).strip() for k in _SETTING_NAMES if getattr(module, k, None)}
    provider = values.get("LLM_PROVIDER", "gemini").lower()
    if provider not in ("gemini", "ollama"):
        return values, f'LLM_PROVIDER in {USER_SETTINGS_FILE} must be "gemini" or "ollama", not "{provider}".'
    key = values.get("GEMINI_API_KEY", "")
    if provider == "gemini" and key and (" " in key or len(key) < 20):
        return values, f"The GEMINI_API_KEY in {USER_SETTINGS_FILE} does not look like a Gemini key (it should start with AIza)."
    return values, ""


def _setting(name: str, env_names: tuple[str, ...], default: str) -> str:
    """utils.py wins; then environment variables; then the default."""
    value = load_user_settings()[0].get(name)
    if value:
        return value
    for env in env_names:
        if os.getenv(env):
            return os.getenv(env, "")
    return default


def _setting_int(name: str, env_names: tuple[str, ...], default: int) -> int:
    try:
        return int(_setting(name, env_names, str(default)))
    except ValueError:
        return default


@dataclass
class LLMConfig:
    # "gemini" (Google API) or "ollama" (local models, e.g. on a T4 GPU)
    provider: str = field(default_factory=lambda: _setting("LLM_PROVIDER", ("MA_PROVIDER",), "gemini").lower())

    # --- Gemini ---
    api_key: str = field(default_factory=lambda: _setting("GEMINI_API_KEY", ("GEMINI_API_KEY", "GOOGLE_API_KEY"), ""))
    model: str = field(default_factory=lambda: _setting("GEMINI_MODEL", ("MA_MODEL",), "gemini-3.8-flash"))
    # "" = model default, or "low" | "medium" | "high"
    thinking_level: str = field(default_factory=lambda: _setting("GEMINI_THINKING_LEVEL", ("MA_THINKING_LEVEL",), ""))

    # --- Ollama ---
    ollama_host: str = field(default_factory=lambda: _setting("OLLAMA_HOST", ("OLLAMA_HOST",), "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _setting("OLLAMA_MODEL", ("MA_OLLAMA_MODEL",), "qwen3:14b"))
    ollama_embed_model: str = field(
        default_factory=lambda: _setting("OLLAMA_EMBED_MODEL", ("MA_OLLAMA_EMBED_MODEL",), "nomic-embed-text"))
    # Context window per request (tokens). 16k fits qwen3:14b + q8_0 KV cache on a 16 GB T4 with room to spare.
    num_ctx: int = field(default_factory=lambda: _setting_int("OLLAMA_CONTEXT", ("MA_OLLAMA_CONTEXT",), 16384))
    ollama_keep_alive: str = field(default_factory=lambda: _env_str("MA_OLLAMA_KEEP_ALIVE", "30m"))

    # --- Shared ---
    # Gemini counts thinking tokens against this limit, so keep it generous; Ollama uses its own cap.
    max_tokens: int = field(default_factory=lambda: _env_int("MA_MAX_TOKENS", 16000))
    max_retries: int = field(default_factory=lambda: _env_int("MA_MAX_RETRIES", 5))
    timeout_s: float = field(default_factory=lambda: _env_float("MA_TIMEOUT_S", 900.0))
    # Parallel model calls. Gemini: 4. Ollama: match OLLAMA_NUM_PARALLEL on the server (2 on a T4).
    max_parallel: int = field(default_factory=lambda: _env_int(
        "MA_MAX_PARALLEL", _setting_int("OLLAMA_PARALLEL", (), 2)
        if _setting("LLM_PROVIDER", ("MA_PROVIDER",), "gemini").lower() == "ollama" else 4))
    use_disk_cache: bool = field(
        default_factory=lambda: _env_str("MA_USE_CACHE", "1") not in ("0", "false", "False")
    )
    cache_dir: Path = field(default_factory=lambda: PROJECT_ROOT / ".cache" / "llm")

    @property
    def is_local(self) -> bool:
        return self.provider == "ollama"

    @property
    def active_model(self) -> str:
        return self.ollama_model if self.is_local else self.model

    def output_tokens(self, requested: int | None = None) -> int:
        """Output cap for one call. Local models get a cap that leaves room for the prompt."""
        n = requested or self.max_tokens
        return min(n, max(1024, self.num_ctx // 4)) if self.is_local else n

    def prompt_budget_words(self) -> int:
        """How many document words fit in one request alongside instructions, schema and the answer.

        Gemini Flash has a 1M-token window, so the whole agreement usually fits. A local model with a
        16k window gets ~7,000 words; longer documents are handled with retrieval instead.
        """
        if not self.is_local:
            return 150_000
        reserve = 3_500 + self.output_tokens()  # instructions + JSON schema + the answer
        return max(1_500, int((self.num_ctx - reserve) * 0.72))  # ~0.72 words per token for legal English

    def prompt_budget_chars(self) -> int:
        """Character budget for document excerpts in one prompt (~6.2 characters per word)."""
        return int(self.prompt_budget_words() * 6.2)


@dataclass
class ChunkingConfig:
    chunk_words: int = field(default_factory=lambda: _env_int("MA_CHUNK_WORDS", 500))
    overlap_words: int = field(default_factory=lambda: _env_int("MA_CHUNK_OVERLAP", 100))
    # Chunks shorter than this are merged into the previous chunk.
    min_chunk_words: int = 60


@dataclass
class RetrievalConfig:
    # "auto": Ollama embeddings when the provider is Ollama, else sentence-transformers, else TF-IDF.
    embedding_backend: str = field(default_factory=lambda: _env_str("MA_EMBEDDINGS", "auto"))
    embedding_model: str = field(
        default_factory=lambda: _env_str("MA_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    top_k: int = 6
    rrf_k: int = 60  # Reciprocal-rank-fusion constant
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    index_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "indexes")


@dataclass
class SummaryConfig:
    # Documents up to this many words are summarised in one call from full text.
    single_pass_max_words: int = 150_000  # Gemini Flash has a 1M-token context window
    # Group size (in chunks) for the map step of map-reduce summarisation.
    map_group_chunks: int = 12
    short_doc_words: int = 4_000
    long_doc_words: int = 25_000


@dataclass
class ScoringWeights:
    """Maximum points per category. Must sum to 100."""

    financial_risk: float = 30.0
    legal_liability: float = 25.0
    regulatory_compliance: float = 20.0
    synergy_realization: float = 25.0

    def as_dict(self) -> dict[str, float]:
        return {
            "Financial Risk": self.financial_risk,
            "Legal Liability": self.legal_liability,
            "Regulatory Compliance": self.regulatory_compliance,
            "Synergy Realization": self.synergy_realization,
        }


@dataclass
class SimulationConfig:
    horizon_months: int = 36
    monte_carlo_runs: int = 2000
    random_seed: int = 42
    annual_discount_rate: float = 0.10


@dataclass
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    summary: SummaryConfig = field(default_factory=SummaryConfig)
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)


def get_settings() -> Settings:
    """Build a fresh Settings object (re-reads utils.py and environment variables)."""
    s = Settings()
    if s.llm.is_local:
        # Small context window: smaller retrieval sets keep each prompt well inside it.
        s.retrieval.top_k = 5
        s.summary.single_pass_max_words = s.llm.prompt_budget_words()
    return s
