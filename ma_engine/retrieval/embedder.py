"""Dense embedding backends.

Dense vectors are computed locally (no API cost), from either:

* ``sentence-transformers`` (local model, default ``BAAI/bge-small-en-v1.5``) -
  best quality, needs ``pip install sentence-transformers`` and a one-time
  model download; or
* ``ollama`` - an embedding model served by the local Ollama server (default
  ``nomic-embed-text``), used automatically when the app runs on Ollama; or
* ``tfidf`` - TF-IDF + LSA (truncated SVD) built with scikit-learn. No model
  download, works fully offline, and is a solid fallback for legal text.

``auto`` picks Ollama embeddings when the app runs on Ollama, then
sentence-transformers when it is installed and loads, otherwise TF-IDF. All vectors are L2-normalised so a dot product is cosine similarity.
"""

from __future__ import annotations

import logging
import pickle
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class Embedder(ABC):
    name: str = "base"

    def fit(self, texts: list[str]) -> None:  # noqa: B027 - optional hook
        """Learn corpus statistics (only needed by corpus-fitted backends)."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray: ...

    @abstractmethod
    def to_state(self) -> dict[str, Any]: ...


class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers"
    _BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str, model: Any | None = None):
        self.model_name = model_name
        if model is None:
            from sentence_transformers import SentenceTransformer  # optional dependency

            model = SentenceTransformer(model_name)
        self.model = model

    def _encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        return _l2_normalize(np.asarray(vecs, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        prefix = self._BGE_QUERY_PREFIX if "bge" in self.model_name.lower() else ""
        return self._encode([prefix + text])[0]

    def to_state(self) -> dict[str, Any]:
        return {"backend": self.name, "model_name": self.model_name}


class TfidfEmbedder(Embedder):
    """TF-IDF, reduced with LSA when the corpus is big enough to make it meaningful."""

    name = "tfidf"

    def __init__(self, max_components: int = 256, min_docs_for_lsa: int = 40):
        self.max_components = max_components
        self.min_docs_for_lsa = min_docs_for_lsa
        self.vectorizer = None
        self.svd = None

    def fit(self, texts: list[str]) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            max_df=0.95 if len(texts) > 20 else 1.0,
        )
        tfidf = self.vectorizer.fit_transform(texts)
        n_comp = min(self.max_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
        self.svd = None
        if len(texts) >= self.min_docs_for_lsa and n_comp >= 16:
            self.svd = TruncatedSVD(n_components=n_comp, random_state=0).fit(tfidf)

    def _transform(self, texts: list[str]) -> np.ndarray:
        if self.vectorizer is None:
            raise RuntimeError("TfidfEmbedder.fit() must be called before embedding.")
        m = self.vectorizer.transform(texts)
        dense = self.svd.transform(m) if self.svd is not None else m.toarray()
        return _l2_normalize(np.asarray(dense, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._transform(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._transform([text])[0]

    def to_state(self) -> dict[str, Any]:
        return {"backend": self.name, "pickle": pickle.dumps((self.vectorizer, self.svd, self.max_components))}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> TfidfEmbedder:
        obj = cls()
        obj.vectorizer, obj.svd, obj.max_components = pickle.loads(state["pickle"])  # noqa: S301 - own index files
        return obj


class OllamaEmbedder(Embedder):
    """Embeddings from a local Ollama server (e.g. nomic-embed-text on the same T4 as the chat model)."""

    name = "ollama"

    def __init__(self, host: str, model_name: str, http: Any | None = None, batch_size: int = 16):
        self.host = host.rstrip("/")
        self.model_name = model_name
        self.batch_size = batch_size
        if http is None:
            import httpx

            http = httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0))
        self._http = http
        nomic = "nomic" in model_name.lower()
        self._doc_prefix = "search_document: " if nomic else ""
        self._query_prefix = "search_query: " if nomic else ""

    def _embed(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            r = self._http.post(f"{self.host}/api/embed",
                                json={"model": self.model_name, "input": texts[i: i + self.batch_size], "truncate": True})
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
        return _l2_normalize(np.asarray(out, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed([self._doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([self._query_prefix + text])[0]

    def to_state(self) -> dict[str, Any]:
        return {"backend": self.name, "model_name": self.model_name, "host": self.host}


def create_embedder(backend: str = "auto", model_name: str = "BAAI/bge-small-en-v1.5",
                    ollama_host: str | None = None, ollama_model: str | None = None) -> Embedder:
    """Pick an embedder. With an Ollama host given, "auto" tries Ollama embeddings first."""
    backend = (backend or "auto").lower()
    if backend == "ollama" or (backend == "auto" and ollama_host and ollama_model):
        try:
            emb = OllamaEmbedder(ollama_host or "http://localhost:11434", ollama_model or "nomic-embed-text")
            emb.embed_query("probe")  # fails fast if the server or the model is missing
            return emb
        except Exception as exc:  # noqa: BLE001
            if backend == "ollama":
                raise RuntimeError(f"Ollama embeddings unavailable: {exc}") from exc
            log.info("Ollama embeddings unavailable (%s); falling back.", exc)
    if backend in ("auto", "sentence-transformers", "st"):
        try:
            return SentenceTransformerEmbedder(model_name)
        except Exception as exc:  # ImportError, download failure, etc.
            if backend != "auto":
                raise
            log.info("sentence-transformers unavailable (%s); using TF-IDF embeddings.", exc)
    if backend in ("auto", "tfidf"):
        return TfidfEmbedder()
    raise ValueError(f"Unknown embedding backend '{backend}'. Use auto, ollama, sentence-transformers or tfidf.")


def embedder_from_state(state: dict[str, Any]) -> Embedder:
    if state["backend"] == TfidfEmbedder.name:
        return TfidfEmbedder.from_state(state)
    if state["backend"] == SentenceTransformerEmbedder.name:
        return SentenceTransformerEmbedder(state["model_name"])
    if state["backend"] == OllamaEmbedder.name:
        return OllamaEmbedder(state["host"], state["model_name"])
    raise ValueError(f"Unknown embedder state backend: {state['backend']}")
