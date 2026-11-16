import numpy as np
import pytest

from ma_engine.retrieval.bm25 import BM25Index
from ma_engine.retrieval.embedder import SentenceTransformerEmbedder, TfidfEmbedder, create_embedder
from ma_engine.retrieval.index_store import IndexStore
from ma_engine.retrieval.indexer import build_index
from ma_engine.retrieval.retriever import HybridRetriever, format_context
from ma_engine.retrieval.tokenizer import tokenize


def test_tokenizer_stems_and_keeps_numbers():
    assert tokenize("The Liabilities and Section 8.3 terminations") == ["liability", "section", "8.3", "termination"]


def test_bm25_ranks_exact_terms():
    idx = BM25Index(["termination fee payable by parent", "escrow amount released", "fee schedule"])
    s = idx.scores("parent termination fee")
    assert s.argmax() == 0 and s[1] == 0


def test_tfidf_embedder_normalized():
    e = TfidfEmbedder()
    texts = ["termination fee", "escrow release", "material adverse effect"]
    e.fit(texts)
    m = e.embed_documents(texts)
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0)
    assert e.embed_query("termination").shape == (m.shape[1],)


def test_sentence_transformer_embedder_with_fake_model():
    class Fake:
        def encode(self, texts, **kw):
            return np.array([[len(t), 1.0] for t in texts])
    e = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5", model=Fake())
    assert e.embed_documents(["a", "bbb"]).shape == (2, 2)
    assert e.to_state() == {"backend": "sentence-transformers", "model_name": "BAAI/bge-small-en-v1.5"}


def test_create_embedder_fallback_and_errors():
    assert create_embedder("tfidf").name == "tfidf"
    with pytest.raises(ValueError):
        create_embedder("nope")


@pytest.fixture
def index(raw_sample, settings):
    return build_index([raw_sample], settings)


def test_hybrid_retrieval_finds_clauses(index):
    r = HybridRetriever(index)
    top = r.search("Parent Termination Fee sole and exclusive remedy", k=2)
    assert any("Parent Termination Fee" in h.chunk.text for h in top)
    top = r.search("change of control credit agreement acceleration", k=3)
    assert any("Company Credit Agreement" in h.chunk.text and "immediately due" in h.chunk.text for h in top)
    assert all(h.score > 0 for h in top)


def test_search_many_and_neighbors(index):
    r = HybridRetriever(index)
    res = r.search_many(["escrow", "indemnification cap"], k=4)
    assert len(res) == 4 and len({x.chunk.chunk_id for x in res}) == 4
    nb = r.neighbors(index.chunks[1].chunk_id)
    assert [c.index for c in nb] == [0, 1, 2]


def test_doc_filter_and_persistence(index, tmp_path, raw_sample):
    from ma_engine.ingestion.loader import load_text
    other = load_text("Unrelated letter of intent with a termination fee of $1 million.", "loi")
    from ma_engine.config import get_settings
    multi = build_index([raw_sample, other], get_settings())
    r = HybridRetriever(multi)
    loi_id = [d for d in multi.documents if d.startswith("loi")][0]
    assert all(h.chunk.doc_id == loi_id for h in r.search("termination fee", doc_id=loi_id))

    path = multi.save(tmp_path / "idx")
    loaded = IndexStore.load(path)
    assert len(loaded) == len(multi) and set(loaded.documents) == set(multi.documents)
    q1 = [h.chunk.chunk_id for h in HybridRetriever(loaded).search("escrow amount", k=3)]
    q2 = [h.chunk.chunk_id for h in r.search("escrow amount", k=3)]
    assert q1 == q2


def test_format_context_escapes_and_orders(index):
    ctx = format_context([index.chunks[2], index.chunks[0], index.chunks[2]])
    assert ctx.count("<excerpt") == 2
    assert ctx.index(index.chunks[0].chunk_id) < ctx.index(index.chunks[2].chunk_id)
    assert "&gt;" in ctx or ">" not in index.chunks[2].section
