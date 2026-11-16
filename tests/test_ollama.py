"""Ollama provider tested over real HTTP against a fake Ollama server."""

from __future__ import annotations

import pytest

from ma_engine.config import LLMConfig, get_settings
from ma_engine.llm.base import LLMError
from ma_engine.llm.connection import check_connection, check_ollama
from ma_engine.llm.factory import get_llm_client
from ma_engine.llm.ollama_client import OllamaClient, estimate_tokens, fit_context
from ma_engine.models import DealTerms, QAAnswer
from ma_engine.pipeline import MAEngine
from ma_engine.retrieval.embedder import OllamaEmbedder, embedder_from_state
from ma_engine.retrieval.index_store import IndexStore
from ma_engine.retrieval.indexer import build_index

from .fake_ollama import FakeOllama


@pytest.fixture
def fake():
    with FakeOllama() as server:
        yield server


def _cfg(fake, tmp_path, **kw) -> LLMConfig:
    return LLMConfig(provider="ollama", ollama_host=fake.url, cache_dir=tmp_path / "c", use_disk_cache=False,
                     max_retries=1, **kw)


@pytest.fixture
def ollama_settings(fake, tmp_path, monkeypatch):
    monkeypatch.delenv("MA_DEMO", raising=False)
    monkeypatch.setenv("MA_EMBEDDINGS", "auto")
    f = tmp_path / "ollama_utils.py"
    f.write_text(f'LLM_PROVIDER = "ollama"\nOLLAMA_HOST = "{fake.url}"\nOLLAMA_MODEL = "qwen3:14b"\n'
                 'OLLAMA_EMBED_MODEL = "nomic-embed-text"\nOLLAMA_CONTEXT = 16384\nOLLAMA_PARALLEL = 2\n')
    monkeypatch.setenv("MA_SETTINGS_FILE", str(f))
    s = get_settings()
    s.llm.cache_dir = tmp_path / "cache"
    s.simulation.monte_carlo_runs = 200
    return s


def test_request_shape(fake, tmp_path):
    c = OllamaClient(_cfg(fake, tmp_path))
    out = c.structured(system="SYS", prompt="Q?", context='<excerpt id="d-c0001" section="">x</excerpt>',
                       output_model=QAAnswer, task="qa")
    assert out.answer
    body = fake.chat_requests()[0]
    assert body["model"] == "qwen3:14b" and body["stream"] is False and body["think"] is False
    assert body["format"]["type"] == "object" and "answer" in body["format"]["properties"]
    assert body["options"]["num_ctx"] == 16384 and body["options"]["num_predict"] == 4096
    assert body["keep_alive"] == "30m"
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    user = body["messages"][1]["content"]
    assert user.startswith("<excerpt") and "JSON schema" in user and '"answer"' in user
    assert c.usage.calls == 1 and c.usage.input_tokens > 0


def test_think_unsupported_falls_back(fake, tmp_path):
    fake.reject_think = True
    c = OllamaClient(_cfg(fake, tmp_path))
    assert c.structured(system="S", prompt="P", output_model=QAAnswer).answer
    assert "think" in fake.chat_requests()[0] and "think" not in fake.chat_requests()[1]
    c.structured(system="S", prompt="P2", output_model=QAAnswer)
    assert "think" not in fake.chat_requests()[2]  # remembered


def test_truncated_answer_retries_with_bigger_budget(fake, tmp_path):
    fake.truncate_once = True
    c = OllamaClient(_cfg(fake, tmp_path))
    c.structured(system="S", prompt="P", output_model=QAAnswer, max_tokens=1500)
    predicts = [b["options"]["num_predict"] for b in fake.chat_requests()]
    assert predicts == [1500, 3000]


def test_invalid_json_is_repaired(fake, tmp_path):
    fake.invalid_once = True
    c = OllamaClient(_cfg(fake, tmp_path))
    assert c.structured(system="S", prompt="P", output_model=QAAnswer).answer
    second = fake.chat_requests()[1]["messages"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "user"]


def test_missing_model_and_server_down_messages(fake, tmp_path):
    c = OllamaClient(_cfg(fake, tmp_path, ollama_model="llama-nope:1b"))
    with pytest.raises(LLMError, match="ollama pull llama-nope:1b"):
        c.structured(system="S", prompt="P", output_model=QAAnswer)
    down = OllamaClient(LLMConfig(provider="ollama", ollama_host="http://127.0.0.1:9", max_retries=1,
                                  use_disk_cache=False, cache_dir=tmp_path))
    with pytest.raises(LLMError, match="not reachable"):
        down.structured(system="S", prompt="P", output_model=QAAnswer)


def test_out_of_memory_message(fake, tmp_path):
    fake.fail_status, fake.fail_body = 500, "llama runner process has terminated: cudaMalloc failed: out of memory"
    c = OllamaClient(_cfg(fake, tmp_path))
    with pytest.raises(LLMError, match="out of memory"):
        c.structured(system="S", prompt="P", output_model=QAAnswer)


def test_context_is_fitted_to_the_window(fake, tmp_path):
    blocks = "".join(f'<excerpt id="d-c{i:04d}" section="">' + ("word " * 900) + "</excerpt>\n\n" for i in range(40))
    kept, dropped = fit_context(blocks, 6000)
    assert dropped > 0 and estimate_tokens(kept) <= 6000 and kept.count("<excerpt") >= 1
    c = OllamaClient(_cfg(fake, tmp_path, num_ctx=8192))
    c.structured(system="S", prompt="P", context=blocks, output_model=DealTerms)
    body = fake.chat_requests()[0]
    total = sum(estimate_tokens(m["content"]) for m in body["messages"]) + body["options"]["num_predict"]
    assert total <= 8192


def test_connection_check(fake, tmp_path):
    cfg = _cfg(fake, tmp_path)
    st = check_ollama(cfg)
    assert st.ok and "qwen3:14b" in st.message and st.detail == ""
    fake.models = ["qwen3:14b"]
    st = check_ollama(cfg)
    assert st.ok and "nomic-embed-text" in st.detail
    fake.models = []
    st = check_ollama(cfg)
    assert st.state == "error" and "ollama pull qwen3:14b" in st.message
    st = check_ollama(LLMConfig(provider="ollama", ollama_host="http://127.0.0.1:9"))
    assert st.state == "error" and "not reachable" in st.message


def test_ollama_embeddings_and_index_round_trip(fake, tmp_path, raw_sample):
    emb = OllamaEmbedder(fake.url, "nomic-embed-text")
    m = emb.embed_documents(["termination fee", "escrow amount"])
    assert m.shape == (2, 64)
    doc_req = next(r for r in fake.requests if r["path"] == "/api/embed")["body"]
    assert doc_req["input"][0].startswith("search_document: ")
    s = get_settings()
    s.llm.provider, s.llm.ollama_host = "ollama", fake.url
    s.retrieval.embedding_backend = "auto"
    index = build_index([raw_sample], s)
    assert index.embedder.name == "ollama"
    loaded = IndexStore.load(index.save(tmp_path / "idx"))
    assert loaded.embedder.name == "ollama" and embedder_from_state(index.embedder.to_state()).host == fake.url


def test_factory_picks_ollama(fake, tmp_path, monkeypatch):
    monkeypatch.delenv("MA_DEMO", raising=False)
    assert isinstance(get_llm_client(_cfg(fake, tmp_path)), OllamaClient)


def test_budget_numbers_fit_a_t4():
    cfg = LLMConfig(provider="ollama", num_ctx=16384)
    assert cfg.output_tokens(16000) == 4096
    assert 5_000 < cfg.prompt_budget_words() < 7_500
    words_tokens = cfg.prompt_budget_words() / 0.72
    assert words_tokens + 3_500 + cfg.output_tokens() <= 16384


def test_full_pipeline_on_ollama(ollama_settings, fake, raw_sample):
    st = check_connection(ollama_settings.llm)
    assert st.ok
    engine = MAEngine(get_llm_client(ollama_settings.llm), ollama_settings)
    assert isinstance(engine.llm, OllamaClient)
    ws = engine.build_workspace([raw_sample])
    assert ws.index.embedder.name == "ollama"
    report = engine.analyze(ws)
    assert report.scorecard and report.summary and report.advice and not report.demo_mode
    assert not any("failed" in w for w in report.warnings), report.warnings
    for body in fake.chat_requests():
        total = sum(len(m["content"]) for m in body["messages"]) / 3.4 + body["options"]["num_predict"]
        assert total <= 16384 * 1.02, total
    ans, hits = engine.ask(ws, "What is the escrow amount?")
    assert ans.answer and hits


def test_long_document_uses_rag_summary_on_ollama(ollama_settings, fake):
    from pathlib import Path

    from ma_engine.ingestion.loader import load_file

    cisco = load_file(Path(__file__).resolve().parent.parent / "sample_data" / "cisco_acacia_merger_agreement.txt")
    engine = MAEngine(get_llm_client(ollama_settings.llm), ollama_settings)
    ws = engine.build_workspace([cisco])
    engine.analyze(ws)
    first_user = fake.chat_requests()[0]["messages"][1]["content"]
    assert "most relevant excerpts" in first_user  # retrieval-based summary, not the 50k-word text
    assert first_user.count("<excerpt") < 20
    assert len(fake.chat_requests()) <= 16  # summary + extraction + 10 risk categories + advice (+ retries)


def test_benchmark_script_runs(ollama_settings, fake, capsys, monkeypatch):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "benchmark_ollama.py"
    spec = importlib.util.spec_from_file_location("bench", path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    monkeypatch.setattr("sys.argv", ["benchmark_ollama.py", "--full"])
    assert bench.main() == 0
    out = capsys.readouterr().out
    assert "tokens/s" in out and "100% on GPU" in out and "Full review" in out


def test_shell_scripts_parse():
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "scripts"
    for name in ("setup_t4.sh", "start_app.sh"):
        assert subprocess.run(["bash", "-n", str(root / name)]).returncode == 0


def test_app_runs_on_ollama(ollama_settings, fake):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception and not at.error
    page = " ".join(m.value for m in at.markdown)
    assert "Running locally on Ollama (qwen3:14b)" in page and "8 to 15 minutes" in page
    labels = at.selectbox[0].options
    at.selectbox[0].select_index(next(i for i, o in enumerate(labels) if "Halcyon" in o)).run()
    next(b for b in at.button if b.label == "Analyze agreement").click().run()
    assert not at.exception, at.exception
    report = at.session_state.report
    assert report is not None and not report.demo_mode and report.advice is not None
    assert any(r["path"] == "/api/chat" for r in fake.requests)


def test_app_shows_ollama_problem_and_blocks_analysis(tmp_path, monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("MA_DEMO", raising=False)
    f = tmp_path / "u.py"
    f.write_text('LLM_PROVIDER = "ollama"\nOLLAMA_HOST = "http://127.0.0.1:9"\n')
    monkeypatch.setenv("MA_SETTINGS_FILE", str(f))
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    assert any("not reachable" in e.value for e in at.error)
    assert next(b for b in at.button if b.label == "Analyze agreement").disabled
