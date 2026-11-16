"""GeminiClient tested against a fake google-genai SDK (no network)."""

import json
from types import SimpleNamespace

import pytest
from google.genai import errors

from ma_engine.config import LLMConfig
from ma_engine.llm.base import LLMError
from ma_engine.llm.factory import get_llm_client
from ma_engine.llm.gemini_client import GeminiClient
from ma_engine.llm.mock_client import MockLLMClient
from ma_engine.llm.schema_utils import gemini_schema, schema_name
from ma_engine.models import AdvisorReport, DealTerms, ExecutiveSummary, QAAnswer, RiskReview


def resp(data, finish="STOP", inp=100, out=20, thoughts=5, cached=0):
    text = data if isinstance(data, str) else json.dumps(data)
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish))],
        usage_metadata=SimpleNamespace(prompt_token_count=inp, candidates_token_count=out,
                                       thoughts_token_count=thoughts, cached_content_token_count=cached),
    )


class FakeModels:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def generate_content(self, *, model, contents, config):
        self.requests.append({"model": model, "contents": contents, "config": config})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _client(tmp_path, replies, **cfg):
    models = FakeModels(replies)
    c = GeminiClient(LLMConfig(api_key="test", cache_dir=tmp_path, **cfg), sdk_client=SimpleNamespace(models=models))
    return c, models


def _api_error(code, message):
    return errors.ClientError(code, {"error": {"code": code, "message": message, "status": "X"}})


def _schema_keywords(node, in_props=False):
    """All JSON-Schema keywords used (property names excluded)."""
    found = set()
    if isinstance(node, dict):
        for k, v in node.items():
            if not in_props:
                found.add(k)
            found |= _schema_keywords(v, in_props=(k == "properties" and not in_props))
    elif isinstance(node, list):
        for v in node:
            found |= _schema_keywords(v)
    return found


def test_schema_is_self_contained_and_gemini_safe():
    allowed = {"type", "description", "enum", "format", "items", "minItems", "maxItems", "minimum", "maximum",
               "properties", "required", "anyOf", "additionalProperties"}
    for model in (ExecutiveSummary, DealTerms, RiskReview, AdvisorReport, QAAnswer):
        s = gemini_schema(model)
        assert _schema_keywords(s) <= allowed, _schema_keywords(s) - allowed
        assert s["type"] == "object" and "properties" in s
    assert "title" in gemini_schema(RiskReview)["properties"]["flags"]["items"]["properties"]  # a real field
    props = gemini_schema(QAAnswer)["properties"]
    assert props["confidence"]["enum"] == ["low", "medium", "high"]
    assert schema_name(DealTerms) == "deal_terms"


def test_request_shape_usage_and_disk_cache(tmp_path):
    c, models = _client(tmp_path, [resp({"answer": "Yes [d-c0001]", "citations": ["d-c0001"]})], model="gemini-3.8-flash",
                        use_disk_cache=True)
    out = c.structured(system="SYS", prompt="Q?", context="<excerpt/>", output_model=QAAnswer)
    assert out.answer.startswith("Yes")
    req = models.requests[0]
    assert req["model"] == "gemini-3.8-flash"
    assert req["config"]["system_instruction"] == "SYS"
    assert req["config"]["response_mime_type"] == "application/json"
    assert req["config"]["response_json_schema"]["type"] == "object"
    assert "thinking_config" not in req["config"]
    assert req["contents"][0]["parts"][0]["text"].startswith("<excerpt/>")  # context first, for caching
    assert c.usage.calls == 1 and c.usage.input_tokens == 100 and c.usage.output_tokens == 25

    c2, models2 = _client(tmp_path, [], use_disk_cache=True)
    again = c2.structured(system="SYS", prompt="Q?", context="<excerpt/>", output_model=QAAnswer)
    assert again == out and models2.requests == [] and c2.usage.cached_responses == 1


def test_thinking_level_sent_when_configured(tmp_path):
    c, models = _client(tmp_path, [resp({"answer": "a"})], thinking_level="low", use_disk_cache=False)
    c.structured(system="S", prompt="P", output_model=QAAnswer)
    assert models.requests[0]["config"]["thinking_config"] == {"thinking_level": "low"}


def test_code_fenced_json_is_accepted(tmp_path):
    c, _ = _client(tmp_path, [resp('```json\n{"answer": "fenced"}\n```')], use_disk_cache=False)
    assert c.structured(system="S", prompt="P", output_model=QAAnswer).answer == "fenced"


def test_validation_repair_loop(tmp_path):
    c, models = _client(tmp_path, [resp({"answer": "x", "confidence": "very"}), resp({"answer": "fixed"})],
                        use_disk_cache=False)
    assert c.structured(system="S", prompt="P", output_model=QAAnswer).answer == "fixed"
    second = models.requests[1]["contents"]
    assert [m["role"] for m in second] == ["user", "model", "user"]
    assert "failed validation" in second[2]["parts"][0]["text"]


def test_invalid_json_is_repaired(tmp_path):
    c, _ = _client(tmp_path, [resp("{not json"), resp({"answer": "ok"})], use_disk_cache=False)
    assert c.structured(system="S", prompt="P", output_model=QAAnswer).answer == "ok"


def test_truncation_retries_with_more_tokens(tmp_path):
    c, models = _client(tmp_path, [resp('{"answer": "par', finish="MAX_TOKENS"), resp({"answer": "full"})],
                        use_disk_cache=False, max_tokens=1000)
    assert c.structured(system="S", prompt="P", output_model=QAAnswer).answer == "full"
    assert [r["config"]["max_output_tokens"] for r in models.requests] == [1000, 2000]


def test_safety_block_raises(tmp_path):
    c, _ = _client(tmp_path, [resp("", finish="SAFETY")], use_disk_cache=False)
    with pytest.raises(LLMError, match="declined"):
        c.structured(system="S", prompt="P", output_model=QAAnswer)


def test_gives_up_after_three_bad_answers(tmp_path):
    bad = resp({"confidence": "nope"})
    c, _ = _client(tmp_path, [bad, bad, bad], use_disk_cache=False)
    with pytest.raises(LLMError, match="valid QAAnswer"):
        c.structured(system="S", prompt="P", output_model=QAAnswer)


@pytest.mark.parametrize("code,message,expected", [
    (400, "API key not valid. Please pass a valid API key.", "utils.py"),
    (404, "models/gemini-x is not found", "not found"),
    (429, "Resource has been exhausted (e.g. check quota).", "rate limit"),
])
def test_error_mapping(tmp_path, code, message, expected):
    c, _ = _client(tmp_path, [_api_error(code, message)], use_disk_cache=False)
    with pytest.raises(LLMError, match=expected):
        c.structured(system="S", prompt="P", output_model=QAAnswer)


def test_network_failure_is_wrapped(tmp_path):
    c, _ = _client(tmp_path, [ConnectionError("no route")], use_disk_cache=False)
    with pytest.raises(LLMError, match="reach the Gemini API"):
        c.structured(system="S", prompt="P", output_model=QAAnswer)


def test_real_sdk_client_builds_without_network():
    c = GeminiClient(LLMConfig(api_key="AIza-test-key"))
    assert c._sdk is not None


def test_factory_and_missing_key(monkeypatch):
    monkeypatch.delenv("MA_DEMO", raising=False)
    assert isinstance(get_llm_client(LLMConfig(api_key="")), MockLLMClient)
    assert isinstance(get_llm_client(LLMConfig(api_key="k"), demo=True), MockLLMClient)
    assert isinstance(get_llm_client(LLMConfig(api_key="k")), GeminiClient)
    monkeypatch.setenv("MA_DEMO", "1")
    assert isinstance(get_llm_client(LLMConfig(api_key="k")), MockLLMClient)
    with pytest.raises(LLMError):
        GeminiClient(LLMConfig(api_key=""))


def test_key_and_model_come_from_utils_py(tmp_path, monkeypatch):
    import ma_engine.config as config

    (tmp_path / "utils.py").write_text('GEMINI_API_KEY = "AIza-from-file"\nGEMINI_MODEL = "gemini-3.7-flash"\n')
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("MA_SETTINGS_FILE", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    cfg = config.LLMConfig()
    assert cfg.api_key == "AIza-from-file" and cfg.model == "gemini-3.7-flash"
    (tmp_path / "utils.py").write_text('GEMINI_API_KEY = ""\n')
    assert config.LLMConfig().api_key == "from-env" and config.LLMConfig().model == "gemini-3.8-flash"


def test_mock_custom_responses():
    m = MockLLMClient(responses={QAAnswer: QAAnswer(answer="canned")})
    assert m.structured(system="", prompt="", output_model=QAAnswer).answer == "canned"
    assert m.calls[0]["model"] == "QAAnswer"


def test_optional_fields_become_nullable_and_required():
    s = gemini_schema(DealTerms)
    assert s["properties"]["deal_value_usd"]["type"] == ["number", "null"]
    assert set(s["required"]) == set(s["properties"])


def test_schema_rejection_falls_back_to_prompt_schema(tmp_path):
    rejected = _api_error(400, "Request contains an invalid argument: response schema too complex")
    c, models = _client(tmp_path, [rejected, resp({"answer": "ok"}), resp({"answer": "again"})], use_disk_cache=False)
    assert c.structured(system="S", prompt="P", output_model=QAAnswer).answer == "ok"
    second = models.requests[1]["config"]
    assert "response_json_schema" not in second and "conforms to this JSON schema" in second["system_instruction"]
    # Remembered for later calls with the same output model.
    c.structured(system="S", prompt="P2", output_model=QAAnswer)
    assert "response_json_schema" not in models.requests[2]["config"]
