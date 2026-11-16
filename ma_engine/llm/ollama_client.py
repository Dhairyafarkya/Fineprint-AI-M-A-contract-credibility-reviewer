"""Local model client for Ollama (tuned for a 16 GB NVIDIA T4).

How it works:
* Calls Ollama's /api/chat with ``format`` set to the output's JSON schema, so
  llama.cpp's grammar-constrained decoding only produces JSON in that shape.
* The schema (with field descriptions) is also written into the prompt: small
  models follow field meanings much better when they can read them.
* Thinking is switched off (``think: false``) for speed; the extraction and review
  prompts are explicit enough without it.
* Prompts are kept inside the context window (``num_ctx``). If the excerpts would
  overflow it, whole excerpts are dropped from the end rather than letting Ollama
  silently cut the start of the prompt.
* Results are validated with Pydantic; invalid output is sent back once or twice
  for correction. Responses are cached on disk keyed by the full request.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from pydantic import ValidationError

from ..config import LLMConfig
from .base import LLMClient, LLMError, T
from .response_cache import ResponseCache
from .schema_utils import gemini_schema

log = logging.getLogger(__name__)

CHARS_PER_TOKEN = 3.4  # conservative for legal English with numbers
_EXCERPT_SPLIT = re.compile(r"(?=<excerpt id=)")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.S)


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def fit_context(context: str, max_tokens: int) -> tuple[str, int]:
    """Keep whole <excerpt> blocks from the start until the token budget is used. Returns (text, dropped)."""
    if estimate_tokens(context) <= max_tokens:
        return context, 0
    blocks = [b for b in _EXCERPT_SPLIT.split(context) if b.strip()]
    if len(blocks) <= 1:  # not excerpt-structured: cut by characters
        return context[: int(max_tokens * CHARS_PER_TOKEN)], 1
    kept, used = [], 0
    for b in blocks:
        t = estimate_tokens(b)
        if used + t > max_tokens:
            break
        kept.append(b)
        used += t
    return "".join(kept), len(blocks) - len(kept)


def _friendly(status: int | None, body: str, model: str, host: str) -> str:
    low = body.lower()
    if status is None:
        return (f"Ollama is not reachable at {host}. Start it with `ollama serve` (or run scripts/setup_t4.sh) "
                "and check OLLAMA_HOST in utils.py.")
    if status == 404 or "not found" in low:
        return f"The Ollama model '{model}' is not installed. Run: ollama pull {model}"
    if "out of memory" in low or "cuda" in low and "memory" in low:
        return (f"The GPU ran out of memory with '{model}'. Use a smaller model (e.g. qwen3:8b) or lower "
                "OLLAMA_CONTEXT in utils.py, and make sure nothing else is using the GPU.")
    return f"Ollama error ({status}): {body[:300]}"


class OllamaClient(LLMClient):
    is_mock = False

    def __init__(self, cfg: LLMConfig | None = None, http: Any | None = None):
        super().__init__()
        self.cfg = cfg or LLMConfig(provider="ollama")
        self.host = self.cfg.ollama_host.rstrip("/")
        if http is None:
            import httpx

            http = httpx.Client(timeout=httpx.Timeout(self.cfg.timeout_s, connect=10.0))
        self._http = http
        self._think_supported = True  # switched off automatically for models without a thinking mode
        self.cache = ResponseCache(self.cfg.cache_dir, enabled=self.cfg.use_disk_cache)

    # ----------------------------------------------------------------- helpers
    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST with retries on connection errors and 5xx (model loading, transient OOM)."""
        import httpx

        last: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                r = self._http.post(f"{self.host}{path}", json=body)
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                last = exc
                if attempt >= min(2, self.cfg.max_retries):
                    raise LLMError(_friendly(None, str(exc), self.cfg.ollama_model, self.host)) from exc
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code < 400:
                return r.json()
            text = r.text
            if r.status_code == 400 and "think" in text.lower() and body.get("think") is not None:
                raise _ThinkUnsupported(text)
            if r.status_code >= 500 and "memory" not in text.lower() and attempt < self.cfg.max_retries:
                time.sleep(2 * (attempt + 1))
                continue
            raise LLMError(_friendly(r.status_code, text, self.cfg.ollama_model, self.host))
        raise LLMError(_friendly(None, str(last), self.cfg.ollama_model, self.host))

    def _body(self, messages: list[dict[str, str]], schema: dict[str, Any], num_predict: int) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.cfg.ollama_model,
            "messages": messages,
            "format": schema,
            "stream": False,
            "keep_alive": self.cfg.ollama_keep_alive,
            "options": {"num_ctx": self.cfg.num_ctx, "num_predict": num_predict, "temperature": 0.1, "top_p": 0.9},
        }
        if self._think_supported:
            body["think"] = False
        return body

    def _chat(self, messages: list[dict[str, str]], schema: dict[str, Any], num_predict: int) -> dict[str, Any]:
        try:
            return self._post("/api/chat", self._body(messages, schema, num_predict))
        except _ThinkUnsupported:
            log.info("Model %s has no thinking mode; sending requests without 'think'.", self.cfg.ollama_model)
            self._think_supported = False
            return self._post("/api/chat", self._body(messages, schema, num_predict))

    # -------------------------------------------------------------- public API
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
        num_predict = self.cfg.output_tokens(max_tokens)
        schema = gemini_schema(output_model)
        schema_text = json.dumps(schema, separators=(",", ":"))
        instructions = (f"{prompt}\n\nRespond with a single JSON object that matches this JSON schema "
                        f"(field descriptions explain what each field means):\n{schema_text}")

        if context:
            fixed = estimate_tokens(system) + estimate_tokens(instructions) + 64
            room = self.cfg.num_ctx - num_predict - fixed
            context, dropped = fit_context(context, max(256, room))
            if dropped:
                log.warning("[%s] context window %d: dropped %d excerpt(s) to fit.", task, self.cfg.num_ctx, dropped)
        user = f"{context}\n\n{instructions}" if context else instructions
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        cache_key = self.cache.key({"provider": "ollama", "model": self.cfg.ollama_model, "num_ctx": self.cfg.num_ctx,
                                    "messages": messages, "schema": schema, "num_predict": num_predict})
        hit = self.cache.get(cache_key)
        if hit is not None:
            try:
                result = output_model.model_validate(hit["output"])
                self._record(cached=True)
                return result
            except (ValidationError, KeyError):
                pass

        last_error = ""
        for attempt in range(3):
            resp = self._chat(messages, schema, num_predict)
            self._record(inp=int(resp.get("prompt_eval_count") or 0), out=int(resp.get("eval_count") or 0))
            text = _FENCE.sub("", ((resp.get("message") or {}).get("content") or "").strip())

            if resp.get("done_reason") == "length" or not text:
                ceiling = max(1024, self.cfg.num_ctx // 2)
                if num_predict >= ceiling:
                    raise LLMError(f"[{task}] The local model's answer did not fit in {num_predict} tokens. "
                                   "Increase OLLAMA_CONTEXT in utils.py.")
                num_predict = min(ceiling, num_predict * 2)
                log.warning("[%s] answer truncated; retrying with num_predict=%d", task, num_predict)
                continue

            try:
                data = json.loads(text)
                result = output_model.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = str(exc)
                log.warning("[%s] invalid output (attempt %d): %s", task, attempt + 1, last_error[:300])
                messages = messages[:2] + [
                    {"role": "assistant", "content": text[:6000]},
                    {"role": "user", "content": "That JSON failed validation. Return the full corrected JSON object "
                                                f"only. Errors:\n{last_error[:2000]}"},
                ]
                continue

            self.cache.put(cache_key, {"output": data})
            return result

        raise LLMError(f"[{task}] The local model did not return a valid {output_model.__name__}. "
                       f"Last error: {last_error[:400]}")


class _ThinkUnsupported(Exception):
    """Internal signal: the model rejected the 'think' option."""
