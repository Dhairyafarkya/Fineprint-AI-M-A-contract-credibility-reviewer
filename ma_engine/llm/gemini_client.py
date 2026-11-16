"""Gemini client that always returns validated structured output.

How it works:
* Each output Pydantic model is converted to a JSON schema and passed as
  ``response_json_schema`` with ``response_mime_type="application/json"``, so
  Gemini's constrained decoding returns JSON in that shape.
* The JSON is validated with Pydantic. If validation fails, the error is sent
  back to Gemini in a follow-up turn and it gets two chances to correct itself.
* Large shared context (document excerpts) goes first in the prompt so Gemini's
  implicit prompt caching can reuse it across calls on the same document.
* Rate limits (429) and transient server errors are retried with exponential
  backoff by the SDK. Responses are cached on disk keyed by the full request.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from ..config import LLMConfig
from .base import LLMClient, LLMError, T
from .response_cache import ResponseCache
from .schema_utils import gemini_schema

log = logging.getLogger(__name__)

MAX_OUTPUT_TOKENS_CEILING = 65_536
RETRYABLE_STATUS = [408, 429, 500, 502, 503, 504]
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.S)


class _SchemaRejected(Exception):
    """Internal signal: Gemini rejected the response schema; retry with the schema in the prompt."""


def _friendly_error(code: int | None, message: str, model: str) -> str:
    low = message.lower()
    if code is None:
        return f"Could not reach the Gemini API ({message[:200]}). Check the internet connection."
    if "api key" in low or "api_key" in low or code in (401, 403):
        return "The Gemini API key was rejected. Check GEMINI_API_KEY in utils.py."
    if code == 404 or "not found" in low:
        return f"Gemini model '{model}' was not found. Check GEMINI_MODEL in utils.py."
    if code == 429 or "quota" in low or "resource_exhausted" in low:
        return "Gemini rate limit or quota reached. Wait a minute and try again, or check the key's quota."
    return f"Gemini API error ({code}): {message[:300]}"


class GeminiClient(LLMClient):
    is_mock = False

    def __init__(self, cfg: LLMConfig | None = None, sdk_client: Any | None = None):
        super().__init__()
        self.cfg = cfg or LLMConfig()
        if sdk_client is None:
            if not self.cfg.api_key:
                raise LLMError("GEMINI_API_KEY is not set. Add it to utils.py in the project folder.")
            from google import genai
            from google.genai import types

            sdk_client = genai.Client(
                api_key=self.cfg.api_key,
                http_options=types.HttpOptions(
                    timeout=int(self.cfg.timeout_s * 1000),
                    retry_options=types.HttpRetryOptions(
                        attempts=self.cfg.max_retries + 1, initial_delay=2.0, max_delay=60.0,
                        http_status_codes=RETRYABLE_STATUS,
                    ),
                ),
            )
        self._sdk = sdk_client
        # If Gemini rejects a response schema as too complex, fall back to plain JSON mode with
        # the schema written into the instructions (still validated by Pydantic afterwards).
        self._schema_in_prompt: set[str] = set()
        self.cache = ResponseCache(self.cfg.cache_dir, enabled=self.cfg.use_disk_cache)

    # ----------------------------------------------------------------- helpers
    def _config(self, system: str, schema: dict[str, Any], max_tokens: int, schema_in_prompt: bool) -> dict[str, Any]:
        config: dict[str, Any] = {
            "system_instruction": system,
            "response_mime_type": "application/json",
            "max_output_tokens": max_tokens,
        }
        if schema_in_prompt:
            config["system_instruction"] = (
                f"{system}\n\nReturn one JSON object that conforms to this JSON schema:\n{json.dumps(schema)}"
            )
        else:
            config["response_json_schema"] = schema
        if self.cfg.thinking_level:
            config["thinking_config"] = {"thinking_level": self.cfg.thinking_level}
        return config

    def _call(self, contents: list[dict[str, Any]], config: dict[str, Any]) -> Any:
        from google.genai import errors

        try:
            return self._sdk.models.generate_content(model=self.cfg.model, contents=contents, config=config)
        except errors.APIError as exc:
            msg = str(exc).lower()
            if getattr(exc, "code", None) == 400 and "response_json_schema" in config and (
                "schema" in msg or "invalid_argument" in msg) and "api key" not in msg:
                raise _SchemaRejected from exc
            log.error("Gemini API error %s: %s", getattr(exc, "code", None), exc)
            raise LLMError(_friendly_error(getattr(exc, "code", None), str(exc), self.cfg.model)) from exc
        except Exception as exc:  # network failures, timeouts
            log.error("Gemini request failed: %s", exc)
            raise LLMError(f"Could not reach the Gemini API: {exc}") from exc

    def _track(self, resp: Any) -> None:
        u = getattr(resp, "usage_metadata", None)
        if u is None:
            return
        self._record(
            inp=getattr(u, "prompt_token_count", 0) or 0,
            out=(getattr(u, "candidates_token_count", 0) or 0) + (getattr(u, "thoughts_token_count", 0) or 0),
            cache_read=getattr(u, "cached_content_token_count", 0) or 0,
        )

    @staticmethod
    def _finish_reason(resp: Any) -> str:
        try:
            fr = resp.candidates[0].finish_reason
        except (AttributeError, IndexError, TypeError):
            return ""
        return str(getattr(fr, "name", fr) or "")

    @staticmethod
    def _text(resp: Any) -> str:
        try:
            text = resp.text
        except (AttributeError, ValueError):
            text = None
        return _FENCE.sub("", (text or "").strip())

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
        max_tokens = max_tokens or self.cfg.max_tokens
        schema = gemini_schema(output_model)
        user_text = f"{context}\n\n{prompt}" if context else prompt
        contents: list[dict[str, Any]] = [{"role": "user", "parts": [{"text": user_text}]}]

        cache_key = self.cache.key({"model": self.cfg.model, "thinking": self.cfg.thinking_level, "system": system,
                                    "contents": contents, "schema": schema, "max_tokens": max_tokens})
        hit = self.cache.get(cache_key)
        if hit is not None:
            try:
                result = output_model.model_validate(hit["output"])
                self._record(cached=True)
                return result
            except (ValidationError, KeyError):
                pass  # schema changed since caching; make a live call

        last_error = ""
        name = output_model.__name__
        for attempt in range(3):
            try:
                resp = self._call(contents, self._config(system, schema, max_tokens, name in self._schema_in_prompt))
            except _SchemaRejected:
                log.warning("[%s] Gemini rejected the %s schema; using JSON mode with the schema in the prompt.", task, name)
                self._schema_in_prompt.add(name)
                resp = self._call(contents, self._config(system, schema, max_tokens, True))
            self._track(resp)
            finish = self._finish_reason(resp)
            text = self._text(resp)

            if finish in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"):
                raise LLMError(f"[{task}] Gemini declined to answer (finish reason {finish}).")
            if finish == "MAX_TOKENS" or not text:
                if max_tokens >= MAX_OUTPUT_TOKENS_CEILING:
                    raise LLMError(f"[{task}] Gemini response was cut off even at {max_tokens} tokens.")
                max_tokens = min(MAX_OUTPUT_TOKENS_CEILING, max_tokens * 2)
                log.warning("[%s] truncated/empty output (%s); retrying with max_output_tokens=%d", task, finish, max_tokens)
                continue

            try:
                data = json.loads(text)
                result = output_model.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = str(exc)
                log.warning("[%s] invalid output (attempt %d): %s", task, attempt + 1, last_error[:300])
                contents = contents + [
                    {"role": "model", "parts": [{"text": text}]},
                    {"role": "user", "parts": [{"text": "That JSON failed validation. Return the full corrected JSON "
                                                        f"object only. Errors:\n{last_error[:4000]}"}]},
                ]
                continue

            self.cache.put(cache_key, {"output": data})
            return result

        raise LLMError(f"[{task}] Gemini did not return a valid {output_model.__name__}. Last error: {last_error[:500]}")
