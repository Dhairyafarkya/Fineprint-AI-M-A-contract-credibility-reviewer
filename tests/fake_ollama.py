"""A tiny stand-in for an Ollama server, speaking the same HTTP API (/api/tags, /api/chat, /api/embed).

/api/chat answers with the smallest JSON object that satisfies the request's `format`
schema, so every structured call in the engine can run end to end without a GPU.
Behaviour switches let tests simulate the failure modes a real server shows.
"""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def minimal_instance(schema: dict[str, Any]) -> Any:
    t = schema.get("type")
    if isinstance(t, list):
        return None if "null" in t else minimal_instance({**schema, "type": t[0]})
    if "enum" in schema:
        return schema["enum"][0]
    if "anyOf" in schema:
        return minimal_instance(schema["anyOf"][0])
    if t == "object":
        return {k: minimal_instance(v) for k, v in schema.get("properties", {}).items()}
    if t == "array":
        return []
    if t == "string":
        return "text from the local model"
    if t in ("number", "integer"):
        return 0
    if t == "boolean":
        return False
    return None


class FakeOllama:
    def __init__(self, models=("qwen3:14b", "nomic-embed-text:latest")):
        self.models = list(models)
        self.requests: list[dict[str, Any]] = []
        self.reject_think = False
        self.truncate_once = False
        self.invalid_once = False
        self.fail_status: int | None = None
        self.fail_body = ""
        self._lock = threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence
                pass

            def _send(self, status: int, payload: Any) -> None:
                body = json.dumps(payload).encode() if not isinstance(payload, str) else payload.encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                if self.path == "/api/tags":
                    self._send(200, {"models": [{"name": m} for m in fake.models]})
                elif self.path == "/api/ps":
                    self._send(200, {"models": [{"name": fake.models[0], "size": 10_000_000_000,
                                                 "size_vram": 10_000_000_000}]} if fake.models else {"models": []})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with fake._lock:
                    fake.requests.append({"path": self.path, "body": body})
                if fake.fail_status:
                    return self._send(fake.fail_status, {"error": fake.fail_body})
                if self.path == "/api/embed":
                    if body["model"].split(":")[0] not in {m.split(":")[0] for m in fake.models}:
                        return self._send(404, {"error": f"model '{body['model']}' not found"})
                    vecs = []
                    for text in body["input"]:
                        v = [0.0] * 64
                        for w in text.lower().split():
                            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
                        vecs.append(v)
                    return self._send(200, {"embeddings": vecs})
                if self.path == "/api/chat":
                    if body["model"] not in fake.models:
                        return self._send(404, {"error": f"model '{body['model']}' not found, try pulling it first"})
                    if fake.reject_think and "think" in body:
                        return self._send(400, {"error": f'"{body["model"]}" does not support thinking'})
                    if fake.truncate_once:
                        fake.truncate_once = False
                        return self._send(200, {"message": {"role": "assistant", "content": '{"answ'},
                                                "done_reason": "length", "prompt_eval_count": 10, "eval_count": 5})
                    if fake.invalid_once:
                        fake.invalid_once = False
                        content = "{not json"
                    else:
                        content = (json.dumps(minimal_instance(body["format"])) if isinstance(body.get("format"), dict)
                               else "A short plain-text answer.")
                    prompt_tokens = sum(len(m["content"]) for m in body["messages"]) // 4
                    return self._send(200, {"message": {"role": "assistant", "content": content},
                                            "done_reason": "stop", "prompt_eval_count": prompt_tokens,
                                            "eval_count": len(content) // 4,
                                            "prompt_eval_duration": 5_000_000, "eval_duration": 50_000_000})
                self._send(404, {"error": "unknown endpoint"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> FakeOllama:
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()

    def chat_requests(self) -> list[dict[str, Any]]:
        return [r["body"] for r in self.requests if r["path"] == "/api/chat"]
