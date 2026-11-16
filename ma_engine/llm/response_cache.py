"""On-disk cache of Gemini responses so re-running an analysis doesn't re-bill tokens."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any


class ResponseCache:
    def __init__(self, directory: Path, enabled: bool = True):
        self.directory = Path(directory)
        self.enabled = enabled
        self._lock = threading.Lock()
        if enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(payload: dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        path = self.directory / f"{key}.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def put(self, key: str, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self.directory / f"{key}.json"
        tmp = path.with_suffix(".tmp")
        with self._lock:
            tmp.write_text(json.dumps(value), encoding="utf-8")
            tmp.replace(path)

    def clear(self) -> int:
        n = 0
        for p in self.directory.glob("*.json"):
            p.unlink()
            n += 1
        return n
