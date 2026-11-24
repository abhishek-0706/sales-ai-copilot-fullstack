"""Append-only JSONL audit log: who asked what, which records reached the model, what guardrails fired."""
from __future__ import annotations

import threading
from collections import deque
from pathlib import Path

from .models import AuditEntry


class AuditLog:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._recent: deque[AuditEntry] = deque(maxlen=500)
        self._lock = threading.Lock()

    def record(self, entry: AuditEntry) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(entry.model_dump_json() + "\n")
            self._recent.append(entry)

    def recent(self, limit: int = 50) -> list[AuditEntry]:
        return list(self._recent)[-limit:][::-1]
