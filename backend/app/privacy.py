"""PII redaction applied to every string before it is sent to any model provider."""
from __future__ import annotations

import re
from typing import Any, Iterable

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d \-()]{8,}\d")
PLACEHOLDER_RE = re.compile(r"\[(EMAIL|PHONE|PERSON)_\d+\]")


def _is_phone(candidate: str) -> bool:
    """International numbers (leading +) need >= 8 digits, others >= 10, so dates and amounts are kept."""
    digits = sum(c.isdigit() for c in candidate)
    return digits >= (8 if candidate.startswith("+") else 10)


class Redactor:
    """Per-request redactor. Placeholders are stable within a request so they can be restored."""

    def __init__(self, known_names: Iterable[str]):
        names = sorted({n for n in known_names if n}, key=len, reverse=True)
        self._name_re = re.compile(r"\b(" + "|".join(map(re.escape, names)) + r")\b") if names else None
        self._forward: dict[str, str] = {}
        self._reverse: dict[str, str] = {}

    @property
    def count(self) -> int:
        return len(self._forward)

    def _token(self, kind: str, original: str) -> str:
        if original not in self._forward:
            n = sum(1 for k in self._reverse if k.startswith(f"[{kind}_")) + 1
            token = f"[{kind}_{n}]"
            self._forward[original] = token
            self._reverse[token] = original
        return self._forward[original]

    def redact(self, text: str) -> str:
        text = EMAIL_RE.sub(lambda m: self._token("EMAIL", m.group(0)), text)
        text = PHONE_RE.sub(
            lambda m: self._token("PHONE", m.group(0)) if _is_phone(m.group(0)) else m.group(0),
            text)
        if self._name_re is not None:
            text = self._name_re.sub(lambda m: self._token("PERSON", m.group(0)), text)
        return text

    def restore(self, text: str) -> str:
        return PLACEHOLDER_RE.sub(lambda m: self._reverse.get(m.group(0), m.group(0)), text)

    def restore_obj(self, obj: Any) -> Any:
        if isinstance(obj, str):
            return self.restore(obj)
        if isinstance(obj, list):
            return [self.restore_obj(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self.restore_obj(v) for k, v in obj.items()}
        return obj


def contains_pii(text: str, known_names: Iterable[str] = ()) -> bool:
    if EMAIL_RE.search(text):
        return True
    if any(_is_phone(m.group(0)) for m in PHONE_RE.finditer(text)):
        return True
    return any(n in text for n in known_names)
