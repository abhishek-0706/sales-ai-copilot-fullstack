"""One interface, three providers.

- openai_compat: any OpenAI-compatible /chat/completions endpoint. Defaults to NVIDIA NIM on
  the NVIDIA API catalog (https://integrate.api.nvidia.com/v1); the same code works with a
  self-hosted NIM, vLLM or Ollama by changing LLM_BASE_URL / LLM_MODEL.
- gemini: Google Generative Language REST API (optional).
- offline: deterministic rule-based/extractive generator (default, no key needed).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal, Protocol

import httpx

from . import offline
from .config import Settings
from .models import Evidence, TokenUsage

Task = Literal["plan", "brief", "meeting_prep", "nba", "answer"]


@dataclass
class GenerationRequest:
    task: Task
    system: str
    prompt: str
    evidence: list[Evidence] = field(default_factory=list)
    question: str | None = None
    extras: dict = field(default_factory=dict)


@dataclass
class LLMResult:
    text: str
    usage: TokenUsage | None
    model: str


class LLMError(RuntimeError):
    pass


class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, req: GenerationRequest) -> LLMResult: ...


def extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start:end + 1])


class OpenAICompatProvider:
    name = "openai_compat"

    def __init__(self, base_url: str, model: str, api_key: str | None, timeout_s: float = 60.0,
                 json_mode: bool = False, transport: httpx.BaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._key = api_key
        self._json_mode = json_mode
        self._client = httpx.Client(timeout=timeout_s, transport=transport)

    def generate(self, req: GenerationRequest) -> LLMResult:
        body: dict = {"model": self.model, "temperature": 0.1, "max_tokens": 1500,
                      "messages": [{"role": "system", "content": req.system},
                                   {"role": "user", "content": req.prompt}]}
        if self._json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        try:
            resp = self._client.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"] or ""
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"{self.name} call failed: {type(exc).__name__}") from exc
        u = data.get("usage") or {}
        usage = TokenUsage(prompt_tokens=u.get("prompt_tokens", 0), completion_tokens=u.get("completion_tokens", 0),
                           total_tokens=u.get("total_tokens", 0)) if u else None
        return LLMResult(text=text, usage=usage, model=data.get("model", self.model))


class GeminiProvider:
    name = "gemini"
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, api_key: str, model: str, timeout_s: float = 60.0,
                 transport: httpx.BaseTransport | None = None):
        self.model = model
        self._key = api_key
        self._client = httpx.Client(timeout=timeout_s, transport=transport)

    def generate(self, req: GenerationRequest) -> LLMResult:
        body = {"systemInstruction": {"parts": [{"text": req.system}]},
                "contents": [{"role": "user", "parts": [{"text": req.prompt}]}],
                "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"}}
        try:
            resp = self._client.post(self.URL.format(model=self.model), json=body,
                                     headers={"x-goog-api-key": self._key})
            resp.raise_for_status()
            data = resp.json()
            text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"{self.name} call failed: {type(exc).__name__}") from exc
        u = data.get("usageMetadata") or {}
        usage = TokenUsage(prompt_tokens=u.get("promptTokenCount", 0),
                           completion_tokens=u.get("candidatesTokenCount", 0),
                           total_tokens=u.get("totalTokenCount", 0)) if u else None
        return LLMResult(text=text, usage=usage, model=self.model)


class OfflineProvider:
    name = "offline"
    model = "deterministic-extractive-v1"

    def __init__(self) -> None:
        self.history: list[GenerationRequest] = []

    def generate(self, req: GenerationRequest) -> LLMResult:
        self.history.append(req)
        x = req.extras
        if req.task == "plan":
            out = offline.plan(req.question or "", x.get("accounts", {}), x.get("account_id"))
        elif req.task == "brief":
            out = offline.brief(req.evidence)
        elif req.task == "meeting_prep":
            out = offline.meeting_prep(req.evidence, x.get("meeting_id"))
        elif req.task == "nba":
            out = offline.next_best_actions(req.evidence, x.get("as_of", "2026-11-23"))
        else:
            out = offline.answer(req.question or "", req.evidence)
        text = json.dumps(out)
        prompt_tokens = (len(req.system) + len(req.prompt)) // 4
        usage = TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=len(text) // 4,
                           total_tokens=prompt_tokens + len(text) // 4, estimated=True)
        return LLMResult(text=text, usage=usage, model=self.model)


def make_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "openai_compat":
        return OpenAICompatProvider(settings.llm_base_url, settings.llm_model, settings.llm_api_key,
                                    settings.llm_timeout_s, settings.llm_json_mode)
    if settings.llm_provider == "gemini":
        if not settings.gemini_api_key:
            raise ValueError("LLM_PROVIDER=gemini requires GEMINI_API_KEY")
        return GeminiProvider(settings.gemini_api_key, settings.gemini_model, settings.llm_timeout_s)
    if settings.llm_provider == "offline":
        return OfflineProvider()
    raise ValueError(f"unknown LLM_PROVIDER '{settings.llm_provider}'")
