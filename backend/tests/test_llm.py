import json

import httpx
import pytest

from app.config import Settings
from app.llm import (GeminiProvider, GenerationRequest, LLMError, OfflineProvider, OpenAICompatProvider,
                     extract_json, make_provider)

REQ = GenerationRequest(task="answer", system="sys", prompt="user prompt")


def test_openai_compat_request_shape_and_usage():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"model": "meta/llama-3.3-70b-instruct",
                                         "choices": [{"message": {"content": "```json\n{\"a\": 1}\n```"}}],
                                         "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}})

    p = OpenAICompatProvider("https://integrate.api.nvidia.com/v1/", "meta/llama-3.3-70b-instruct", "nvapi-test",
                             json_mode=True, transport=httpx.MockTransport(handler))
    res = p.generate(REQ)
    assert seen["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert seen["auth"] == "Bearer nvapi-test"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert extract_json(res.text) == {"a": 1}
    assert res.usage.total_tokens == 16 and not res.usage.estimated


def test_openai_compat_without_key_for_local_servers():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    p = OpenAICompatProvider("http://localhost:11434/v1", "llama3.1:8b", None, transport=httpx.MockTransport(handler))
    res = p.generate(REQ)
    assert seen["auth"] is None and seen["url"] == "http://localhost:11434/v1/chat/completions"
    assert res.usage is None


def test_openai_compat_http_error_becomes_llm_error():
    p = OpenAICompatProvider("http://x/v1", "m", "k", transport=httpx.MockTransport(lambda r: httpx.Response(429)))
    with pytest.raises(LLMError):
        p.generate(REQ)


def test_gemini_request_shape():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-goog-api-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "{\"ok\": true}"}]}}],
                                         "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2,
                                                           "totalTokenCount": 5}})

    res = GeminiProvider("g-key", "gemini-2.5-flash", transport=httpx.MockTransport(handler)).generate(REQ)
    assert seen["url"].endswith("/models/gemini-2.5-flash:generateContent") and seen["key"] == "g-key"
    assert seen["body"]["generationConfig"]["responseMimeType"] == "application/json"
    assert extract_json(res.text) == {"ok": True} and res.usage.total_tokens == 5


@pytest.mark.parametrize("env,provider,base,model", [
    ({}, "offline", "https://integrate.api.nvidia.com/v1", "meta/llama-3.3-70b-instruct"),
    ({"NVIDIA_API_KEY": "nvapi-x"}, "openai_compat", "https://integrate.api.nvidia.com/v1", None),
    ({"LLM_PROVIDER": "openai_compat", "LLM_BASE_URL": "http://localhost:8001/v1/", "LLM_MODEL": "my-vllm-model"},
     "openai_compat", "http://localhost:8001/v1", "my-vllm-model"),
    ({"LLM_PROVIDER": "nim", "LLM_API_KEY": "k"}, "openai_compat", "https://integrate.api.nvidia.com/v1", None),
    ({"GEMINI_API_KEY": "g"}, "gemini", None, None),
])
def test_settings_provider_selection(monkeypatch, env, provider, base, model):
    for k in ["LLM_PROVIDER", "LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY", "NVIDIA_API_KEY", "GEMINI_API_KEY"]:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    s = Settings.from_env()
    assert s.llm_provider == provider
    if base:
        assert s.llm_base_url == base
    if model:
        assert s.llm_model == model


def test_make_provider_defaults_to_offline():
    assert isinstance(make_provider(Settings()), OfflineProvider)
