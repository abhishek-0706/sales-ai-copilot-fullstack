from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    data_dir: Path = BACKEND_DIR / "data"
    var_dir: Path = BACKEND_DIR / "var"
    llm_provider: str = "offline"
    llm_base_url: str = "https://integrate.api.nvidia.com/v1"
    llm_model: str = "meta/llama-3.3-70b-instruct"
    llm_api_key: str | None = None
    llm_json_mode: bool = False
    llm_timeout_s: float = 60.0
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    embedder: str = "hash"
    embedding_dim: int = 384
    model_cache_dir: Path = BACKEND_DIR / ".models"
    vector_db_path: str = ":memory:"
    retrieval_k: int = 6
    audit_log_path: Path = BACKEND_DIR / "var" / "audit.jsonl"
    otel_console: bool = False
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173"])

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = _env("LLM_API_KEY") or _env("NVIDIA_API_KEY")
        gemini_key = _env("GEMINI_API_KEY")
        provider = _env("LLM_PROVIDER")
        if provider is None:
            provider = "openai_compat" if api_key else "gemini" if gemini_key else "offline"
        if provider == "nim":
            provider = "openai_compat"
        var_dir = Path(_env("VAR_DIR", str(BACKEND_DIR / "var")))
        return cls(
            data_dir=Path(_env("DATA_DIR", str(BACKEND_DIR / "data"))),
            var_dir=var_dir,
            llm_provider=provider,
            llm_base_url=_env("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/"),
            llm_model=_env("LLM_MODEL", "meta/llama-3.3-70b-instruct"),
            llm_api_key=api_key,
            llm_json_mode=_env("LLM_JSON_MODE", "false").lower() == "true",
            llm_timeout_s=float(_env("LLM_TIMEOUT_S", "60")),
            gemini_api_key=gemini_key,
            gemini_model=_env("GEMINI_MODEL", "gemini-2.5-flash"),
            embedder=_env("EMBEDDER", "hash"),
            model_cache_dir=Path(_env("MODEL_CACHE_DIR", str(BACKEND_DIR / ".models"))),
            vector_db_path=_env("VECTOR_DB_PATH", ":memory:"),
            retrieval_k=int(_env("RETRIEVAL_K", "6")),
            audit_log_path=Path(_env("AUDIT_LOG_PATH", str(var_dir / "audit.jsonl"))),
            otel_console=_env("OTEL_CONSOLE", "false").lower() == "true",
            cors_origins=_env("CORS_ORIGINS", "http://localhost:5173").split(","),
        )
