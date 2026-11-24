"""Trace IDs, structured JSON logs, Prometheus metrics and OpenTelemetry spans."""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from prometheus_client import CollectorRegistry, Counter, Histogram

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")

REGISTRY = CollectorRegistry()
HTTP_REQUESTS = Counter("salesai_http_requests_total", "HTTP requests", ["method", "route", "status"],
                        registry=REGISTRY)
HTTP_LATENCY = Histogram("salesai_http_request_duration_seconds", "HTTP request latency", ["route"],
                         registry=REGISTRY)
STAGE_LATENCY = Histogram("salesai_stage_duration_seconds", "Pipeline stage latency", ["stage"],
                          buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
                          registry=REGISTRY)
LLM_TOKENS = Counter("salesai_llm_tokens_total", "Tokens reported by the provider (offline mode: estimates)",
                     ["provider", "kind", "estimated"], registry=REGISTRY)
GUARDRAIL_EVENTS = Counter("salesai_guardrail_events_total", "Guardrail events", ["type"], registry=REGISTRY)
LLM_FALLBACKS = Counter("salesai_llm_fallbacks_total", "Fallbacks to the offline generator", ["provider"],
                        registry=REGISTRY)
ACCESS_DENIED = Counter("salesai_access_denied_total", "Requests refused by access policy", ["route"],
                        registry=REGISTRY)

_tracer_ready = False
tracer = trace.get_tracer("sales-ai-copilot")


def setup_tracing(console: bool) -> None:
    global _tracer_ready
    if _tracer_ready:
        return
    provider = TracerProvider(resource=Resource.create({"service.name": "sales-ai-copilot-api"}))
    if console:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _tracer_ready = True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + "Z",
                   "level": record.levelname, "logger": record.name, "event": record.getMessage(),
                   "trace_id": trace_id_var.get()}
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, default=str)


def setup_logging(level: int = logging.INFO) -> None:
    logger = logging.getLogger("salesai")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False


log = logging.getLogger("salesai")


def log_event(event: str, **fields) -> None:
    log.info(event, extra={"fields": fields})


def current_trace_id() -> str:
    tid = trace_id_var.get()
    if tid:
        return tid
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else uuid.uuid4().hex


@contextmanager
def timed_stage(stages: dict[str, float], name: str) -> Iterator[None]:
    start = time.perf_counter()
    with tracer.start_as_current_span(f"stage.{name}"):
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            stages[name] = round(stages.get(name, 0.0) + elapsed * 1000, 2)
            STAGE_LATENCY.labels(name).observe(elapsed)
