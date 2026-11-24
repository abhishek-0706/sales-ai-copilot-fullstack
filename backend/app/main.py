"""FastAPI application. Run: uvicorn app.main:create_app --factory --port 8000 (from backend/)."""
from __future__ import annotations

import time

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry import propagate, trace
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from .agent import Agent, NotFound
from .audit import AuditLog
from .config import Settings
from .data_store import DataStore
from .embeddings import make_embedder
from .guardrails import scan
from .llm import make_provider
from .models import (AccessScope, AccountSummary, AskRequest, AskResponse, AuditEntry, BriefResponse,
                     MeetingPrepRequest, MeetingPrepResponse, NextBestActionsResponse, Role, SourceRecord, User)
from .observability import (ACCESS_DENIED, HTTP_LATENCY, HTTP_REQUESTS, REGISTRY, log_event, setup_logging,
                            setup_tracing, trace_id_var, tracer)
from .security import can_read, scope_for
from .tools import Tools
from .vectorstore import HybridStore


class Health(BaseModel):
    status: str
    index_chunks: int
    embedder: str
    provider: str
    model: str
    data_as_of: str


class Me(BaseModel):
    user: User
    scope: AccessScope


class AccountListItem(AccountSummary):
    opportunity_stage: str | None = None
    next_meeting: str | None = None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    setup_logging()
    setup_tracing(settings.otel_console)
    store = DataStore(settings.data_dir)
    index = HybridStore(make_embedder(settings.embedder, settings.embedding_dim, settings.model_cache_dir),
                        settings.vector_db_path)
    index.build(store.chunks())

    llm = make_provider(settings)
    agent = Agent(store, index, Tools(store), llm, AuditLog(settings.audit_log_path), settings)
    log_event("app.started", provider=llm.name, model=llm.model, embedder=index.embedder.name,
              chunks=index.count())

    app = FastAPI(title="Sales AI Copilot API", version="0.1.0",
                  description="Account briefs, meeting prep, next-best-actions and Q&A over synthetic CRM data, "
                              "with role/territory access control, PII redaction, guardrails and citations.")
    app.state.agent = agent
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"],
                       allow_headers=["*"], expose_headers=["X-Trace-Id"])

    @app.middleware("http")
    async def observe(request: Request, call_next):
        start = time.perf_counter()
        ctx = propagate.extract(dict(request.headers))
        with tracer.start_as_current_span(f"{request.method} {request.url.path}", context=ctx) as span:
            sc = span.get_span_context()
            token = trace_id_var.set(format(sc.trace_id, "032x"))
            status = 500
            try:
                response = await call_next(request)
                status = response.status_code
            finally:
                route = request.scope.get("route")
                path = getattr(route, "path", "unmatched")
                elapsed = time.perf_counter() - start
                HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
                HTTP_LATENCY.labels(path).observe(elapsed)
                log_event("http.request", method=request.method, route=path, status=status,
                          latency_ms=round(elapsed * 1000, 2))
                trace_id = trace_id_var.get()
                trace_id_var.reset(token)
        response.headers["X-Trace-Id"] = trace_id
        return response

    def current_user(x_user_id: str = Header(..., description="Demo identity; production would use OIDC/JWT")) -> User:
        user = store.users.get(x_user_id)
        if user is None:
            raise HTTPException(401, "unknown user")
        span = trace.get_current_span()
        span.set_attribute("enduser.id", user.id)
        span.set_attribute("enduser.role", user.role.value)
        return user

    def not_found(route: str, what: str) -> HTTPException:
        ACCESS_DENIED.labels(route).inc()
        return HTTPException(404, f"{what} not found or not accessible")

    @app.get("/api/health", response_model=Health, tags=["ops"])
    def health() -> Health:
        return Health(status="ok", index_chunks=index.count(), embedder=index.embedder.name, provider=llm.name,
                      model=llm.model, data_as_of=store.meta["as_of"])

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    @app.get("/api/users", response_model=list[User], tags=["identity"])
    def users() -> list[User]:
        return list(store.users.values())

    @app.get("/api/me", response_model=Me, tags=["identity"])
    def me(user: User = Depends(current_user)) -> Me:
        return Me(user=user, scope=scope_for(user))

    @app.get("/api/accounts", response_model=list[AccountListItem], tags=["accounts"])
    def accounts(user: User = Depends(current_user)) -> list[AccountListItem]:
        out = []
        for acc in store.accounts.values():
            if not can_read(user, territory=acc.territory, record_type="account", sensitivity=acc.sensitivity):
                continue
            opp = next((o for o in store.opportunities.values() if o.account_id == acc.id and can_read(
                user, territory=o.territory, record_type="opportunity", sensitivity=o.sensitivity)), None)
            mtg = next((m for m in sorted(store.meetings.values(), key=lambda m: m.start) if m.account_id == acc.id
                        and can_read(user, territory=m.territory, record_type="meeting", sensitivity=m.sensitivity)),
                       None)
            out.append(AccountListItem(**store.summary(acc).model_dump(),
                                       opportunity_stage=opp.stage if opp else None,
                                       next_meeting=mtg.start if mtg else None))
        return out

    @app.get("/api/accounts/{account_id}", response_model=AccountSummary, tags=["accounts"])
    def account(account_id: str, user: User = Depends(current_user)) -> AccountSummary:
        try:
            return store.summary(agent.account_for(user, account_id))
        except NotFound:
            raise not_found("/api/accounts/{account_id}", "account")

    @app.post("/api/accounts/{account_id}/brief", response_model=BriefResponse, tags=["copilot"])
    def brief(account_id: str, user: User = Depends(current_user)) -> BriefResponse:
        try:
            return agent.account_brief(user, account_id)
        except NotFound:
            raise not_found("/api/accounts/{account_id}/brief", "account")

    @app.post("/api/accounts/{account_id}/meeting-prep", response_model=MeetingPrepResponse, tags=["copilot"])
    def meeting_prep(account_id: str, body: MeetingPrepRequest | None = None,
                     user: User = Depends(current_user)) -> MeetingPrepResponse:
        try:
            return agent.meeting_prep(user, account_id, body.meeting_id if body else None)
        except NotFound:
            raise not_found("/api/accounts/{account_id}/meeting-prep", "account or meeting")

    @app.post("/api/accounts/{account_id}/next-best-actions", response_model=NextBestActionsResponse,
              tags=["copilot"])
    def next_best_actions(account_id: str, user: User = Depends(current_user)) -> NextBestActionsResponse:
        try:
            return agent.next_best_actions(user, account_id)
        except NotFound:
            raise not_found("/api/accounts/{account_id}/next-best-actions", "account")

    @app.post("/api/ask", response_model=AskResponse, tags=["copilot"])
    def ask(body: AskRequest, user: User = Depends(current_user)) -> AskResponse:
        try:
            return agent.ask(user, body.question, body.account_id)
        except NotFound:
            raise not_found("/api/ask", "account")

    @app.get("/api/sources/{source_id}", response_model=SourceRecord, tags=["copilot"])
    def source(source_id: str, user: User = Depends(current_user)) -> SourceRecord:
        ev = store.by_source_id(source_id)
        if ev is None or not can_read(user, territory=ev.territory, record_type=ev.source_type,
                                      sensitivity=ev.sensitivity):
            raise not_found("/api/sources/{source_id}", "source")
        return SourceRecord(source_id=ev.source_id, record_id=ev.record_id, source_type=ev.source_type,
                            title=ev.title, text=ev.text, account_id=ev.account_id, territory=ev.territory,
                            sensitivity=ev.sensitivity, flagged_injection=bool(scan(ev.text)))

    @app.get("/api/audit", response_model=list[AuditEntry], tags=["ops"])
    def audit(limit: int = 50, user: User = Depends(current_user)) -> list[AuditEntry]:
        if user.role is not Role.MANAGER:
            ACCESS_DENIED.labels("/api/audit").inc()
            raise HTTPException(403, "audit log is restricted to sales managers")
        return agent.audit.recent(min(max(limit, 1), 200))

    return app
