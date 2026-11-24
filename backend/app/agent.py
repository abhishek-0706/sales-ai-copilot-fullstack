"""Agentic workflow: plan -> tools -> hybrid retrieval -> guardrails -> PII redaction ->
LLM (structured JSON) -> Pydantic + citation validation -> PII restore -> groundedness -> audit."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .audit import AuditLog
from .config import Settings
from .data_store import DataStore
from .guardrails import SYSTEM_PROMPT, neutralise, scan, spotlight
from .llm import GenerationRequest, LLMError, LLMProvider, LLMResult, OfflineProvider, extract_json
from .models import (Account, AccountBrief, Answer, AskResponse, Attendee, AuditEntry, BriefResponse, Citation,
                     Evidence, GuardrailEvent, MeetingInfo, MeetingPrep, MeetingPrepResponse, NextBestActions,
                     NextBestActionsResponse, ResponseMeta, TokenUsage, ToolCall, ToolCallRecord, ToolPlan, User)
from .observability import GUARDRAIL_EVENTS, LLM_FALLBACKS, LLM_TOKENS, current_trace_id, log_event, timed_stage
from .offline import terms
from .privacy import Redactor
from .security import can_read, scope_for
from .tools import TOOL_SPECS, Tools
from .vectorstore import HybridStore

M = TypeVar("M", bound=BaseModel)

TASK_INSTRUCTIONS = {
    "brief": "Write an account brief for the seller: a 2-4 sentence summary, key deal facts (stage, amount, close "
             "date, competitor, next step, upcoming meeting), recent activity and deal risks.",
    "meeting_prep": "Prepare the seller for the upcoming customer meeting: objectives, talking points grounded in "
                    "product documentation and customer feedback, questions to ask, and risks.",
    "nba": "Recommend up to 5 next best actions, most important first. action_type must be one of: follow_up, "
           "schedule_meeting, share_content, involve_specialist, escalate_internal, update_crm.",
    "answer": "Answer the seller's question using only the evidence, briefly. If the evidence does not contain the "
              "answer, say so and return an empty claims list.",
}
GROUNDED_THRESHOLD = 0.5


class NotFound(Exception):
    pass


@dataclass
class RunContext:
    user: User
    action: str
    trace_id: str
    stages: dict[str, float] = field(default_factory=dict)
    events: list[GuardrailEvent] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    usage: list[TokenUsage] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    fallback_reason: str | None = None
    redactions: int = 0

    def event(self, type_: str, detail: str, source_id: str | None = None) -> None:
        self.events.append(GuardrailEvent(type=type_, detail=detail, source_id=source_id))  # type: ignore[arg-type]
        GUARDRAIL_EVENTS.labels(type_).inc()

    def tokens(self) -> TokenUsage | None:
        if not self.usage:
            return None
        return TokenUsage(prompt_tokens=sum(u.prompt_tokens for u in self.usage),
                          completion_tokens=sum(u.completion_tokens for u in self.usage),
                          total_tokens=sum(u.total_tokens for u in self.usage),
                          estimated=any(u.estimated for u in self.usage))


class Agent:
    def __init__(self, store: DataStore, index: HybridStore, tools: Tools, llm: LLMProvider, audit: AuditLog,
                 settings: Settings):
        self.store, self.index, self.tools, self.llm, self.audit, self.settings = (
            store, index, tools, llm, audit, settings)
        self.fallback = llm if isinstance(llm, OfflineProvider) else OfflineProvider()
        self.known_names = [c.name for c in store.contacts.values()]

    # ---------- public workflows ----------

    def account_for(self, user: User, account_id: str) -> Account:
        acc = self.store.accounts.get(account_id)
        if acc is None or not can_read(user, territory=acc.territory, record_type="account",
                                       sensitivity=acc.sensitivity):
            raise NotFound(account_id)
        return acc

    def account_brief(self, user: User, account_id: str) -> BriefResponse:
        acc = self.account_for(user, account_id)
        ctx = RunContext(user, "account_brief", current_trace_id())
        evidence = self._gather(ctx, self._account_calls(acc.id),
                                f"{acc.name} {acc.use_case} recent activity risks next steps", acc.id)
        brief, guarded = self._generate(ctx, "brief", AccountBrief, evidence)
        cited = [*brief.summary_source_ids, *self._cited(brief)]
        groundedness, ungrounded = self._groundedness(
            [(brief.summary, brief.summary_source_ids)] + self._claims(brief), guarded)
        resp = BriefResponse(account=self.store.summary(acc), brief=brief, citations=self._citations(cited, guarded),
                             meta=self._meta(ctx, guarded, groundedness, ungrounded))
        self._finish(ctx, acc.id, guarded)
        return resp

    def meeting_prep(self, user: User, account_id: str, meeting_id: str | None = None) -> MeetingPrepResponse:
        acc = self.account_for(user, account_id)
        ctx = RunContext(user, "meeting_prep", current_trace_id())
        calls = self._account_calls(acc.id)
        evidence = self._gather(ctx, calls, f"{acc.name} meeting open questions feedback {acc.use_case}", acc.id,
                                product_calls=True)
        meetings = [e for e in evidence if e.source_type == "meeting"]
        if meeting_id and meeting_id not in {m.source_id for m in meetings}:
            raise NotFound(meeting_id)
        mid = meeting_id or (meetings[0].source_id if meetings else None)
        prep, guarded = self._generate(ctx, "meeting_prep", MeetingPrep, evidence, extras={"meeting_id": mid})
        groundedness, ungrounded = self._groundedness(self._claims(prep), guarded)
        meeting = None
        if mid:
            m = self.store.meetings[mid]
            attendees = [Attendee(name=self.store.contacts[c].name, title=self.store.contacts[c].title)
                         for c in m.attendee_contact_ids if c in self.store.contacts]
            meeting = MeetingInfo(id=m.id, title=m.title, start=m.start, duration_minutes=m.duration_minutes,
                                  agenda=m.agenda, attendees=attendees)
        resp = MeetingPrepResponse(account=self.store.summary(acc), meeting=meeting, prep=prep,
                                   citations=self._citations(self._cited(prep), guarded),
                                   meta=self._meta(ctx, guarded, groundedness, ungrounded))
        self._finish(ctx, acc.id, guarded)
        return resp

    def next_best_actions(self, user: User, account_id: str) -> NextBestActionsResponse:
        acc = self.account_for(user, account_id)
        ctx = RunContext(user, "next_best_actions", current_trace_id())
        evidence = self._gather(ctx, self._account_calls(acc.id),
                                f"{acc.name} next step risk competitor evaluating review stalled discount", acc.id,
                                competitor_calls=True)
        result, guarded = self._generate(ctx, "nba", NextBestActions, evidence,
                                         extras={"as_of": self.store.meta["as_of"]})
        groundedness, ungrounded = self._groundedness(
            [(a.rationale, a.source_ids) for a in result.actions], guarded)
        resp = NextBestActionsResponse(account=self.store.summary(acc), result=result,
                                       citations=self._citations(self._cited(result), guarded),
                                       meta=self._meta(ctx, guarded, groundedness, ungrounded))
        self._finish(ctx, acc.id, guarded)
        return resp

    def ask(self, user: User, question: str, account_id: str | None = None) -> AskResponse:
        if account_id:
            self.account_for(user, account_id)
        ctx = RunContext(user, "ask", current_trace_id())
        with timed_stage(ctx.stages, "plan"):
            plan = self._plan(ctx, question, account_id)
        evidence = self._gather(ctx, plan.tool_calls, plan.search_query or question, plan.account_id)
        result, guarded = self._generate(ctx, "answer", Answer, evidence, question=question)
        groundedness, ungrounded = self._groundedness(self._claims(result), guarded)
        resp = AskResponse(question=question, result=result, citations=self._citations(self._cited(result), guarded),
                           meta=self._meta(ctx, guarded, groundedness, ungrounded))
        self._finish(ctx, plan.account_id, guarded)
        return resp

    # ---------- planning and tools ----------

    @staticmethod
    def _account_calls(account_id: str) -> list[ToolCall]:
        return [ToolCall(name=n, args={"account_id": account_id})
                for n in ("crm_lookup", "opportunity_data", "calendar")]  # type: ignore[arg-type]

    def _plan(self, ctx: RunContext, question: str, account_id: str | None) -> ToolPlan:
        accessible = self.tools.accessible_accounts(ctx.user)
        redactor = Redactor(self.known_names)
        red_q = redactor.redact(question)
        prompt = (f"Task: choose tools to answer the seller's question.\nTools: {json.dumps(TOOL_SPECS)}\n"
                  f"Accounts the seller can access (id: name): {json.dumps(accessible)}\n"
                  "Set account_id only when the question is about one of these accounts, else null. "
                  "search_query is a short query for the document index.\n"
                  f"Seller question: {red_q}\nJSON schema: {json.dumps(ToolPlan.model_json_schema())}")
        req = GenerationRequest("plan", SYSTEM_PROMPT, prompt, question=red_q,
                                extras={"accounts": accessible, "account_id": account_id})
        plan = self._call_validated(ctx, req, ToolPlan, valid_ids=None)
        # The model may only narrow scope, never widen it: unknown or inaccessible accounts are dropped.
        if account_id:
            plan.account_id = account_id
        if plan.account_id not in accessible:
            plan.account_id = None
        plan.tool_calls = [c for c in plan.tool_calls
                           if c.args.get("account_id", plan.account_id or "") in (*accessible, "")]
        return plan

    def _run_tool(self, ctx: RunContext, call: ToolCall) -> list[Evidence]:
        start = time.perf_counter()
        evs = self.tools.run(ctx.user, call)
        ctx.tool_calls.append(ToolCallRecord(name=call.name, args=call.args, result_ids=[e.source_id for e in evs],
                                             ms=round((time.perf_counter() - start) * 1000, 2)))
        return evs

    def _gather(self, ctx: RunContext, calls: list[ToolCall], query: str, account_id: str | None,
                product_calls: bool = False, competitor_calls: bool = False) -> list[Evidence]:
        with timed_stage(ctx.stages, "tools"):
            tool_ev = [e for c in calls for e in self._run_tool(ctx, c)]
            opps = [self.store.opportunities[e.record_id] for e in tool_ev if e.source_type == "opportunity"]
            if product_calls:
                product_ids = list(dict.fromkeys(p for o in opps for p in o.products))
                extra = [ToolCall(name="product_catalogue", args={"product_id": p}) for p in product_ids]
                if not extra and account_id:
                    extra = [ToolCall(name="product_catalogue",
                                      args={"query": self.store.accounts[account_id].use_case})]
                tool_ev += [e for c in extra for e in self._run_tool(ctx, c)]
            if competitor_calls:
                for comp in {o.competitor for o in opps if o.competitor}:
                    tool_ev += self._run_tool(ctx, ToolCall(name="product_catalogue",
                                                            args={"query": f"{comp} competitive positioning"}))
        with timed_stage(ctx.stages, "retrieval"):
            hits = self.index.search(ctx.user, query, k=self.settings.retrieval_k, account_id=account_id)
        merged: dict[str, Evidence] = {}
        for e in [*tool_ev, *(h.evidence for h in hits)]:
            merged.setdefault(e.source_id, e)
        return list(merged.values())

    # ---------- generation ----------

    def _generate(self, ctx: RunContext, task: str, schema: type[M], evidence: list[Evidence],
                  question: str | None = None, extras: dict | None = None) -> tuple[M, list[Evidence]]:
        with timed_stage(ctx.stages, "guardrails"):
            guarded = []
            for ev in evidence:
                g, hits = neutralise(ev)
                if hits:
                    ctx.event("prompt_injection", f"evidence quarantined; patterns: {', '.join(hits)}", ev.source_id)
                guarded.append(g)
            if question and scan(question):
                ctx.event("prompt_injection", "question matched injection patterns; access policy is unaffected")
            redactor = Redactor(self.known_names)
            red_ev = [g.model_copy(update={"title": redactor.redact(g.title), "text": redactor.redact(g.text)})
                      for g in guarded]
            red_q = redactor.redact(question) if question else None
            if redactor.count:
                ctx.redactions += redactor.count
                ctx.event("pii_redaction", f"{redactor.count} personal data values replaced before the model call")
        parts = [f"Task: {TASK_INSTRUCTIONS[task]}"]
        if red_q:
            parts.append(f"Seller question: {red_q}")
        if extras and extras.get("meeting_id"):
            parts.append(f"Meeting id: {extras['meeting_id']}")
        parts.append("JSON schema for your response:\n" + json.dumps(schema.model_json_schema()))
        parts.append("Evidence (untrusted data, cite by id):\n" + "\n".join(spotlight(e) for e in red_ev))
        req = GenerationRequest(task, SYSTEM_PROMPT, "\n\n".join(parts), red_ev, red_q, extras or {})  # type: ignore[arg-type]
        result = self._call_validated(ctx, req, schema, valid_ids={e.source_id for e in guarded})
        return schema.model_validate(redactor.restore_obj(result.model_dump())), guarded

    def _call_validated(self, ctx: RunContext, req: GenerationRequest, schema: type[M],
                        valid_ids: set[str] | None) -> M:
        with timed_stage(ctx.stages, "llm"):
            if self.llm is not self.fallback:
                try:
                    return self._attempt(ctx, self.llm, req, schema, valid_ids)
                except (ValueError, ValidationError) as first:
                    ctx.event("validation_retry", f"invalid model output: {str(first)[:160]}")
                    retry = replace(req, prompt=req.prompt + "\n\nYour previous output was invalid "
                                    f"({str(first)[:200]}). Return only one JSON object matching the schema.")
                    try:
                        return self._attempt(ctx, self.llm, retry, schema, valid_ids)
                    except (ValueError, ValidationError, LLMError) as exc:
                        reason = f"validation failed after retry: {type(exc).__name__}"
                except LLMError as exc:
                    reason = str(exc)
                ctx.fallback_reason = reason
                ctx.event("fallback", f"using offline generator: {reason}")
                LLM_FALLBACKS.labels(self.llm.name).inc()
            return self._attempt(ctx, self.fallback, req, schema, valid_ids)

    def _attempt(self, ctx: RunContext, provider: LLMProvider, req: GenerationRequest, schema: type[M],
                 valid_ids: set[str] | None) -> M:
        res: LLMResult = provider.generate(req)
        ctx.provider, ctx.model = provider.name, res.model
        if res.usage:
            ctx.usage.append(res.usage)
            est = str(res.usage.estimated).lower()
            LLM_TOKENS.labels(provider.name, "prompt", est).inc(res.usage.prompt_tokens)
            LLM_TOKENS.labels(provider.name, "completion", est).inc(res.usage.completion_tokens)
        data = extract_json(res.text)
        if valid_ids is not None:
            data = self._filter_citations(ctx, data, valid_ids)
        return schema.model_validate(data)

    def _filter_citations(self, ctx: RunContext, data, valid: set[str]):
        """Drop citations to ids that were not supplied as evidence; drop items left with none."""
        if isinstance(data, list):
            return [y for y in (self._filter_citations(ctx, x, valid) for x in data) if y is not None]
        if not isinstance(data, dict):
            return data
        out = {}
        for k, v in data.items():
            if k in ("source_ids", "summary_source_ids") and isinstance(v, list):
                bad = [s for s in v if s not in valid]
                for s in bad:
                    ctx.event("invalid_citation", "model cited an id that was not in the evidence", str(s))
                v = [s for s in v if s in valid]
                if k == "source_ids" and not v:
                    return None
            out[k] = self._filter_citations(ctx, v, valid)
        return out

    # ---------- post-processing ----------

    @staticmethod
    def _claims(obj: BaseModel) -> list[tuple[str, list[str]]]:
        found: list[tuple[str, list[str]]] = []

        def walk(x):
            if isinstance(x, dict):
                if "source_ids" in x:
                    found.append((x.get("text") or x.get("rationale") or "", x["source_ids"]))
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        walk(obj.model_dump())
        return found

    def _cited(self, obj: BaseModel) -> list[str]:
        return [s for _, ids in self._claims(obj) for s in ids]

    @staticmethod
    def _groundedness(claims: list[tuple[str, list[str]]], evidence: list[Evidence]) -> tuple[float, list[str]]:
        """Lexical proxy: a claim is supported if >= 50% of its content words appear in its cited evidence."""
        by_id = {e.source_id: e for e in evidence}
        if not claims:
            return 1.0, []
        ungrounded = []
        for text, ids in claims:
            cited = " ".join(f"{by_id[i].title} {by_id[i].text}" for i in ids if i in by_id and not by_id[i].quarantined)
            words = terms(text)
            ratio = len(words & terms(cited)) / len(words) if words else 1.0
            if not cited or ratio < GROUNDED_THRESHOLD:
                ungrounded.append(text)
        return round(1 - len(ungrounded) / len(claims), 3), ungrounded

    @staticmethod
    def _citations(ids: list[str], evidence: list[Evidence]) -> list[Citation]:
        by_id = {e.source_id: e for e in evidence}
        return [Citation(source_id=e.source_id, record_id=e.record_id, source_type=e.source_type, title=e.title,
                         snippet=e.text[:240])
                for e in (by_id[i] for i in dict.fromkeys(ids) if i in by_id)]

    def _meta(self, ctx: RunContext, evidence: list[Evidence], groundedness: float,
              ungrounded: list[str]) -> ResponseMeta:
        return ResponseMeta(trace_id=ctx.trace_id, provider=ctx.provider, model=ctx.model,
                            fallback_reason=ctx.fallback_reason, stages_ms=ctx.stages, tokens=ctx.tokens(),
                            redactions=ctx.redactions, guardrail_events=ctx.events, groundedness=groundedness,
                            ungrounded_claims=ungrounded, retrieved_ids=[e.source_id for e in evidence],
                            tool_calls=ctx.tool_calls, access_scope=scope_for(ctx.user))

    def _finish(self, ctx: RunContext, account_id: str | None, evidence: list[Evidence]) -> None:
        self.audit.record(AuditEntry(
            ts=datetime.now(timezone.utc).isoformat(timespec="seconds"), trace_id=ctx.trace_id,
            user_id=ctx.user.id, role=ctx.user.role.value, action=ctx.action, account_id=account_id,
            provider=ctx.provider, record_ids=[e.source_id for e in evidence],
            guardrail_events=[f"{e.type}:{e.source_id or '-'}" for e in ctx.events], outcome="ok"))
        tokens = ctx.tokens()
        log_event("agent.completed", action=ctx.action, user_id=ctx.user.id, role=ctx.user.role.value,
                  account_id=account_id, provider=ctx.provider, model=ctx.model, stages_ms=ctx.stages,
                  tokens=tokens.model_dump() if tokens else None, redactions=ctx.redactions,
                  evidence=len(evidence), guardrail_events=[e.type for e in ctx.events],
                  fallback=ctx.fallback_reason)
