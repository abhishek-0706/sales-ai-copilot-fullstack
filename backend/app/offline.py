"""Deterministic offline "model": rule-based planning and extractive synthesis.

It is not an LLM. It exists so the full pipeline (tools, retrieval, guardrails, redaction,
validation, citations, audit) runs with no API key, and so tests and CI are reproducible.
It only sees the same redacted, guarded evidence a real model would see.
"""
from __future__ import annotations

import math
import re

from .embeddings import tokenize
from .models import Evidence

SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
RISK_TERMS = {"risk", "concern", "pending", "stalled", "review", "budget", "evaluating", "requires", "freeze",
              "depends", "gdpr", "residency", "security", "approval", "procurement", "payback", "intermittent",
              "strict", "validated", "mandatory"}
SPECIALIST_RE = re.compile(r"security|gdpr|privacy|compliance|model risk|residency|validated systems|validation team|"
                           r"intellectual property|data processing", re.IGNORECASE)


def terms(text: str) -> set[str]:
    return {t[:-1] if len(t) > 4 and t.endswith("s") else t for t in tokenize(text)}


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_RE.split(text) if len(s.strip()) > 3]


def usable(evidence: list[Evidence]) -> list[Evidence]:
    return [e for e in evidence if not e.quarantined]


def _date(ev: Evidence) -> str:
    m = DATE_RE.search(ev.title) or DATE_RE.search(ev.text)
    return m.group(0) if m else ""


def claim(text: str, *evs: Evidence) -> dict:
    return {"text": text[:600], "source_ids": [e.source_id for e in evs]}


def _of(ev: list[Evidence], *types: str) -> list[Evidence]:
    return [e for e in ev if e.source_type in types]


def risk_claims(ev: list[Evidence], limit: int) -> list[dict]:
    seen, out = set(), []
    for e in sorted(_of(ev, "activity", "email", "opportunity"), key=_date, reverse=True):
        for s in sentences(e.text):
            if terms(s) & RISK_TERMS and s.lower() not in seen and not s.lower().startswith("regards"):
                seen.add(s.lower())
                out.append(claim(s, e))
    return out[:limit]


def plan(question: str, accounts: dict[str, str], account_id: str | None) -> dict:
    q = question.lower()
    if not account_id:
        for aid, name in accounts.items():
            if name.lower() in q or re.search(rf"\b{re.escape(name.split()[0].lower())}\b", q):
                account_id = aid
                break
    calls: list[dict] = []
    if account_id:
        calls.append({"name": "crm_lookup", "args": {"account_id": account_id}})
        if re.search(r"deal|opportunit|pipeline|stage|amount|close|forecast|compet|discount|pric|next step|worth", q):
            calls.append({"name": "opportunity_data", "args": {"account_id": account_id}})
        if re.search(r"meeting|calendar|schedule|agenda|upcoming|when", q):
            calls.append({"name": "calendar", "args": {"account_id": account_id}})
    elif re.search(r"meeting|calendar|schedule|agenda|upcoming", q):
        calls.append({"name": "calendar", "args": {}})
    if re.search(r"product|feature|platform|capabilit|support|latency|redact|guard|vector|edge|document ai|"
                 r"pric|discount|policy|compet|position", q):
        calls.append({"name": "product_catalogue", "args": {"query": question}})
    return {"account_id": account_id, "tool_calls": calls, "search_query": question}


def brief(evidence: list[Evidence]) -> dict:
    ev = usable(evidence)
    acc = next(iter(_of(ev, "account")), None)
    opps = _of(ev, "opportunity")
    summary, ids = [], []
    if acc:
        summary.append(" ".join(sentences(acc.text)[:2]))
        ids.append(acc.source_id)
    if opps:
        summary.append(sentences(opps[0].text)[0])
        ids.append(opps[0].source_id)
    if not ids and ev:
        summary.append(sentences(ev[0].text)[0])
        ids.append(ev[0].source_id)
    key_facts = [claim(s, o) for o in opps for s in sentences(o.text)][:5]
    key_facts += [claim(sentences(m.text)[0], m) for m in _of(ev, "meeting")[:1]]
    acts = sorted(_of(ev, "activity"), key=_date, reverse=True)
    recent = [claim(" ".join(sentences(a.text)[:2]), a) for a in acts][:4]
    return {"summary": " ".join(summary) or "No accessible records were found for this account.",
            "summary_source_ids": ids or ["none"], "key_facts": key_facts, "recent_activity": recent,
            "risks": risk_claims(ev, 4)}


def meeting_prep(evidence: list[Evidence], meeting_id: str | None) -> dict:
    ev = usable(evidence)
    meetings = _of(ev, "meeting")
    meeting = next((m for m in meetings if m.source_id == meeting_id), meetings[0] if meetings else None)
    objectives = []
    if meeting and "Agenda:" in meeting.text:
        agenda = meeting.text.split("Agenda:", 1)[1]
        objectives = [claim(_upper_first(p.strip().rstrip(".")), meeting) for p in agenda.split(";") if p.strip()]
    if not objectives:
        for o in _of(ev, "opportunity", "activity"):
            m = re.search(r"next step: ([^.]+)", o.text, re.IGNORECASE)
            if m:
                objectives = [claim(f"Agree next step: {m.group(1)}", o)]
                break
    if not objectives and ev:
        objectives = [claim(sentences(ev[0].text)[0], ev[0])]
    talking = [claim(sentences(d.text)[0], d) for d in _of(ev, "product_doc")[:3]]
    for a in _of(ev, "activity"):
        m = re.search(r"Feedback: ([^.]+)\.", a.text)
        if m:
            talking.append(claim(f"Feedback: {m.group(1)}", a))
    questions = []
    for e in _of(ev, "activity", "email"):
        for m in re.finditer(r"(?:Open question|clarity on):? ([^.]+)\.", e.text):
            q = f"Can we resolve {m.group(1).strip()}?"
            if q not in questions:
                questions.append(q)
    return {"objectives": objectives[:5], "talking_points": talking[:6], "questions_to_ask": questions[:6],
            "risks": risk_claims(ev, 3)}


def next_best_actions(evidence: list[Evidence], as_of: str) -> dict:
    ev = usable(evidence)
    actions: list[dict] = []

    def add(action: str, kind: str, priority: str, rationale: str, *evs: Evidence) -> None:
        actions.append({"action": action, "action_type": kind, "priority": priority, "rationale": rationale,
                        "source_ids": [e.source_id for e in evs]})

    for m in _of(ev, "meeting")[:1]:
        add(f"Confirm attendees and agenda for '{m.title}'", "schedule_meeting", "high", sentences(m.text)[0], m)
    for o in _of(ev, "opportunity")[:1]:
        step = re.search(r"Next step: ([^.]+)", o.text)
        close = re.search(r"close date (\d{4}-\d{2}-\d{2})", o.text)
        urgent = bool(close and close.group(1) <= _plus_days(as_of, 45))
        if step:
            add(f"Complete the agreed next step: {step.group(1)}", "follow_up", "high" if urgent else "medium",
                sentences(o.text)[0], o)
    for a in _of(ev, "activity"):
        m = re.search(r"is also evaluating ([A-Z][\w ]+?)\.", a.text)
        if m:
            brief_doc = [d for d in _of(ev, "product_doc") if m.group(1) in d.title]
            add(f"Share competitive positioning against {m.group(1)}", "share_content", "medium",
                sentences(a.text)[0], a, *brief_doc[:1])
            break
    for e in sorted(_of(ev, "activity", "email", "opportunity"), key=_date, reverse=True):
        hit = next((s for s in sentences(e.text) if SPECIALIST_RE.search(s)), None)
        if hit:
            add("Bring a security and compliance specialist to the next call", "involve_specialist", "high", hit, e)
            break
    for a in _of(ev, "activity"):
        if re.search(r"stalled|economic buyer", a.text):
            add("Ask your sales manager for an executive sponsor to re-engage the economic buyer",
                "escalate_internal", "high", sentences(a.text)[0], a)
            break
    for a in _of(ev, "activity"):
        if "Deal desk guidance" in a.text:
            add("Keep pricing within the approved deal desk guidance before sending the proposal",
                "escalate_internal", "medium", sentences(a.text)[0], a)
            break
    if not actions and ev:
        add("Log a discovery call to capture the customer's priorities", "update_crm", "low",
            sentences(ev[0].text)[0], ev[0])
    return {"actions": actions[:5]}


def answer(question: str, evidence: list[Evidence]) -> dict:
    """Extractive answer: rank evidence sentences by IDF-weighted overlap with the question."""
    q = terms(question)
    units = [(order, e, s, terms(s), terms(e.title)) for order, e in enumerate(usable(evidence))
             for s in sentences(e.text)]
    df: dict[str, int] = {}
    for *_, st, tt in units:
        for t in st | tt:
            df[t] = df.get(t, 0) + 1
    idf = {t: math.log(1 + len(units) / n) for t, n in df.items()}
    # Terms present in most units (typically the account name) cannot on their own make a sentence relevant.
    common = {t for t, n in df.items() if n > 0.5 * len(units)}
    scored = []
    for order, e, s, st, tt in units:
        in_sentence = q & st
        title_only = (q & tt) - in_sentence
        if len(in_sentence) + len(title_only) >= min(2, len(q)) and ((in_sentence | title_only) - common):
            score = sum(idf[t] for t in in_sentence) + 0.5 * sum(idf[t] for t in title_only)
            scored.append((score, order, s, e))
    scored.sort(key=lambda x: (-x[0], x[1]))
    picked, used = [], set()
    for _, _, s, e in scored:
        if s not in used and len(picked) < 3:
            picked.append(claim(s, e))
            used.add(s)
    if not picked:
        return {"answer": "I could not find this in the records you are allowed to access.", "claims": []}
    return {"answer": " ".join(c["text"] for c in picked), "claims": picked}


def _upper_first(text: str) -> str:
    return text[:1].upper() + text[1:]


def _plus_days(day: str, n: int) -> str:
    from datetime import date, timedelta
    return (date.fromisoformat(day) + timedelta(days=n)).isoformat()
