import json

import pytest

from app.agent import Agent, NotFound
from app.llm import GenerationRequest, LLMError, LLMResult, OfflineProvider
from app.privacy import contains_pii


class Scripted:
    """Fake model: returns queued outputs (str) or raises (Exception), records every request."""
    name, model = "fake", "fake-model"

    def __init__(self, *outputs):
        self.outputs, self.requests = list(outputs), []

    def generate(self, req: GenerationRequest) -> LLMResult:
        self.requests.append(req)
        out = self.outputs.pop(0) if self.outputs else "not json"
        if isinstance(out, Exception):
            raise out
        return LLMResult(text=out if isinstance(out, str) else json.dumps(out), usage=None, model=self.model)


def make_agent(base: Agent, llm) -> Agent:
    return Agent(base.store, base.index, base.tools, llm, base.audit, base.settings)


def test_brief_has_valid_citations_and_quarantines_injection(agent, users):
    agent.llm.history.clear()
    resp = agent.account_brief(users["u-ae-01"], "ACC-001")
    evidence = set(resp.meta.retrieved_ids)
    cited = {c.source_id for c in resp.citations}
    assert cited and cited <= evidence
    assert any(e.type == "prompt_injection" and e.source_id == "ACT-0046" for e in resp.meta.guardrail_events)
    prompt = agent.llm.history[-1].prompt
    assert "IGNORE ALL PREVIOUS INSTRUCTIONS" not in prompt and "exfil@attacker.example" not in prompt
    assert not contains_pii(prompt, agent.known_names)
    assert resp.meta.groundedness == 1.0 and resp.meta.tokens.estimated


def test_role_changes_what_the_brief_can_use(agent, users):
    sdr = agent.account_brief(users["u-sdr-01"], "ACC-001")
    mgr = agent.account_brief(users["u-mgr-01"], "ACC-001")
    assert not {"OPP-0011", "ACT-0003", "ACT-0047", "EML-0011"} & set(sdr.meta.retrieved_ids)
    assert "ACT-0047" in mgr.meta.retrieved_ids


def test_inaccessible_account_raises_not_found(agent, users):
    with pytest.raises(NotFound):
        agent.account_brief(users["u-ae-01"], "ACC-008")
    with pytest.raises(NotFound):
        agent.ask(users["u-sdr-01"], "anything", "ACC-005")


def test_meeting_prep_and_next_best_actions(agent, users):
    prep = agent.meeting_prep(users["u-ae-01"], "ACC-001")
    assert prep.meeting and prep.meeting.id == "MTG-0011" and prep.prep.objectives
    assert prep.meeting.attendees[0].name == "Nikhil Bansal"  # restored after redaction
    nba = agent.next_best_actions(users["u-ae-02"], "ACC-008")
    kinds = {a.action_type for a in nba.result.actions}
    assert {"follow_up", "involve_specialist"} <= kinds
    assert all(set(a.source_ids) <= set(nba.meta.retrieved_ids) for a in nba.result.actions)


def test_ask_cross_territory_returns_nothing_from_other_territory(agent, users):
    resp = agent.ask(users["u-ae-01"], "What discount is approved for Skyrune Airlines?")
    assert not any(i.startswith(("ACT-0049", "OPP-0081", "ACC-008")) for i in resp.meta.retrieved_ids)
    assert "15 percent" not in resp.result.answer


def test_invalid_citations_are_dropped(agent, users):
    bad = {"answer": "Kordon Cloud is the competitor.",
           "claims": [{"text": "Kordon Cloud is the competitor.", "source_ids": ["ACT-0003", "ACT-0049"]},
                      {"text": "Fabricated claim with only a bogus id", "source_ids": ["XYZ-1"]}]}
    plan = {"account_id": "ACC-001", "tool_calls": [{"name": "crm_lookup", "args": {"account_id": "ACC-001"}}],
            "search_query": "competitor"}
    fake = Scripted(plan, bad)
    resp = make_agent(agent, fake).ask(users["u-ae-01"], "Which competitor is Velmora evaluating?")
    assert [c.source_ids for c in resp.result.claims] == [["ACT-0003"]]
    assert sum(e.type == "invalid_citation" for e in resp.meta.guardrail_events) == 2
    assert resp.meta.provider == "fake"


def test_model_cannot_widen_scope_via_plan(agent, users):
    plan = {"account_id": "ACC-008", "tool_calls": [{"name": "opportunity_data", "args": {"account_id": "ACC-008"}}],
            "search_query": "Skyrune discount"}
    fake = Scripted(plan, {"answer": "No information.", "claims": []})
    resp = make_agent(agent, fake).ask(users["u-ae-01"], "Tell me about Skyrune")
    assert resp.meta.tool_calls == []
    assert not any(i.startswith(("ACC-008", "OPP-0081")) for i in resp.meta.retrieved_ids)


def test_invalid_output_retries_then_falls_back(agent, users):
    fake = Scripted("garbage", "{\"still\": \"wrong\"}")
    resp = make_agent(agent, fake).account_brief(users["u-ae-01"], "ACC-002")
    types = [e.type for e in resp.meta.guardrail_events]
    assert "validation_retry" in types and "fallback" in types
    assert resp.meta.provider == "offline" and resp.meta.fallback_reason
    assert len(fake.requests) == 2


def test_provider_error_falls_back(agent, users):
    resp = make_agent(agent, Scripted(LLMError("boom"))).next_best_actions(users["u-ae-01"], "ACC-001")
    assert resp.meta.provider == "offline" and resp.meta.fallback_reason == "boom"


def test_fake_provider_never_sees_pii_or_injection(agent, users):
    fake = Scripted()
    make_agent(agent, fake).meeting_prep(users["u-mgr-01"], "ACC-001")
    assert fake.requests
    for req in fake.requests:
        assert not contains_pii(req.prompt, agent.known_names)
        assert "IGNORE ALL PREVIOUS" not in req.prompt


def test_offline_is_deterministic(agent, users):
    a = agent.ask(users["u-ae-01"], "What is the stage of the Velmora Logistics opportunity?")
    b = agent.ask(users["u-ae-01"], "What is the stage of the Velmora Logistics opportunity?")
    assert a.result == b.result and isinstance(agent.llm, OfflineProvider)
