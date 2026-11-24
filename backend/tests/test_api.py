import json
import re

from conftest import hdr


def test_health_and_openapi(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["provider"] == "offline" and h["index_chunks"] > 100
    spec = client.get("/openapi.json").json()
    for path in ["/api/accounts", "/api/accounts/{account_id}/brief", "/api/accounts/{account_id}/meeting-prep",
                 "/api/accounts/{account_id}/next-best-actions", "/api/ask", "/api/sources/{source_id}"]:
        assert path in spec["paths"]
    assert "BriefResponse" in spec["components"]["schemas"]


def test_unknown_or_missing_user_is_rejected(client):
    assert client.get("/api/accounts", headers=hdr("nobody")).status_code == 401
    assert client.get("/api/accounts").status_code == 422


def test_account_list_is_filtered_by_territory_and_role(client):
    sdr = client.get("/api/accounts", headers=hdr("u-sdr-01")).json()
    mgr = client.get("/api/accounts", headers=hdr("u-mgr-01")).json()
    emea = client.get("/api/accounts", headers=hdr("u-ae-02")).json()
    assert {a["territory"] for a in sdr} == {"APAC-West"} and len(sdr) == 4
    assert all(a["opportunity_stage"] is None for a in sdr)
    assert {a["territory"] for a in mgr} == {"APAC-West", "APAC-South"} and len(mgr) == 7
    assert {a["id"] for a in emea} == {"ACC-008", "ACC-009", "ACC-010"}


def test_inaccessible_records_return_404(client):
    assert client.get("/api/accounts/ACC-008", headers=hdr("u-ae-01")).status_code == 404
    assert client.post("/api/accounts/ACC-008/brief", headers=hdr("u-ae-01")).status_code == 404
    assert client.get("/api/sources/ACT-0047", headers=hdr("u-ae-01")).status_code == 404
    assert client.get("/api/sources/OPP-0011", headers=hdr("u-sdr-01")).status_code == 404
    assert client.get("/api/sources/ACT-0047", headers=hdr("u-mgr-01")).status_code == 200


def test_source_viewer_flags_injection(client):
    src = client.get("/api/sources/ACT-0046", headers=hdr("u-ae-01")).json()
    assert src["flagged_injection"] is True


def test_brief_endpoint_and_trace_header(client):
    r = client.post("/api/accounts/ACC-001/brief", headers=hdr("u-ae-01"))
    assert r.status_code == 200
    body = r.json()
    assert re.fullmatch(r"[0-9a-f]{32}", r.headers["x-trace-id"])
    assert body["meta"]["trace_id"] == r.headers["x-trace-id"]
    assert {"tools", "retrieval", "guardrails", "llm"} <= set(body["meta"]["stages_ms"])
    assert body["citations"] and body["account"]["synthetic"] is True


def test_traceparent_is_propagated(client):
    tid = "4bf92f3577b34da6a3ce929d0e0e4736"
    r = client.get("/api/health", headers={"traceparent": f"00-{tid}-00f067aa0ba902b7-01"})
    assert r.headers["x-trace-id"] == tid


def test_meeting_prep_validation(client):
    ok = client.post("/api/accounts/ACC-001/meeting-prep", headers=hdr("u-ae-01"), json={"meeting_id": "MTG-0011"})
    assert ok.status_code == 200 and ok.json()["meeting"]["id"] == "MTG-0011"
    other = client.post("/api/accounts/ACC-001/meeting-prep", headers=hdr("u-ae-01"), json={"meeting_id": "MTG-0081"})
    assert other.status_code == 404


def test_ask_validation_and_answer(client):
    assert client.post("/api/ask", headers=hdr("u-ae-01"), json={"question": ""}).status_code == 422
    r = client.post("/api/ask", headers=hdr("u-ae-01"),
                    json={"question": "Which competitor is Velmora Logistics evaluating?"}).json()
    assert "Kordon Cloud" in r["result"]["answer"]
    assert {c["source_id"] for c in r["citations"]} <= set(r["meta"]["retrieved_ids"])


def test_audit_is_manager_only_and_records_requests(client):
    client.post("/api/accounts/ACC-002/next-best-actions", headers=hdr("u-ae-01"))
    assert client.get("/api/audit", headers=hdr("u-ae-01")).status_code == 403
    entries = client.get("/api/audit", headers=hdr("u-mgr-01")).json()
    assert any(e["action"] == "next_best_actions" and e["user_id"] == "u-ae-01" for e in entries)


def test_metrics_endpoint(client):
    client.get("/api/health")
    text = client.get("/metrics").text
    for name in ["salesai_http_requests_total", "salesai_stage_duration_seconds_bucket",
                 "salesai_guardrail_events_total", "salesai_llm_tokens_total"]:
        assert name in text


def test_audit_log_file_is_jsonl(client, settings):
    client.post("/api/ask", headers=hdr("u-sdr-01"), json={"question": "What is Torvane's pain point?"})
    lines = settings.audit_log_path.read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(lines[-1])["user_id"] == "u-sdr-01"
