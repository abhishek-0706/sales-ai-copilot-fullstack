"""Checks on the LLM fine-tuning dataset (llm_finetune/): frozen test set, schema, no test leakage, the scorer, and
the endpoint evaluator against a stub OpenAI-compatible server."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FT = ROOT / "llm_finetune"
DATA = FT / "data"
sys.path.insert(0, str(FT))

import scoring  # noqa: E402
import stub_server  # noqa: E402
from app.models import Answer  # noqa: E402
from eval.run_eval import record_meta  # noqa: E402

REQUIRED = {"id", "split", "account_id", "intent", "wording", "user", "question", "system", "prompt", "prompt_sha256",
            "evidence", "expected", "abstain_reason", "facts", "facts_restored", "fact_kind", "supporting_ids",
            "target", "offline_generator_output"}


def rows(split: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture(scope="module")
def splits() -> dict[str, list[dict]]:
    return {s: rows(s) for s in ("train", "val", "test")}


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads((DATA / "dataset_report.json").read_text(encoding="utf-8"))


def test_test_set_is_frozen(report):
    digest = hashlib.sha256((DATA / "test.jsonl").read_bytes()).hexdigest()
    assert digest == (DATA / "test.sha256").read_text().split()[0] == report["test_sha256"]


def test_rows_follow_the_schema(splits):
    for split, rs in splits.items():
        assert rs
        for r in rs:
            assert REQUIRED <= r.keys() and r["split"] == split
            assert hashlib.sha256(r["prompt"].encode()).hexdigest() == r["prompt_sha256"]
            target = Answer.model_validate_json(r["target"])
            evidence = {e["source_id"] for e in r["evidence"]}
            cited = {s for c in target.claims for s in c.source_ids}
            assert cited <= evidence
            if r["expected"] == "answer":
                assert target.claims and cited <= set(r["supporting_ids"]) and r["abstain_reason"] is None
            else:
                assert not target.claims and target.answer == scoring.ABSTAIN_TEXT and r["abstain_reason"]
                assert not r["supporting_ids"]


def test_splits_are_by_account_and_do_not_leak(splits, report):
    accounts = {s: {r["account_id"] for r in rs} for s, rs in splits.items()}
    assert not (accounts["train"] & accounts["val"]) and not (accounts["train"] & accounts["test"])
    assert not (accounts["val"] & accounts["test"])
    names = report["account_names"]
    held_out = accounts["val"] | accounts["test"]
    for r in splits["train"]:
        assert not any(e["source_id"].split(":")[0] in report["record_meta"] and e["account_id"] in held_out
                       for e in r["evidence"]), r["id"]
        assert not any(names[a] in r["prompt"] or names[a].split()[0] in r["question"] for a in held_out), r["id"]
        assert r["wording"] == "seen"
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        assert not ({r["prompt_sha256"] for r in splits[a]} & {r["prompt_sha256"] for r in splits[b]})
        assert not ({r["question"].lower() for r in splits[a]} & {r["question"].lower() for r in splits[b]})
    assert {r["wording"] for r in splits["test"]} == {"seen", "held_out"}


def test_no_personal_data_in_prompts_or_targets(splits):
    contacts = json.loads((ROOT / "backend" / "data" / "contacts.json").read_text(encoding="utf-8"))
    pii = [c[k] for c in contacts for k in ("name", "email", "phone")]
    for rs in splits.values():
        for r in rs:
            blob = r["prompt"] + r["target"]
            assert not any(v in blob for v in pii), r["id"]


def test_targets_score_perfectly_and_wrong_outputs_do_not():
    meta = record_meta()
    test = rows("test")
    scores = [scoring.score_row(r, r["target"], meta) for r in test]
    summary = scoring.summarize(test, scores)
    for k in ("schema_valid", "behaviour_accuracy", "answer_accuracy", "correct_abstention", "citation_hit"):
        assert summary[k]["mean"] == 1.0, k
    answer_row = next(r for r in test if r["expected"] == "answer")
    abstain_row = next(r for r in test if r["expected"] == "abstain")
    assert not scoring.score_row(answer_row, "not json", meta)["valid"]
    abstained = scoring.score_row(answer_row, json.dumps({"answer": scoring.ABSTAIN_TEXT, "claims": []}), meta)
    assert abstained["false_abstain"] and not abstained["behaviour_ok"]
    made_up = json.dumps({"answer": "x", "claims": [{"text": "made up", "source_ids": ["OPP-9999"]}]})
    s = scoring.score_row(abstain_row, made_up, meta)
    # the app drops the claim with the unknown id and accepts what is left; the scorer still counts the bad citation
    assert s["valid"] and s["app_accepts"] and s["cited"] == 1 and s["cited_valid"] == 0 and not s["behaviour_ok"]


def test_endpoint_evaluator_through_the_apps_provider(monkeypatch, tmp_path):
    import eval_endpoint

    test = rows("test")[:6]
    server = stub_server.serve({r["prompt_sha256"]: r["target"] for r in test}, model="stub")
    port = server.server_address[1]
    for k, v in {"LLM_PROVIDER": "openai_compat", "LLM_BASE_URL": f"http://127.0.0.1:{port}/v1",
                 "LLM_MODEL": "stub", "LLM_API_KEY": ""}.items():
        monkeypatch.setenv(k, v)
    try:
        assert eval_endpoint.main(["--mode", "direct", "--limit", "6", "--out-dir", str(tmp_path)]) == 0
        assert eval_endpoint.main(["--mode", "app", "--limit", "6", "--out-dir", str(tmp_path)]) == 0
    finally:
        server.shutdown()
    direct = json.loads((tmp_path / "endpoint_stub_direct.json").read_text(encoding="utf-8"))
    app = json.loads((tmp_path / "endpoint_stub_app.json").read_text(encoding="utf-8"))
    assert direct["metrics"]["all"]["behaviour_accuracy"]["mean"] == 1.0 and direct["metrics"]["http_errors"] == 0
    assert app["metrics"]["prompt_matches_frozen"]["mean"] == 1.0 and app["metrics"]["fallback"]["mean"] == 0.0
    assert app["metrics"]["behaviour_ok"]["mean"] == 1.0


def test_endpoint_evaluator_counts_fallbacks(monkeypatch, tmp_path):
    import eval_endpoint

    server = stub_server.serve({}, model="stub", mode="garbage")
    for k, v in {"LLM_PROVIDER": "openai_compat", "LLM_BASE_URL": f"http://127.0.0.1:{server.server_address[1]}/v1",
                 "LLM_MODEL": "stub", "LLM_API_KEY": ""}.items():
        monkeypatch.setenv(k, v)
    try:
        assert eval_endpoint.main(["--mode", "app", "--limit", "3", "--out-dir", str(tmp_path)]) == 0
    finally:
        server.shutdown()
    app = json.loads((tmp_path / "endpoint_stub_app.json").read_text(encoding="utf-8"))
    assert app["metrics"]["fallback"]["mean"] == 1.0 and app["metrics"]["validation_retry_rate"]["mean"] == 1.0
