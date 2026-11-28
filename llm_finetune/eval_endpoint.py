"""Score any OpenAI-compatible endpoint (vLLM, NIM, Ollama, llama.cpp ...) on the frozen test set with the app's own
provider class and validators. Writes llm_finetune/results/endpoint_<name>_<mode>.json.

  LLM_PROVIDER=openai_compat LLM_BASE_URL=http://host:port/v1 LLM_MODEL=<served name> [LLM_API_KEY=...]
  python llm_finetune/eval_endpoint.py [--mode direct|app] [--limit N] [--workers 4] [--name NAME]

direct  each frozen prompt (system + user message, exactly as the app built it) is sent through
        app.llm.OpenAICompatProvider, the class the app uses, and the raw output is scored with scoring.py.
app     each test question is asked end to end through the unchanged app: Agent.ask as the row's persona, with the
        endpoint as the LLM provider (validation, one retry, fallback to the offline generator, PII restore). It
        reports whether the prompt the app built today is identical to the frozen one, how often the app retried or
        fell back, and behaviour, fact and citation checks on the final response. By default the tool-planning step
        stays with the rule planner (--planner offline), because the fine-tuned model was trained on the answer
        task only; --planner endpoint sends planning to the endpoint too.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent / "backend", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402

import scoring  # noqa: E402
from app.config import Settings  # noqa: E402
from app.llm import GenerationRequest, LLMError, OfflineProvider, make_provider  # noqa: E402
from app.main import create_app  # noqa: E402
from app.models import User  # noqa: E402
from eval.run_eval import allowed, record_meta  # noqa: E402

DATA = HERE / "data"


def load_test(limit: int | None = None) -> list[dict]:
    raw = (DATA / "test.jsonl").read_bytes()
    frozen = (DATA / "test.sha256").read_text().split()[0]
    if hashlib.sha256(raw).hexdigest() != frozen:
        raise SystemExit("data/test.jsonl does not match data/test.sha256: the frozen test set was modified")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def run_direct(rows: list[dict], settings: Settings, workers: int) -> tuple[dict, list[dict]]:
    provider = make_provider(settings)
    meta = record_meta()

    def one(r: dict) -> tuple[str, float, str | None]:
        t0 = time.perf_counter()
        try:
            text, err = provider.generate(GenerationRequest("answer", r["system"], r["prompt"])).text, None
        except LLMError as exc:
            text, err = "", str(exc)
        return text, round(time.perf_counter() - t0, 3), err

    with ThreadPoolExecutor(max_workers=workers) as pool:
        outs = list(pool.map(one, rows))
    scores = [scoring.score_row(r, text, meta) for r, (text, _, _) in zip(rows, outs)]
    items = [{"id": r["id"], "raw": text, "seconds": s, "http_error": err, "score": sc}
             for r, (text, s, err), sc in zip(rows, outs, scores)]
    metrics = scoring.report(rows, scores)
    metrics["http_errors"] = sum(1 for _, _, e in outs if e)
    return metrics, items


class _Router:
    """Planning to the rule planner, everything else to the endpoint (see --planner)."""

    def __init__(self, endpoint, planner):
        self.endpoint, self.planner = endpoint, planner
        self.name, self.model = endpoint.name, endpoint.model

    def generate(self, req):
        return (self.planner if req.task == "plan" else self.endpoint).generate(req)


class _Recorder:
    def __init__(self, inner):
        self.inner, self.name, self.model = inner, inner.name, inner.model
        self.answer_prompts: list[str] = []

    def generate(self, req):
        if req.task == "answer":
            self.answer_prompts.append(req.prompt)
        return self.inner.generate(req)


def _mean_ci(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {"mean": round(float(arr.mean()), 4), "ci95": scoring._ci(arr), "n": len(values)} if values else {}


def run_app(rows: list[dict], settings: Settings, planner: str) -> tuple[dict, list[dict]]:
    logging.getLogger("salesai").disabled = True
    meta = record_meta()
    with tempfile.TemporaryDirectory() as tmp:
        # hash embedder: the frozen prompts were built with it, so retrieval must match
        app = create_app(replace(settings, embedder="hash", var_dir=Path(tmp), audit_log_path=Path(tmp) / "a.jsonl"))
        agent = app.state.agent
        endpoint = agent.llm if planner == "endpoint" else _Router(agent.llm, OfflineProvider())
        rec = _Recorder(endpoint)
        agent.llm = rec
        items = []
        for r in rows:
            u = r["user"]
            user = User(id=u["id"], name=u["id"], role=u["role"], territories=u["territories"])
            rec.answer_prompts.clear()
            t0 = time.perf_counter()
            resp = agent.ask(user, r["question"])
            seconds = round(time.perf_counter() - t0, 3)
            events = [e.type for e in resp.meta.guardrail_events]
            claims = resp.result.claims
            text = " ".join([resp.result.answer, *(c.text for c in claims)])
            cited = [s for c in claims for s in c.source_ids]
            evidence = set(resp.meta.retrieved_ids)
            abstained = not claims
            expect_abstain = r["expected"] == "abstain"
            items.append({
                "id": r["id"], "seconds": seconds,
                "prompt_matches_frozen": bool(rec.answer_prompts) and
                hashlib.sha256(rec.answer_prompts[0].encode()).hexdigest() == r["prompt_sha256"],
                "fallback": resp.meta.fallback_reason is not None, "fallback_reason": resp.meta.fallback_reason,
                "validation_retries": events.count("validation_retry"),
                "invalid_citations_removed": events.count("invalid_citation"),
                "abstained": abstained, "behaviour_ok": abstained == expect_abstain,
                "answer_ok": None if expect_abstain else
                (not abstained and scoring.fact_found(text, r["facts_restored"], r["fact_kind"])),
                "abstain_ok": abstained if expect_abstain else None,
                "citations_valid": all(s in evidence and allowed(u, s, meta) for s in cited),
                "answer": resp.result.answer,
            })
    metrics = {k: _mean_ci([float(i[k]) for i in items if i[k] is not None])
               for k in ("prompt_matches_frozen", "fallback", "behaviour_ok", "answer_ok", "abstain_ok",
                         "citations_valid")}
    metrics["validation_retry_rate"] = _mean_ci([float(i["validation_retries"] > 0) for i in items])
    metrics["planner"] = planner
    return metrics, items


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["direct", "app"], default="direct")
    ap.add_argument("--planner", choices=["offline", "endpoint"], default="offline", help="app mode only")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N test rows")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests (direct mode)")
    ap.add_argument("--name", default=None, help="output file name (default: LLM_MODEL)")
    ap.add_argument("--out-dir", type=Path, default=HERE / "results")
    a = ap.parse_args(argv)
    settings = Settings.from_env()
    if settings.llm_provider != "openai_compat":
        raise SystemExit("set LLM_PROVIDER=openai_compat, LLM_BASE_URL and LLM_MODEL (and LLM_API_KEY if needed)")
    rows = load_test(a.limit)
    t0 = time.perf_counter()
    if a.mode == "direct":
        metrics, items = run_direct(rows, settings, a.workers)
    else:
        metrics, items = run_app(rows, settings, a.planner)
    name = re.sub(r"[^\w.-]+", "_", a.name or settings.llm_model)
    out = {"endpoint": settings.llm_base_url, "model": settings.llm_model, "mode": a.mode,
           "test_sha256": (DATA / "test.sha256").read_text().split()[0], "rows": len(rows),
           "seconds": round(time.perf_counter() - t0, 1), "metrics": metrics, "items": items}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    path = a.out_dir / f"endpoint_{name}_{a.mode}.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if a.mode == "direct":
        m = metrics["all"]
        keys = ("schema_valid", "behaviour_accuracy", "answer_accuracy", "correct_abstention", "citation_validity",
                "rouge_l_f1_answer_rows")
        print(", ".join(f"{k} {m[k]['mean']} {m[k]['ci95']}" for k in keys if k in m),
              f"| http errors {metrics['http_errors']}")
    else:
        print(", ".join(f"{k} {v['mean']}" for k, v in metrics.items() if isinstance(v, dict) and v))
    print(f"{len(rows)} rows in {out['seconds']} s -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
