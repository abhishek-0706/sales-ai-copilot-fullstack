"""Retrieval metrics (hit@1/3/6, MRR@10) through the app's own HybridStore, plus paired comparisons.

Run from backend/ with the app venv:
  python -m finetune.eval_retrieval --embedders hash,fastembed,base-onnx,finetuned
  python -m finetune.eval_retrieval --embedders fastembed,finetuned --queries finetune/data/val.jsonl --split val
  python -m finetune.eval_retrieval --compare fastembed finetuned
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

from app.config import BACKEND_DIR
from app.data_store import DataStore
from app.embeddings import FineTunedEmbedder, make_embedder
from app.models import Role, User
from app.security import can_read
from app.vectorstore import HybridStore

HERE = Path(__file__).resolve().parent
TEST = HERE / "data" / "test_queries.json"
TEST_HASH = HERE / "data" / "test_queries.sha256"
RESULTS = HERE / "results"
MODELS = BACKEND_DIR / ".models"
BASE_ONNX_DIR = MODELS / "bge-small-en-v1.5-base-onnx"
HOLDOUT_DIR = MODELS / "bge-small-en-v1.5-sales-ft-holdout"
KS = (1, 3, 6)
MODES = ("vector", "hybrid", "keyword")
TEST_PERSONAS = {"t-ae-na": User(id="t-ae-na", name="Test AE NA-East", role=Role.AE, territories=["NA-East"])}


def load_test() -> list[dict]:
    raw = TEST.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(raw).hexdigest() != TEST_HASH.read_text().split()[0]:
        sys.exit("test_queries.json no longer matches the hash recorded when it was frozen")
    return json.loads(raw)["queries"]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_embedder(key: str):
    if key == "base-onnx":
        return FineTunedEmbedder(BASE_ONNX_DIR, name="onnx:bge-small-en-v1.5-base-export")
    if key == "finetuned-holdout":
        return FineTunedEmbedder(HOLDOUT_DIR, name="finetuned:bge-small-en-v1.5-sales-ft-holdout")
    return make_embedder(key, 384, MODELS)


def first_rank(index: HybridStore, user: User, question: str, relevant: set[str], mode: str) -> int | None:
    hits = index.search(user, question, k=10, mode=mode)
    return next((i for i, h in enumerate(hits, start=1) if h.evidence.source_id in relevant), None)


def metrics(ranks: list[int | None]) -> dict:
    n = len(ranks)
    out: dict = {f"hit@{k}": round(sum(1 for r in ranks if r and r <= k) / n, 3) for k in KS}
    out["mrr@10"] = round(sum(1 / r for r in ranks if r) / n, 3)
    out["n"] = n
    return out


def evaluate(key: str, queries: list[dict], split: str, settings: list[str]) -> dict:
    store = DataStore(BACKEND_DIR / "data")
    chunks = store.chunks()
    by_id = {c.source_id: c for c in chunks}
    users = {**store.users, **TEST_PERSONAS}
    territories = sorted({c.territory for c in chunks})
    all_access = User(id="t-all", name="All access (test)", role=Role.MANAGER, territories=territories)
    for q in queries:
        unknown = set(q["relevant"]) - by_id.keys()
        if unknown:
            sys.exit(f"{q['id']}: unknown chunk ids {unknown}")
        if "filtered" in settings and not any(
                can_read(users[q["user"]], territory=by_id[s].territory, record_type=by_id[s].source_type,
                         sensitivity=by_id[s].sensitivity) for s in q["relevant"]):
            sys.exit(f"{q['id']}: user {q['user']} cannot see any relevant chunk")

    embedder = build_embedder(key)
    index = HybridStore(embedder)
    index.build(chunks)
    out = {"embedder": embedder.name, "key": key, "split": split, "n_queries": len(queries),
           "n_chunks": len(chunks), "metrics": {}, "by_style": {}, "ranks": {}}
    for setting in settings:
        for mode in MODES:
            label = f"{setting}/{mode}"
            ranks = [first_rank(index, users[q["user"]] if setting == "filtered" else all_access, q["question"],
                                set(q["relevant"]), mode) for q in queries]
            out["metrics"][label] = metrics(ranks)
            out["ranks"][label] = {q["id"]: r for q, r in zip(queries, ranks)}
            styles = sorted({q["style"] for q in queries if "style" in q})
            out["by_style"][label] = {s: metrics([r for q, r in zip(queries, ranks) if q.get("style") == s])
                                      for s in styles}
    return out


def print_table(results: list[dict]) -> None:
    print(f"{'embedder':40s} {'setting/mode':20s} {'hit@1':>6s} {'hit@3':>6s} {'hit@6':>6s} {'MRR@10':>7s}  n")
    for res in results:
        for label, m in res["metrics"].items():
            print(f"{res['embedder']:40s} {label:20s} {m['hit@1']:6.3f} {m['hit@3']:6.3f} {m['hit@6']:6.3f}"
                  f" {m['mrr@10']:7.3f}  {m['n']}")


def sign_test_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def query_accounts() -> dict[str, set[str]]:
    groups = {c.source_id: c.account_id or c.record_id for c in DataStore(BACKEND_DIR / "data").chunks()}
    return {q["id"]: {groups[r] for r in q["relevant"]} for q in load_test()}


def compare(split: str, a_key: str, b_key: str, accounts: set[str], n_boot: int = 10_000) -> None:
    a = json.loads((RESULTS / f"{split}_{a_key}.json").read_text(encoding="utf-8"))
    b = json.loads((RESULTS / f"{split}_{b_key}.json").read_text(encoding="utf-8"))
    rng = random.Random(0)
    stats = {"hit@1": lambda r: 1.0 if r and r <= 1 else 0.0, "hit@3": lambda r: 1.0 if r and r <= 3 else 0.0,
             "hit@6": lambda r: 1.0 if r and r <= 6 else 0.0, "mrr@10": lambda r: 1 / r if r else 0.0}
    subset = f" on questions about {sorted(accounts)}" if accounts else ""
    print(f"paired comparison on {split}{subset}: {b['embedder']} vs {a['embedder']} (difference B - A with 95% CI"
          f" from {n_boot} paired bootstrap resamples; exact sign test on questions whose rank changed)")
    report = {"a": a["embedder"], "b": b["embedder"], "split": split, "accounts": sorted(accounts), "labels": {}}
    qa = query_accounts() if accounts else {}
    for label in b["ranks"]:
        if label.endswith("keyword"):
            continue
        ids = [i for i in b["ranks"][label] if not accounts or qa[i] & accounts]
        ra, rb = a["ranks"][label], b["ranks"][label]
        better = sum(1 for i in ids if (rb[i] or 99) < (ra[i] or 99))
        worse = sum(1 for i in ids if (rb[i] or 99) > (ra[i] or 99))
        row = {"n": len(ids), "better": better, "worse": worse, "sign_test_p": round(sign_test_p(better, worse), 3)}
        parts = []
        for name, f in stats.items():
            diffs = [f(rb[i]) - f(ra[i]) for i in ids]
            boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n_boot))
            lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
            va, vb = sum(f(ra[i]) for i in ids) / len(ids), sum(f(rb[i]) for i in ids) / len(ids)
            row[name] = {"a": round(va, 3), "b": round(vb, 3), "diff": round(vb - va, 3),
                         "ci95": [round(lo, 3), round(hi, 3)]}
            parts.append(f"{name} {va:.3f}->{vb:.3f} ({vb - va:+.3f} [{lo:+.3f}, {hi:+.3f}])")
        report["labels"][label] = row
        print(f"  {label:18s} n={len(ids)} better {better} worse {worse} p={row['sign_test_p']:.3f} | "
              + " | ".join(parts))
    suffix = "_" + "-".join(sorted(accounts)) if accounts else ""
    out = RESULTS / f"compare_{split}_{a_key}_vs_{b_key}{suffix}.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedders", default="hash,fastembed")
    ap.add_argument("--queries", type=Path, help="jsonl split (train/val); default is the frozen test set")
    ap.add_argument("--split", default="test")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--accounts", default="", help="with --compare: only test questions about these accounts")
    args = ap.parse_args()
    if args.compare:
        compare(args.split, *args.compare, accounts={a for a in args.accounts.split(",") if a})
        return 0
    if args.queries:
        queries, settings = load_jsonl(args.queries), ["unfiltered"]
    else:
        queries, settings = load_test(), ["filtered", "unfiltered"]
    RESULTS.mkdir(exist_ok=True)
    results = []
    for key in args.embedders.split(","):
        res = evaluate(key, queries, args.split, settings)
        (RESULTS / f"{args.split}_{key}.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        results.append(res)
    print_table(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
