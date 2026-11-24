"""Builds synthetic (question, relevant chunks) training data from the CRM corpus.

Two sources: an LLM writes questions per account (or product group) and labels every chunk that answers
each one; templates add short keyword-style questions with known answers. Dropped and reported: questions
that give no cue to which account they are about, questions close to the frozen test set or the golden
eval set by string similarity, and paraphrases of a test question (base-model cosine >= 0.8) that share
its answer chunk.

Run from backend/ with the app venv. LLM calls need LLM_BASE_URL, LLM_MODEL and LLM_API_KEY (any
OpenAI-compatible endpoint); generated questions are cached in data/llm_questions.jsonl and reused.
  python -m finetune.make_training_data [--regenerate]
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx
import numpy as np

from app.config import BACKEND_DIR
from app.data_store import DataStore
from app.embeddings import tokenize
from app.models import Evidence

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
LLM_CACHE = DATA / "llm_questions.jsonl"
SEED = 42
VAL_FRACTION = 0.1
INJECTION_NOTE = "ACT-0046"
MAX_RATIO, MAX_JACCARD = 0.7, 0.5
MAX_COSINE = 0.8

SYSTEM = "You write realistic search queries for a sales team's CRM assistant. Reply with JSON only."
PROMPT = """Records ({label}), each with its id in brackets:
{records}

Write {n} different questions a salesperson might type into the CRM assistant. Each question must be answered by one or more of these records.
Rules:
- Cover every record at least twice.
- Mix styles: about a third name the {entity}; about a third describe it indirectly (by industry, city, product interest or situation) without its name; the rest are short, informal or keyword-like, as people type in a hurry.
- Prefer synonyms and everyday wording over copying phrases from the records.
- For each question list ALL record ids whose text contains the answer (not records that only mention the topic).
Return {{"questions": [{{"q": "...", "ids": ["..."]}}]}}"""


def passage(c: Evidence) -> str:
    return f"{c.title}. {c.text}"


def group_of(c: Evidence) -> str:
    return c.account_id or c.record_id


def kind_of(c: Evidence) -> str:
    if c.source_type != "activity":
        return c.source_type
    if c.title.startswith("Manager Note"):
        return "discount"
    if c.text.startswith("Competitive note"):
        return "competitor"
    if c.text.startswith("Deal risk"):
        return "risk"
    return c.title.split()[0].lower()


def call_llm(client: httpx.Client, prompt: str) -> list[dict]:
    body = {"model": os.environ["LLM_MODEL"], "temperature": 0.9, "max_tokens": 4000, "reasoning_effort": "low",
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
    for attempt in range(10):
        r = client.post("/chat/completions", json=body)
        if r.status_code == 429:
            wait = float(r.headers.get("retry-after", "20"))
            print(f"  rate limited, waiting {wait:.0f}s", flush=True)
            time.sleep(wait + 1)
            continue
        if r.status_code >= 500:
            time.sleep(10 * (attempt + 1))
            continue
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        try:
            return json.loads(content)["questions"]
        except (json.JSONDecodeError, KeyError):
            print("  unparseable reply, retrying", flush=True)
    raise RuntimeError("LLM call failed after retries")


def llm_bundles(chunks: list[Evidence], accounts: dict) -> list[tuple[str, str, list[Evidence], int]]:
    by_group: dict[str, list[Evidence]] = defaultdict(list)
    for c in chunks:
        if c.source_id != INJECTION_NOTE:
            by_group[group_of(c)].append(c)
    bundles = [(g, f"customer account {accounts[g].name}", cs, 36) for g, cs in by_group.items() if g in accounts]
    products = [c for g, cs in by_group.items() if g not in accounts for c in cs]
    half = len(products) // 2
    bundles += [("products-a", "product documentation", products[:half], 30),
                ("products-b", "product documentation", products[half:], 30)]
    return bundles


def generate_llm_questions(chunks: list[Evidence], accounts: dict) -> list[dict]:
    headers = {"Authorization": f"Bearer {os.environ['LLM_API_KEY']}", "User-Agent": "sales-ai-finetune/1.0"}
    rows = []
    with httpx.Client(base_url=os.environ["LLM_BASE_URL"], headers=headers, timeout=120) as client:
        for key, label, cs, n in llm_bundles(chunks, accounts):
            records = "\n".join(f"[{c.source_id}] {passage(c)}" for c in cs)
            entity = "account" if key.startswith("ACC") else "product"
            print(f"LLM: {key} ({len(cs)} records)", flush=True)
            valid = {c.source_id for c in cs}
            for item in call_llm(client, PROMPT.format(label=label, records=records, n=n, entity=entity)):
                ids = [i for i in item.get("ids", []) if i in valid]
                if isinstance(item.get("q"), str) and item["q"].strip() and ids:
                    rows.append({"question": item["q"].strip(), "relevant": ids, "source": "llm", "bundle": key})
    return rows


def template_questions(chunks: list[Evidence], accounts: dict) -> list[dict]:
    kinds: dict[str, dict[str, str]] = defaultdict(dict)
    for c in chunks:
        if c.account_id and c.source_id != INJECTION_NOTE:
            kinds[c.account_id][kind_of(c)] = c.source_id
    templates = [
        ("{name} use case", ["account", "call", "opportunity", "email", "meeting"]),
        ("where is {name} based", ["account"]), ("how many employees at {short}", ["account"]),
        ("who owns the {name} account", ["account"]), ("{short} deal size", ["opportunity"]),
        ("{name} opportunity stage", ["opportunity"]), ("{short} close date", ["opportunity"]),
        ("win probability {name}", ["opportunity"]), ("products quoted to {name}", ["opportunity"]),
        ("{name} competitor", ["opportunity", "competitor"]),
        ("what is {name} comparing us with and why", ["competitor"]),
        ("{name} main pain point", ["call"]), ("who did we meet on the {short} discovery call", ["call", "email"]),
        ("{name} demo feedback", ["demo"]), ("open questions from {name}", ["demo", "email", "meeting"]),
        ("{name} deal risk", ["risk", "email"]), ("next step with {name}", ["risk", "opportunity", "meeting"]),
        ("{name} approved discount", ["discount"]), ("latest email from {name}", ["email"]),
        ("when is the next {short} meeting", ["meeting"]), ("{short} meeting agenda", ["meeting"]),
    ]
    rows = []
    for acc_id, ks in kinds.items():
        name = accounts[acc_id].name
        for tpl, wanted in templates:
            ids = [ks[k] for k in wanted if k in ks]
            if ids and wanted[0] in ks:
                rows.append({"question": tpl.format(name=name, short=name.split()[0]), "relevant": ids,
                             "source": "template", "bundle": acc_id})
    for c in chunks:
        if c.source_type == "product_doc":
            product, section = c.title.rsplit(" - ", 1)
            for tpl in ("{p} {s}", "what does the {p} doc say about {s}"):
                rows.append({"question": tpl.format(p=product, s=section.lower()), "relevant": [c.source_id],
                             "source": "template", "bundle": c.record_id})
    return rows


def clauses(text: str) -> list[str]:
    parts = re.split(r"[.;:!?'()]\s*|,\s+", text.lower())
    return [p.strip() for p in parts if len(p.split()) >= 4]


def related_chunks(chunks: list[Evidence]) -> dict[str, set[str]]:
    """Chunks in the same group that repeat a clause of each other (templated emails and meetings
    restate activity notes); excluded as hard negatives so a duplicate answer is never a negative."""
    texts = {c.source_id: passage(c).lower() for c in chunks}
    groups: dict[str, list[str]] = defaultdict(list)
    for c in chunks:
        groups[group_of(c)].append(c.source_id)
    rel: dict[str, set[str]] = defaultdict(set)
    for ids in groups.values():
        for a in ids:
            for b in ids:
                if a != b and any(cl in texts[b] for cl in clauses(texts[a])):
                    rel[a].add(b)
                    rel[b].add(a)
    return rel


def stems(text: str) -> set[str]:
    return {t[:6] for t in tokenize(text) if len(t) > 2}


def account_signatures(chunks: list[Evidence]) -> dict[str, set[str]]:
    """Word stems that occur in one account's records and no other account's."""
    vocab: dict[str, set[str]] = defaultdict(set)
    for c in chunks:
        if c.account_id:
            vocab[c.account_id] |= stems(passage(c))
    return {a: v - set().union(*(w for b, w in vocab.items() if b != a)) for a, v in vocab.items()}


def jaccard(a: str, b: str) -> float:
    x, y = set(tokenize(a)), set(tokenize(b))
    return len(x & y) / len(x | y) if x | y else 0.0


def semantic_neighbours(questions: list[str], test: list[dict]) -> list[tuple[float, dict]]:
    """Nearest test question for each question by base bge-small cosine similarity."""
    from app.embeddings import FastEmbedEmbedder

    emb = FastEmbedEmbedder(BACKEND_DIR / ".models")
    t = np.vstack([emb.embed_query(q["question"]) for q in test])
    sims = np.vstack([emb.embed_query(q) for q in questions]) @ t.T
    return [(float(s.max()), test[int(s.argmax())]) for s in sims]


def closest(q: str, refs: list[str]) -> tuple[float, float, str]:
    best = max(refs, key=lambda r: difflib.SequenceMatcher(None, q.lower(), r.lower()).ratio())
    ratio = difflib.SequenceMatcher(None, q.lower(), best.lower()).ratio()
    jac = max(jaccard(q, r) for r in refs)
    return ratio, jac, best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true", help="call the LLM even if a cache exists")
    args = ap.parse_args()
    store = DataStore(BACKEND_DIR / "data")
    chunks = store.chunks()
    DATA.mkdir(exist_ok=True)

    with (DATA / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"id": c.source_id, "group": group_of(c), "text": passage(c)}) + "\n")

    if args.regenerate or not LLM_CACHE.exists():
        llm_rows = generate_llm_questions(chunks, store.accounts)
        LLM_CACHE.write_text("".join(json.dumps(r) + "\n" for r in llm_rows), encoding="utf-8")
    llm_rows = [json.loads(line) for line in LLM_CACHE.read_text(encoding="utf-8").splitlines()]
    rows = llm_rows + template_questions(chunks, store.accounts)

    test_qs = [q["question"] for q in json.loads((DATA / "test_queries.json").read_text(encoding="utf-8"))["queries"]]
    golden_qs = [q["question"] for q in json.loads((BACKEND_DIR / "eval" / "golden.json").read_text(encoding="utf-8"))]
    refs = test_qs + golden_qs
    group = {c.source_id: group_of(c) for c in chunks}
    related = related_chunks(chunks)

    signatures = account_signatures(chunks)
    seen, kept, dropped, ambiguous = set(), [], [], []
    for r in rows:
        norm = " ".join(r["question"].lower().split())
        if norm in seen:
            continue
        seen.add(norm)
        acc = r["bundle"] if r["bundle"] in signatures else None
        if acc and not stems(r["question"]) & signatures[acc]:
            ambiguous.append(r["question"])
            continue
        ratio, jac, ref = closest(r["question"], refs)
        r.update(max_ratio=round(ratio, 3), max_jaccard=round(jac, 3), nearest_ref=ref)
        (dropped if ratio >= MAX_RATIO or jac >= MAX_JACCARD else kept).append(r)

    test = json.loads((DATA / "test_queries.json").read_text(encoding="utf-8"))["queries"]
    paraphrases, survivors = [], []
    for r, (sim, t) in zip(kept, semantic_neighbours([r["question"] for r in kept], test)):
        r.update(max_cosine_to_test=round(sim, 3), nearest_test=t["id"],
                 shares_test_answer=bool(set(r["relevant"]) & set(t["relevant"])))
        (paraphrases if sim >= MAX_COSINE and r["shares_test_answer"] else survivors).append(r)
    cosine_profile = {th: sum(1 for r in kept if r["max_cosine_to_test"] >= th and r["shares_test_answer"])
                      for th in (0.75, 0.8, 0.85, 0.9)}
    kept = survivors

    rng = random.Random(SEED)
    rng.shuffle(kept)
    n_val = round(len(kept) * VAL_FRACTION)
    for i, r in enumerate(kept):
        r["id"] = f"{'v' if i < n_val else 'tr'}{i:04d}"
        r["groups"] = sorted({group[s] for s in r["relevant"]})
        r["related"] = sorted(set().union(*(related[s] for s in r["relevant"])) - set(r["relevant"]))
    val, train = kept[:n_val], kept[n_val:]
    for name, part in (("train", train), ("val", val)):
        (DATA / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in part), encoding="utf-8")

    near = sorted(kept, key=lambda r: -r["max_ratio"])[:10]
    report = {
        "seed": SEED, "thresholds": {"difflib_ratio": MAX_RATIO, "token_jaccard": MAX_JACCARD},
        "generated": {"llm": len(llm_rows), "template": len(rows) - len(llm_rows)},
        "after_dedup": len(kept) + len(dropped) + len(ambiguous),
        "dropped_ambiguous_no_account_cue": len(ambiguous),
        "dropped_near_test_or_golden": len(dropped),
        "dropped_nearest_is_test": sum(1 for r in dropped if r["nearest_ref"] in test_qs),
        "dropped_semantic_paraphrase_of_test": len(paraphrases),
        "same_answer_questions_by_cosine_threshold": cosine_profile,
        "kept_max_cosine_same_answer": max((r["max_cosine_to_test"] for r in kept if r["shares_test_answer"]),
                                           default=0.0),
        "train_questions": len(train), "val_questions": len(val),
        "train_pairs": sum(len(r["relevant"]) for r in train), "val_pairs": sum(len(r["relevant"]) for r in val),
        "kept_max_ratio": max(r["max_ratio"] for r in kept), "kept_max_jaccard": max(r["max_jaccard"] for r in kept),
        "exact_test_matches_in_train_or_val": sum(1 for r in kept if r["question"].lower() in
                                                  {q.lower() for q in test_qs}),
        "dropped": [{k: r[k] for k in ("question", "nearest_ref", "max_ratio", "max_jaccard")} for r in dropped],
        "closest_kept": [{k: r[k] for k in ("question", "nearest_ref", "max_ratio", "max_jaccard")} for r in near],
        "semantic_dropped": [{"question": r["question"], "nearest_test": r["nearest_test"],
                              "cosine": r["max_cosine_to_test"]} for r in paraphrases],
        "ambiguous_examples": ambiguous[:30],
    }
    (DATA / "leakage_report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    lists = ("dropped", "closest_kept", "semantic_dropped", "ambiguous_examples")
    print(json.dumps({k: v for k, v in report.items() if k not in lists}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
