"""Scores answers to the app's Ask task with the app's own validators.

Used by make_dataset.py (offline-generator reference and a target self-check), eval_endpoint.py and the notebook,
which carries a verbatim copy (build_notebook.py swaps the "app bindings" block for the source of those functions).

Per test row:
- schema valid: the raw output parses with app.llm.extract_json and validates as app.models.Answer, as returned;
- app accepts: the output passes the app's citation filter (Agent._filter_citations) and Answer validation, so the
  app would use it without a retry;
- abstained: a valid answer with an empty claims list, which is what the app's task instruction asks for when the
  evidence does not hold the answer; behaviour is correct when the model abstains exactly on the abstain rows;
- fact: the expected fact appears in the answer and claim text (short facts: one of the accepted surface forms as a
  substring; long facts: at least LONG_FACT_RECALL of the fact's content words, using the app's own `terms`);
- citation validity: every cited id was in the evidence supplied with the prompt and the acting user may read it,
  checked with the independent policy oracle in eval/run_eval.py (not app.security);
- citation hit: at least one cited id holds the fact;
- groundedness: the app's lexical proxy (Agent._groundedness) on the claims and their cited evidence;
- ROUGE-L F1 of the answer text against the target answer text.
"""
from __future__ import annotations

import json
import re
from types import SimpleNamespace

import numpy as np

# >>> app bindings (build_notebook.py replaces this block with verbatim copies of these functions)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.agent import GROUNDED_THRESHOLD, Agent  # noqa: E402,F401
from app.llm import extract_json  # noqa: E402
from app.models import Answer  # noqa: E402
from app.offline import terms  # noqa: E402
from eval.run_eval import allowed  # noqa: E402

_filter_citations = Agent._filter_citations
_groundedness = Agent._groundedness
# <<< app bindings

ABSTAIN_TEXT = "I could not find this in the records you are allowed to access."
LONG_FACT_RECALL = 0.6
BOOTSTRAP = 10_000

# name: (per-row field, rows it is defined on: all rows, rows expecting an answer, rows expecting an abstention,
# or rows where the field is not None)
METRICS = {
    "schema_valid": ("valid", "all"),
    "app_accepts_without_retry": ("app_accepts", "all"),
    "behaviour_accuracy": ("behaviour_ok", "all"),
    "answer_accuracy": ("answer_ok", "answer"),
    "correct_abstention": ("abstain_ok", "abstain"),
    "false_abstention": ("false_abstain", "answer"),
    "hidden_fact_in_output": ("hidden_fact_leak", "abstain_with_fact"),
    "citation_hit": ("citation_hit", "answer"),
    "rows_with_all_citations_valid": ("all_citations_valid", "defined"),
    "groundedness_proxy": ("groundedness", "defined"),
    "rouge_l_f1_answer_rows": ("rouge_l", "answer"),
    "rouge_l_f1_all_rows": ("rouge_l", "all"),
}


class _Ctx:
    """Stands in for agent.RunContext: the citation filter only records guardrail events on it."""

    def event(self, *args, **kwargs) -> None:
        pass


class _CitationFilter:
    _filter_citations = _filter_citations


def parse_strict(raw: str) -> tuple[dict | None, str | None]:
    try:
        return Answer.model_validate(extract_json(raw)).model_dump(), None
    except (ValueError, TypeError) as exc:  # json.JSONDecodeError and pydantic.ValidationError are ValueErrors
        return None, type(exc).__name__


def app_accepts(raw: str, evidence_ids: list[str]) -> bool:
    try:
        data = _CitationFilter()._filter_citations(_Ctx(), extract_json(raw), set(evidence_ids))
        Answer.model_validate(data)
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def fact_found(text: str, groups: list[list[str]], kind: str) -> bool:
    """groups: every group must match; a group matches when one of its surface forms does."""
    if not groups:
        return False
    if kind == "long":
        words = terms(text)
        return all(max(len(terms(a) & words) / max(len(terms(a)), 1) for a in g) >= LONG_FACT_RECALL
                   for g in groups)
    t = _norm(text)
    return all(any(_norm(a) in t for a in g) for g in groups)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def rouge_l(candidate: str, reference: str) -> float:
    a, b = _tokens(candidate), _tokens(reference)
    if not a or not b:
        return float(a == b)
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return round(2 * p * r / (p + r), 4)


def score_row(row: dict, raw: str, record_meta: dict) -> dict:
    obj, err = parse_strict(raw)
    evidence_ids = [e["source_id"] for e in row["evidence"]]
    expect_abstain = row["expected"] == "abstain"
    target = json.loads(row["target"])
    out = {"id": row["id"], "valid": obj is not None, "error": err, "app_accepts": app_accepts(raw, evidence_ids),
           "abstained": None, "behaviour_ok": False, "answer_ok": None, "abstain_ok": None, "false_abstain": None,
           "hidden_fact_leak": None, "citation_hit": None, "cited": 0, "cited_valid": 0,
           "all_citations_valid": None, "groundedness": None, "rouge_l": 0.0}
    text = raw if obj is None else " ".join([obj["answer"], *(c["text"] for c in obj["claims"])])
    if expect_abstain:
        out["abstain_ok"] = False
        if row["facts"]:
            out["hidden_fact_leak"] = fact_found(text, row["facts"], row["fact_kind"])
    else:
        out["answer_ok"] = out["citation_hit"] = out["false_abstain"] = False
    if obj is None:
        return out
    claims = obj["claims"]
    abstained = not claims
    cited = [s for c in claims for s in c["source_ids"]]
    valid_ids = set(evidence_ids)
    out.update(abstained=abstained, behaviour_ok=abstained == expect_abstain, cited=len(cited),
               cited_valid=sum(1 for s in cited if s in valid_ids and s.split(":")[0] in record_meta
                               and allowed(row["user"], s, record_meta)),
               rouge_l=rouge_l(obj["answer"], target["answer"]))
    if cited:
        out["all_citations_valid"] = out["cited_valid"] == len(cited)
        evs = [SimpleNamespace(**e) for e in row["evidence"]]
        out["groundedness"] = _groundedness([(c["text"], c["source_ids"]) for c in claims], evs)[0]
    if expect_abstain:
        out["abstain_ok"] = abstained
    else:
        out["false_abstain"] = abstained
        out["answer_ok"] = not abstained and fact_found(text, row["facts"], row["fact_kind"])
        out["citation_hit"] = bool(set(cited) & set(row["supporting_ids"]))
    return out


def _defined(row: dict, s: dict, scope: str, field: str) -> bool:
    if scope == "all":
        return True
    if scope == "answer":
        return row["expected"] == "answer"
    if scope == "abstain":
        return row["expected"] == "abstain"
    if scope == "abstain_with_fact":
        return row["expected"] == "abstain" and bool(row["facts"])
    return s[field] is not None


def _ci(values: np.ndarray, seed: int = 0, n_boot: int = BOOTSTRAP) -> list[float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def _ratio_ci(num: np.ndarray, den: np.ndarray, seed: int = 0, n_boot: int = BOOTSTRAP) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), (n_boot, len(num)))
    d = den[idx].sum(axis=1)
    ratios = np.where(d > 0, num[idx].sum(axis=1) / np.maximum(d, 1), np.nan)
    return [round(float(np.nanpercentile(ratios, 2.5)), 4), round(float(np.nanpercentile(ratios, 97.5)), 4)]


def summarize(rows: list[dict], scores: list[dict], seed: int = 0) -> dict:
    """Mean and bootstrap 95% CI (percentile, resampling rows) for every metric on these rows."""
    out: dict = {"n": len(rows), "n_expect_answer": sum(r["expected"] == "answer" for r in rows),
                 "n_expect_abstain": sum(r["expected"] == "abstain" for r in rows)}
    for name, (field, scope) in METRICS.items():
        vals = [float(s[field]) for r, s in zip(rows, scores) if _defined(r, s, scope, field)]
        if vals:
            arr = np.asarray(vals)
            out[name] = {"mean": round(float(arr.mean()), 4), "ci95": _ci(arr, seed), "n": len(vals)}
    num = np.asarray([s["cited_valid"] for s in scores], dtype=float)
    den = np.asarray([s["cited"] for s in scores], dtype=float)
    if den.sum():
        out["citation_validity"] = {"mean": round(float(num.sum() / den.sum()), 4), "ci95": _ratio_ci(num, den, seed),
                                    "n_cited": int(den.sum())}
    return out


def compare(rows: list[dict], a: list[dict], b: list[dict], seed: int = 0) -> dict:
    """b minus a on the same rows (paired bootstrap 95% CI) for every metric defined on all rows of its scope."""
    out = {}
    for name, (field, scope) in METRICS.items():
        if scope == "defined":
            continue
        pairs = [(float(sa[field]), float(sb[field])) for r, sa, sb in zip(rows, a, b) if _defined(r, sa, scope, field)]
        if pairs:
            diff = np.asarray([y - x for x, y in pairs])
            out[name] = {"diff": round(float(diff.mean()), 4), "ci95": _ci(diff, seed), "n": len(pairs)}
    return out


def subsets(rows: list[dict]) -> dict[str, list[int]]:
    """Row indices per reporting subset."""
    groups: dict[str, list[int]] = {"all": list(range(len(rows)))}
    for i, r in enumerate(rows):
        groups.setdefault(f"wording={r['wording']}", []).append(i)
        groups.setdefault(f"expected={r['expected']}", []).append(i)
        if r["expected"] == "abstain":
            groups.setdefault(f"abstain_reason={r['abstain_reason']}", []).append(i)
    return groups


def report(rows: list[dict], scores: list[dict], seed: int = 0) -> dict:
    return {name: summarize([rows[i] for i in idx], [scores[i] for i in idx], seed)
            for name, idx in subsets(rows).items()}
