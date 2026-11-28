"""Supervised fine-tuning data for the app's grounded-answer (Ask) task, built with the app's own pipeline.

Every example is one seller question about one account, asked as one persona. The question goes through
`Agent.ask` with the deterministic offline provider, exactly as in the app: rule planner, tools and hybrid
retrieval run as that persona (the access predicate is inside the SQL), suspicious evidence is quarantined and
personal data is redacted. The request the app then sends to a model for the `answer` task (system prompt, user
prompt, redacted evidence) is captured unchanged and paired with a deterministic target in the app's `Answer`
schema, built from the structured record that holds the fact:

- if the fact is in the evidence the persona received, the target states it and cites the evidence ids that hold
  it (with PII placeholders such as [PERSON_1] where the evidence has them);
- otherwise the target abstains (the app's own wording, empty claims list), and the reason is recorded: the
  persona may not read the record (role, sensitivity or territory), the app did not retrieve it, or no record
  holds the fact at all.

Splits are by account, so test accounts are never seen in training. One question wording per intent is held out
for the test set only. The test file's SHA-256 is written next to it; a rebuild that would change it is refused
unless --refreeze is given.

  python llm_finetune/make_dataset.py           # writes llm_finetune/data/
  python llm_finetune/make_dataset.py --check   # rebuilds in memory, verifies data/ is byte-identical
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BACKEND = ROOT / "backend"
for p in (BACKEND, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import scoring  # noqa: E402
from app.agent import Agent  # noqa: E402
from app.config import Settings  # noqa: E402
from app.guardrails import SYSTEM_PROMPT, neutralise  # noqa: E402
from app.llm import LLMResult  # noqa: E402
from app.main import create_app  # noqa: E402
from app.models import Answer, Evidence, Role, User  # noqa: E402
from app.privacy import Redactor  # noqa: E402
from app.security import POLICIES, can_read  # noqa: E402
from eval.run_eval import record_meta  # noqa: E402

DATA = HERE / "data"
SEED = 20261127
TEST_HASH_FILE = DATA / "test.sha256"
GOLDEN = BACKEND / "eval" / "golden.json"
OOT_INTENTS = ("deal_stage", "next_meeting", "pain_point", "contact_by_title")
ABSTAIN = scoring.ABSTAIN_TEXT

# ---------------------------------------------------------------------------------------------- question intents
# Four wordings per intent; the last one is held out (test split only). {acc} = account name, {short} = its first
# word (the app's planner recognises both), {title} = a contact's job title.
PHRASINGS: dict[str, list[str]] = {
    "deal_stage": ["What stage is the {acc} deal in?", "Which sales stage has the {acc} opportunity reached?",
                   "Where does the {acc} opportunity sit in the pipeline?", "How far along is our deal with {short}?"],
    "deal_amount": ["How much is the {acc} opportunity worth?", "What is the deal amount for {acc}?",
                    "What is the value of the {acc} opportunity?", "What's the {short} deal size in dollars?"],
    "close_date": ["When is the {acc} deal expected to close?", "What is the close date of the {acc} opportunity?",
                   "By what date do we expect to close {acc}?", "What's the target signing date for the {short} deal?"],
    "win_probability": ["What is the win probability of the {acc} opportunity?",
                        "How likely are we to win the {acc} deal?",
                        "What probability is recorded for the {acc} opportunity?",
                        "What are the odds we win the {short} deal?"],
    "competitor": ["Who are we competing against at {acc}?", "Which other vendor is {acc} considering?",
                   "What competitor is recorded on the {acc} deal?",
                   "Which rival vendor is competing for the {short} deal?"],
    "deal_products": ["Which products are in the {acc} opportunity?", "What products does the {acc} deal include?",
                      "Which products are on the {acc} deal?", "What are we selling to {short} in this deal?"],
    "next_step": ["What is the next step on the {acc} deal?", "What did we agree to do next with {acc}?",
                  "What should happen next on the {acc} opportunity?", "What's the follow-up action we owe {short}?"],
    "pain_point": ["What problem is {acc} trying to solve?", "What is {acc}'s biggest pain point?",
                   "Which operational pain did {acc} describe on the discovery call?",
                   "What's hurting {short} the most today?"],
    "demo_feedback": ["How did {acc} react to the product demo?", "What did {acc} say after the demo?",
                      "What demo feedback is recorded for {acc}?",
                      "Did the {short} team like what they saw in the demo?"],
    "open_question": ["What open question did {acc} raise?", "What does {acc} still need clarity on?",
                      "Which unresolved question came up with {acc}?",
                      "What is {short} still unsure about before committing?"],
    "deal_risk": ["What is the main risk on the {acc} deal?", "What deal risk is recorded for {acc}?",
                  "What could block the {acc} opportunity?", "What worries {short} about moving ahead?"],
    "discount_guidance": ["What pricing guidance does the deal desk have for {acc}?",
                          "How much discount can we offer {acc}?", "What is the approved discount ceiling for {acc}?",
                          "How low can we go on price with {short}?"],
    "next_meeting": ["When is the next meeting with {acc}?", "When is our upcoming call with {acc} scheduled?",
                     "What is the date of the next {acc} meeting?", "When do we next sit down with {short}?"],
    "account_owner": ["Who owns the {acc} account?", "Which account executive is responsible for {acc}?",
                      "Who is the account owner for {acc}?", "Whose patch is {short} on?"],
    "account_location": ["Where is {acc} based?", "Which city is {acc} headquartered in?",
                         "In what city is {acc} located?", "Where would I fly to visit {short}?"],
    "employee_count": ["How many employees does {acc} have?", "How large is {acc} by headcount?",
                       "What is the employee count at {acc}?", "Roughly how many people work at {short}?"],
    "use_case": ["What use case is {acc} interested in?", "What does {acc} want to use AI for?",
                 "Which use case are we pursuing with {acc}?", "What is {short} hoping to build with us?"],
    "account_tier": ["What tier is the {acc} account?", "Which customer tier is {acc} in?",
                     "Is {acc} a Strategic, Enterprise or Commercial account?",
                     "How is {short} segmented in our account tiers?"],
    "contact_by_title": ["Who is the {title} at {acc}?", "Who holds the {title} role at {acc}?",
                         "What is the name of {acc}'s {title}?", "Which person at {short} is the {title}?"],
    "contact_email": ["What is the email address of {acc}'s {title}?", "How can I email the {title} at {acc}?",
                      "What email do we have for the {title} at {acc}?", "Give me the {title}'s email at {short}."],
    "revenue": ["What was {acc}'s annual revenue last year?", "How much revenue does {acc} make?",
                "What is {acc}'s yearly turnover?", "How big is {short}'s top line?"],
    "ceo": ["Who is the chief executive of {acc}?", "Who runs {acc} as CEO?", "What is the name of {acc}'s CEO?",
            "Who's at the very top of {short}'s org chart?"],
    "stock_ticker": ["What is {acc}'s stock ticker symbol?", "Under which ticker does {acc} trade?",
                     "Is {acc} listed on a stock exchange, and under what symbol?",
                     "What symbol would I search to find {short}'s share price?"],
    "erp_system": ["Which ERP system does {acc} run?", "What ERP software is {acc} using?",
                   "Is {acc} on SAP or Oracle for ERP?", "What back-office system does {short} use for finance?"],
}
HELD_OUT = 3
# Intents with no record behind them: the only correct answer is an abstention. NOT_IN_RECORDS_WORDS must not occur
# in any record of the account (checked at build time).
NOT_IN_RECORDS_WORDS = {"revenue": ["revenue", "turnover"], "ceo": ["chief executive", "ceo"],
                        "stock_ticker": ["ticker", "stock exchange"], "erp_system": ["erp", "sap", "oracle"]}


@dataclass
class Fact:
    intent: str
    key: str                       # unique per account, e.g. contact_by_title:CON-001-1
    match: list[str]               # raw strings that must all occur in one record for it to hold the fact
    accept: list[list[str]]        # scoring: every group must match one surface form
    kind: str                      # "short" (substring) or "long" (content-word recall)
    answer: Callable[[Callable[[str], str]], str]  # builds the answer text; argument redacts a raw value
    prefer: tuple[str, ...]        # evidence types to cite first
    fmt: dict = field(default_factory=dict)  # extra phrasing fields, e.g. {"title": ...}


def _after(prefix: str, text: str) -> str:
    m = re.search(re.escape(prefix) + r"\s*([^.]+)\.", text)
    if not m:
        raise ValueError(f"'{prefix}' not found in: {text}")
    return m.group(1).strip()


def _date_forms(iso_day: str) -> list[str]:
    from datetime import date
    d = date.fromisoformat(iso_day[:10])
    month = d.strftime("%B")
    return [iso_day[:10], f"{d.day} {month} {d.year}", f"{month} {d.day}, {d.year}", f"{month[:3]} {d.day}"]


def facts_for(store, acc) -> list[Fact]:
    opp = next(o for o in store.opportunities.values() if o.account_id == acc.id)
    acts = sorted((a for a in store.activities.values() if a.account_id == acc.id), key=lambda a: a.id)
    call = next(a for a in acts if a.text.startswith("Discovery call"))
    demo = next(a for a in acts if a.text.startswith("Product demo"))
    risk = next(a for a in acts if a.text.startswith("Deal risk:"))
    desk = next((a for a in acts if a.type == "manager_note"), None)
    meeting = min((m for m in store.meetings.values() if m.account_id == acc.id), key=lambda m: m.start)
    contacts = sorted((c for c in store.contacts.values() if c.account_id == acc.id), key=lambda c: c.id)
    amount = f"{opp.amount_usd:,}"
    products = [store.products[p].name for p in opp.products]
    n = acc.name
    F = Fact
    out = [
        F("deal_stage", "deal_stage", [f"stage {opp.stage},"], [[opp.stage]], "short",
          lambda r: f"The {opp.name} opportunity is at the {opp.stage} stage.", ("opportunity",)),
        F("deal_amount", "deal_amount", [f"amount USD {amount}"],
          [[f"USD {amount}", f"${amount}", amount, str(opp.amount_usd)]], "short",
          lambda r: f"The {opp.name} opportunity is worth USD {amount}.", ("opportunity",)),
        F("close_date", "close_date", [f"close date {opp.close_date}"], [_date_forms(opp.close_date)], "short",
          lambda r: f"The {opp.name} opportunity has a close date of {opp.close_date}.", ("opportunity",)),
        F("win_probability", "win_probability", [f"probability {opp.probability} percent"],
          [[f"{opp.probability} percent", f"{opp.probability}%", f"{opp.probability} per cent"]], "short",
          lambda r: f"The {opp.name} opportunity has a win probability of {opp.probability} percent.",
          ("opportunity",)),
        F("deal_products", "deal_products", products, [[p] for p in products], "short",
          lambda r: f"The {opp.name} opportunity includes {', '.join(products[:-1]) + ' and ' if len(products) > 1 else ''}"
                    f"{products[-1]}.", ("opportunity",)),
        F("next_step", "next_step", [opp.next_step], [[opp.next_step]], "long",
          lambda r: f"The agreed next step with {n} is to {opp.next_step}.", ("opportunity", "activity", "meeting")),
        F("pain_point", "pain_point", [_after("Main pain point:", call.text)],
          [[_after("Main pain point:", call.text)]], "long",
          lambda r: f"The main pain point at {n} is {_after('Main pain point:', call.text)}.", ("activity",)),
        F("demo_feedback", "demo_feedback", [_after("Feedback:", demo.text)], [[_after("Feedback:", demo.text)]],
          "long", lambda r: f"Demo feedback from {n}: {_after('Feedback:', demo.text)}.", ("activity",)),
        F("open_question", "open_question", [_after("Open question:", demo.text)],
          [[_after("Open question:", demo.text)]], "long",
          lambda r: f"The open question from {n} is {_after('Open question:', demo.text)}.",
          ("activity", "email", "meeting")),
        F("deal_risk", "deal_risk", [_after("Deal risk:", risk.text)], [[_after("Deal risk:", risk.text)]], "long",
          lambda r: f"Deal risk recorded for {n}: {_after('Deal risk:', risk.text)}.", ("activity", "email")),
        F("next_meeting", "next_meeting", [f"starts {meeting.start}"], [_date_forms(meeting.start)], "short",
          lambda r: f"The next meeting with {n}, '{meeting.title}', starts {meeting.start} and lasts "
                    f"{meeting.duration_minutes} minutes.", ("meeting",)),
        F("account_owner", "account_owner", [f"owned by {acc.owner_name}"], [[acc.owner_name]], "short",
          lambda r: f"{n} is owned by {acc.owner_name}.", ("account",)),
        F("account_location", "account_location", [f"based in {acc.city}"], [[acc.city]], "short",
          lambda r: f"{n} is based in {acc.city}.", ("account",)),
        F("employee_count", "employee_count", [f"{acc.employees:,} employees"],
          [[f"{acc.employees:,}", str(acc.employees)]], "short",
          lambda r: f"{n} has about {acc.employees:,} employees.", ("account",)),
        F("use_case", "use_case", [acc.use_case], [[acc.use_case]], "long",
          lambda r: f"{n} wants {acc.use_case}.", ("account", "activity", "meeting")),
        F("account_tier", "account_tier", [f"is a {acc.tier} {acc.industry} account"], [[acc.tier]], "short",
          lambda r: f"{n} is in the {acc.tier} tier.", ("account",)),
    ]
    if opp.competitor:
        comp = opp.competitor
        out.append(F("competitor", "competitor", [comp], [[comp]], "short",
                     lambda r: f"{n} is also evaluating {comp}.", ("activity", "opportunity")))
    else:
        out.append(F("competitor", "competitor", ["Competitor: none recorded"],
                     [["no competitor", "none recorded"]], "short",
                     lambda r: f"No competitor is recorded on the {opp.name} opportunity.", ("opportunity",)))
    if desk:
        guidance = desk.text.split(": ", 1)[1].rstrip(".")
        pct = re.search(r"(\d+) percent", guidance).group(1)
        out.append(F("discount_guidance", "discount_guidance", [guidance], [[f"{pct} percent", f"{pct}%"]], "short",
                     lambda r: f"The deal desk guidance for {n} is: {guidance}.", ("activity",)))
    for c in contacts:
        out.append(F("contact_by_title", f"contact_by_title:{c.id}", [c.name, c.title], [[c.name]], "short",
                     lambda r, c=c: f"The {c.title} at {n} is {r(c.name)}.", ("contact", "activity"),
                     {"title": c.title}))
    c0 = contacts[0]
    out.append(F("contact_email", f"contact_email:{c0.id}", [c0.email], [[c0.email]], "short",
                 lambda r: f"The email address of {r(c0.name)}, {c0.title} at {n}, is {r(c0.email)}.",
                 ("contact", "email"), {"title": c0.title}))
    for intent in NOT_IN_RECORDS_WORDS:
        out.append(F(intent, intent, [], [], "short", lambda r: ABSTAIN, ()))
    return out


# ---------------------------------------------------------------------------------------------- personas and split
def personas(store) -> dict[str, dict[str, User]]:
    """Per territory: an SDR, an AE and a sales manager. Real users from users.json where one exists; otherwise a
    dataset-only persona (id prefix ds-) with the same role, so every account is asked about by all three roles."""
    real = list(store.users.values())
    out: dict[str, dict[str, User]] = {}
    for terr in sorted({a.territory for a in store.accounts.values()}):
        out[terr] = {}
        for role in (Role.SDR, Role.AE, Role.MANAGER):
            u = next((u for u in real if u.role == role and terr in u.territories), None)
            out[terr][role.value] = u or User(id=f"ds-{role.value.lower()}-{terr.lower()}",
                                              name=f"Dataset {role.value} ({terr})", role=role, territories=[terr])
    return out


# Out-of-territory askers. Training uses a persona whose territory has no accounts, so its evidence holds only
# global product docs and no record of a validation or test account can enter a training prompt. Validation and
# test use real AEs from another territory, whose evidence holds other accounts' records (harder distractors).
OOT_TRAIN = User(id="ds-ae-latam", name="Dataset AE (LATAM)", role=Role.AE, territories=["LATAM"])
OOT_EVAL = {"APAC-West": "u-ae-02", "APAC-South": "u-ae-02", "EMEA-North": "u-ae-01", "NA-East": "u-ae-01"}


def split_accounts(store) -> dict[str, str]:
    """Seeded, before any training: test takes one account from each of APAC-West, APAC-South and EMEA-North;
    validation one from APAC-West and one from NA-East; the remaining seven are training accounts."""
    rng = random.Random(SEED)
    by_terr: dict[str, list[str]] = {}
    for a in sorted(store.accounts.values(), key=lambda a: a.id):
        by_terr.setdefault(a.territory, []).append(a.id)
    for ids in by_terr.values():
        rng.shuffle(ids)
    split = {aid: "train" for aid in store.accounts}
    for terr in ("APAC-West", "APAC-South", "EMEA-North"):
        split[by_terr[terr].pop()] = "test"
    for terr in ("APAC-West", "NA-East"):
        split[by_terr[terr].pop()] = "val"
    return split


# ---------------------------------------------------------------------------------------------- capture
class Capture:
    """Wraps the offline provider: keeps the last `answer` request and the offline generator's raw output."""

    def __init__(self, inner):
        self.inner, self.name, self.model = inner, inner.name, inner.model
        self.last = None

    def generate(self, req) -> LLMResult:
        res = self.inner.generate(req)
        if req.task == "answer":
            self.last = (req, res.text)
        return res


def build_app(tmp: Path):
    logging.getLogger("salesai").disabled = True
    settings = replace(Settings(), llm_provider="offline", embedder="hash", var_dir=tmp,
                       audit_log_path=tmp / "audit.jsonl")
    app = create_app(settings)
    agent: Agent = app.state.agent
    cap = Capture(agent.llm)
    agent.llm = agent.fallback = cap
    return agent, cap


def reproduce_redactor(agent: Agent, red_ev: list[Evidence], question: str) -> Redactor:
    """Rebuild the request's redaction mapping (same order as Agent._generate) and check it reproduces the
    captured evidence exactly, so placeholders in targets match what the model sees."""
    redactor = Redactor(agent.known_names)
    for got in red_ev:
        orig, _ = neutralise(agent.store.by_source_id(got.source_id))
        title, text = redactor.redact(orig.title), redactor.redact(orig.text)
        assert (title, text) == (got.title, got.text), f"redaction mismatch for {got.source_id}"
    redactor.redact(question)
    return redactor


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower())


def abstain_reason(user: User, fact: Fact, records: list[Evidence]) -> str:
    if not fact.match:
        return "not_in_records"
    holders = [e for e in records if all(_norm(m) in _norm(f"{e.title} {e.text}") for m in fact.match)]
    if not holders:
        raise ValueError(f"no record holds {fact.key}")
    if any(can_read(user, territory=e.territory, record_type=e.source_type, sensitivity=e.sensitivity)
           for e in holders):
        return "not_retrieved"
    if not any(e.territory == "GLOBAL" or e.territory in user.territories for e in holders):
        return "access_territory"
    if not any(e.source_type in POLICIES[user.role].record_types for e in holders):
        return "access_role"
    return "access_sensitivity"


def make_row(agent: Agent, cap: Capture, user: User, acc, fact: Fact, phrasing: int, split: str,
             records: list[Evidence]) -> dict:
    question = PHRASINGS[fact.intent][phrasing].format(acc=acc.name, short=acc.name.split()[0], **fact.fmt)
    cap.last = None
    agent.ask(user, question)
    req, offline_raw = cap.last
    redactor = reproduce_redactor(agent, req.evidence, question)
    red = redactor.redact
    support = []
    if fact.match:
        need = [_norm(red(m)) for m in fact.match]
        support = [e for e in req.evidence if e.account_id == acc.id and not e.quarantined
                   and all(m in _norm(f"{e.title} {e.text}") for m in need)]
        order = {t: i for i, t in enumerate(fact.prefer)}
        support.sort(key=lambda e: order.get(e.source_type, len(order)))
    if support:
        text = red(fact.answer(red))
        cite = [e.source_id for e in support[:2]]
        target = {"answer": text, "claims": [{"text": text, "source_ids": cite}]}
        expected, reason = "answer", None
    else:
        target = {"answer": ABSTAIN, "claims": []}
        expected, reason = "abstain", abstain_reason(user, fact, records)
    target_json = json.dumps(Answer.model_validate(target).model_dump(), ensure_ascii=False)
    evidence = [{"source_id": e.source_id, "source_type": e.source_type, "account_id": e.account_id,
                 "title": e.title, "text": e.text, "quarantined": e.quarantined} for e in req.evidence]
    return {
        "id": f"{split}-{acc.id}-{fact.key}-{user.role.value}-{user.id}",
        "split": split, "account_id": acc.id, "intent": fact.intent, "fact_key": fact.key,
        "phrasing": phrasing, "wording": "held_out" if phrasing == HELD_OUT else "seen",
        "user": {"id": user.id, "role": user.role.value, "territories": list(user.territories)},
        "question": question, "system": req.system, "prompt": req.prompt,
        "prompt_sha256": hashlib.sha256(req.prompt.encode()).hexdigest(),
        "evidence": evidence, "expected": expected, "abstain_reason": reason,
        # answer rows: the fact as the model sees it (placeholders for personal data); abstain rows: the real value
        # the model must not produce
        "facts": [[red(a) for a in g] for g in fact.accept] if support else fact.accept,
        "facts_restored": fact.accept,
        "fact_kind": fact.kind, "supporting_ids": [e.source_id for e in support],
        "target": target_json, "offline_generator_output": offline_raw,
    }


# ---------------------------------------------------------------------------------------------- build
def build() -> tuple[dict[str, list[dict]], dict]:
    with tempfile.TemporaryDirectory() as tmp:
        agent, cap = build_app(Path(tmp))
        store = agent.store
        split_of = split_accounts(store)
        teams = personas(store)
        users = {u["id"]: u for u in json.loads((BACKEND / "data" / "users.json").read_text(encoding="utf-8"))}
        all_records = store.chunks() + [store.contact_evidence(c) for c in store.contacts.values()]
        rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
        k = 0
        for acc in sorted(store.accounts.values(), key=lambda a: a.id):
            split = split_of[acc.id]
            records = [e for e in all_records if e.account_id == acc.id]
            for words in NOT_IN_RECORDS_WORDS.values():
                for w in words:
                    assert not any(re.search(rf"\b{w}\b", _norm(f"{e.title} {e.text}")) for e in records), (acc.id, w)
            for fact in facts_for(store, acc):
                for role in ("SDR", "AE", "MANAGER"):
                    phr = (HELD_OUT if k % 2 else k // 2 % 3) if split == "test" else k % 3
                    k += 1
                    rows[split].append(make_row(agent, cap, teams[acc.territory][role], acc, fact, phr, split, records))
                if fact.intent in OOT_INTENTS:
                    oot = OOT_TRAIN if split == "train" else agent.store.users[OOT_EVAL[acc.territory]]
                    phr = (HELD_OUT if k % 2 else k // 2 % 3) if split == "test" else k % 3
                    k += 1
                    rows[split].append(make_row(agent, cap, oot, acc, fact, phr, split, records))
        report = checks_and_report(rows, split_of, store, users)
    return rows, report


def _golden_overlap(rows: list[dict], golden: list[dict], store) -> list[dict]:
    """Rows whose target would teach a golden-eval answer: same account as a golden item and the target answer
    contains its expected fact (answer_contains) or a value a golden negative must not reveal (answer_excludes)."""
    def accounts(ids: list[str]) -> set[str]:
        return {e.account_id for e in map(store.by_source_id, ids) if e is not None and e.account_id}

    items = [(g["id"], accounts(g.get("expected_sources", []) + g.get("must_not_retrieve", [])),
              [s.lower() for s in g.get("answer_contains", []) + g.get("answer_excludes", [])]) for g in golden]
    out = []
    for r in rows:
        answer = json.loads(r["target"])["answer"].lower()
        for gid, accs, facts in items:
            hit = next((f for f in facts if f in answer), None)
            if r["account_id"] in accs and hit:
                out.append({"id": r["id"], "question": r["question"], "golden_id": gid, "golden_fact": hit})
                break
    return out


def checks_and_report(rows: dict[str, list[dict]], split_of: dict, store, users) -> dict:
    names = {a.id: a.name for a in store.accounts.values()}
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    eval_accounts = {a for a, s in split_of.items() if s != "train"}
    test_accounts = {a for a, s in split_of.items() if s == "test"}
    meta = record_meta()

    def acc_of(source_id: str) -> str | None:
        e = store.by_source_id(source_id)
        return e.account_id if e else None

    # 1. rows that would teach a golden-eval answer are dropped from train and validation, so eval/golden.json stays
    #    clean for an LLM-mode run with the fine-tuned model; test rows (never trained on) are only reported
    dropped_golden = {s: _golden_overlap(rows[s], golden, store) for s in ("train", "val")}
    for s, d in dropped_golden.items():
        ids = {x["id"] for x in d}
        rows[s] = [r for r in rows[s] if r["id"] not in ids]
    test_golden = _golden_overlap(rows["test"], golden, store)

    # 2. no record of a validation or test account in any training prompt (evidence or question)
    train_eval_records = [r["id"] for r in rows["train"] if any(acc_of(e["source_id"]) in eval_accounts
                                                                for e in r["evidence"])]
    train_eval_names = [r["id"] for r in rows["train"]
                        if any(names[a] in r["prompt"] or names[a].split()[0] in r["question"] for a in eval_accounts)]
    assert not train_eval_records and not train_eval_names, (train_eval_records[:3], train_eval_names[:3])
    val_test_records = [r["id"] for r in rows["val"] if any(acc_of(e["source_id"]) in test_accounts
                                                           for e in r["evidence"])]

    # 3. exact duplicates across splits; held-out wording only in test
    prompts = {s: {r["prompt_sha256"] for r in rows[s]} for s in rows}
    questions = {s: {r["question"].lower() for r in rows[s]} for s in rows}
    held_out_outside_test = [r["id"] for s in ("train", "val") for r in rows[s] if r["wording"] == "held_out"]
    assert not held_out_outside_test

    # 4. every target validates, cites only supplied evidence, is grounded by the app's own proxy, and the
    #    scorer gives it full marks (self-check of scoring.py); every abstention has no fact-holding evidence
    for s in rows:
        for r in rows[s]:
            t = json.loads(r["target"])
            ids = {e["source_id"] for e in r["evidence"]}
            assert all(x in ids for c in t["claims"] for x in c["source_ids"]), r["id"]
            sc = scoring.score_row(r, r["target"], meta)
            assert sc["valid"] and sc["app_accepts"] and sc["behaviour_ok"], r["id"]
            if r["expected"] == "answer":
                assert sc["answer_ok"] and sc["citation_hit"] and sc["all_citations_valid"], r["id"]
                assert sc["groundedness"] == 1.0, (r["id"], t)
            else:
                assert not r["supporting_ids"] and not sc["hidden_fact_leak"], r["id"]

    # 5. test facts that also occur in training targets (shared vocabulary such as stage or tier names, or the
    #    same owner); reported, not removed: the model still has to find them in the test evidence
    train_targets = " ".join(json.loads(r["target"])["answer"].lower() for r in rows["train"])
    shared = Counter()
    for r in rows["test"]:
        if r["expected"] == "answer":
            for g in r["facts_restored"]:
                if any(a.lower() in train_targets for a in g):
                    shared[r["intent"]] += 1
                    break

    def counts(rs: list[dict]) -> dict:
        return {"rows": len(rs), "accounts": sorted({r["account_id"] for r in rs}),
                "expected": dict(Counter(r["expected"] for r in rs)),
                "abstain_reason": dict(sorted(Counter(r["abstain_reason"] for r in rs if r["abstain_reason"]).items())),
                "role": dict(Counter(r["user"]["role"] for r in rs)),
                "wording": dict(Counter(r["wording"] for r in rs)),
                "intent": dict(sorted(Counter(r["intent"] for r in rs).items())),
                "prompt_chars": {"min": min(len(r["prompt"]) for r in rs),
                                 "mean": round(sum(len(r["prompt"]) for r in rs) / len(rs)),
                                 "max": max(len(r["prompt"]) for r in rs)},
                "target_chars_max": max(len(r["target"]) for r in rs)}

    test = rows["test"]
    offline_scores = [scoring.score_row(r, r["offline_generator_output"], meta) for r in test]
    oracle_scores = [scoring.score_row(r, r["target"], meta) for r in test]
    personas_used = {r["user"]["id"]: {"role": r["user"]["role"], "territories": r["user"]["territories"],
                                       "source": "users.json" if r["user"]["id"] in users else "dataset-only"}
                     for s in rows for r in rows[s]}
    return {
        "what": "SFT data for the Ask task (grounded answer with citations) built with the app's own pipeline",
        "seed": SEED, "system_prompt": SYSTEM_PROMPT, "abstain_text": ABSTAIN,
        "split_accounts": {s: sorted(a for a, x in split_of.items() if x == s) for s in ("train", "val", "test")},
        "account_names": names,
        "personas": dict(sorted(personas_used.items())),
        "phrasings": PHRASINGS, "held_out_phrasing_index": HELD_OUT,
        "fact_matching": {"short": "case-insensitive substring of one accepted surface form per group",
                          "long": f"at least {scoring.LONG_FACT_RECALL} of the fact's content words (app.offline.terms)"},
        "splits": {s: counts(rs) for s, rs in rows.items()},
        "leakage": {
            "golden_eval_answers_dropped": {s: len(d) for s, d in dropped_golden.items()},
            "golden_eval_answers_dropped_list": dropped_golden,
            "test_rows_with_golden_eval_answers_reported_only": len(test_golden),
            "test_rows_with_golden_eval_answers_list": test_golden,
            "train_prompts_with_val_or_test_account_records": len(train_eval_records),
            "train_prompts_naming_val_or_test_accounts": len(train_eval_names),
            "val_prompts_with_test_account_records": len(val_test_records),
            "prompt_overlap": {"train_test": len(prompts["train"] & prompts["test"]),
                               "train_val": len(prompts["train"] & prompts["val"]),
                               "val_test": len(prompts["val"] & prompts["test"])},
            "question_overlap": {"train_test": len(questions["train"] & questions["test"]),
                                 "train_val": len(questions["train"] & questions["val"]),
                                 "val_test": len(questions["val"] & questions["test"])},
            "held_out_wording_rows_outside_test": len(held_out_outside_test),
            "test_answer_rows_whose_fact_also_appears_in_a_training_target": dict(sorted(shared.items())),
        },
        "reference_on_test": {
            "offline_extractive_generator": scoring.report(test, offline_scores),
            "targets_self_check": scoring.summarize(test, oracle_scores),
        },
        "record_meta": {k: list(v) for k, v in sorted(meta.items())},
    }


def _jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify data/ matches a fresh build; write nothing")
    ap.add_argument("--refreeze", action="store_true", help="allow the frozen test set to change")
    a = ap.parse_args(argv)
    rows, report = build()
    files = {f"{s}.jsonl": _jsonl(rows[s]) for s in rows}
    test_hash = hashlib.sha256(files["test.jsonl"]).hexdigest()
    report["test_sha256"] = test_hash
    files["dataset_report.json"] = (json.dumps(report, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    files[TEST_HASH_FILE.name] = f"{test_hash}  test.jsonl\n".encode()
    if a.check:
        bad = [n for n, b in files.items() if not (DATA / n).exists() or (DATA / n).read_bytes() != b]
        print("data/ matches a fresh build" if not bad else f"differs from a fresh build: {bad}")
        return 1 if bad else 0
    if TEST_HASH_FILE.exists() and TEST_HASH_FILE.read_text().split()[0] != test_hash and not a.refreeze:
        raise SystemExit(f"the frozen test set would change ({TEST_HASH_FILE}); rerun with --refreeze if intended")
    DATA.mkdir(exist_ok=True)
    for n, b in files.items():
        (DATA / n).write_bytes(b)
    for s in rows:
        c = report["splits"][s]
        print(f"{s:5} {c['rows']:4} rows  accounts {c['accounts']}  {c['expected']}  {c['abstain_reason']}")
    print("leakage", json.dumps({k: v for k, v in report["leakage"].items() if not k.endswith("_list")}))
    off = report["reference_on_test"]["offline_extractive_generator"]["all"]
    print("offline generator on test:", {k: v["mean"] for k, v in off.items() if isinstance(v, dict)})
    print(f"test sha256 {test_hash}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
