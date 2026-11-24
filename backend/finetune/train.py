"""Fine-tunes BAAI/bge-small-en-v1.5 on the synthetic pairs with MultipleNegativesRankingLoss, on CPU.

Run from backend/ with the training venv (see README.md):
  finetune\\.venv-train\\Scripts\\python finetune/train.py --run sales-ft
Options: --no-hard-negatives (in-batch negatives only), --holdout-accounts ACC-003,ACC-007 (drop those
accounts from training to measure transfer to unseen documents).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(BACKEND / ".models" / "hf"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,  # noqa: E402
                                   SentenceTransformerTrainingArguments, losses)
from sentence_transformers.evaluation import InformationRetrievalEvaluator  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
CHECKPOINTS = BACKEND / ".models" / "st"
BASE_MODEL = "BAAI/bge-small-en-v1.5"
INJECTION_NOTE = "ACT-0046"
SEED = 42


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class GroupBatchSampler(torch.utils.data.Sampler):
    """At most one row per account or product in a batch. Templated emails and meetings restate
    activity notes, so a same-account passage in the batch could be a correct answer used as a negative."""

    def __init__(self, groups: list[frozenset[str]], batch_size: int, seed: int):
        self.groups, self.batch_size, self.seed, self.epoch = groups, batch_size, seed, 0
        self.drop_last = False

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        pending = list(range(len(self.groups)))
        rng.shuffle(pending)
        while pending:
            batch, used, rest = [], set(), []
            for i in pending:
                if len(batch) < self.batch_size and not self.groups[i] & used:
                    batch.append(i)
                    used |= self.groups[i]
                else:
                    rest.append(i)
            yield batch
            pending = rest

    def __len__(self) -> int:
        return sum(1 for _ in GroupBatchSampler(self.groups, self.batch_size, self.seed))


class GroupTrainer(SentenceTransformerTrainer):
    def __init__(self, *args, row_groups: list[frozenset[str]], **kwargs):
        self.row_groups = row_groups
        super().__init__(*args, **kwargs)

    def get_batch_sampler(self, dataset, batch_size, *args, **kwargs):
        if len(dataset) == len(self.row_groups):
            return GroupBatchSampler(self.row_groups, batch_size, SEED)
        return super().get_batch_sampler(dataset, batch_size, *args, **kwargs)


def mine_hard_negatives(model: SentenceTransformer, rows: list[dict], corpus: dict[str, dict]) -> list[str]:
    """Most similar passage (base model) that is not a labelled answer and does not restate one,
    preferring the anchor's own account so the negative never answers another row in the batch."""
    ids = [i for i in corpus if i != INJECTION_NOTE]
    doc_emb = model.encode([corpus[i]["text"] for i in ids], normalize_embeddings=True, batch_size=64)
    q_emb = model.encode([r["question"] for r in rows], normalize_embeddings=True, batch_size=64)
    sims = q_emb @ doc_emb.T
    negatives = []
    for r, s in zip(rows, sims):
        banned = set(r["relevant"]) | set(r["related"])
        ranked = [ids[j] for j in np.argsort(-s) if ids[j] not in banned]
        same = [i for i in ranked if corpus[i]["group"] in r["groups"]]
        negatives.append(same[0] if same else ranked[0])
    return negatives


def ir_evaluator(rows: list[dict], corpus: dict[str, dict], name: str) -> InformationRetrievalEvaluator:
    return InformationRetrievalEvaluator(
        queries={r["id"]: r["question"] for r in rows}, corpus={i: d["text"] for i, d in corpus.items()},
        relevant_docs={r["id"]: set(r["relevant"]) for r in rows}, name=name, accuracy_at_k=[1, 3, 6],
        precision_recall_at_k=[6], mrr_at_k=[10], ndcg_at_k=[10], map_at_k=[10], show_progress_bar=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="sales-ft")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-hard-negatives", action="store_true")
    ap.add_argument("--holdout-accounts", default="")
    args = ap.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(args.threads)

    corpus = {d["id"]: d for d in read_jsonl(DATA / "corpus.jsonl")}
    train_q, val_q = read_jsonl(DATA / "train.jsonl"), read_jsonl(DATA / "val.jsonl")
    holdout = {a for a in args.holdout_accounts.split(",") if a}
    if holdout:
        train_q = [r for r in train_q if not holdout & set(r["groups"])]
        val_q = [r for r in val_q if not holdout & set(r["groups"])]
        corpus_train = {i: d for i, d in corpus.items() if d["group"] not in holdout}
    else:
        corpus_train = corpus

    model = SentenceTransformer(BASE_MODEL, device="cpu")
    model.max_seq_length = 128
    pairs = [{**r, "positive": p} for r in train_q for p in r["relevant"]]
    cols = {"anchor": [p["question"] for p in pairs], "positive": [corpus[p["positive"]]["text"] for p in pairs]}
    row_groups = [frozenset(p["groups"]) for p in pairs]
    if not args.no_hard_negatives:
        negs = mine_hard_negatives(model, pairs, corpus_train)
        cols["negative"] = [corpus[n]["text"] for n in negs]
        row_groups = [g | {corpus[n]["group"]} for g, n in zip(row_groups, negs)]
    train_ds = Dataset.from_dict(cols)

    val_eval = ir_evaluator(val_q, corpus, "val")
    train_sample = random.Random(SEED).sample(train_q, min(len(val_q), len(train_q)))
    train_eval = ir_evaluator(train_sample, corpus, "train")
    before = {**val_eval(model), **train_eval(model)}

    out_dir = CHECKPOINTS / args.run
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir / "checkpoints"), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, learning_rate=args.lr, warmup_steps=0.1,
        weight_decay=0.01, seed=SEED, data_seed=SEED, eval_strategy="epoch", save_strategy="no",
        logging_steps=10, report_to="none", use_cpu=True, dataloader_num_workers=0)
    loss = losses.MultipleNegativesRankingLoss(model)
    trainer = GroupTrainer(model=model, args=targs, train_dataset=train_ds, loss=loss, evaluator=val_eval,
                           row_groups=row_groups)
    start = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - start
    after = {**val_eval(model), **train_eval(model)}
    model.save(str(out_dir / "final"))

    keys = [k for k in before if any(m in k for m in ("accuracy@1", "accuracy@3", "accuracy@6", "mrr@10"))]
    report = {
        "run": args.run, "base_model": BASE_MODEL, "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "hard_negatives": not args.no_hard_negatives, "holdout_accounts": sorted(holdout),
        "train_questions": len(train_q), "train_pairs": len(pairs), "val_questions": len(val_q),
        "train_sample_for_overfit_check": len(train_sample), "train_seconds": round(seconds, 1),
        "torch": torch.__version__, "threads": torch.get_num_threads(),
        "before": {k: round(before[k], 4) for k in keys}, "after": {k: round(after[k], 4) for k in keys},
        "log": [e for e in trainer.state.log_history if "loss" in e or "eval_val_cosine_mrr@10" in e],
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"train_{args.run}.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "log"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
