"""Write sales_llm_qlora_vllm.ipynb: a self-contained Kaggle / Colab notebook with llm_finetune/data embedded
(gzip + base64), verbatim copies of the app code it scores with, and scoring.py. Only the .ipynb is uploaded.

  python llm_finetune/build_notebook.py     (app venv; no torch needed)
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import inspect
import json
import re
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent / "backend", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import eval.run_eval as run_eval  # noqa: E402
from app import embeddings, llm, models, offline  # noqa: E402
from app.agent import GROUNDED_THRESHOLD, Agent  # noqa: E402

DATA = HERE / "data"
OUT = HERE / "sales_llm_qlora_vllm.ipynb"
PINS = {"transformers": "5.22.2", "peft": "0.21.3", "trl": "1.17.1", "datasets": "5.1.0", "accelerate": "1.15.0",
        "bitsandbytes": "0.50.2"}
MODEL = ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
SMOKE_MODEL = ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775")
VLLM = "0.32.0"
VLLM_CU129_WHEEL = ("https://github.com/vllm-project/vllm/releases/download/v0.32.0/"
                    "vllm-0.32.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl")
LLMC = "0.12.0.3"
UV = "0.13.5"
CLOUDFLARED = "2026.10.7"

INTRO = """# QLoRA fine-tune of an open model for the Sales AI Copilot, then vLLM serving on a T4

**Status: not yet run on a GPU; no accuracy or throughput numbers.** The notebook has only been run end to end as a
CPU smoke test (0.5B model, 3 optimiser steps, a stub server in place of vLLM); see `llm_finetune/README.md`.

What it does, on a free Kaggle (2x T4) or Colab (1x T4) GPU:

1. **Data**: {n_train} training, {n_val} validation and {n_test} test examples of the app's grounded-answer task
   (`POST /api/ask`), built by `llm_finetune/make_dataset.py` with the app's own retrieval, access control,
   prompt-injection quarantine and PII redaction. Test accounts never appear in training. The test file was frozen
   (SHA-256 checked below) before any training.
2. **QLoRA**: {model} (Apache-2.0) with a 4-bit NF4 base (bitsandbytes) and LoRA adapters, with transformers, peft
   and trl.
3. **Evaluation**: base vs fine-tuned on the frozen test set with the app's own validators (copied verbatim below):
   JSON schema validity, citation validity (cited ids were in the evidence and the acting user may read them),
   correct abstention, fact accuracy, ROUGE-L, with bootstrap 95% confidence intervals.
4. **Inference optimisation**: the merged model is served by vLLM {vllm} (OpenAI-compatible server, fp16) and compared
   with a 4-bit GPTQ (W4A16) variant made with llm-compressor, at concurrency 1, 8 and 16: tokens/s, latency p50/p95,
   time to first token, peak GPU memory; plus tensor_parallel_size 2 vs 1 when two GPUs are present.
5. Everything goes to `results.json` with the GPU name and library versions.

**Run**: Kaggle: Settings > Accelerator > GPU T4 x2, Internet on, then Run all. Colab: Runtime > Change runtime type >
T4 GPU, then Runtime > Run all. The time estimate in the README (roughly 1.5 to 2 hours) is not a measurement.
"""

CONFIG = """import os, sys, json, math, time, platform, subprocess, shutil, re, gc, random

SMOKE = os.environ.get("SALES_LLM_SMOKE") == "1"  # tiny CPU run used to test this notebook before release


def gpu_names():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=60).stdout
        return [line.strip() for line in out.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


GPUS = gpu_names()  # every physical GPU, read before CUDA_VISIBLE_DEVICES narrows this kernel to one
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # training and HF evaluation use one GPU (no DataParallel)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_ID, MODEL_REVISION = <<MODEL>>
SEED = 42
QLORA = True  # 4-bit NF4 base (bitsandbytes) during training; False = LoRA on an fp16 base
EPOCHS = 2
LEARNING_RATE = 2e-4
BATCH_SIZE, GRAD_ACCUM = 2, 8
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MAX_LEN = 2048
MAX_NEW_TOKENS = 320
GEN_BATCH = 8
MAX_STEPS = -1
TRAIN_LIMIT = VAL_LIMIT = TEST_LIMIT = None
# inference optimisation with vLLM (section 7; runs in separate virtualenvs, so it cannot disturb the training stack)
RUN_VLLM = True
RUN_QUANT = True   # GPTQ W4A16 with llm-compressor; if that fails, vLLM's FP8 weight-only quantization instead
RUN_TP2 = True     # tensor_parallel_size=2 vs 1, only when two GPUs are present
VLLM_VERSION, LLMC_VERSION, UV_VERSION = <<VLLM>>, <<LLMC>>, <<UV>>
VLLM_CU129_WHEEL = <<WHEEL>>
SERVED = "sales-qwen"
VLLM_PORT = 8000
GPU_MEM_UTIL = 0.80
KV_CACHE_BYTES = 4 * 1024 ** 3  # fixed KV cache per GPU, so peak memory is comparable across variants
MAX_MODEL_LEN = 4096
MAX_NUM_SEQS = 16
CONCURRENCY = (1, 8, 16)
C1_REQUESTS = 32
CALIB_SAMPLES = 128
RUN_TUNNEL = False  # optional last cell: HTTPS tunnel to a vLLM server for llm_finetune/eval_endpoint.py
CLOUDFLARED_VERSION = <<CLOUDFLARED>>
if SMOKE:
    MODEL_ID, MODEL_REVISION = <<SMOKE_MODEL>>
    EPOCHS, MAX_STEPS, BATCH_SIZE, GRAD_ACCUM, GEN_BATCH, MAX_NEW_TOKENS = 1, 3, 1, 2, 2, 48
    TRAIN_LIMIT, VAL_LIMIT, TEST_LIMIT, C1_REQUESTS = 12, 2, 6, 4
    RUN_VLLM = False  # no GPU: section 7 runs its client and scoring code against a stub server instead
HOST = "kaggle" if os.path.isdir("/kaggle/working") else ("colab" if os.path.isdir("/content") else "local")
OUT_DIR = os.environ.get("SALES_LLM_OUT") or {"kaggle": "/kaggle/working/sales_llm",
                                             "colab": "/content/sales_llm"}.get(HOST, os.path.abspath("sales_llm_out"))
WORK_DIR = os.environ.get("SALES_LLM_WORK") or ("/tmp/sales_llm_work" if HOST != "local" else os.path.join(OUT_DIR, "_work"))
for d in (OUT_DIR, WORK_DIR):
    os.makedirs(d, exist_ok=True)
TIMES = {}
T_START = time.time()
print("model", MODEL_ID, "| smoke" if SMOKE else "", "| GPUs", GPUS or "none", "| outputs in", OUT_DIR)
"""

INSTALL = """import importlib.metadata as md

PINS = <<PINS>>


def installed(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


need = [f"{p}=={v}" for p, v in PINS.items() if installed(p) != v]
if int((installed("pydantic") or "0").split(".")[0]) < 2:
    need.append("pydantic==2.14.4")
if need:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *need])
print({p: installed(p) for p in [*PINS, "pydantic", "torch"]})
assert all(installed(p) == v for p, v in PINS.items()), "pinned versions are not active: restart the session, run all"
"""

DATA_CELL = """# llm_finetune/data/*.jsonl and dataset_report.json (make_dataset.py), embedded by build_notebook.py
import base64, gzip, hashlib
from collections import Counter

DATA_B64 = <<BLOB>>
FILES = json.loads(gzip.decompress(base64.b64decode(DATA_B64)))
FROZEN_TEST_SHA256 = <<TEST_SHA>>
TEST_SHA256 = hashlib.sha256(FILES["test.jsonl"].encode("utf-8")).hexdigest()
assert TEST_SHA256 == FROZEN_TEST_SHA256, "embedded test set does not match the frozen hash"
REPORT = json.loads(FILES["dataset_report.json"])
RECORD_META = REPORT["record_meta"]


def rows_of(name):
    return [json.loads(line) for line in FILES[name].splitlines() if line.strip()]


train_rows, val_rows, test_rows = rows_of("train.jsonl"), rows_of("val.jsonl"), rows_of("test.jsonl")
print("test set sha256", TEST_SHA256, "(frozen before any training)")
for name, rs in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
    print(f"{name:5} {len(rs):4} rows, accounts {sorted({r['account_id'] for r in rs})}, "
          f"{dict(Counter(r['expected'] for r in rs))}")
print("leakage checks", json.dumps({k: v for k, v in REPORT["leakage"].items() if not k.endswith("_list")}, indent=1))
"""

APP_FOOTER = """

SCHEMA_JSON = json.dumps(Answer.model_json_schema())
print("prompts carry this Answer schema:", all(SCHEMA_JSON in r["prompt"] for r in test_rows))
assert all(score_row(r, r["target"], RECORD_META)["behaviour_ok"] for r in test_rows), "scoring self-check failed"
print("every test target passes the app's validators and scores as correct")
"""

MODEL_CELL = """import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

set_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE != "cuda" and not SMOKE:
    raise RuntimeError("No GPU. Kaggle: Settings > Accelerator > GPU T4 x2. Colab: Runtime > Change runtime type > T4 GPU.")
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32  # a T4 has no bf16
ENV = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
       "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu", "gpus_physical": GPUS, "host": HOST,
       **{p: installed(p) for p in PINS}, "pydantic": installed("pydantic")}
print(ENV)
tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)


def messages(r):
    return [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["prompt"]}]


def chat_text(r):
    return tok.apply_chat_template(messages(r), tokenize=False, add_generation_prompt=True)


def full_text(r):
    return tok.apply_chat_template(messages(r) + [{"role": "assistant", "content": r["target"]}], tokenize=False)


def n_tokens(text):
    return len(tok(text, add_special_tokens=False).input_ids)


full = [n_tokens(full_text(r)) for r in train_rows + val_rows]
prompt_only = [n_tokens(chat_text(r)) for r in test_rows]
TOKENS = {"train_val_max": max(full), "train_val_mean": round(float(np.mean(full)), 1),
          "train_val_p95": int(np.percentile(full, 95)), "train_tokens_per_epoch": int(sum(full[:len(train_rows)])),
          "test_prompt_max": max(prompt_only), "test_prompt_mean": round(float(np.mean(prompt_only)), 1)}
print("tokens", TOKENS)
assert TOKENS["train_val_max"] <= MAX_LEN, "an example is longer than MAX_LEN and would be truncated"
assert TOKENS["test_prompt_max"] + MAX_NEW_TOKENS <= MAX_MODEL_LEN
"""

EVAL_HELPERS = """TEST = test_rows[:TEST_LIMIT] if TEST_LIMIT else test_rows
RESULTS = {"test": {}, "compare": {}}
RAW, SCORES, TRAIN_INFO = {}, {}, {}
INFER = {"status": "not run", "variants": {}}
RESULTS_PATH = os.path.join(OUT_DIR, "results.json")


def greedy(m):
    # Qwen ships sampling defaults and repetition_penalty 1.05; evaluation is greedy without a penalty
    g = m.generation_config
    g.do_sample, g.temperature, g.top_p, g.top_k, g.repetition_penalty = False, None, None, None, 1.0
    g.max_new_tokens, g.pad_token_id = MAX_NEW_TOKENS, tok.pad_token_id
    return m


@torch.no_grad()
def generate(m, rows):
    m.eval()
    tok.padding_side = "left"
    order = sorted(range(len(rows)), key=lambda i: -len(rows[i]["prompt"]))  # similar lengths per batch
    texts = {}
    for s in range(0, len(order), GEN_BATCH):
        idx = order[s:s + GEN_BATCH]
        enc = tok([chat_text(rows[i]) for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(m.device)
        gen = greedy(m).generate(**enc)
        for i, t in zip(idx, tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            texts[i] = t
    return [texts[i] for i in range(len(rows))]


def record(tag, rows, raws):
    RAW[tag] = raws
    SCORES[tag] = [score_row(r, raw, RECORD_META) for r, raw in zip(rows, raws)]
    RESULTS["test"][tag] = report(rows, SCORES[tag])


HEAD = ("schema_valid", "behaviour_accuracy", "answer_accuracy", "correct_abstention", "citation_validity",
        "citation_hit", "rouge_l_f1_answer_rows")


def table(subset="all"):
    print(f"subset: {subset}")
    print(f"{'system':46}" + "".join(f"{h[:19]:>21}" for h in HEAD))
    for tag, rep in RESULTS["test"].items():
        s = rep.get(subset, {})
        cell = lambda k: f"{s[k]['mean']:.3f} [{s[k]['ci95'][0]:.2f},{s[k]['ci95'][1]:.2f}]" if k in s else "-"
        print(f"{tag[:46]:46}" + "".join(f"{cell(h):>21}" for h in HEAD))


def write_results(status="partial"):
    keep = ("valid", "app_accepts", "behaviour_ok", "answer_ok", "abstain_ok", "citation_hit", "all_citations_valid",
            "rouge_l")
    items = [{"id": r["id"], "intent": r["intent"], "role": r["user"]["role"], "wording": r["wording"],
              "expected": r["expected"], "abstain_reason": r["abstain_reason"],
              "outputs": {t: RAW[t][i] for t in RAW if t != TARGETS_TAG},
              "scores": {t: {k: SCORES[t][i][k] for k in keep} for t in SCORES if t != TARGETS_TAG}}
             for i, r in enumerate(TEST)]
    results = {
        "what": "QLoRA fine-tune of an open model for the app's grounded-answer task, evaluated on a frozen test set "
                "with the app's validators, then served with vLLM (fp16 vs quantized, tensor parallel 1 vs 2)",
        "status": status, "smoke_run": SMOKE, "model": MODEL_ID, "model_revision": MODEL_REVISION, "env": ENV,
        "config": {"seed": SEED, "qlora": QLORA, "epochs": EPOCHS, "max_steps": MAX_STEPS,
                   "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
                   "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
                   "lora_targets": LORA_TARGETS, "max_len": MAX_LEN, "max_new_tokens": MAX_NEW_TOKENS,
                   "decoding": "greedy, repetition_penalty 1.0 (HF); temperature 0 (vLLM)",
                   "train_examples": len(TRAIN) if "TRAIN" in globals() else None,
                   "val_examples": len(VAL) if "VAL" in globals() else None, "test_rows": len(TEST),
                   "vllm": {"version": VLLM_VERSION, "max_model_len": MAX_MODEL_LEN, "max_num_seqs": MAX_NUM_SEQS,
                            "kv_cache_bytes": KV_CACHE_BYTES, "concurrency": CONCURRENCY,
                            "c1_requests": C1_REQUESTS}},
        "data": {"test_sha256": TEST_SHA256, "frozen_test_sha256": FROZEN_TEST_SHA256, "tokens": TOKENS,
                 "splits": {k: {x: v[x] for x in ("rows", "accounts", "expected", "abstain_reason", "wording")}
                            for k, v in REPORT["splits"].items()},
                 "leakage": {k: v for k, v in REPORT["leakage"].items() if not k.endswith("_list")}},
        "training": TRAIN_INFO, "quality_test": RESULTS, "inference": INFER, "seconds": TIMES, "items": items,
        "notes": ["fact accuracy: short facts need one accepted surface form, long facts 60% of their content words",
                  "abstained = valid JSON with an empty claims list (the app's own instruction)",
                  "citation validity uses the policy oracle in backend/eval/run_eval.py, not app.security",
                  "single seed; greedy decoding; one training run"],
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)


OFFLINE_TAG, TARGETS_TAG = "offline extractive generator (app default)", "targets (self-check)"
record(OFFLINE_TAG, TEST, [r["offline_generator_output"] for r in TEST])
record(TARGETS_TAG, TEST, [r["target"] for r in TEST])
table()
"""

BASE_EVAL = """t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=DTYPE).to(DEVICE)
TIMES["load_base_s"] = round(time.time() - t0, 1)
BASE_TAG = f"base {MODEL_ID.split('/')[-1]} (HF, greedy)"
t0 = time.time()
record(BASE_TAG, TEST, generate(model, TEST))
TIMES["eval_base_s"] = round(time.time() - t0, 1)
table()
del model
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()
write_results()
"""

TRAIN_CELL = """from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def as_prompt_completion(r):
    return {"prompt": messages(r), "completion": [{"role": "assistant", "content": r["target"]}]}


TRAIN = train_rows[:TRAIN_LIMIT] if TRAIN_LIMIT else train_rows
VAL = val_rows[:VAL_LIMIT] if VAL_LIMIT else val_rows
train_ds = Dataset.from_list([as_prompt_completion(r) for r in TRAIN]).shuffle(seed=SEED)
val_ds = Dataset.from_list([as_prompt_completion(r) for r in VAL])
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=DTYPE) if QLORA else None
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=DTYPE, quantization_config=quant,
                                             device_map={"": 0} if DEVICE == "cuda" else None)
if QLORA:
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
steps = MAX_STEPS if MAX_STEPS > 0 else math.ceil(len(TRAIN) / (BATCH_SIZE * GRAD_ACCUM)) * EPOCHS
per = "steps" if SMOKE else "epoch"
args = SFTConfig(
    output_dir=os.path.join(WORK_DIR, "checkpoints"), num_train_epochs=EPOCHS, max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM, learning_rate=LEARNING_RATE, lr_scheduler_type="cosine",
    warmup_steps=max(1, round(0.05 * steps)), weight_decay=0.0, max_grad_norm=1.0, optim="adamw_torch",
    fp16=DEVICE == "cuda", bf16=False, gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}, max_length=MAX_LEN, packing=False,
    completion_only_loss=True, eval_strategy=per, save_strategy=per, **({"eval_steps": MAX_STEPS,
    "save_steps": MAX_STEPS} if SMOKE else {}), load_best_model_at_end=True, metric_for_best_model="eval_loss",
    greater_is_better=False, save_total_limit=1, save_only_model=True, logging_steps=1 if SMOKE else 4,
    seed=SEED, data_seed=SEED, report_to="none")
peft_config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGETS,
                         bias="none", task_type="CAUSAL_LM")
tok.padding_side = "right"
trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, processing_class=tok,
                     peft_config=peft_config)
# TRL 1.17.1 casts QLoRA adapter weights to bf16 (trl/trainer/sft_trainer.py, "When using QLoRA ..."). A T4 has no
# bf16 and fp16 loss scaling cannot unscale bf16 gradients, so the trainable adapters are kept in fp32.
recast = 0
for _, p in trainer.model.named_parameters():
    if p.requires_grad and p.dtype != torch.float32:
        p.data = p.data.float()
        recast += 1
print("adapter tensors recast to fp32:", recast, "| 4-bit base:", getattr(trainer.model, "is_loaded_in_4bit", False))
trainer.model.print_trainable_parameters()
TIMES["load_train_model_s"] = round(time.time() - t0, 1)
t0 = time.time()
train_out = trainer.train()
TIMES["train_s"] = round(time.time() - t0, 1)
final_eval = trainer.evaluate()
assert math.isfinite(train_out.training_loss), "training loss is not finite (fp16 overflow?): see README"
TRAIN_INFO = {"train_loss": train_out.training_loss, "final_eval": final_eval, "optimizer_steps": trainer.state.global_step,
              "best_checkpoint": trainer.state.best_model_checkpoint, "log_history": list(trainer.state.log_history),
              "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if DEVICE == "cuda" else None}
print("train loss", round(train_out.training_loss, 4), "| eval loss (best checkpoint)",
      round(final_eval["eval_loss"], 4), "| steps", trainer.state.global_step, "| minutes", round(TIMES["train_s"] / 60, 1),
      "| peak GPU memory GiB", TRAIN_INFO["peak_gpu_memory_gib"])
write_results()
"""

MERGE_CELL = """from peft import PeftModel

ADAPTER_DIR = os.path.join(OUT_DIR, "adapter")
MERGED_DIR = os.path.join(WORK_DIR, "merged")
trainer.model.save_pretrained(ADAPTER_DIR)
tok.save_pretrained(ADAPTER_DIR)
del trainer, model
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()
t0 = time.time()
# QLoRA trains against the 4-bit base; the adapter is merged into the fp16 base, which is what gets deployed
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=DTYPE).to(DEVICE)
ft = PeftModel.from_pretrained(base, ADAPTER_DIR).merge_and_unload()
greedy(ft)  # saved generation defaults: greedy, no repetition penalty, so vLLM serves it the way it was evaluated
ft.save_pretrained(MERGED_DIR)
tok.padding_side = "left"
tok.save_pretrained(MERGED_DIR)
TIMES["merge_save_s"] = round(time.time() - t0, 1)
FT_TAG = f"fine-tuned {MODEL_ID.split('/')[-1]} (QLoRA, merged fp16, HF)"
t0 = time.time()
record(FT_TAG, TEST, generate(ft, TEST))
TIMES["eval_finetuned_s"] = round(time.time() - t0, 1)
RESULTS["compare"]["fine-tuned minus base"] = compare(TEST, SCORES[BASE_TAG], SCORES[FT_TAG])
RESULTS["compare"]["fine-tuned minus offline generator"] = compare(TEST, SCORES[OFFLINE_TAG], SCORES[FT_TAG])
for subset in ("all", "wording=held_out", "expected=answer", "expected=abstain"):
    table(subset)
    print()
for name, diff in RESULTS["compare"].items():
    print(name, {k: (v["diff"], v["ci95"]) for k, v in diff.items()})
del ft, base
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()
write_results()
"""

INFER_INTRO = """## 7. Inference optimisation with vLLM

vLLM {vllm} runs as an OpenAI-compatible server in its own virtualenv (it pins torch 2.13 and transformers < 5.20,
which would break the training stack above). The benchmark client below sends the frozen test prompts the way the
app's `openai_compat` provider does and scores the answers with the same validators, so speed and quality are
measured on the same requests.

* **fp16, tensor_parallel_size 1**: the merged model as trained and evaluated above.
* **GPTQ W4A16** (4-bit weights, fp16 activations), made with llm-compressor {llmc} from 128 *training* prompts.
  Why GPTQ: vLLM {vllm}'s quantization table lists GPTQ and its Marlin kernels as supported on Turing (sm75), and
  its compressed-tensors W4A16 scheme needs compute capability 7.5. FP8 W8A8 needs Ada or newer; bitsandbytes moved
  to an out-of-tree plugin (`vllm-bnb-plugin`) in {vllm}; AutoAWQ is archived, and llm-compressor's AWQ writes the
  same W4A16 format, so it would exercise the same kernels. llm-compressor {llmc} is used because it pins
  compressed-tensors 0.17.1, next to the
  0.17.0 that vLLM {vllm} pins (newer releases jump to 0.20). If the GPTQ step fails, the notebook serves vLLM's
  online FP8 quantization (`fp8_per_tensor`, weight-only on a T4) instead and records which variant ran.
* **fp16, tensor_parallel_size 2** vs 1, when the session has two GPUs (Kaggle T4 x2; PCIe, no NVLink).

Every server uses the same fixed KV cache size, so peak GPU memory differs only by weights, activations and graphs.
Load: concurrency {conc} with the test prompts (concurrency 1 uses the first {c1}); temperature 0; up to
{max_new} new tokens. Nothing in this section was run before release: it is written against vLLM {vllm}'s CLI and
OpenAI-compatible API (flags and log lines checked in its source), but it is unverified on a GPU.
"""

INFER_HELPERS = """import signal, secrets, threading, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

STUB_SOURCE = <<STUB>>


def tail(path, n=40):
    try:
        with open(path, errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


def run_logged(cmd, log_path, env=None, timeout=None):
    print("$", " ".join(map(str, cmd))[:240], flush=True)
    with open(log_path, "a") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    if p.returncode:
        raise RuntimeError(f"{cmd[0]} exited with {p.returncode}; last lines of {log_path}:\\n{tail(log_path)}")


def driver_info():
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    cuda, drv = re.search(r"CUDA Version:\\s*([\\d.]+)", out), re.search(r"Driver Version:\\s*([\\d.]+)", out)
    return {"driver": drv.group(1) if drv else None, "driver_max_cuda": cuda.group(1) if cuda else None}


def cuda13_ok():
    v = driver_info().get("driver_max_cuda") or "0"
    return int(v.split(".")[0]) >= 13


def make_venv(name, steps):
    \"\"\"A uv-managed Python 3.12 virtualenv; steps are `uv pip install` argument lists run in order.\"\"\"
    venv = os.path.join(WORK_DIR, name)
    py, log = os.path.join(venv, "bin", "python"), os.path.join(WORK_DIR, f"{name}-install.log")
    if os.path.exists(venv + ".ok"):
        return py
    t0 = time.time()
    run_logged([sys.executable, "-m", "pip", "install", "-q", f"uv=={UV_VERSION}"], log)
    run_logged([sys.executable, "-m", "uv", "venv", "--python", "3.12", venv], log)
    for args in steps:
        run_logged([sys.executable, "-m", "uv", "pip", "install", "--python", py, *args], log)
    open(venv + ".ok", "w").close()
    TIMES[f"install_{name}_s"] = round(time.time() - t0, 1)
    return py


def versions(py, mods):
    code = ("import importlib.metadata as m, json; print(json.dumps({p: m.version(p) for p in %r}))" % (mods,))
    try:
        return json.loads(subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=120).stdout)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"error": repr(exc)}


class GpuMem:
    \"\"\"Polls nvidia-smi; peak memory.used per GPU (MiB) while the block runs.\"\"\"

    def __init__(self, every=0.5):
        self.peak, self._stop = {}, threading.Event()
        self._t = threading.Thread(target=self._loop, args=(every,), daemon=True)

    @staticmethod
    def read():
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=30).stdout
            return {int(a): int(b) for a, b in (line.split(",") for line in out.strip().splitlines())}
        except (OSError, ValueError, subprocess.SubprocessError):
            return {}

    def _loop(self, every):
        while not self._stop.is_set():
            for i, v in self.read().items():
                self.peak[i] = max(self.peak.get(i, 0), v)
            self._stop.wait(every)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(5)


def start_server(cmd, env, log_path, port, timeout=1500):
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                            **({"start_new_session": True} if os.name == "posix" else {}))
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}; last lines of its log:\\n{tail(log_path)}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return proc, round(time.time() - t0, 1)
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(3)
    stop_server(proc)
    raise TimeoutError(f"server not healthy after {timeout} s; last lines of its log:\\n{tail(log_path)}")


def stop_server(proc):
    if proc is None or proc.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(proc.pid, signal.SIGTERM)
    else:
        proc.terminate()
    try:
        proc.wait(90)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL) if os.name == "posix" else proc.kill()
        proc.wait(30)


def parse_vllm_log(path):
    try:
        text = open(path, errors="replace").read()
    except OSError:
        return {}

    def last(pattern, cast=float):
        found = re.findall(pattern, text)
        return cast(found[-1].replace(",", "")) if found else None

    backend = re.findall(r"Using (\\w+) attention backend", text) or re.findall(r"Using (\\w+) backend\\.", text)
    return {"model_weights_gib_per_gpu": last(r"Model loading took ([\\d.]+) GiB"),
            "kv_cache_tokens": last(r"KV cache size: ([\\d,]+) tokens", int),
            "max_concurrency_at_max_model_len": last(r"Maximum concurrency for [\\d,]+ tokens per request: ([\\d.]+)x"),
            "init_engine_s": last(r"init engine \\(profile, create kv cache, warmup model\\) took ([\\d.]+)"),
            "attention_backend": backend[-1] if backend else None,
            "linear_kernels": sorted(set(re.findall(r"(\\w+LinearKernel)", text))),
            "warnings": [line.strip()[:300] for line in text.splitlines() if "WARNING" in line][:25]}


def chat_request(base, row, max_tokens, api_key=None):
    body = {"model": SERVED, "messages": messages(row), "temperature": 0.0, "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True}}
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {api_key}"} if api_key else {})}
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(), headers)
    t0, first, parts, usage = time.perf_counter(), None, [], {}
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            usage = chunk.get("usage") or usage
            for ch in chunk.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    if first is None:
                        first = time.perf_counter() - t0
                    parts.append(piece)
    return {"text": "".join(parts), "latency_s": time.perf_counter() - t0, "ttft_s": first,
            "completion_tokens": usage.get("completion_tokens"), "prompt_tokens": usage.get("prompt_tokens")}


def load_test(base, rows, concurrency, max_tokens, api_key=None):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(chat_request, base, r, max_tokens, api_key) for r in rows]
    wall = time.perf_counter() - t0
    res, errors = [], []
    for f in futures:
        try:
            res.append(f.result())
        except Exception as exc:  # recorded, not raised: one failed request should not lose the whole run
            res.append(None)
            errors.append(repr(exc)[:300])
    ok = [r for r in res if r]
    lat = np.array([r["latency_s"] for r in ok])
    ttft = np.array([r["ttft_s"] for r in ok if r["ttft_s"] is not None])
    decode = np.array([(r["completion_tokens"] - 1) / (r["latency_s"] - r["ttft_s"]) for r in ok
                       if r["ttft_s"] is not None and (r["completion_tokens"] or 0) > 1 and r["latency_s"] > r["ttft_s"]])
    out_tok = sum(r["completion_tokens"] or 0 for r in ok)
    in_tok = sum(r["prompt_tokens"] or 0 for r in ok)
    pct = lambda a, q: round(float(np.percentile(a, q)), 4) if len(a) else None
    stats = {"concurrency": concurrency, "requests": len(rows), "errors": len(errors), "error_samples": errors[:3],
             "wall_s": round(wall, 3), "requests_per_s": round(len(ok) / wall, 3),
             "output_tokens_per_s": round(out_tok / wall, 1), "prompt_tokens_per_s": round(in_tok / wall, 1),
             "mean_output_tokens": round(out_tok / max(len(ok), 1), 1),
             "mean_prompt_tokens": round(in_tok / max(len(ok), 1), 1),
             "latency_p50_s": pct(lat, 50), "latency_p95_s": pct(lat, 95),
             "latency_mean_s": round(float(lat.mean()), 4) if len(lat) else None,
             "ttft_p50_s": pct(ttft, 50), "ttft_p95_s": pct(ttft, 95),
             "decode_tokens_per_s_per_request_p50": pct(decode, 50)}
    return stats, [r["text"] if r else "" for r in res]


def vllm_cmd(py, model_path, extra=()):
    return [os.path.join(os.path.dirname(py), "vllm"), "serve", model_path, "--served-model-name", SERVED,
            "--port", str(VLLM_PORT), "--dtype", "half", "--max-model-len", str(MAX_MODEL_LEN),
            "--max-num-seqs", str(MAX_NUM_SEQS), "--gpu-memory-utilization", str(GPU_MEM_UTIL),
            "--kv-cache-memory-bytes", str(KV_CACHE_BYTES), "--seed", "0", *extra]


def bench_variant(name, cmd, gpus="0", is_vllm=True):
    base = f"http://127.0.0.1:{VLLM_PORT}/v1"
    log_path = os.path.join(WORK_DIR, "server-" + re.sub(r"[^\\w]+", "-", name).strip("-") + ".log")
    res = {"variant": name, "server_cmd": cmd, "gpus": gpus, "log": log_path, "error": None}
    INFER["variants"][name] = res
    proc = None
    try:
        res["gpu_mem_before_mib"] = GpuMem.read()
        with GpuMem() as mem:
            proc, res["startup_s"] = start_server(cmd, dict(os.environ, CUDA_VISIBLE_DEVICES=gpus), log_path,
                                                  VLLM_PORT)
            res["server_log"] = parse_vllm_log(log_path) if is_vllm else {}
            load_test(base, TEST[:8], min(8, max(CONCURRENCY)), MAX_NEW_TOKENS)  # warm-up, not measured
            res["loads"] = {}
            for c in CONCURRENCY:
                rows = TEST[:C1_REQUESTS] if c == 1 else TEST
                stats, texts = load_test(base, rows, c, MAX_NEW_TOKENS)
                res["loads"][f"c{c}"] = stats
                print(f"{name} c={c}: {stats['output_tokens_per_s']} output tok/s, latency p50/p95 "
                      f"{stats['latency_p50_s']}/{stats['latency_p95_s']} s, TTFT p50 {stats['ttft_p50_s']} s, "
                      f"errors {stats['errors']}", flush=True)
                if c == 8 or (c == max(CONCURRENCY) and 8 not in CONCURRENCY):
                    record(f"vLLM: {name}", TEST, texts)  # quality of the served model on the full test set
        res["gpu_mem_peak_mib"] = mem.peak
        res["quality_vs_hf_finetuned"] = compare(TEST, SCORES[FT_TAG], SCORES[f"vLLM: {name}"])
    except Exception as exc:  # one variant failing must not lose the others
        res["error"] = f"{type(exc).__name__}: {exc}"[:4000]
        print("FAILED", name, res["error"][:2000])
    finally:
        stop_server(proc)
        write_results()
    return res
"""

INFER_RUN = """CALIB_SCRIPT = '''import json, sys, time
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

src, out, calib, max_len = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
tok = AutoTokenizer.from_pretrained(src)
model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float16, device_map="auto")
texts = [json.loads(line)["text"] for line in open(calib, encoding="utf-8") if line.strip()]
enc = [tok(t, add_special_tokens=False, truncation=True, max_length=max_len) for t in texts]
ds = Dataset.from_list([{"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]} for e in enc])
t0 = time.time()
oneshot(model=model, dataset=ds, recipe=GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
        max_seq_length=max_len, num_calibration_samples=len(texts))
model.save_pretrained(out, save_compressed=True)
tok.save_pretrained(out)
print(f"GPTQ W4A16 in {time.time() - t0:.0f} s ->", out)
'''


def dir_gb(path):
    return round(sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs) / 1e9, 3)


def gptq_w4a16(merged_dir):
    torch_index = "https://download.pytorch.org/whl/" + ("cu130" if cuda13_ok() else "cu129")
    py = make_venv("llmc-venv", [["torch==2.11.0", "--default-index", torch_index], [f"llmcompressor=={LLMC_VERSION}"]])
    INFER["llmc_env"] = {"torch_index": torch_index, **versions(py, ["llmcompressor", "compressed-tensors", "torch",
                                                                     "transformers"])}
    calib = os.path.join(WORK_DIR, "calib.jsonl")
    with open(calib, "w", encoding="utf-8") as f:  # training rows only: the test set never touches the quantizer
        for r in random.Random(SEED).sample(train_rows, min(CALIB_SAMPLES, len(train_rows))):
            f.write(json.dumps({"text": full_text(r)}) + "\\n")
    script = os.path.join(WORK_DIR, "gptq_w4a16.py")
    with open(script, "w") as f:
        f.write(CALIB_SCRIPT)
    out = os.path.join(WORK_DIR, "gptq-w4a16")
    t0 = time.time()
    run_logged([py, script, merged_dir, out, calib, str(MAX_LEN)], os.path.join(WORK_DIR, "gptq.log"),
               env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
    INFER["gptq"] = {"seconds": round(time.time() - t0, 1), "calibration_rows": CALIB_SAMPLES,
                     "size_gb": dir_gb(out), "fp16_size_gb": dir_gb(merged_dir)}
    return out


INFER["driver"] = driver_info()
if SMOKE:
    # No GPU here: run the client, scoring and results code against a stub server that replays the test targets.
    # It is NOT vLLM and its numbers mean nothing; this only checks the code path.
    stub_path, answers = os.path.join(WORK_DIR, "stub_server.py"), os.path.join(WORK_DIR, "stub_answers.jsonl")
    with open(stub_path, "w") as f:
        f.write(STUB_SOURCE)
    with open(answers, "w", encoding="utf-8") as f:
        for r in TEST:
            f.write(json.dumps({"prompt_sha256": r["prompt_sha256"], "text": r["target"]}) + "\\n")
    INFER["status"] = "smoke test: stub server, not vLLM"
    bench_variant("stub server (smoke test only)", [sys.executable, stub_path, "--answers", answers, "--port",
                                                   str(VLLM_PORT), "--model", SERVED, "--token-delay", "0.002"],
                  is_vllm=False)
elif RUN_VLLM and DEVICE == "cuda":
    INFER["status"] = "started"
    try:
        if cuda13_ok():  # PyPI wheel, built for CUDA 13 (needs driver >= 580)
            steps, route = [[f"vllm=={VLLM_VERSION}"]], "PyPI (CUDA 13 build)"
        else:  # CUDA 12.9 build from the vLLM release page, with torch 2.13.0+cu129
            steps = [["torch==2.13.0", "torchvision==0.28.0", "torchaudio==2.11.0", "--default-index",
                      "https://download.pytorch.org/whl/cu129"], [VLLM_CU129_WHEEL]]
            route = "GitHub release +cu129 wheel"
        VLLM_PY = make_venv("vllm-venv", steps)
        INFER["vllm_env"] = {"route": route, **versions(VLLM_PY, ["vllm", "torch", "transformers", "compressed-tensors",
                                                                  "triton"])}
        print(INFER["vllm_env"])
    except Exception as exc:
        VLLM_PY = None
        INFER["status"] = f"vLLM install failed: {exc}"[:4000]
        print(INFER["status"])
    if VLLM_PY:
        bench_variant("fp16, TP=1", vllm_cmd(VLLM_PY, MERGED_DIR))
        if RUN_QUANT:
            try:
                bench_variant("GPTQ W4A16 (llm-compressor), TP=1", vllm_cmd(VLLM_PY, gptq_w4a16(MERGED_DIR)))
                quant_failed = bool(INFER["variants"]["GPTQ W4A16 (llm-compressor), TP=1"]["error"])
            except Exception as exc:
                INFER["gptq_error"] = f"{type(exc).__name__}: {exc}"[:4000]
                print("GPTQ step failed:", INFER["gptq_error"][:2000])
                quant_failed = True
            if quant_failed:  # vLLM 0.32 online FP8; on a T4 the weights are FP8 and the matmuls run as W8A16 (Marlin)
                bench_variant("FP8 weight-only (vLLM online), TP=1",
                              vllm_cmd(VLLM_PY, MERGED_DIR, ["--quantization", "fp8_per_tensor"]))
        if RUN_TP2 and len(GPUS) >= 2:
            # T4 pairs on Kaggle are connected over PCIe; custom all-reduce needs P2P/NVLink, so it is disabled
            bench_variant("fp16, TP=2", vllm_cmd(VLLM_PY, MERGED_DIR, ["--tensor-parallel-size", "2",
                                                                       "--disable-custom-all-reduce"]), gpus="0,1")
        else:
            INFER["tp2"] = f"skipped: {len(GPUS)} GPU(s) in this session"
        INFER["status"] = "done"
else:
    INFER["status"] = "skipped (RUN_VLLM is False or no GPU)"
write_results()
"""

RESULTS_CELL = """TIMES["total_s"] = round(time.time() - T_START, 1)
write_results("complete")
table()
print()
cols = ("startup_s", "weights GiB", "KV tokens", "peak MiB", "c1 p50 s", "c1 p95 s", "c1 TTFT s", "c8 tok/s",
        "c8 p95 s", "c16 tok/s", "c16 p95 s", "behaviour", "answer acc")
print(f"{'variant':38}" + "".join(f"{c:>12}" for c in cols))
for name, v in INFER["variants"].items():
    if v.get("error"):
        print(f"{name:38} FAILED: {v['error'][:150]}")
        continue
    L, S = v.get("loads", {}), v.get("server_log", {})
    q = RESULTS["test"].get(f"vLLM: {name}", {}).get("all", {})
    peak = max(v["gpu_mem_peak_mib"].values()) if v.get("gpu_mem_peak_mib") else None
    vals = (v.get("startup_s"), S.get("model_weights_gib_per_gpu"), S.get("kv_cache_tokens"), peak,
            L.get("c1", {}).get("latency_p50_s"), L.get("c1", {}).get("latency_p95_s"), L.get("c1", {}).get("ttft_p50_s"),
            L.get("c8", {}).get("output_tokens_per_s"), L.get("c8", {}).get("latency_p95_s"),
            L.get("c16", {}).get("output_tokens_per_s"), L.get("c16", {}).get("latency_p95_s"),
            q.get("behaviour_accuracy", {}).get("mean"), q.get("answer_accuracy", {}).get("mean"))
    print(f"{name[:38]:38}" + "".join(f"{str(x):>12}" for x in vals))
ADAPTER_ZIP = shutil.make_archive(os.path.join(OUT_DIR, "sales_qlora_adapter"), "zip", ADAPTER_DIR)
print("saved", RESULTS_PATH, "and", ADAPTER_ZIP, "| total minutes", round(TIMES["total_s"] / 60, 1))
try:
    from google.colab import files
    files.download(RESULTS_PATH)
    files.download(ADAPTER_ZIP)
except ImportError:
    print("Kaggle: open the Output panel on the right (" + OUT_DIR + ") and download results.json and "
          "sales_qlora_adapter.zip")
"""

TUNNEL_CELL = """# Optional (RUN_TUNNEL = True): serve the fine-tuned model and open a temporary public HTTPS tunnel, so that
# llm_finetune/eval_endpoint.py on your own machine can score it through the app's openai_compat provider.
# The server requires a random API key. The key is printed below: clear this cell's output before sharing the
# notebook. Stop both processes with stop_server(TUNNEL_SERVER); TUNNEL.terminate()
if RUN_TUNNEL and not SMOKE and "VLLM_PY" in globals() and VLLM_PY:
    key = secrets.token_urlsafe(24)
    TUNNEL_SERVER, _ = start_server(vllm_cmd(VLLM_PY, MERGED_DIR, ["--api-key", key]),
                                    dict(os.environ, CUDA_VISIBLE_DEVICES="0"),
                                    os.path.join(WORK_DIR, "server-tunnel.log"), VLLM_PORT)
    cf = os.path.join(WORK_DIR, "cloudflared")
    if not os.path.exists(cf):
        urllib.request.urlretrieve("https://github.com/cloudflare/cloudflared/releases/download/"
                                   f"{CLOUDFLARED_VERSION}/cloudflared-linux-amd64", cf)
        os.chmod(cf, 0o755)
    tlog = os.path.join(WORK_DIR, "tunnel.log")
    TUNNEL = subprocess.Popen([cf, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{VLLM_PORT}"],
                              stdout=open(tlog, "w"), stderr=subprocess.STDOUT)
    url = None
    for _ in range(60):
        found = re.findall(r"https://[a-z0-9-]+\\.trycloudflare\\.com", tail(tlog, 200))
        if found:
            url = found[-1]
            break
        time.sleep(2)
    print("tunnel:", url or "not up yet, see " + tlog)
    print(f"LLM_PROVIDER=openai_compat LLM_BASE_URL={url}/v1 LLM_MODEL={SERVED} LLM_API_KEY={key}")
else:
    print("tunnel skipped (set RUN_TUNNEL = True in the first code cell; needs the vLLM section to have run)")
"""


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


def fill(template: str, **values) -> str:
    for k, v in values.items():
        template = template.replace(f"<<{k}>>", repr(v))
    assert not re.search(r"<<[A-Z_]+>>", template), "unfilled placeholder"
    return template


def _function(fn) -> str:
    """Source of a function or method as a top-level function (dedented, @staticmethod dropped)."""
    src = textwrap.dedent(inspect.getsource(fn))
    return "".join(line for line in src.splitlines(keepends=True) if line.strip() != "@staticmethod")


def app_code() -> str:
    oracle = ", ".join(f"{role!r}: ({{{', '.join(repr(t) for t in sorted(types))}}}, {level})"
                       for role, (types, level) in run_eval.ORACLE.items())
    scoring_src = (HERE / "scoring.py").read_text(encoding="utf-8")
    body = scoring_src.split("# <<< app bindings\n", 1)[1]
    parts = [
        "# Verbatim copies of the app code used for validation and scoring, generated by llm_finetune/build_notebook.py\n"
        "# from backend/app/{embeddings,offline,models,llm,agent}.py and backend/eval/run_eval.py, followed by\n"
        "# llm_finetune/scoring.py.\n",
        "from __future__ import annotations\n\nimport json\nimport re\nfrom types import SimpleNamespace\n\n"
        "import numpy as np\nfrom pydantic import BaseModel, Field\n\n",
        f"STOPWORDS = frozenset({sorted(embeddings.STOPWORDS)!r})\nTOKEN_RE = re.compile({embeddings.TOKEN_RE.pattern!r})\n\n\n",
        inspect.getsource(embeddings.tokenize), "\n\n", inspect.getsource(offline.terms), "\n\n",
        inspect.getsource(models.Claim), "\n\n", inspect.getsource(models.Answer), "\n\n",
        inspect.getsource(llm.extract_json), "\n\n", f"GROUNDED_THRESHOLD = {GROUNDED_THRESHOLD!r}\n\n\n",
        _function(Agent._filter_citations), "\n\n", _function(Agent._groundedness), "\n\n",
        f"ORACLE = {{{oracle}}}\nLEVEL = {run_eval.LEVEL!r}\n\n\n", inspect.getsource(run_eval.allowed), body,
        APP_FOOTER,
    ]
    return "".join(parts)


def build(data: Path = DATA, out: Path = OUT) -> Path:
    names = ("train.jsonl", "val.jsonl", "test.jsonl", "dataset_report.json")
    files = {n: (data / n).read_text(encoding="utf-8") for n in names}
    frozen = (data / "test.sha256").read_text().split()[0]
    assert hashlib.sha256((data / "test.jsonl").read_bytes()).hexdigest() == frozen, "test.jsonl changed"
    raw = json.dumps(files, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    blob = base64.b64encode(gzip.compress(raw, compresslevel=9, mtime=0)).decode("ascii")
    counts = {n: sum(1 for line in files[n].splitlines() if line.strip()) for n in names[:3]}
    fmt = dict(vllm=VLLM, llmc=LLMC)
    cells = [
        _md(INTRO.format(n_train=counts["train.jsonl"], n_val=counts["val.jsonl"], n_test=counts["test.jsonl"],
                         model=MODEL[0], **fmt)),
        _code(fill(CONFIG, MODEL=MODEL, SMOKE_MODEL=SMOKE_MODEL, VLLM=VLLM, LLMC=LLMC, UV=UV, WHEEL=VLLM_CU129_WHEEL,
                   CLOUDFLARED=CLOUDFLARED)),
        _md("## 1. Install pinned training libraries (the preinstalled CUDA build of torch is used)"),
        _code(fill(INSTALL, PINS=PINS)),
        _md("## 2. Embedded data and the app code used for validation and scoring"),
        _code(fill(DATA_CELL, BLOB=blob, TEST_SHA=frozen)),
        _code(app_code()),
        _md("## 3. Tokenizer, prompt format and token lengths\n\nThe user message is the exact prompt the app "
            "builds for the `answer` task (task, question, the `Answer` JSON schema, spotlighted evidence), with the "
            "app's system prompt, wrapped in the model's chat template."),
        _code(MODEL_CELL),
        _code(EVAL_HELPERS),
        _md("## 4. Base model on the frozen test set\n\nThe app's offline extractive generator (its default, not an "
            "LLM) and the targets themselves (a scorer self-check, all 1.0) are listed for reference."),
        _code(BASE_EVAL),
        _md("## 5. QLoRA fine-tuning (4-bit NF4 base, LoRA adapters, fp16 autocast, loss on the answer tokens only)"),
        _code(TRAIN_CELL),
        _md("## 6. Merge the adapter into the fp16 base, evaluate the merged model\n\nPaired bootstrap 95% "
            "intervals for fine-tuned minus base are printed at the end; `wording=held_out` is the half of the test "
            "set asked with a question wording never used in training."),
        _code(MERGE_CELL),
        _md(INFER_INTRO.format(conc=", ".join(map(str, (1, 8, 16))), c1=32, max_new=320, **fmt)),
        _code(fill(INFER_HELPERS, STUB=(HERE / "stub_server.py").read_text(encoding="utf-8"))),
        _code(INFER_RUN),
        _md("## 8. Results"),
        _code(RESULTS_CELL),
        _md("## 9. Optional: tunnel for `llm_finetune/eval_endpoint.py`"),
        _code(TUNNEL_CELL),
    ]
    nb = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
        "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        c["id"] = hashlib.sha1(f"{i}".encode()).hexdigest()[:8]
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB; data blob {len(blob) / 1024:.0f} KiB, "
          f"train {counts['train.jsonl']}, val {counts['val.jsonl']}, test {counts['test.jsonl']})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    build(a.data, a.out)
