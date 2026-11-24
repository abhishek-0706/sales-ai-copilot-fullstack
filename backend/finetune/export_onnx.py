"""Exports a sentence-transformers BERT model to ONNX in the layout fastembed loads from a directory,
then checks the ONNX output against the torch model.

Run from backend/ with the training venv (see README.md for the full commands):
  python finetune/export_onnx.py --model .models/st/sales-ft/final --out .models/bge-small-en-v1.5-sales-ft
  python finetune/export_onnx.py --model BAAI/bge-small-en-v1.5 --out .models/bge-small-en-v1.5-base-onnx
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(BACKEND / ".models" / "hf"))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

CHECK_TEXTS = ["Which competitor is Velmora Logistics evaluating?",
               "Deal risk: the hospital security review is pending and patient data residency is mandatory.",
               "edge appliance latency"]


class LastHiddenState(torch.nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask,
                         token_type_ids=token_type_ids).last_hidden_state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    st = SentenceTransformer(args.model, device="cpu")
    bert, tok = st[0].auto_model.eval(), st.tokenizer
    args.out.mkdir(parents=True, exist_ok=True)

    # Trace with a padded batch: tracing freezes transformers' "mask is all ones, drop it" branch, and an
    # unpadded trace input would export a graph that ignores padding.
    enc = tok(CHECK_TEXTS, padding=True, return_tensors="pt")
    inputs = (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"])
    names = ["input_ids", "attention_mask", "token_type_ids"]
    axes = {n: {0: "batch", 1: "sequence"} for n in names} | {"last_hidden_state": {0: "batch", 1: "sequence"}}
    with torch.no_grad():
        torch.onnx.export(LastHiddenState(bert), inputs, str(args.out / "model.onnx"), input_names=names,
                          output_names=["last_hidden_state"], dynamic_axes=axes, opset_version=17, dynamo=False)
    tok.save_pretrained(str(args.out))
    bert.config.save_pretrained(str(args.out))
    cfg_path = args.out / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["model_max_length"] = st.max_seq_length
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")

    sess = ort.InferenceSession(str(args.out / "model.onnx"), providers=["CPUExecutionProvider"])
    diff = 0.0
    for texts in (CHECK_TEXTS, CHECK_TEXTS[2:], [t * 4 for t in CHECK_TEXTS] + CHECK_TEXTS):
        batch = tok(texts, padding=True, truncation=True, max_length=st.max_seq_length, return_tensors="np")
        cls = sess.run(None, {n: batch[n].astype(np.int64) for n in names})[0][:, 0]
        onnx_emb = cls / np.linalg.norm(cls, axis=1, keepdims=True)
        diff = max(diff, float(np.abs(onnx_emb - st.encode(texts, normalize_embeddings=True)).max()))
    print(f"exported to {args.out}; max abs difference ONNX vs torch over 3 batch shapes = {diff:.2e}")
    if diff > 1e-4:
        raise SystemExit("ONNX output does not match the torch model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
