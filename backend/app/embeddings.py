"""Offline embedders.

- HashingEmbedder (default): deterministic feature hashing of words, word bigrams and
  character trigrams. No model, no download, identical results on every machine. It
  captures lexical overlap and some morphology, not real semantics.
- FastEmbedEmbedder (EMBEDDER=fastembed): BAAI/bge-small-en-v1.5 (ONNX, 384 dims) run
  locally through fastembed. Needs a one-off model download, then works offline.
- FineTunedEmbedder (EMBEDDER=finetuned): bge-small-en-v1.5 fine-tuned on this corpus
  (backend/finetune), exported to ONNX and loaded from disk through fastembed. No torch at runtime.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Protocol

import numpy as np

STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the their this to was were what "
    "which who with about does do did how when where why our we they them is are any can will would should "
    "me my you your i tell give show list".split())
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _features(self, text: str) -> list[tuple[str, float]]:
        toks = tokenize(text)
        feats = [(f"w:{t}", 1.0) for t in toks]
        feats += [(f"b:{a}_{b}", 0.6) for a, b in zip(toks, toks[1:])]
        for t in toks:
            padded = f"#{t}#"
            feats += [(f"c:{padded[i:i + 3]}", 0.25) for i in range(len(padded) - 2)]
        return feats

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for feat, weight in self._features(text):
            h = hashlib.blake2b(feat.encode(), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if h[4] & 1 else -1.0
            vec[idx] += sign * weight
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._embed(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)


class FastEmbedEmbedder:
    MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, cache_dir: Path):
        from fastembed import TextEmbedding  # optional dependency

        self._model = TextEmbedding(self.MODEL, cache_dir=str(cache_dir))
        self.dim = 384
        self.name = f"fastembed:{self.MODEL}"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(list(self._model.passage_embed(texts)), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(next(iter(self._model.query_embed([text]))), dtype=np.float32)


FINETUNED_DIR_NAME = "bge-small-en-v1.5-sales-ft"


class FineTunedEmbedder(FastEmbedEmbedder):
    FASTEMBED_NAME = "local/bge-small-en-v1.5-onnx-dir"
    REQUIRED_FILES = ("model.onnx", "tokenizer.json", "tokenizer_config.json")

    def __init__(self, model_dir: Path, name: str = f"finetuned:{FINETUNED_DIR_NAME}"):
        missing = [f for f in self.REQUIRED_FILES if not (model_dir / f).is_file()]
        if missing:
            raise FileNotFoundError(
                f"EMBEDDER=finetuned but {', '.join(missing)} not found in {model_dir}. Build the model with "
                "backend/finetune (see backend/finetune/README.md) or set EMBEDDER=fastembed or hash.")
        from fastembed import TextEmbedding  # optional dependency
        from fastembed.common.model_description import ModelSource, PoolingType

        if not any(m["model"] == self.FASTEMBED_NAME for m in TextEmbedding.list_supported_models()):
            TextEmbedding.add_custom_model(model=self.FASTEMBED_NAME, pooling=PoolingType.CLS, normalization=True,
                                           sources=ModelSource(hf=self.FASTEMBED_NAME), dim=384,
                                           model_file="model.onnx")
        self._model = TextEmbedding(self.FASTEMBED_NAME, specific_model_path=str(model_dir))
        self.dim = 384
        self.name = name


def make_embedder(kind: str, dim: int, cache_dir: Path) -> Embedder:
    if kind == "fastembed":
        return FastEmbedEmbedder(cache_dir)
    if kind == "finetuned":
        return FineTunedEmbedder(cache_dir / FINETUNED_DIR_NAME)
    if kind == "hash":
        return HashingEmbedder(dim)
    raise ValueError(f"unknown EMBEDDER '{kind}' (expected 'hash', 'fastembed' or 'finetuned')")
