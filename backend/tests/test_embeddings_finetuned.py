import numpy as np
import pytest

from app.config import BACKEND_DIR
from app.embeddings import FINETUNED_DIR_NAME, FineTunedEmbedder, make_embedder
from app.vectorstore import HybridStore

MODEL_DIR = BACKEND_DIR / ".models" / FINETUNED_DIR_NAME


def test_missing_model_files_fail_with_instructions(tmp_path):
    with pytest.raises(FileNotFoundError) as err:
        make_embedder("finetuned", 384, tmp_path)
    msg = str(err.value)
    assert "model.onnx" in msg and "backend/finetune" in msg and "EMBEDDER=fastembed" in msg


def test_unknown_embedder_lists_finetuned(tmp_path):
    with pytest.raises(ValueError, match="finetuned"):
        make_embedder("nope", 384, tmp_path)


@pytest.fixture(scope="module")
def finetuned():
    pytest.importorskip("fastembed")
    if not (MODEL_DIR / "model.onnx").is_file():
        pytest.skip(f"fine-tuned model not built at {MODEL_DIR}")
    return FineTunedEmbedder(MODEL_DIR)


def test_finetuned_embedder_returns_normalised_384_dim_vectors(finetuned):
    docs = finetuned.embed_documents(["Deal risk: procurement requires three vendor quotes.", "Edge latency"])
    q = finetuned.embed_query("what does the insurer's purchasing team insist on")
    assert docs.shape == (2, 384) and q.shape == (384,)
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-4) and abs(np.linalg.norm(q) - 1.0) < 1e-4
    assert float(docs[0] @ q) > float(docs[1] @ q)


def test_finetuned_embedder_works_in_hybrid_store(finetuned, agent, users):
    store = HybridStore(finetuned)
    store.build(agent.store.chunks())
    hits = store.search(users["u-ae-02"], "Which competitor hosts the airline's contact centre?", k=6, mode="vector")
    assert "ACT-0029" in [h.evidence.source_id for h in hits]
