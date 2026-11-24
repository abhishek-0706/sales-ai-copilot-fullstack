import pytest

from app.embeddings import HashingEmbedder
from app.security import can_read

QUERIES = ["discount approved deal desk", "competitor evaluating", "security review", "opportunity amount stage",
           "email concern", "meeting agenda", "pricing policy discount authority", "GDPR passenger data"]


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder(384)
    a, b = e.embed_query("vector search for sales"), e.embed_query("vector search for sales")
    assert (a == b).all() and abs(float((a * a).sum()) - 1.0) < 1e-5


def test_hybrid_search_finds_relevant_record(agent, users):
    hits = agent.index.search(users["u-ae-01"], "Velmora Logistics competitor Kordon Cloud", k=6)
    assert "ACT-0003" in [h.evidence.source_id for h in hits]
    assert all(h.vector_rank or h.keyword_rank for h in hits)


@pytest.mark.parametrize("mode", ["hybrid", "vector", "keyword"])
def test_access_filter_is_applied_inside_search(agent, users, mode):
    for user in users.values():
        for q in QUERIES:
            for h in agent.index.search(user, q, k=10, mode=mode):
                e = h.evidence
                assert can_read(user, territory=e.territory, record_type=e.source_type, sensitivity=e.sensitivity), \
                    (user.id, q, e.source_id)


def test_sdr_never_retrieves_restricted_or_confidential(agent, users):
    ids = {h.evidence.record_id for h in agent.index.search(users["u-sdr-01"], "maximum approved discount", k=20)}
    assert not ids & {"ACT-0047", "PRD-007", "OPP-0011", "ACT-0003"}


def test_account_scoped_search(agent, users):
    hits = agent.index.search(users["u-mgr-01"], "deal risk next step", k=10, account_id="ACC-005")
    assert {h.evidence.account_id for h in hits} <= {"ACC-005", None}
