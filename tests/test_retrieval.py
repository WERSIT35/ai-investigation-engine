from app.bm25 import BM25Index, tokenize
from app.fusion import diversify_by_doc, reciprocal_rank_fusion


def test_tokenize_stems_and_drops_stopwords():
    assert tokenize("The services were delayed") == ["servic", "delay"]
    assert "payments.settled" in tokenize("topic payments.settled")


def test_bm25_ranks_and_supports_removal():
    ix = BM25Index()
    ix.add("a", "ledger service connection pool exhausted")
    ix.add("b", "notification service webhooks delayed")
    ix.add("c", "ledger ledger ledger")
    res = ix.search("ledger pool", 3)
    assert res[0][0] == "a"
    ix.remove("a")
    assert "a" not in ix and all(k != "a" for k, _ in ix.search("ledger pool", 3))
    assert ix.search("zzz", 3) == []


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion({"dense": ["x", "y", "z"], "bm25": ["y", "q"]}, k=60)
    keys = [k for k, _, _ in fused]
    assert keys[0] == "y"  # rank 2 + rank 1 beats rank 1 alone
    assert fused[0][2] == {"dense": 2, "bm25": 1}


def test_diversify_caps_per_doc_and_backfills():
    items = [("d1", 1), ("d1", 2), ("d1", 3), ("d2", 4)]
    out = diversify_by_doc(items, lambda x: x[0], max_per_doc=1, limit=3)
    assert out == [("d1", 1), ("d2", 4), ("d1", 2)]


def test_hybrid_search_finds_synthetic_doc(loaded_engine):
    hits = loaded_engine.search("Which Kafka topic does notification-service consume?", top_k=5)
    assert "syn::architecture-overview" in [h.doc_id for h in hits]
    for mode in ("dense", "bm25"):
        assert loaded_engine.search("ledger rollback runbook", top_k=3, mode=mode)
