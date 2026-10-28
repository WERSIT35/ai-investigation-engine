from app.models import Document


def test_ingest_update_delete_and_dedup(engine):
    d1 = Document("d1", "Doc one", "Alpha beta gamma. " * 5, "s", "txt")
    rep = engine.ingest([d1])
    assert rep.added == ["d1"] and rep.chunks_added >= 1
    assert engine.ingest([Document("d1", "Doc one", "Alpha beta gamma. " * 5, "s", "txt")]).unchanged == ["d1"]

    # a second document whose chunk duplicates d1's content (after normalisation) is skipped
    dup = Document("d2", "Doc one", "alpha BETA gamma " * 5, "s", "txt")
    rep = engine.ingest([dup])
    assert rep.duplicate_chunks_skipped == 1

    rep = engine.ingest([Document("d1", "Doc one", "Completely new content about delta.", "s", "txt")])
    assert rep.updated == ["d1"]
    assert "delta" in engine.get_document_chunks("d1")[0].text
    assert engine.search("alpha beta gamma", mode="bm25") == []  # old chunks are gone from BM25 too

    v = engine.version
    assert engine.delete("d1") and not engine.delete("d1")
    assert engine.version == v + 1 and engine.store.count() == 0


def test_rebuild_from_store_restores_state(loaded_engine):
    before = (len(loaded_engine.docs), len(loaded_engine.chunks))
    loaded_engine.rebuild_from_store()
    assert (len(loaded_engine.docs), len(loaded_engine.chunks)) == before
    assert loaded_engine.docs["syn::postmortem-inc-1042"].n_chunks >= 1
