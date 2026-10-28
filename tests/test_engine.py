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
    assert rep.duplicate_docs == ["d2"] and rep.added == [] and "d2" not in engine.docs  # reported, not "added"

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


def test_search_is_safe_during_concurrent_ingest(engine):
    # FastAPI runs requests in a thread pool: a search must not crash while an upload mutates the BM25 index
    import threading

    engine.ingest([Document(f"d{i}", f"ledger {i}", f"ledger service payment {i} " * 20, "s", "txt") for i in range(100)])
    errors = []

    def writer():
        for i in range(100, 400):
            engine.ingest([Document(f"d{i}", f"ledger {i}", f"ledger payment incident {i} " * 5, "s", "txt")])

    def reader():
        for _ in range(800):
            try:
                engine.search("ledger payment", mode="hybrid")
                engine.find_docs("ledger 1")
            except Exception as exc:  # noqa: BLE001 - any exception is the failure being tested
                errors.append(repr(exc))
                return

    threads = [threading.Thread(target=writer)] + [threading.Thread(target=reader) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(engine.docs) == 400


def test_failed_update_keeps_the_old_version(engine):
    engine.ingest([Document("d1", "Doc", "Original text about the ledger.", "s", "txt")])

    class Broken:
        dim, name = engine.embedder.dim, "broken"

        def embed_documents(self, texts):
            raise RuntimeError("embedding service down")

        def embed_query(self, text):
            raise RuntimeError("embedding service down")

    good, engine.embedder = engine.embedder, Broken()
    try:
        engine.ingest([Document("d1", "Doc", "Replacement text about kafka.", "s", "txt")])
    except RuntimeError:
        pass
    engine.embedder = good
    assert "Original" in engine.get_document_chunks("d1")[0].text
    assert engine.store.count() == 1
    engine.rebuild_from_store()  # Qdrant still holds the old copy too
    assert "Original" in engine.get_document_chunks("d1")[0].text


def test_update_that_shrinks_or_repeats_content(engine):
    long = " ".join(f"Sentence number {i} about the ledger service." for i in range(40))
    engine.ingest([Document("d1", "Doc", long, "s", "txt")], strategy="fixed", size=200, overlap=0)
    assert engine.docs["d1"].n_chunks > 3

    # keeps the first chunk verbatim, drops the rest: the unchanged chunk must not be skipped as a "duplicate"
    first = engine.get_document_chunks("d1")[0].text
    rep = engine.ingest([Document("d1", "Doc", first + "\n\nA new closing line.", "s", "txt")],
                        strategy="fixed", size=200, overlap=0)
    assert rep.updated == ["d1"] and rep.duplicate_chunks_skipped == 0
    texts = [h.text for h in engine.get_document_chunks("d1")]
    assert texts[0] == first and "closing line" in texts[-1]
    assert engine.store.count() == len(texts) == engine.docs["d1"].n_chunks  # stale positions were deleted
    assert not any("number 30 " in t for t in texts)  # dropped content is gone from the index
