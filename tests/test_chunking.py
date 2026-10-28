import pytest

from app.chunking import chunk_document, fixed_chunks, recursive_chunks
from app.models import Document

TEXT = ("Para one sentence one. Para one sentence two is a bit longer.\n\n"
        "Second paragraph starts here. It has two sentences.\n\n" + "Word " * 120)


@pytest.mark.parametrize("fn", [fixed_chunks, recursive_chunks])
def test_chunks_respect_size_and_cover_text(fn):
    chunks = fn(TEXT, 120, 30)
    assert all(len(c) <= 120 for c in chunks)
    joined = " ".join(chunks)
    for w in ("one.", "longer.", "starts", "sentences."):
        assert w in joined


def test_recursive_keeps_sentences_whole():
    chunks = recursive_chunks(TEXT, 120, 0)
    assert "Para one sentence one. Para one sentence two is a bit longer." in chunks[0]


def test_fixed_overlap_repeats_words():
    chunks = fixed_chunks(" ".join(f"w{i}" for i in range(100)), 60, 20)
    assert len(chunks) > 1
    assert set(chunks[0].split()) & set(chunks[1].split())


def test_chunk_document_ids_are_deterministic():
    d = Document("d1", "T", TEXT, "s", "txt")
    a, b = chunk_document(d, "recursive", 120, 20), chunk_document(d, "recursive", 120, 20)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert a[0].metadata["chunk_strategy"] == "recursive"
    with pytest.raises(ValueError):
        chunk_document(d, "nope")
