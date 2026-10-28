import json

import pytest

from app.loaders import ParseError, UnsupportedFormatError, load_bytes


def test_markdown_front_matter_sets_id_title_metadata():
    data = b"---\ndoc_id: x1\ntitle: Hello\ndate: 2026-03-01\n---\n# Ignored heading\nBody text."
    [d] = load_bytes("a.md", data)
    assert (d.doc_id, d.title, d.metadata["date"], d.fmt) == ("x1", "Hello", "2026-03-01", "md")
    assert "Body text." in d.text


def test_txt_defaults_from_filename():
    [d] = load_bytes("Notes.TXT", b"first line\n\nsecond   paragraph")
    assert d.doc_id == "file::notes.txt"
    assert d.text == "first line\n\nsecond paragraph"


def test_json_list_and_jsonl():
    recs = [{"id": "a", "title": "A", "text": "alpha", "service": "s1"}, {"content": "beta"}, {"title": "empty"}]
    docs = load_bytes("r.json", json.dumps(recs).encode())
    assert [d.doc_id for d in docs] == ["a", "file::r.json#1"]
    assert docs[0].metadata["service"] == "s1"
    docs = load_bytes("r.jsonl", b'{"doc_id":"x","text":"one"}\n\n{"doc_id":"y","text":"two"}\n')
    assert [d.doc_id for d in docs] == ["x", "y"]


def test_html_strips_scripts_and_reads_meta():
    html = b"<html><head><title>T</title><meta name='doc_id' content='h1'></head><body><p>Hi</p><script>evil()</script></body></html>"
    [d] = load_bytes("p.html", html)
    assert d.doc_id == "h1" and d.title == "T" and "evil" not in d.text and "Hi" in d.text


def test_pdf(synthetic_docs):
    pdf = [d for d in synthetic_docs if d.fmt == "pdf"]
    assert pdf and "99.95%" in pdf[0].text and pdf[0].metadata["pages"] >= 1


def test_errors():
    with pytest.raises(UnsupportedFormatError):
        load_bytes("x.exe", b"MZ")
    with pytest.raises(ParseError):
        load_bytes("bad.json", b"{not json")
    with pytest.raises(ParseError):
        load_bytes("broken.pdf", b"%PDF-garbage")
