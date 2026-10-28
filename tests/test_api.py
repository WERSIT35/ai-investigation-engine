from dataclasses import replace

from fastapi.testclient import TestClient

from app.main import create_app


def make_client(settings, engine):
    return TestClient(create_app(settings, engine=engine))


def test_upload_list_search_ask_delete(settings, engine):
    with make_client(settings, engine) as c:
        r = c.post("/documents", files=[("files", ("notes.md", b"# Notes\nThe vault door code changed on Monday.", "text/markdown")),
                                        ("files", ("x.json", b'[{"doc_id":"j1","text":"Kafka lag on Friday"}]', "application/json"))])
        assert r.status_code == 200 and r.json()["counts"]["added"] == 2
        assert c.get("/documents").json()["total"] == 2
        assert c.get("/documents/file::notes.md").json()["chunks"]
        hits = c.post("/search", json={"query": "vault door code", "mode": "bm25"}).json()["hits"]
        assert hits[0]["doc_id"] == "file::notes.md"
        a = c.post("/ask", json={"question": "When did the vault door code change?"}).json()
        assert a["status"] == "degraded"  # no LLM configured in tests
        r = c.put("/documents/file::notes.md", files={"file": ("n.md", b"# Notes\nNew text.", "text/markdown")})
        assert r.json()["updated"] == ["file::notes.md"]
        assert c.delete("/documents/j1").status_code == 200
        assert c.delete("/documents/j1").status_code == 404
        assert c.get("/health").json()["documents"] == 1


def test_validation_and_auth(settings, engine):
    with make_client(settings, engine) as c:
        assert c.post("/documents", files={"files": ("a.exe", b"x", "application/octet-stream")}).status_code == 415
        assert c.post("/documents", files={"files": ("a.json", b"{bad", "application/json")}).status_code == 422
        assert c.post("/ask", json={"question": ""}).status_code == 422
    secured = replace(settings, admin_token="s3cret")
    with make_client(secured, engine) as c:
        assert c.post("/documents/text", json={"title": "t", "text": "x"}).status_code == 401
        ok = c.post("/documents/text", json={"title": "t", "text": "x"}, headers={"X-Admin-Token": "s3cret"})
        assert ok.status_code == 200
        assert c.get("/").status_code == 200


def test_rate_limit(settings, engine):
    with make_client(replace(settings, ask_rate_per_min=2), engine) as c:
        codes = [c.post("/ask", json={"question": "anything here"}).status_code for _ in range(3)]
        assert codes == [200, 200, 429]


def test_rate_limit_is_per_client_behind_a_proxy(settings, engine):
    with make_client(replace(settings, ask_rate_per_min=1, trust_proxy=True), engine) as c:
        def ask(xff):
            return c.post("/ask", json={"question": "anything here"}, headers={"X-Forwarded-For": xff}).status_code
        assert ask("1.1.1.1") == 200
        assert ask("2.2.2.2") == 200  # another visitor has their own bucket, not the proxy's shared one
        assert ask("9.9.9.9, 1.1.1.1") == 429  # a forged left-most entry doesn't reset the real client's bucket
    with make_client(replace(settings, ask_rate_per_min=1), engine) as c:  # proxy not trusted: header ignored
        codes = [c.post("/ask", json={"question": "anything here"}, headers={"X-Forwarded-For": ip}).status_code
                 for ip in ("1.1.1.1", "2.2.2.2")]
        assert codes == [200, 429]
