"""API smoke tests (no LLM calls: only endpoints that don't need one)."""
from fastapi.testclient import TestClient

from dsagent.api.main import app

client = TestClient(app)


def test_ui_and_sources_served():
    r = client.get("/")
    assert r.status_code == 200 and "Dataset Agent" in r.text
    s = client.get("/api/sources")
    assert s.status_code == 200 and isinstance(s.json(), list)


def test_unknown_job_and_bad_input():
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.post("/api/jobs", json={"problem": ""}).status_code == 422
    assert client.get("/api/jobs/nope/download/../../.env").status_code == 404
