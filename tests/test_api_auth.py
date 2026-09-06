"""The bearer-token gate on /api/*.

Once deployed, every accepted request spends GPU time on Modal, and the
per-IP rate limiting lives in the portfolio's proxy rather than here — so
an unprotected public URL has no budget ceiling at all.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.auth import API_KEY_ENV, configured_api_key

KEY = "test-key-abcdef0123456789"


def _post(client: TestClient, headers: dict | None = None):
    return client.post(
        "/api/match",
        files={"job_listing": ("job.txt", b"Requirements React", "text/plain")},
        headers=headers or {},
    )


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setenv(API_KEY_ENV, KEY)
    # Never reached when auth rejects; patched so an accepted request fails
    # on something obvious rather than calling a real model.
    monkeypatch.setattr("src.api.routes.load_serving_cv", lambda: None)
    return TestClient(create_app())


def test_request_without_a_token_is_401(client):
    response = _post(client)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_wrong_token_is_403_not_401(client):
    """403, not 401: a 401 invites a retry with different credentials, and
    there is exactly one valid token."""
    assert _post(client, {"Authorization": f"Bearer {KEY}x"}).status_code == 403


@pytest.mark.parametrize(
    "header",
    ["", "Bearer", "Bearer ", KEY, f"Basic {KEY}", f"bearer{KEY}"],
)
def test_malformed_authorization_headers_are_rejected(client, header):
    assert _post(client, {"Authorization": header}).status_code in (401, 403)


def test_correct_token_gets_past_the_gate(client):
    """Not a 401/403 — it fails later, inside the handler, which is what
    proves the dependency let it through."""
    assert _post(client, {"Authorization": f"Bearer {KEY}"}).status_code not in (401, 403)


def test_lowercase_bearer_scheme_is_accepted(client):
    """RFC 7235 makes the scheme case-insensitive; rejecting "bearer" would
    be a confusing failure against a conformant client."""
    assert _post(client, {"Authorization": f"bearer {KEY}"}).status_code not in (401, 403)


def test_unset_key_leaves_the_api_open_for_local_use(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    monkeypatch.setattr("src.api.routes.load_serving_cv", lambda: None)

    assert configured_api_key() == ""
    assert _post(TestClient(create_app())).status_code not in (401, 403)


def test_starting_without_a_key_warns(monkeypatch):
    """"Forgot to set it" is the dangerous case, so it must not be silent."""
    monkeypatch.delenv(API_KEY_ENV, raising=False)

    with pytest.warns(UserWarning, match="unauthenticated"):
        create_app()


def test_upload_endpoints_are_also_behind_the_gate(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, KEY)
    monkeypatch.setenv("ALLOW_CV_UPLOAD", "1")
    client = TestClient(create_app())

    for path in ("/api/compare", "/api/ingest"):
        response = client.post(path, files={"candidate_cv": ("cv.txt", b"x", "text/plain")})
        assert response.status_code == 401, f"{path} is reachable without a token"


def test_openapi_stays_open_for_the_health_check(monkeypatch):
    """The portfolio's /api/cv/health pings /openapi.json with no auth
    header, so gating it would break health checks. It exposes the API's
    shape and nothing else — no data, no model call, no GPU."""
    monkeypatch.setenv(API_KEY_ENV, KEY)

    assert TestClient(create_app()).get("/openapi.json").status_code == 200
