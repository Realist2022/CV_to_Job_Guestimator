import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app, cv_upload_enabled
from src.api.routes import _api_evaluation
from src.schemas.ingestion import WITHHELD_SPAN_TEXT, IngestionResult, RedactedCV
from src.schemas.pii import TextSpan
from src.services.cv_store import CVNotFoundError, ServingCVUnavailableError
from src.web_app import app
from tests.factories import build_pipeline_result


def _upload_client(monkeypatch) -> TestClient:
    """A client for an app with the CV-upload endpoints mounted.

    They are off by default (see cv_upload_enabled), so a test that
    exercises /api/compare or /api/ingest has to opt in exactly the way a
    developer running locally does. Using the module-level `app` here would
    quietly test 404s instead.
    """
    monkeypatch.setenv("ALLOW_CV_UPLOAD", "1")
    return TestClient(create_app())


def _serving_cv() -> RedactedCV:
    """Stands in for serving/redacted_cv.json — span values already withheld,
    exactly as scripts/build_serving_cv.py writes them."""
    return RedactedCV.from_raw_text(
        raw_text="Jane Doe\nReact developer",
        redacted_text="[PERSON_NAME]\nReact developer",
        pii_spans=[TextSpan(kind="person_name", text=WITHHELD_SPAN_TEXT)],
        pii_engine="presidio:test",
    )


class FakePipeline:
    def __init__(self, *_args, **_kwargs):
        pass

    def run(self, listing, cv, *, verbose=True, on_ingested=None):
        assert "React" in listing.text
        assert "React" in cv.text
        if on_ingested is not None:
            redacted_cv = RedactedCV.from_raw_text(
                raw_text=cv.text,
                redacted_text=cv.text,
                pii_spans=[],
                pii_engine="fake-pii",
            )
            on_ingested(
                IngestionResult(
                    cv_id=redacted_cv.cv_id,
                    pii_engine="fake-pii",
                    execution_seconds=0.01,
                    pii_spans=[],
                    redacted_cv=redacted_cv,
                )
            )
        return build_pipeline_result(execution_seconds=0.01, pillar_b_applicable=False)


class FakeLogger:
    def __init__(self, *_args, **_kwargs):
        self.last_run_number = None

    # Signatures mirror the real ArtifactLogger, evaluation kwarg included:
    # a double that silently accepts fewer arguments turns a wiring change
    # into a 500 at runtime instead of a failure at the call site.
    def log_run(self, _result, evaluation=None, config=None):
        self.last_run_number = 1
        return "artifacts/run-test.json"

    def log_ingestion_run(self, _result, config=None, evaluation=None):
        self.last_run_number = 1
        return "artifacts/run-ingest-test.json"


class FakeClient:
    def __init__(self, model: str, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature


def test_compare_endpoint_accepts_text_uploads_without_real_model_calls(monkeypatch):
    monkeypatch.setattr("src.api.routes.ExtractionPipeline", FakePipeline)
    monkeypatch.setattr("src.api.routes.ArtifactLogger", FakeLogger)
    # /api/compare's on_ingested hook persists through
    # src.services.ingestion_persistence.persist_ingestion, not routes.py's
    # own CVIngestionStore/ArtifactLogger names — patch it there too, or
    # this "without_real_model_calls" test silently writes real files.
    monkeypatch.setattr("src.services.ingestion_persistence.CVIngestionStore", FakeCVIngestionStore)
    monkeypatch.setattr("src.services.ingestion_persistence.ArtifactLogger", FakeLogger)
    monkeypatch.setattr(
        "src.api.routes.client_for_role",
        lambda role: FakeClient(model=f"fake-{role}"),
    )

    client = _upload_client(monkeypatch)
    response = client.post(
        "/api/compare",
        files={
            "job_listing": ("job.txt", b"Requirements\nReact", "text/plain"),
            "candidate_cv": ("cv.txt", b"Jane Doe\nReact developer", "text/plain"),
        },
        data={"skills_weight": "0.8", "work_experience_weight": "0.2"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["artifact_path"] == "artifacts/run-test.json"
    assert payload["metrics"]["total_requirements"] == 1
    assert payload["scoring_weights"] == {"skills_match": 0.8, "work_experience": 0.2}
    assert payload["skills_evaluation"]["matched_cv_skills"] == ["React"]


def test_compare_endpoint_rejects_unsupported_uploads(monkeypatch):
    client = _upload_client(monkeypatch)
    response = client.post(
        "/api/compare",
        files={
            "job_listing": ("job.docx", b"Requirements\nReact", "application/octet-stream"),
            "candidate_cv": ("cv.txt", b"React developer", "text/plain"),
        },
    )

    assert response.status_code == 400
    assert "must be a PDF or TXT" in response.json()["detail"]


class FakeIngestionPipeline:
    def __init__(self, *_args, **_kwargs):
        pass

    def run(self, cv, *, verbose=True):
        assert "Jane Doe" in cv.text
        redacted_cv = RedactedCV.from_raw_text(
            raw_text=cv.text,
            redacted_text=cv.text.replace("Jane Doe", "[PERSON_NAME]"),
            pii_spans=[TextSpan(kind="person_name", text="Jane Doe")],
            pii_engine="fake-pii",
        )
        return IngestionResult(
            cv_id=redacted_cv.cv_id,
            pii_engine="fake-pii",
            execution_seconds=0.01,
            pii_spans=redacted_cv.pii_spans,
            redacted_cv=redacted_cv,
        )


class FakeCVIngestionStore:
    saved: dict = {}

    def __init__(self, *_args, **_kwargs):
        pass

    def save(self, redacted_cv):
        FakeCVIngestionStore.saved[redacted_cv.cv_id] = redacted_cv
        return f"redacted_cvs/{redacted_cv.cv_id}.json"

    def load(self, cv_id):
        try:
            return FakeCVIngestionStore.saved[cv_id]
        except KeyError:
            raise CVNotFoundError(f"No ingested CV found for cv_id '{cv_id}'.") from None


class FakeMatchingPipeline:
    def __init__(self, *_args, **_kwargs):
        pass

    def run(self, listing, redacted_cv, *, verbose=True):
        assert "React" in listing.text
        assert "[PERSON_NAME]" in redacted_cv.text
        return build_pipeline_result(
            pii_engine=redacted_cv.pii_engine,
            execution_seconds=0.01,
            pillar_b_applicable=False,
            redacted_cv_trace_id=redacted_cv.ingestion_trace_id,
            pii_spans=redacted_cv.pii_spans,
        )


class FakeArtifactLogger:
    def __init__(self, *_args, **_kwargs):
        self.last_run_number = None

    def log_run(self, _result, config=None, evaluation=None):
        self.last_run_number = 1
        return "artifacts/run-test.json"

    def log_ingestion_run(self, _result, config=None, evaluation=None):
        self.last_run_number = 1
        return "artifacts/run-ingest-test.json"


def test_ingest_endpoint_persists_redacted_cv_and_returns_cv_id(monkeypatch):
    monkeypatch.setattr("src.api.routes.IngestionPipeline", FakeIngestionPipeline)
    # /api/ingest persists through
    # src.services.ingestion_persistence.persist_ingestion, so that's where
    # CVIngestionStore/ArtifactLogger need patching, not routes.py.
    monkeypatch.setattr("src.services.ingestion_persistence.CVIngestionStore", FakeCVIngestionStore)
    monkeypatch.setattr("src.services.ingestion_persistence.ArtifactLogger", FakeArtifactLogger)
    monkeypatch.setattr(
        "src.api.routes.client_for_role", lambda role: FakeClient(model=f"fake-{role}")
    )

    client = _upload_client(monkeypatch)
    response = client.post(
        "/api/ingest",
        files={"candidate_cv": ("cv.txt", b"Jane Doe\nReact developer", "text/plain")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["artifact_path"] == "artifacts/run-ingest-test.json"
    assert payload["pii_span_count"] == 1
    assert "cv_id" in payload
    # The raw PII value never appears anywhere in the ingest response.
    assert "Jane Doe" not in response.text


def _match_client(monkeypatch) -> TestClient:
    monkeypatch.setattr("src.api.routes.MatchingPipeline", FakeMatchingPipeline)
    monkeypatch.setattr("src.api.routes.ArtifactLogger", FakeArtifactLogger)
    monkeypatch.setattr("src.api.routes.load_serving_cv", _serving_cv)
    monkeypatch.setattr(
        "src.api.routes.client_for_role", lambda role: FakeClient(model=f"fake-{role}")
    )
    return TestClient(app)


def test_match_endpoint_uses_the_pinned_cv_with_no_pii_call(monkeypatch):
    client = _match_client(monkeypatch)

    response = client.post(
        "/api/match",
        files={"job_listing": ("job.txt", b"Requirements React", "text/plain")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["artifact_path"] == "artifacts/run-test.json"
    assert payload["skills_evaluation"]["matched_cv_skills"] == ["React"]


def test_match_endpoint_needs_only_a_job_listing(monkeypatch):
    """The lock-down in one assertion: a visitor sends a job listing and
    nothing else, and gets a full result. No CV, no cv_id, no PII."""
    client = _match_client(monkeypatch)

    response = client.post(
        "/api/match",
        files={"job_listing": ("job.txt", b"Requirements React", "text/plain")},
    )

    assert response.status_code == 200


def test_match_endpoint_ignores_a_caller_supplied_cv_id(monkeypatch):
    """cv_id is gone from the signature, so sending one selects nothing.

    Worth asserting rather than assuming: the danger of dropping a
    parameter is that FastAPI silently ignores the extra form field and a
    caller keeps believing it still steers which CV gets served. It does
    not — the pinned CV is used regardless.
    """
    served: list = []

    class RecordingMatchingPipeline(FakeMatchingPipeline):
        def run(self, listing, redacted_cv, *, verbose=True):
            served.append(redacted_cv)
            return super().run(listing, redacted_cv, verbose=verbose)

    client = _match_client(monkeypatch)
    monkeypatch.setattr("src.api.routes.MatchingPipeline", RecordingMatchingPipeline)

    response = client.post(
        "/api/match",
        files={"job_listing": ("job.txt", b"Requirements React", "text/plain")},
        data={"cv_id": "some-other-persons-cv"},
    )

    assert response.status_code == 200
    assert [cv.cv_id for cv in served] == [_serving_cv().cv_id]


def test_match_endpoint_reports_503_when_the_pinned_cv_is_missing(monkeypatch):
    """A deployment fault, not a bad request: the visitor's listing was
    fine and no change on their side would help."""

    def _unavailable():
        raise ServingCVUnavailableError("No serving CV at serving/redacted_cv.json.")

    monkeypatch.setattr("src.api.routes.load_serving_cv", _unavailable)
    monkeypatch.setattr(
        "src.api.routes.client_for_role", lambda role: FakeClient(model=f"fake-{role}")
    )

    response = TestClient(app).post(
        "/api/match",
        files={"job_listing": ("job.txt", b"Requirements React", "text/plain")},
    )

    assert response.status_code == 503
    assert "serving" in response.json()["detail"].lower()


def test_cv_upload_endpoints_are_absent_unless_explicitly_enabled(monkeypatch):
    """The public deployment must not expose anything that takes a CV.

    Asserted against the module-level `app` — built with ALLOW_CV_UPLOAD
    unset, exactly as the container builds it — so this fails if the
    default posture ever flips.
    """
    monkeypatch.delenv("ALLOW_CV_UPLOAD", raising=False)
    client = TestClient(create_app())

    for path in ("/api/compare", "/api/ingest"):
        response = client.post(
            path,
            files={"candidate_cv": ("cv.txt", b"Jane Doe", "text/plain")},
        )
        assert response.status_code == 404, f"{path} is reachable with uploads disabled"


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe", "TRUE-ish"])
def test_cv_upload_stays_off_for_anything_but_an_explicit_opt_in(monkeypatch, value):
    monkeypatch.setenv("ALLOW_CV_UPLOAD", value)
    assert not cv_upload_enabled()


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " on "])
def test_cv_upload_opt_in_accepts_the_documented_values(monkeypatch, value):
    monkeypatch.setenv("ALLOW_CV_UPLOAD", value)
    assert cv_upload_enabled()


def _ingestion_result(span_count: int = 1) -> IngestionResult:
    redacted = RedactedCV.from_raw_text(
        raw_text="Jane Doe\nReact developer",
        redacted_text="[PERSON_NAME]\nReact developer",
        pii_spans=[TextSpan(kind="person_name", text="Jane Doe")] * span_count,
        pii_engine="fake-pii",
    )
    return IngestionResult(
        cv_id=redacted.cv_id,
        pii_engine="fake-pii",
        execution_seconds=0.01,
        pii_spans=redacted.pii_spans,
        redacted_cv=redacted,
    )


def test_api_evaluation_returns_none_when_nothing_configured(monkeypatch):
    monkeypatch.setattr("src.api.routes.load_default_evaluation_criteria", dict)
    assert _api_evaluation(build_pipeline_result()) is None


def test_api_evaluation_judges_a_pipeline_result(monkeypatch):
    monkeypatch.setattr(
        "src.api.routes.load_default_evaluation_criteria",
        lambda: {"min_final_relevance": 90},
    )
    report = _api_evaluation(build_pipeline_result(final_relevance=45.0))
    assert not report.passed
    assert report.checks[0].name == "min_final_relevance"


def test_api_evaluation_drops_criteria_the_result_shape_cannot_answer(monkeypatch):
    # An IngestionResult has no scorecard, so min_final_relevance must be
    # skipped rather than raising AttributeError and 500-ing /api/ingest
    # over a config default aimed at the scored endpoints.
    monkeypatch.setattr(
        "src.api.routes.load_default_evaluation_criteria",
        lambda: {"min_final_relevance": 90, "min_pii_spans": 1},
    )
    report = _api_evaluation(_ingestion_result())
    assert report is not None
    assert [check.name for check in report.checks] == ["min_pii_spans"]
    assert report.passed

