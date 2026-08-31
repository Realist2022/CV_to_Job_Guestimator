from pathlib import Path

from fastapi.testclient import TestClient

from src.api.routes import _api_evaluation
from src.schemas.ingestion import IngestionResult, RedactedCV
from src.schemas.pii import TextSpan
from src.services.cv_store import CVNotFoundError
from src.web_app import app
from tests.factories import build_pipeline_result


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

    client = TestClient(app)
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


def test_compare_endpoint_rejects_unsupported_uploads():
    client = TestClient(app)
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

    client = TestClient(app)
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


def test_match_endpoint_uses_previously_ingested_cv_with_no_pii_call(monkeypatch):
    monkeypatch.setattr("src.api.routes.IngestionPipeline", FakeIngestionPipeline)
    # /api/match's own CVIngestionStore().load(cv_id)/ArtifactLogger().log_run
    # live in routes.py, but /api/ingest's save happens through
    # src.services.ingestion_persistence.persist_ingestion — both need
    # patching to the same FakeCVIngestionStore so the two calls share state.
    monkeypatch.setattr("src.api.routes.CVIngestionStore", FakeCVIngestionStore)
    monkeypatch.setattr("src.services.ingestion_persistence.CVIngestionStore", FakeCVIngestionStore)
    monkeypatch.setattr("src.services.ingestion_persistence.ArtifactLogger", FakeArtifactLogger)
    monkeypatch.setattr("src.api.routes.MatchingPipeline", FakeMatchingPipeline)
    monkeypatch.setattr("src.api.routes.ArtifactLogger", FakeArtifactLogger)
    monkeypatch.setattr(
        "src.api.routes.client_for_role", lambda role: FakeClient(model=f"fake-{role}")
    )

    client = TestClient(app)
    ingest_response = client.post(
        "/api/ingest",
        files={"candidate_cv": ("cv.txt", b"Jane Doe\nReact developer", "text/plain")},
    )
    cv_id = ingest_response.json()["cv_id"]

    match_response = client.post(
        "/api/match",
        files={"job_listing": ("job.txt", b"Requirements\nReact", "text/plain")},
        data={"cv_id": cv_id},
    )

    assert match_response.status_code == 200
    payload = match_response.json()
    assert payload["artifact_path"] == "artifacts/run-test.json"
    assert payload["skills_evaluation"]["matched_cv_skills"] == ["React"]


def test_match_endpoint_rejects_unknown_cv_id(monkeypatch):
    monkeypatch.setattr("src.api.routes.CVIngestionStore", FakeCVIngestionStore)
    monkeypatch.setattr(
        "src.api.routes.client_for_role", lambda role: FakeClient(model=f"fake-{role}")
    )

    client = TestClient(app)
    response = client.post(
        "/api/match",
        files={"job_listing": ("job.txt", b"Requirements\nReact", "text/plain")},
        data={"cv_id": "never-ingested"},
    )

    assert response.status_code == 400


def test_web_ui_renders_requirement_skill_names():
    static_html = Path("web/index.html").read_text(encoding="utf-8")

    assert "requirement.skill_name" in static_html
    assert "requirement.capability" not in static_html


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
