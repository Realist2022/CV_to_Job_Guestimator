from pathlib import Path

from fastapi.testclient import TestClient

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

    def log_run(self, _result, config=None, evaluation=None):
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
    # Pipelines and clients resolve in src.api.harness_adapter; the run
    # itself (and its ArtifactLogger.log_run) happens in
    # src.harness.runner — patch each where it's looked up.
    monkeypatch.setattr("src.api.harness_adapter.ExtractionPipeline", FakePipeline)
    monkeypatch.setattr("src.harness.runner.ArtifactLogger", FakeLogger)
    # /api/compare's on_ingested hook persists through
    # src.services.ingestion_persistence.persist_ingestion, not the
    # runner's own CVIngestionStore/ArtifactLogger names — patch it there
    # too, or this "without_real_model_calls" test silently writes real
    # files.
    monkeypatch.setattr("src.services.ingestion_persistence.CVIngestionStore", FakeCVIngestionStore)
    monkeypatch.setattr("src.services.ingestion_persistence.ArtifactLogger", FakeLogger)
    monkeypatch.setattr(
        "src.api.harness_adapter.client_for_role",
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
    monkeypatch.setattr("src.api.harness_adapter.IngestionPipeline", FakeIngestionPipeline)
    # /api/ingest persists through
    # src.services.ingestion_persistence.persist_ingestion, so that's where
    # CVIngestionStore/ArtifactLogger need patching, not the runner.
    monkeypatch.setattr("src.services.ingestion_persistence.CVIngestionStore", FakeCVIngestionStore)
    monkeypatch.setattr("src.services.ingestion_persistence.ArtifactLogger", FakeArtifactLogger)
    monkeypatch.setattr(
        "src.api.harness_adapter.client_for_role",
        lambda role: FakeClient(model=f"fake-{role}"),
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
    monkeypatch.setattr("src.api.harness_adapter.IngestionPipeline", FakeIngestionPipeline)
    # /api/match's CVIngestionStore().load(cv_id) lives in
    # src.api.harness_adapter, but /api/ingest's save happens through
    # src.services.ingestion_persistence.persist_ingestion — both need
    # patching to the same FakeCVIngestionStore so the two calls share state.
    monkeypatch.setattr("src.api.harness_adapter.CVIngestionStore", FakeCVIngestionStore)
    monkeypatch.setattr("src.services.ingestion_persistence.CVIngestionStore", FakeCVIngestionStore)
    monkeypatch.setattr("src.services.ingestion_persistence.ArtifactLogger", FakeArtifactLogger)
    monkeypatch.setattr("src.api.harness_adapter.MatchingPipeline", FakeMatchingPipeline)
    monkeypatch.setattr("src.harness.runner.ArtifactLogger", FakeArtifactLogger)
    monkeypatch.setattr(
        "src.api.harness_adapter.client_for_role",
        lambda role: FakeClient(model=f"fake-{role}"),
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
    monkeypatch.setattr("src.api.harness_adapter.CVIngestionStore", FakeCVIngestionStore)
    monkeypatch.setattr(
        "src.api.harness_adapter.client_for_role",
        lambda role: FakeClient(model=f"fake-{role}"),
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