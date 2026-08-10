from fastapi.testclient import TestClient

from src.schemas.experience import OverallExperienceOutput, SkillTenureOutput
from src.schemas.pipeline import PipelineMetrics, PipelineResult
from src.schemas.pii import TextSpan
from src.schemas.requirements import SkillMatchResult
from src.schemas.scoring import Scorecard
from src.web_app import app


class FakePipeline:
    def __init__(self, *_args, **_kwargs):
        pass

    def run(self, listing, cv, *, verbose=True):
        assert "React" in listing.text
        assert "React" in cv.text
        return PipelineResult(
            engine="fake-eval",
            pii_engine="fake-pii",
            execution_seconds=0.01,
            skills_eval=SkillMatchResult(
                job_requirements=[{"capability": "React"}],
                matched_cv_skills=["React"],
                missing_cv_skills=[],
                rationale="React is present.",
            ),
            skill_tenure=SkillTenureOutput(skills=[]),
            overall_experience=OverallExperienceOutput(
                target_job_title="Full Stack Developer",
                target_overall_years=2.0,
                candidate_roles=[],
            ),
            scorecard=Scorecard(
                final_relevance=45.0,
                pillar_a={"score": 100.0, "raw": "1/1 skills"},
                pillar_b={
                    "score": 0.0,
                    "raw": "No explicit commercial-tenure requirements",
                    "applicable": False,
                },
                pillar_c={"score": 0.0, "raw": "No relevant roles"},
                counted_roles=[],
            ),
            metrics=PipelineMetrics(
                total_requirements=1,
                total_matched=1,
                match_percentage=100.0,
                final_relevance=45.0,
            ),
            redacted_cv="React developer",
            pii_spans=[TextSpan(kind="person_name", text="Jane Doe")],
        )


class FakeLogger:
    def __init__(self, *_args, **_kwargs):
        pass

    def log_run(self, _result):
        return "artifacts/run-test.json"


def test_compare_endpoint_accepts_text_uploads_without_real_model_calls(monkeypatch):
    monkeypatch.setattr("src.web_app.ExtractionPipeline", FakePipeline)
    monkeypatch.setattr("src.web_app.ArtifactLogger", FakeLogger)

    client = TestClient(app)
    response = client.post(
        "/api/compare",
        files={
            "job_listing": ("job.txt", b"Requirements\nReact", "text/plain"),
            "candidate_cv": ("cv.txt", b"Jane Doe\nReact developer", "text/plain"),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["artifact_path"] == "artifacts/run-test.json"
    assert payload["metrics"]["total_requirements"] == 1
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