from pathlib import Path

from fastapi.testclient import TestClient

from src.schemas.experience import OverallExperienceOutput
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
                job_requirements=[{"skill_name": "React"}],
                matched_cv_skills=["React"],
                missing_cv_skills=[],
                rationale="React is present.",
            ),
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
                    "raw": "No relevant roles",
                    "applicable": False,
                },
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
    monkeypatch.setattr("src.api.routes.ExtractionPipeline", FakePipeline)
    monkeypatch.setattr("src.api.routes.ArtifactLogger", FakeLogger)

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


def test_web_ui_renders_requirement_skill_names():
    static_html = Path("web/index.html").read_text(encoding="utf-8")

    assert "requirement.skill_name" in static_html
    assert "requirement.capability" not in static_html