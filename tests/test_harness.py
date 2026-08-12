from pathlib import Path

import pytest

from src.harness.evaluator import ThresholdEvaluator
from src.harness.registry import Registry
from src.harness.task_loader import EvaluationCriteria, load_task
from src.schemas.experience import OverallExperienceOutput
from src.schemas.pii import TextSpan
from src.schemas.pipeline import PipelineMetrics, PipelineResult
from src.schemas.requirements import SkillMatchResult
from src.schemas.scoring import Scorecard


def _fake_result(final_relevance=45.0, match_percentage=100.0) -> PipelineResult:
    return PipelineResult(
        engine="fake-eval",
        pii_engine="fake-pii",
        execution_seconds=0.5,
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
            final_relevance=final_relevance,
            pillar_a={"score": match_percentage, "raw": "1/1 skills"},
            pillar_b={"score": 0.0, "raw": "No relevant roles", "applicable": False},
            counted_roles=[],
        ),
        metrics=PipelineMetrics(
            total_requirements=1,
            total_matched=1,
            match_percentage=match_percentage,
            final_relevance=final_relevance,
        ),
        redacted_cv="React developer",
        pii_spans=[TextSpan(kind="person_name", text="Jane Doe")],
    )


def test_load_task_parses_all_shipped_tasks():
    for path in Path("tasks").glob("*.yaml"):
        task = load_task(path)
        assert task.name
        assert task.models.evaluation
        assert task.inputs.job_listing
        assert task.inputs.candidate_cv


def test_threshold_evaluator_passes_when_criteria_met():
    criteria = EvaluationCriteria(
        min_final_relevance=40, min_skills_match=90, min_pii_spans=1
    )
    report = ThresholdEvaluator(criteria).evaluate(_fake_result())
    assert report.passed
    assert len(report.checks) == 3


def test_threshold_evaluator_fails_below_threshold():
    criteria = EvaluationCriteria(min_final_relevance=90)
    report = ThresholdEvaluator(criteria).evaluate(_fake_result(final_relevance=45.0))
    assert not report.passed
    failed = [check for check in report.checks if not check.passed]
    assert failed[0].name == "min_final_relevance"


def test_registry_rejects_unknown_and_duplicate_names():
    registry = Registry("widget")
    registry.register("a", lambda **_: "made-a")
    assert registry.create("a") == "made-a"
    with pytest.raises(ValueError):
        registry.register("a", lambda **_: "again")
    with pytest.raises(KeyError):
        registry.create("missing")


def test_runner_registers_default_components():
    from src.harness import runner  # noqa: F401  (import triggers registration)
    from src.harness.registry import pipelines, pii_detectors

    assert "extraction" in pipelines.names()
    assert {"regex", "model"} <= set(pii_detectors.names())
