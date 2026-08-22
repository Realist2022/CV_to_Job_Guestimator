from pathlib import Path

import pytest

from src.harness.evaluator import ThresholdEvaluator
from src.harness.registry import Registry
from src.harness.task_loader import EvaluationCriteria, load_task
from tests.factories import build_pipeline_result


def _fake_result(final_relevance=45.0, match_percentage=100.0):
    return build_pipeline_result(
        final_relevance=final_relevance,
        match_percentage=match_percentage,
        pillar_b_applicable=False,
    )


def test_load_task_parses_all_shipped_tasks():
    for path in Path("tasks").glob("*.yaml"):
        task = load_task(path)
        assert task.name
        # models.evaluation/.pii and inputs.job_listing/.candidate_cv are
        # each optional now (see task_loader.py): which ones a task must
        # set depends on its pipeline, since "ingestion" never calls an
        # evaluation model and "matching" never calls a PII model or takes
        # a raw candidate_cv path.
        if task.pipeline != "ingestion":
            assert task.models.evaluation
            assert task.inputs.job_listing
        if task.pipeline != "matching":
            assert task.models.pii
        if task.pipeline == "matching":
            assert task.inputs.redacted_cv_id
        else:
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
    from src.harness.registry import pii_detectors, pipelines

    assert "extraction" in pipelines.names()
    assert {"regex", "model", "presidio"} <= set(pii_detectors.names())
