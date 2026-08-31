from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.harness.evaluator import ThresholdEvaluator, resolve_criteria
from src.harness.registry import Registry
from src.harness.task_loader import EvaluationCriteria, load_task
from src.schemas.pipeline import TraceSpan
from tests.factories import build_ingestion_result, build_pipeline_result


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
        # models.evaluation and inputs.job_listing/.candidate_cv are each
        # optional now (see task_loader.py): which ones a task must set
        # depends on its pipeline, since "ingestion" never calls an
        # evaluation model. There's no models.pii at all — PII redaction
        # runs entirely through presidio, with no LLM model to select for
        # any pipeline shape.
        if task.pipeline != "ingestion":
            assert task.models.evaluation
            assert task.inputs.job_listing
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
    assert {"presidio"} <= set(pii_detectors.names())


def _traced_result(*attempts: int | None):
    """A result whose trace has one span per given attempts value.

    None stands for a step with no LLM call (pii_redaction is local), which
    _peak_attempts must skip rather than count.
    """
    spans = [
        TraceSpan(
            step=f"step_{index}",
            started_at=datetime(2026, 8, 30, tzinfo=timezone.utc),
            duration_seconds=0.1,
            attempts=value,
        )
        for index, value in enumerate(attempts)
    ]
    return build_pipeline_result(trace=spans)


def test_max_attempts_passes_when_no_step_retried():
    report = ThresholdEvaluator(EvaluationCriteria(max_attempts=1)).evaluate(
        _traced_result(None, 1, 1, 1)
    )
    assert report.passed
    assert report.checks[0].actual == "1"


def test_max_attempts_fails_when_any_step_retried():
    # One retried step is enough: it means the model returned output that
    # failed schema validation, whatever the other steps managed.
    report = ThresholdEvaluator(EvaluationCriteria(max_attempts=1)).evaluate(
        _traced_result(None, 1, 3, 1)
    )
    assert not report.passed
    assert report.checks[0].actual == "3"


def test_max_attempts_ignores_steps_with_no_llm_call():
    # An all-local trace reports 0 attempts, not a fabricated 1.
    report = ThresholdEvaluator(EvaluationCriteria(max_attempts=1)).evaluate(
        _traced_result(None, None)
    )
    assert report.passed
    assert report.checks[0].actual == "0"


def test_resolve_criteria_layers_task_over_defaults():
    criteria = resolve_criteria(
        _fake_result(),
        task=EvaluationCriteria(min_final_relevance=30),
        defaults={"max_attempts": 1, "max_execution_seconds": 60},
    )
    assert criteria.min_final_relevance == 30
    assert criteria.max_attempts == 1
    assert criteria.max_execution_seconds == 60


def test_resolve_criteria_lets_a_task_override_a_default():
    criteria = resolve_criteria(
        _fake_result(),
        task=EvaluationCriteria(max_attempts=5),
        defaults={"max_attempts": 1},
    )
    assert criteria.max_attempts == 5


def test_resolve_criteria_lets_a_task_opt_out_with_null():
    # `max_attempts: null` in a task's YAML is *setting* the key, so it wins
    # over the default rather than being treated as absent.
    criteria = resolve_criteria(
        _fake_result(),
        task=EvaluationCriteria.model_validate({"max_attempts": None}),
        defaults={"max_attempts": 1},
    )
    assert criteria.max_attempts is None


def test_resolve_criteria_drops_inherited_defaults_the_result_cannot_answer():
    ingestion = build_ingestion_result()
    criteria = resolve_criteria(
        ingestion, task=None, defaults={"min_final_relevance": 30, "max_attempts": 1}
    )
    assert criteria.min_final_relevance is None  # no scorecard on an IngestionResult
    assert criteria.max_attempts == 1


def test_resolve_criteria_keeps_explicit_task_criteria_the_result_cannot_answer():
    # A hand-written threshold against the wrong pipeline shape must fail
    # loudly, not be silently dropped while appearing to be enforced.
    ingestion = build_ingestion_result()
    criteria = resolve_criteria(
        ingestion, task=EvaluationCriteria(min_final_relevance=30), defaults={}
    )
    assert criteria.min_final_relevance == 30
    with pytest.raises(AttributeError):
        ThresholdEvaluator(criteria).evaluate(ingestion)
