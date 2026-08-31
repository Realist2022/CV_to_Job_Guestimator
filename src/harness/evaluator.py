"""Judges a pipeline result against the thresholds declared in a task."""

from operator import ge, le

from src.harness.task_loader import EvaluationCriteria
from src.schemas.evaluation import CheckResult, EvaluationReport
from src.schemas.ingestion import IngestionResult
from src.schemas.pipeline import PipelineResult

# criterion name -> (comparison symbol, operator, value taken from the result)
_CHECKS = [
    ("min_final_relevance", ">=", ge, lambda r: r.scorecard.final_relevance),
    ("min_skills_match", ">=", ge, lambda r: r.metrics.match_percentage),
    ("min_pii_spans", ">=", ge, lambda r: len(r.pii_spans)),
    ("max_execution_seconds", "<=", le, lambda r: r.execution_seconds),
    ("max_attempts", "<=", le, lambda r: _peak_attempts(r)),
]

# criterion name -> the result attribute its check reaches for. Only used to
# decide whether an *inherited* default applies to a given result shape; a
# criterion written explicitly in a task is never filtered out this way.
_CRITERION_REQUIRES = {
    "min_final_relevance": "scorecard",
    "min_skills_match": "metrics",
    "min_pii_spans": "pii_spans",
    "max_execution_seconds": "execution_seconds",
    "max_attempts": "trace",
}


def resolve_criteria(
    result: PipelineResult | IngestionResult,
    *,
    task: EvaluationCriteria | None = None,
    defaults: dict | None = None,
) -> EvaluationCriteria:
    """Combine configs/pipeline.yaml's `default_evaluation` with a task's own.

    One definition, both entry points: the CLI harness passes `task` and the
    web API (which has no task file) passes only `defaults`, so an everyday
    `uv run main.py` and an equivalent /api/compare are judged by the same
    bar instead of drifting apart.

    Precedence is per key, not all-or-nothing: a task that sets one criterion
    still inherits the rest. Setting a key to `null` in a task counts as
    setting it -- that's how a task opts out of an inherited default rather
    than being stuck with it.

    Inherited defaults that `result`'s shape can't answer are dropped:
    a global min_final_relevance shouldn't crash an ingestion run that has no
    scorecard. Criteria the task states explicitly are never dropped -- a
    threshold someone wrote by hand should fail loudly against the wrong
    pipeline shape, not be silently ignored while appearing to be enforced.
    """
    stated = task.model_fields_set if task else set()
    values = {name: getattr(task, name) for name in stated} if task else {}
    for name, value in (defaults or {}).items():
        if name in stated or value is None:
            continue
        if name in _CRITERION_REQUIRES and not hasattr(result, _CRITERION_REQUIRES[name]):
            continue
        values[name] = value
    return EvaluationCriteria.model_validate(values)


def _peak_attempts(result: PipelineResult | IngestionResult) -> int:
    """Most attempts any single LLM step in this run needed.

    Spans with `attempts: None` had no LLM call (pii_redaction is entirely
    local), so they're skipped rather than counted as 1 -- otherwise a
    run whose trace is all non-LLM steps would report a fabricated attempt
    that never happened. A run with no LLM steps at all reports 0, which
    passes any max_attempts threshold: nothing retried because nothing ran.
    """
    return max((span.attempts for span in result.trace if span.attempts is not None), default=0)


class ThresholdEvaluator:
    def __init__(self, criteria: EvaluationCriteria):
        self.criteria = criteria

    def evaluate(self, result: PipelineResult | IngestionResult) -> EvaluationReport:
        # Attribute access happens lazily per criterion, so an
        # IngestionResult (no scorecard/metrics) is fine as long as the
        # task only sets the criteria its result shape supports; a
        # mismatch (e.g. min_final_relevance on an ingestion task)
        # raises AttributeError here rather than being caught at load
        # time.
        checks: list[CheckResult] = []
        for name, symbol, compare, actual_of in _CHECKS:
            threshold = getattr(self.criteria, name)
            if threshold is None:
                continue
            actual = actual_of(result)
            checks.append(
                CheckResult(
                    name=name,
                    expected=f"{symbol} {threshold}",
                    actual=str(actual),
                    passed=compare(actual, threshold),
                )
            )

        return EvaluationReport(
            passed=all(check.passed for check in checks),
            checks=checks,
        )
