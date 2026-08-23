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
]


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
