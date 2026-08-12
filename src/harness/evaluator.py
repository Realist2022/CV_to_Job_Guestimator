"""Judges a pipeline result against the thresholds declared in a task."""

from pydantic import BaseModel

from src.harness.task_loader import EvaluationCriteria
from src.schemas.pipeline import PipelineResult


class CheckResult(BaseModel):
    name: str
    expected: str
    actual: str
    passed: bool


class EvaluationReport(BaseModel):
    passed: bool
    checks: list[CheckResult]


class ThresholdEvaluator:
    def __init__(self, criteria: EvaluationCriteria):
        self.criteria = criteria

    def evaluate(self, result: PipelineResult) -> EvaluationReport:
        checks: list[CheckResult] = []
        criteria = self.criteria

        if criteria.min_final_relevance is not None:
            checks.append(
                _check(
                    "min_final_relevance",
                    f">= {criteria.min_final_relevance}",
                    result.scorecard.final_relevance,
                    result.scorecard.final_relevance >= criteria.min_final_relevance,
                )
            )
        if criteria.min_skills_match is not None:
            checks.append(
                _check(
                    "min_skills_match",
                    f">= {criteria.min_skills_match}",
                    result.metrics.match_percentage,
                    result.metrics.match_percentage >= criteria.min_skills_match,
                )
            )
        if criteria.min_pii_spans is not None:
            checks.append(
                _check(
                    "min_pii_spans",
                    f">= {criteria.min_pii_spans}",
                    len(result.pii_spans),
                    len(result.pii_spans) >= criteria.min_pii_spans,
                )
            )
        if criteria.max_execution_seconds is not None:
            checks.append(
                _check(
                    "max_execution_seconds",
                    f"<= {criteria.max_execution_seconds}",
                    result.execution_seconds,
                    result.execution_seconds <= criteria.max_execution_seconds,
                )
            )

        return EvaluationReport(
            passed=all(check.passed for check in checks),
            checks=checks,
        )


def _check(name: str, expected: str, actual: object, passed: bool) -> CheckResult:
    return CheckResult(name=name, expected=expected, actual=str(actual), passed=passed)
