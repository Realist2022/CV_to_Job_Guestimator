from pydantic import BaseModel


class CheckResult(BaseModel):
    name: str
    expected: str
    actual: str
    passed: bool


class EvaluationReport(BaseModel):
    passed: bool
    checks: list[CheckResult]