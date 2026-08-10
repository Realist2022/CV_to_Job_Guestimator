from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class JobRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability: str = Field(
        min_length=1,
        description="One atomic technical or operational capability.",
    )
    minimum_commercial_years: Optional[float] = Field(
        default=None,
        gt=0.0,
        description=(
            "Explicit minimum commercial years required for this capability, "
            "or null when no capability-specific duration is stated."
        ),
    )


class JobRequirementsOutput(BaseModel):
    job_requirements: list[JobRequirement] = Field(
        description="Unique atomic capabilities from the job description."
    )

    @field_validator("job_requirements")
    @classmethod
    def validate_job_requirements(
        cls, requirements: list[JobRequirement]
    ) -> list[JobRequirement]:
        names = [requirement.capability.strip().casefold() for requirement in requirements]
        if len(names) != len(set(names)):
            raise ValueError("job_requirements must not contain duplicates")
        return requirements


class SkillEvaluationOutput(BaseModel):
    matched_cv_skills: list[str] = Field(
        description="Items from job_requirements that are satisfied or present in the CV."
    )
    missing_cv_skills: list[str] = Field(
        description="Items from job_requirements that are absent or unsatisfied in the CV."
    )
    rationale: str = Field(
        description="Concise 1-sentence explanation detailing key matches and gaps."
    )


class RequirementEvaluation(BaseModel):
    requirement_id: int = Field(description="The supplied numeric requirement ID.")
    matched: bool = Field(description="Whether the CV satisfies this requirement.")


class SkillEvaluationDecision(BaseModel):
    evaluations: list[RequirementEvaluation] = Field(
        description="One match decision for every supplied requirement ID."
    )


class SkillMatchResult(BaseModel):
    requirement_category: str = Field(
        default="Core Competencies & Skills",
        description="Category name for the skills evaluation.",
    )
    job_requirements: list[JobRequirement] = Field(
        description="Unique atomic capabilities required by the job."
    )
    matched_cv_skills: list[str] = Field(
        description="Skills from job_requirements that are satisfied or present in the CV."
    )
    missing_cv_skills: list[str] = Field(
        description="Skills required by the job that are missing from the CV."
    )
    rationale: str = Field(
        description="Concise 1-sentence explanation detailing key matches and gaps."
    )

    @model_validator(mode="after")
    def validate_requirements_coverage(self):
        requirement_names = [
            requirement.capability for requirement in self.job_requirements
        ]
        requirements = set(requirement_names)
        matched = set(self.matched_cv_skills)
        missing = set(self.missing_cv_skills)

        if len(requirements) != len(requirement_names):
            raise ValueError("job_requirements must not contain duplicates")
        if len(matched) != len(self.matched_cv_skills):
            raise ValueError("matched_cv_skills must not contain duplicates")
        if len(missing) != len(self.missing_cv_skills):
            raise ValueError("missing_cv_skills must not contain duplicates")
        if matched & missing:
            raise ValueError("A requirement cannot be both matched and missing")
        if matched | missing != requirements:
            raise ValueError(
                "matched_cv_skills and missing_cv_skills must exactly partition job_requirements"
            )
        return self

    @property
    def total_job_requirements(self) -> int:
        return len(self.job_requirements)

    @property
    def total_matched_skills(self) -> int:
        return len(self.matched_cv_skills)

    @property
    def match_percentage(self) -> float:
        if not self.job_requirements:
            return 0.0
        return round((self.total_matched_skills / self.total_job_requirements) * 100, 2)