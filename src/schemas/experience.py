from typing import Optional

from pydantic import BaseModel, Field, field_validator


class WorkRole(BaseModel):
    role_title: str = Field(description="Title of the candidate's position.")
    start_date: Optional[str] = Field(
        default=None,
        description="Role start date in YYYY-MM format, or null when missing.",
    )
    end_date: Optional[str] = Field(
        default=None,
        description="Role end date in YYYY-MM format, Present, or null when missing.",
    )
    match_rationale: str = Field(
        description="Brief comparison of this role with the target job."
    )
    is_relevant: bool = Field(
        description="Whether the role provides directly relevant target-job experience."
    )


class OverallExperienceOutput(BaseModel):
    target_job_title: str = Field(description="Target role title from the listing.")
    target_overall_years: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Explicit overall experience required, or null when unspecified.",
    )
    candidate_roles: list[WorkRole] = Field(
        description="Professional roles supported by the candidate CV."
    )

    @field_validator("target_overall_years")
    @classmethod
    def _zero_years_means_unspecified(cls, value: Optional[float]) -> Optional[float]:
        """Read a 0-year requirement as "no requirement stated".

        No listing asks for "at least 0 years", so a 0 here is a model saying
        "none given" in the wrong way. Seen on 2026-11-27: the fine-tuned model
        returned 0 instead of null for a listing with no experience bar. The
        scorer floors the target at 0.1 years, so 0.2 years of relevant work
        scored a perfect 100 on experience, and the portfolio showed
        "0 yrs wanted". Normalising here fixes both the score and the display.
        """
        return None if value == 0 else value


class OverallExperienceResponse(BaseModel):
    overall_experience: OverallExperienceOutput = Field(
        description="Extracted overall target and candidate role experience."
    )