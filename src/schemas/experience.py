from typing import Optional

from pydantic import BaseModel, Field


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


class OverallExperienceResponse(BaseModel):
    overall_experience: OverallExperienceOutput = Field(
        description="Extracted overall target and candidate role experience."
    )