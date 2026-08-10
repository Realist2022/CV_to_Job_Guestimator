from typing import Optional

from pydantic import BaseModel, Field


class SkillTenureRecord(BaseModel):
    requirement_id: int = Field(description="The supplied numeric requirement ID.")
    target_years: float = Field(
        default=1.0,
        ge=0.0,
        description="Explicit minimum years required, or 1.0 when unspecified.",
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Earliest supported use in YYYY-MM format.",
    )
    end_date: Optional[str] = Field(
        default=None,
        description="Latest supported use in YYYY-MM format or Present.",
    )
    evidence: str = Field(description="Concise CV evidence supporting the date range.")


class SkillTenureOutput(BaseModel):
    skills: list[SkillTenureRecord] = Field(
        description="One tenure record for every supplied matched requirement ID."
    )


class SkillTenureEvidenceRecord(BaseModel):
    requirement_id: int = Field(description="The supplied numeric requirement ID.")
    role_ids: list[int] = Field(
        description="IDs of dated roles that explicitly demonstrate the requirement."
    )
    evidence: str = Field(description="Concise CV evidence supporting the role links.")


class SkillTenureEvidenceOutput(BaseModel):
    skills: list[SkillTenureEvidenceRecord] = Field(
        description="One role-association record for every matched requirement ID."
    )


class WorkRole(BaseModel):
    role_title: str = Field(description="Title of the candidate's position.")
    start_date: str = Field(description="Role start date in YYYY-MM format.")
    end_date: str = Field(description="Role end date in YYYY-MM format or Present.")
    match_rationale: str = Field(
        description="Brief comparison of this role with the target job."
    )
    is_relevant: bool = Field(
        description="Whether the role provides directly relevant target-job experience."
    )


class OverallExperienceOutput(BaseModel):
    target_job_title: str = Field(description="Target role title from the listing.")
    target_overall_years: float = Field(
        default=2.0,
        ge=0.0,
        description="Explicit overall experience required, or 2.0 when unspecified.",
    )
    candidate_roles: list[WorkRole] = Field(
        description="Professional roles supported by the candidate CV."
    )