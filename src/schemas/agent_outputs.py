from typing import List
from pydantic import BaseModel, Field

# --- AGENT 1 SCHEMAS ---
class JobRequirement(BaseModel):
    skill_name: str = Field(description="The exact name of the competency, tool, language, license, or platform.")
    target_years: float = Field(default=1.0, description="Minimum years required. Set to 1.0 if not specified.")

class JobRequirementsOutput(BaseModel):
    job_title: str = Field(description="The title of the role from the job description.")
    required_skills: List[JobRequirement] = Field(description="Exhaustive list of technical skills required.")

# --- AGENT 2 SCHEMAS ---
class CandidateSkillMatch(BaseModel):
    skill_name: str = Field(description="The skill name matching one from the required list.")
    start_date: str = Field(description="YYYY-MM when the candidate first used this tool.")
    end_date: str = Field(description="YYYY-MM when last used, or 'Present'.")

class CVSkillMatchOutput(BaseModel):
    matched_skills: List[CandidateSkillMatch] = Field(description="List of required skills found in the candidate's CV.")

# --- AGENT 3 SCHEMAS ---
class WorkRole(BaseModel):
    role_title: str = Field(description="Title of the candidate's position.")
    start_date: str = Field(description="Format strictly as YYYY-MM.")
    end_date: str = Field(description="Format strictly as YYYY-MM or 'Present'.")
    match_rationale: str = Field(description="Brief explanation comparing this role to target job.")
    is_relevant: bool = Field(description="True ONLY if role is directly relevant to target job.")

class OverallExperienceOutput(BaseModel):
    target_overall_years: float = Field(default=2.0, description="Minimum overall career years demanded.")
    candidate_roles: List[WorkRole] = Field(description="List of professional positions found in CV.")