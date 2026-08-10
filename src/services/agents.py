import json
import re
from typing import List, Optional, Type
from pydantic import create_model, model_validator
from src.schemas.experience import (
    OverallExperienceOutput,
    SkillTenureEvidenceOutput,
    SkillTenureRecord,
    SkillTenureOutput,
)
from src.schemas.pii import PIIOutput
from src.schemas.requirements import (
    JobRequirement,
    JobRequirementsOutput,
    SkillEvaluationDecision,
    SkillEvaluationOutput,
)
from src.prompts.templates import (
    JOB_REQUIREMENTS_SYSTEM_PROMPT,
    OVERALL_EXPERIENCE_SYSTEM_PROMPT,
    PII_SYSTEM_PROMPT,
    SKILL_MATCHER_SYSTEM_PROMPT,
    SKILL_TENURE_SYSTEM_PROMPT,
)
from src.services.document_parser import JobListing, CandidateCV
from src.services.llm_client import InstructorClient


_YEARS_PATTERN = re.compile(
    r"(?P<minimum>\d+(?:\.\d+)?)\s*"
    r"(?:(?:-|–|—|to)\s*\d+(?:\.\d+)?)?\s*\+?\s*years?\b",
    re.IGNORECASE,
)

_CAPABILITY_ALIASES = {
    "react": ("react", "react.js", "reactjs"),
    "node.js": ("node.js", "nodejs", "node js"),
    "javascript or typescript": ("javascript", "typescript"),
    "rest apis": ("rest api", "rest apis"),
    "postgresql": ("postgresql", "postgres"),
    "git": ("git",),
}


def _aliases_for(capability: str) -> tuple[str, ...]:
    return _CAPABILITY_ALIASES.get(capability.strip().casefold(), (capability,))


def _contains_capability(text: str, capability: str) -> bool:
    return any(
        re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text, re.IGNORECASE)
        for alias in _aliases_for(capability)
    )


def _source_commercial_years(capability: str, source: str) -> Optional[float]:
    for line in source.splitlines():
        if "commercial" not in line.casefold():
            continue
        if not _contains_capability(line, capability):
            continue
        match = _YEARS_PATTERN.search(line)
        if match:
            return float(match.group("minimum"))
    return None


def _constrained_evaluation_model(
    job_requirements: List[JobRequirement],
) -> Type[SkillEvaluationDecision]:
    expected_ids = set(range(len(job_requirements)))

    def validate_requirements_coverage(self: SkillEvaluationDecision):
        received_ids = [evaluation.requirement_id for evaluation in self.evaluations]
        if len(received_ids) != len(set(received_ids)):
            raise ValueError("Evaluation contained duplicate requirement IDs")
        if set(received_ids) != expected_ids:
            raise ValueError(
                f"Evaluation must contain exactly these requirement IDs: {sorted(expected_ids)}"
            )
        return self

    return create_model(
        "ConstrainedSkillEvaluationDecision",
        __base__=SkillEvaluationDecision,
        __validators__={
            "validate_requirements_coverage": model_validator(mode="after")(
                validate_requirements_coverage
            )
        },
    )


def _constrained_tenure_evidence_model(
    requirement_ids: set[int], role_count: int
) -> Type[SkillTenureEvidenceOutput]:
    expected_ids = requirement_ids

    def validate_tenure_coverage(self: SkillTenureEvidenceOutput):
        received_ids = [skill.requirement_id for skill in self.skills]
        if len(received_ids) != len(set(received_ids)):
            raise ValueError("Skill tenure output contained duplicate requirement IDs")
        if set(received_ids) != expected_ids:
            raise ValueError(
                f"Skill tenure output must contain exactly these requirement IDs: {sorted(expected_ids)}"
            )
        for skill in self.skills:
            if len(skill.role_ids) != len(set(skill.role_ids)):
                raise ValueError("Skill tenure output contained duplicate role IDs")
            invalid_role_ids = [
                role_id
                for role_id in skill.role_ids
                if role_id < 0 or role_id >= role_count
            ]
            if invalid_role_ids:
                raise ValueError(
                    f"Skill tenure output contained invalid role IDs: {invalid_role_ids}"
                )
        return self

    return create_model(
        "ConstrainedSkillTenureEvidenceOutput",
        __base__=SkillTenureEvidenceOutput,
        __validators__={
            "validate_tenure_coverage": model_validator(mode="after")(
                validate_tenure_coverage
            )
        },
    )


class JobRequirementsAgent:
    """Extracts the authoritative requirement list from a job description."""

    system_prompt = JOB_REQUIREMENTS_SYSTEM_PROMPT

    def __init__(self, client: InstructorClient):
        self.client = client

    def run(self, listing: JobListing) -> Optional[JobRequirementsOutput]:
        user_message = f"JOB DESCRIPTION:\n{listing.requirements_section}"
        result = self.client.complete(
            system_prompt=self.system_prompt,
            user_prompt=user_message,
            response_model=JobRequirementsOutput,
        )
        if result is None:
            return None

        grounded_requirements = []
        for requirement in result.job_requirements:
            source_years = _source_commercial_years(
                requirement.capability, listing.requirements_section
            )
            grounded_requirements.append(
                requirement.model_copy(
                    update={"minimum_commercial_years": source_years}
                )
            )
        return JobRequirementsOutput(job_requirements=grounded_requirements)


class SkillMatcherAgent:
    """Classifies an authoritative requirement list against a candidate CV."""

    system_prompt = SKILL_MATCHER_SYSTEM_PROMPT

    def __init__(self, client: InstructorClient):
        self.client = client

    def run(
        self, job_requirements: List[JobRequirement], cv: CandidateCV
    ) -> Optional[SkillEvaluationOutput]:
        response_model = _constrained_evaluation_model(job_requirements)
        requirements = json.dumps(
            [
                {
                    "requirement_id": requirement_id,
                    "capability": requirement.capability,
                }
                for requirement_id, requirement in enumerate(job_requirements)
            ],
            ensure_ascii=False,
        )
        user_message = f"JOB REQUIREMENTS:\n{requirements}\n\nCANDIDATE CV:\n{cv.text}"
        decision = self.client.complete(
            system_prompt=self.system_prompt,
            user_prompt=user_message,
            response_model=response_model,
        )
        if decision is None:
            return None

        matched_ids = {
            evaluation.requirement_id
            for evaluation in decision.evaluations
            if evaluation.matched
        }
        matched_ids.update(
            requirement_id
            for requirement_id, requirement in enumerate(job_requirements)
            if _contains_capability(cv.text, requirement.capability)
        )
        matched_skills = [
            requirement.capability
            for requirement_id, requirement in enumerate(job_requirements)
            if requirement_id in matched_ids
        ]
        missing_skills = [
            requirement.capability
            for requirement_id, requirement in enumerate(job_requirements)
            if requirement_id not in matched_ids
        ]
        rationale = f"Matched {len(matched_skills)} of {len(job_requirements)} requirements."
        if missing_skills:
            rationale = f"{rationale[:-1]}; missing: {', '.join(missing_skills)}."

        return SkillEvaluationOutput(
            matched_cv_skills=matched_skills,
            missing_cv_skills=missing_skills,
            rationale=rationale,
        )


class SkillTenureAgent:
    """Measures required and evidenced tenure for matched requirements."""

    system_prompt = SKILL_TENURE_SYSTEM_PROMPT

    def __init__(self, client: InstructorClient):
        self.client = client

    def run(
        self,
        job_requirements: List[JobRequirement],
        overall_experience: OverallExperienceOutput,
        cv: CandidateCV,
    ) -> Optional[SkillTenureOutput]:
        roles = overall_experience.candidate_roles
        commercial_requirements = [
            (requirement_id, requirement)
            for requirement_id, requirement in enumerate(job_requirements)
            if requirement.minimum_commercial_years is not None
        ]
        requirement_ids = {
            requirement_id for requirement_id, _ in commercial_requirements
        }
        response_model = _constrained_tenure_evidence_model(
            requirement_ids, len(roles)
        )
        requirements = json.dumps(
            [
                {
                    "requirement_id": requirement_id,
                    "capability": requirement.capability,
                    "minimum_commercial_years": requirement.minimum_commercial_years,
                }
                for requirement_id, requirement in commercial_requirements
            ],
            ensure_ascii=False,
        )
        dated_roles = json.dumps(
            [
                {
                    "role_id": role_id,
                    "role_title": role.role_title,
                    "start_date": role.start_date,
                    "end_date": role.end_date,
                    "relevance_rationale": role.match_rationale,
                }
                for role_id, role in enumerate(roles)
            ],
            ensure_ascii=False,
        )
        user_message = (
            f"COMMERCIAL TENURE REQUIREMENTS:\n{requirements}"
            f"\n\nDATED CV ROLES:\n{dated_roles}"
            f"\n\nCANDIDATE CV:\n{cv.text}"
        )
        evidence_output = self.client.complete(
            system_prompt=self.system_prompt,
            user_prompt=user_message,
            response_model=response_model,
        )
        if evidence_output is None:
            return None

        evidence_by_requirement = {
            skill.requirement_id: skill for skill in evidence_output.skills
        }
        tenure_records = []
        for requirement_id, requirement in commercial_requirements:
            evidence = evidence_by_requirement[requirement_id]
            referenced_roles = [
                roles[role_id]
                for role_id in evidence.role_ids
                if _contains_capability(
                    " ".join(
                        (
                            roles[role_id].role_title,
                            roles[role_id].match_rationale,
                            evidence.evidence,
                        )
                    ),
                    requirement.capability,
                )
            ]
            start_date = (
                min(role.start_date for role in referenced_roles)
                if referenced_roles
                else None
            )
            if not referenced_roles:
                end_date = None
            elif any(
                role.end_date.lower() in {"present", "current", "now"}
                for role in referenced_roles
            ):
                end_date = "Present"
            else:
                end_date = max(role.end_date for role in referenced_roles)

            tenure_records.append(
                SkillTenureRecord(
                    requirement_id=requirement_id,
                    target_years=requirement.minimum_commercial_years,
                    start_date=start_date,
                    end_date=end_date,
                    evidence=(
                        evidence.evidence
                        if referenced_roles
                        else "No dated role-specific evidence."
                    ),
                )
            )

        return SkillTenureOutput(skills=tenure_records)


class OverallExperienceAgent:
    """Extracts and classifies career roles against the target job."""

    system_prompt = OVERALL_EXPERIENCE_SYSTEM_PROMPT

    def __init__(self, client: InstructorClient):
        self.client = client

    def run(
        self, listing: JobListing, cv: CandidateCV
    ) -> Optional[OverallExperienceOutput]:
        user_message = (
            f"JOB DESCRIPTION:\n{listing.text}\n\nCANDIDATE CV:\n{cv.text}"
        )
        return self.client.complete(
            system_prompt=self.system_prompt,
            user_prompt=user_message,
            response_model=OverallExperienceOutput,
        )


class PIIAgent:
    """Identifies personally identifying information in a CV."""

    system_prompt = PII_SYSTEM_PROMPT

    def __init__(self, client: InstructorClient):
        self.client = client

    def run(self, cv: CandidateCV) -> Optional[PIIOutput]:
        user_message = f"CV:\n{cv.text}"
        return self.client.complete(
            system_prompt=self.system_prompt,
            user_prompt=user_message,
            response_model=PIIOutput,
        )