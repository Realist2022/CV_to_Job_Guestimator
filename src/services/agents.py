import json
import re
from typing import List, Optional, Type

from pydantic import create_model, model_validator

from src.prompts.templates import (
    JOB_REQUIREMENTS_SYSTEM_PROMPT,
    OVERALL_EXPERIENCE_SYSTEM_PROMPT,
    SKILL_MATCHER_SYSTEM_PROMPT,
)
from src.schemas.experience import (
    OverallExperienceOutput,
    OverallExperienceResponse,
)
from src.schemas.ingestion import RedactedCV
from src.schemas.requirements import (
    JobRequirement,
    JobRequirementsOutput,
    SkillEvaluationDecision,
    SkillEvaluationOutput,
)
from src.services.document_parser import JobListing
from src.services.llm_client import CompletionClient

_SKILL_NAME_ALIASES = {
    "react": ("react", "react.js", "reactjs"),
    "node.js": ("node.js", "nodejs", "node js"),
    "javascript or typescript": ("javascript", "typescript"),
    "rest apis": ("rest api", "rest apis"),
    "postgresql": ("postgresql", "postgres"),
    "git": ("git",),
}

_EXPERIENCE_ROLE_BLOCK_PATTERN = re.compile(
    r"(?P<header>[^\n]+?)\s*[•*]\s*"
    r"(?P<start>[A-Za-z]+\s+\d{4}|\d{4}-\d{2})\s*"
    r"(?:-|–|—|to)\s*"
    r"(?P<end>Present|Current|Now|[A-Za-z]+\s+\d{4}|\d{4}-\d{2})"
    r"(?P<body>.*?)(?=\n\s*[^\n]+?\s*[•*]\s*"
    r"(?:[A-Za-z]+\s+\d{4}|\d{4}-\d{2})\s*(?:-|–|—|to)|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_ROLE_LINE_PATTERN = re.compile(r"^\s*Role:\s*(?P<title>.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def _aliases_for(skill_name: str) -> tuple[str, ...]:
    return _SKILL_NAME_ALIASES.get(skill_name.strip().casefold(), (skill_name,))


def _contains_skill_name(text: str, skill_name: str) -> bool:
    return any(
        re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text, re.IGNORECASE)
        for alias in _aliases_for(skill_name)
    )


def _normalize_title(value: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", value.casefold())
    return {word for word in words if word not in {"and", "full", "stack", "developer"}}


def _role_titles_overlap(left: str, right: str) -> bool:
    left_words = _normalize_title(left)
    right_words = _normalize_title(right)
    return bool(left_words and right_words and left_words & right_words)


def _explicit_cv_role_dates(cv_text: str) -> list[dict[str, str]]:
    role_dates = []
    for match in _EXPERIENCE_ROLE_BLOCK_PATTERN.finditer(cv_text):
        role_line = _ROLE_LINE_PATTERN.search(match.group("body"))
        if role_line is None:
            continue
        role_dates.append(
            {
                "title": role_line.group("title"),
                "start_date": match.group("start"),
                "end_date": match.group("end"),
            }
        )
    return role_dates


def _backfill_overall_experience_dates(
    overall_experience: OverallExperienceOutput, cv_text: str
) -> OverallExperienceOutput:
    explicit_role_dates = _explicit_cv_role_dates(cv_text)
    if not explicit_role_dates:
        return overall_experience

    candidate_roles = []
    for role in overall_experience.candidate_roles:
        if role.start_date and role.end_date:
            candidate_roles.append(role)
            continue

        matching_dates = next(
            (
                role_dates
                for role_dates in explicit_role_dates
                if _role_titles_overlap(role.role_title, role_dates["title"])
            ),
            None,
        )
        if matching_dates is None:
            candidate_roles.append(role)
            continue

        candidate_roles.append(
            role.model_copy(
                update={
                    "start_date": role.start_date or matching_dates["start_date"],
                    "end_date": role.end_date or matching_dates["end_date"],
                }
            )
        )

    return overall_experience.model_copy(update={"candidate_roles": candidate_roles})


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


class JobRequirementsAgent:
    """Extracts the authoritative requirement list from a job description."""

    system_prompt = JOB_REQUIREMENTS_SYSTEM_PROMPT

    def __init__(self, client: CompletionClient):
        self.client = client

    def run(self, listing: JobListing) -> Optional[JobRequirementsOutput]:
        user_message = f"JOB DESCRIPTION:\n{listing.requirements_section}"
        return self.client.complete(
            system_prompt=self.system_prompt,
            user_prompt=user_message,
            response_model=JobRequirementsOutput,
        )


class SkillMatcherAgent:
    """Classifies an authoritative requirement list against a candidate CV."""

    system_prompt = SKILL_MATCHER_SYSTEM_PROMPT

    def __init__(self, client: CompletionClient):
        self.client = client

    def run(
        self, job_requirements: List[JobRequirement], cv: RedactedCV
    ) -> Optional[SkillEvaluationOutput]:
        response_model = _constrained_evaluation_model(job_requirements)
        requirements = json.dumps(
            [
                {
                    "requirement_id": requirement_id,
                    "skill_name": requirement.skill_name,
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
            if _contains_skill_name(cv.text, requirement.skill_name)
        )
        matched_skills, missing_skills = [], []
        for requirement_id, requirement in enumerate(job_requirements):
            bucket = matched_skills if requirement_id in matched_ids else missing_skills
            bucket.append(requirement.skill_name)
        rationale = f"Matched {len(matched_skills)} of {len(job_requirements)} requirements."
        if missing_skills:
            rationale = f"{rationale[:-1]}; missing: {', '.join(missing_skills)}."

        return SkillEvaluationOutput(
            matched_cv_skills=matched_skills,
            missing_cv_skills=missing_skills,
            rationale=rationale,
        )


class OverallExperienceAgent:
    """Extracts and classifies career roles against the target job."""

    system_prompt = OVERALL_EXPERIENCE_SYSTEM_PROMPT

    def __init__(self, client: CompletionClient):
        self.client = client

    def run(
        self, listing: JobListing, cv: RedactedCV
    ) -> Optional[OverallExperienceOutput]:
        user_message = (
            f"JOB DESCRIPTION:\n{listing.text}\n\nCANDIDATE CV:\n{cv.text}"
        )
        result = self.client.complete(
            system_prompt=self.system_prompt,
            user_prompt=user_message,
            response_model=OverallExperienceResponse,
        )
        if result is None:
            return None
        output = (
            result
            if isinstance(result, OverallExperienceOutput)
            else result.overall_experience
        )
        return _backfill_overall_experience_dates(output, cv.text)