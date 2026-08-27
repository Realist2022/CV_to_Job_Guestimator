import re
from datetime import datetime
from typing import Optional

from dateutil import parser as date_parser

from src.config import load_scoring_weights
from src.schemas.experience import OverallExperienceOutput
from src.schemas.requirements import SkillMatchResult
from src.schemas.scoring import Scorecard


class RelevanceScoringEngine:
    def __init__(self, weights: Optional[dict[str, float]] = None):
        self.weights = weights or load_scoring_weights()
        if abs(sum(self.weights.values()) - 1.0) > 1e-9:
            raise ValueError("Scoring weights must add up to 1.0")

    @staticmethod
    def calculate_duration_in_years(
        start_date: Optional[str], end_date: Optional[str]
    ) -> float:
        start_text = (start_date or "").strip()
        end_text = (end_date or "").strip()
        if start_text and not end_text:
            range_parts = re.split(r"\s+(?:-|–|—|to)\s+", start_text, maxsplit=1)
            if len(range_parts) == 2:
                start_text, end_text = range_parts
        if not start_text or not end_text:
            return 0.0
        try:
            start = RelevanceScoringEngine._parse_date(start_text)
            end = RelevanceScoringEngine._parse_date(end_text)
        except (TypeError, ValueError):
            return 0.0
        return max(round((end - start).days / 365.25, 2), 0.0)

    @staticmethod
    def _parse_date(value: str) -> datetime:
        text = value.strip()
        if text.lower() in {"present", "current", "now"}:
            return datetime.now()
        try:
            return datetime.strptime(text, "%Y-%m")
        except ValueError:
            return date_parser.parse(text, default=datetime(1900, 1, 1), fuzzy=True)

    def calculate_scorecard(
        self,
        skills_result: SkillMatchResult,
        overall_experience: OverallExperienceOutput,
    ) -> Scorecard:
        skills_match_score = skills_result.match_percentage

        relevant_roles = [
            role for role in overall_experience.candidate_roles if role.is_relevant
        ]

        # A model that repeats a role would otherwise have its years counted twice.
        seen: set = set()
        deduped_roles = []
        for role in relevant_roles:
            key = (role.role_title, role.start_date, role.end_date)
            if key in seen:
                continue
            seen.add(key)
            deduped_roles.append(role)
        relevant_roles = deduped_roles


        total_career_years = sum(
            self.calculate_duration_in_years(role.start_date, role.end_date)
            for role in relevant_roles
        )
        work_experience_applicable = overall_experience.target_overall_years is not None
        if work_experience_applicable:
            target_career_years = max(overall_experience.target_overall_years, 0.1)
            work_experience_score = min(
                total_career_years / target_career_years * 100, 100.0
            )
            work_experience_raw = (
                f"{round(total_career_years, 1)} years vs "
                f"{target_career_years} years required"
            )
        else:
            target_career_years = None
            work_experience_score = 0.0
            work_experience_raw = "No explicit overall experience requirement"

        weighted_score = (
            self.weights["skills_match"] * skills_match_score
        )
        active_weight = self.weights["skills_match"]
        if work_experience_applicable:
            weighted_score += self.weights["work_experience"] * work_experience_score
            active_weight += self.weights["work_experience"]
        final_relevance = weighted_score / active_weight
        return Scorecard(
            final_relevance=round(final_relevance, 1),
            pillar_a={
                "score": round(skills_match_score, 1),
                "raw": (
                    f"{skills_result.total_matched_skills}/"
                    f"{skills_result.total_job_requirements} skills"
                ),
            },
            pillar_b={
                "score": round(work_experience_score, 1),
                "raw": work_experience_raw,
                "applicable": work_experience_applicable,
            },
            counted_roles=[role.role_title for role in relevant_roles],
        )