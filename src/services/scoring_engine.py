from datetime import datetime
from typing import Optional

from src.config import (
    SKILLS_MATCH_WEIGHT,
    SKILL_TENURE_WEIGHT,
    WORK_EXPERIENCE_WEIGHT,
)
from src.schemas.experience import OverallExperienceOutput, SkillTenureOutput
from src.schemas.requirements import SkillMatchResult
from src.schemas.scoring import Scorecard


class RelevanceScoringEngine:
    def __init__(self):
        self.weights = {
            "skills_match": SKILLS_MATCH_WEIGHT,
            "skill_tenure": SKILL_TENURE_WEIGHT,
            "work_experience": WORK_EXPERIENCE_WEIGHT,
        }
        if abs(sum(self.weights.values()) - 1.0) > 1e-9:
            raise ValueError("Scoring weights must add up to 1.0")

    @staticmethod
    def calculate_duration_in_years(
        start_date: Optional[str], end_date: Optional[str]
    ) -> float:
        if not start_date or not end_date:
            return 0.0
        try:
            start = datetime.strptime(start_date.strip(), "%Y-%m")
            end_text = end_date.strip()
            end = (
                datetime.now()
                if end_text.lower() in {"present", "current", "now"}
                else datetime.strptime(end_text, "%Y-%m")
            )
        except (TypeError, ValueError):
            return 0.0
        return max(round((end - start).days / 365.25, 2), 0.0)

    def calculate_scorecard(
        self,
        skills_result: SkillMatchResult,
        skill_tenure: SkillTenureOutput,
        overall_experience: OverallExperienceOutput,
    ) -> Scorecard:
        skills_match_score = skills_result.match_percentage

        tenure_scores = []
        for skill in skill_tenure.skills:
            target_years = max(skill.target_years, 0.1)
            candidate_years = self.calculate_duration_in_years(
                skill.start_date, skill.end_date
            )
            tenure_scores.append(min(candidate_years / target_years * 100, 100.0))
        skill_tenure_score = (
            sum(tenure_scores) / len(tenure_scores) if tenure_scores else 0.0
        )
        skill_tenure_applicable = bool(tenure_scores)

        relevant_roles = [
            role for role in overall_experience.candidate_roles if role.is_relevant
        ]
        total_career_years = sum(
            self.calculate_duration_in_years(role.start_date, role.end_date)
            for role in relevant_roles
        )
        target_career_years = max(overall_experience.target_overall_years, 0.1)
        work_experience_score = min(
            total_career_years / target_career_years * 100, 100.0
        )

        weighted_score = (
            self.weights["skills_match"] * skills_match_score
            + self.weights["work_experience"] * work_experience_score
        )
        active_weight = (
            self.weights["skills_match"] + self.weights["work_experience"]
        )
        if skill_tenure_applicable:
            weighted_score += self.weights["skill_tenure"] * skill_tenure_score
            active_weight += self.weights["skill_tenure"]
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
                "score": round(skill_tenure_score, 1),
                "raw": (
                    (
                        "Average commercial tenure fit across "
                        f"{len(tenure_scores)} explicit requirements"
                    )
                    if skill_tenure_applicable
                    else "No explicit commercial-tenure requirements"
                ),
                "applicable": skill_tenure_applicable,
            },
            pillar_c={
                "score": round(work_experience_score, 1),
                "raw": (
                    f"{round(total_career_years, 1)} years vs "
                    f"{target_career_years} years required"
                ),
            },
            counted_roles=[role.role_title for role in relevant_roles],
        )