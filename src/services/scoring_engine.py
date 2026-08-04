from datetime import datetime
from dateutil import parser
from src.config import SKILLS_MATCH_WEIGHT, SKILL_TENURE_WEIGHT, WORK_EXPERIENCE_WEIGHT
from src.schemas.agent_outputs import JobRequirementsOutput, CVSkillMatchOutput, OverallExperienceOutput

class RelevanceScoringEngine:
    def __init__(self):
        self.weights = {
            "skills_match": SKILLS_MATCH_WEIGHT,
            "skill_tenure": SKILL_TENURE_WEIGHT,
            "work_exp": WORK_EXPERIENCE_WEIGHT,
        }

    @staticmethod
    def calculate_duration_in_years(start_str: str, end_str: str) -> float:
        try:
            start = parser.parse(start_str.strip())
            end_clean = end_str.strip().lower()
            end = datetime.now() if end_clean in ["present", "current", "now"] else parser.parse(end_clean)
            days = (end - start).days
            return max(round(days / 365.25, 2), 0.0)
        except (ValueError, TypeError):
            return 0.0

    def calculate_scorecard(self, pipeline_output: dict) -> dict:
        job_reqs: JobRequirementsOutput = pipeline_output["job_requirements"]
        matched_cv: CVSkillMatchOutput = pipeline_output["matched_skills"]
        overall_exp: OverallExperienceOutput = pipeline_output["overall_experience"]

        # Pillar A: Skills Match
        target_dict = {req.skill_name.lower(): req.target_years for req in job_reqs.required_skills}
        total_job_skills = len(target_dict)
        valid_matches = [s for s in matched_cv.matched_skills if s.skill_name.lower() in target_dict]
        total_found = len(valid_matches)
        skills_match_score = (total_found / total_job_skills * 100) if total_job_skills > 0 else 0.0

        # Pillar B: Skill Tenure
        tenure_scores = []
        for skill in valid_matches:
            target_yrs = max(target_dict.get(skill.skill_name.lower(), 1.0), 0.1)
            candidate_yrs = self.calculate_duration_in_years(skill.start_date, skill.end_date)
            tenure_scores.append(min((candidate_yrs / target_yrs) * 100, 100.0))
        skill_tenure_score = (sum(tenure_scores) / len(tenure_scores)) if tenure_scores else 0.0

        # Pillar C: Overall Tenure
        relevant_roles = [r for r in overall_exp.candidate_roles if r.is_relevant]
        total_career_years = sum([self.calculate_duration_in_years(r.start_date, r.end_date) for r in relevant_roles])
        target_career_years = max(overall_exp.target_overall_years, 0.1)
        work_exp_score = min((total_career_years / target_career_years) * 100, 100.0)

        # Final Score
        final_relevance = (
            (self.weights["skills_match"] * skills_match_score)
            + (self.weights["skill_tenure"] * skill_tenure_score)
            + (self.weights["work_exp"] * work_exp_score)
        )

        return {
            "final_relevance": round(final_relevance, 1),
            "pillar_a": {"score": round(skills_match_score, 1), "raw": f"{total_found}/{total_job_skills} skills"},
            "pillar_b": {"score": round(skill_tenure_score, 1), "raw": f"Avg tenure fit across {total_found} tools"},
            "pillar_c": {"score": round(work_exp_score, 1), "raw": f"{round(total_career_years, 1)} yrs vs {target_career_years} yrs required"},
            "validated_skills": [s.skill_name for s in valid_matches],
            "counted_roles": [r.role_title for r in relevant_roles],
        }