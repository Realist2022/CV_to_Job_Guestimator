import unittest

from src.schemas.experience import OverallExperienceOutput
from src.schemas.requirements import (
    JobRequirement,
    SkillMatchResult,
)
from src.services.agents import SkillMatcherAgent
from src.services.agents import OverallExperienceAgent
from src.services.document_parser import CandidateCV, JobListing
from src.services.scoring_engine import RelevanceScoringEngine


class StaticClient:
    def __init__(self, response):
        self.response = response

    def complete(self, system_prompt, user_prompt, response_model, max_retries=2):
        if callable(self.response):
            return self.response(response_model)
        return self.response


class RequirementScoringTest(unittest.TestCase):
    def test_explicit_react_js_alias_matches_react_skill_name(self):
        client = StaticClient(
            lambda model: model(
                evaluations=[{"requirement_id": 0, "matched": False}]
            )
        )

        result = SkillMatcherAgent(client).run(
            [JobRequirement(skill_name="React")],
            CandidateCV("Frontend skills: React.js"),
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.matched_cv_skills, ["React"])
        self.assertEqual(result.missing_cv_skills, [])

    def test_overall_experience_backfills_dates_from_explicit_role_block(self):
        client = StaticClient(
            lambda model: model(
                overall_experience={
                    "target_job_title": "Intermediate Full Stack Developer",
                    "target_overall_years": 2.0,
                    "candidate_roles": [
                        {
                            "role_title": "Software Engineer/Full stack Developer",
                            "start_date": None,
                            "end_date": None,
                            "match_rationale": "Directly relevant software role.",
                            "is_relevant": True,
                        }
                    ],
                }
            )
        )

        result = OverallExperienceAgent(client).run(
            JobListing("Intermediate Full Stack Developer"),
            CandidateCV(
                "Experience\n\n"
                "FOODSTUFFS • July 2025 – October 2025\n\n"
                "Role: Software Engineer\n\n"
                "Engineered scalable REST API endpoints."
            ),
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.candidate_roles[0].start_date, "July 2025")
        self.assertEqual(result.candidate_roles[0].end_date, "October 2025")

    def test_scorecard_uses_skill_name_match_and_overall_experience(self):
        requirements = [
            JobRequirement(skill_name="React"),
            JobRequirement(skill_name="Node.js"),
            JobRequirement(skill_name="REST APIs"),
        ]
        skills_result = SkillMatchResult(
            job_requirements=requirements,
            matched_cv_skills=["React", "Node.js", "REST APIs"],
            missing_cv_skills=[],
            rationale="All capabilities are present.",
        )
        overall_experience = OverallExperienceOutput(
            target_job_title="Full Stack Developer",
            target_overall_years=2.0,
            candidate_roles=[
                {
                    "role_title": "Software Engineer",
                    "start_date": "2025-07",
                    "end_date": "2025-10",
                    "match_rationale": "Comparable software role.",
                    "is_relevant": True,
                }
            ],
        )

        scorecard = RelevanceScoringEngine().calculate_scorecard(
            skills_result,
            overall_experience,
        )

        self.assertEqual(skills_result.match_percentage, 100.0)
        self.assertEqual(scorecard.pillar_a.score, 100.0)
        self.assertEqual(scorecard.pillar_b.score, 12.5)
        self.assertEqual(scorecard.final_relevance, 65.0)

    def test_missing_overall_experience_requirement_is_not_applicable(self):
        skills_result = SkillMatchResult(
            job_requirements=[JobRequirement(skill_name="Python")],
            matched_cv_skills=["Python"],
            missing_cv_skills=[],
            rationale="All capabilities are present.",
        )
        overall_experience = OverallExperienceOutput(
            target_job_title="Python Developer",
            target_overall_years=None,
            candidate_roles=[
                {
                    "role_title": "Python Developer",
                    "start_date": "2020-01",
                    "end_date": "2022-01",
                    "match_rationale": "Built Python services matching the job responsibilities.",
                    "is_relevant": True,
                }
            ],
        )

        scorecard = RelevanceScoringEngine().calculate_scorecard(
            skills_result,
            overall_experience,
        )

        self.assertFalse(scorecard.pillar_b.applicable)
        self.assertEqual(scorecard.pillar_b.raw, "No explicit overall experience requirement")
        self.assertEqual(scorecard.final_relevance, 100.0)

    def test_duration_accepts_month_name_dates_and_range_in_start_field(self):
        engine = RelevanceScoringEngine()

        self.assertEqual(
            engine.calculate_duration_in_years("Nov 2024 – Nov 2025", ""),
            1.0,
        )
        self.assertEqual(
            engine.calculate_duration_in_years("July 2025", "October 2025"),
            0.25,
        )


if __name__ == "__main__":
    unittest.main()