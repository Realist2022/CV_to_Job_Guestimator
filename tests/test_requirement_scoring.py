import unittest

from src.schemas.experience import OverallExperienceOutput, SkillTenureOutput
from src.schemas.requirements import (
    JobRequirement,
    JobRequirementsOutput,
    SkillMatchResult,
)
from src.services.agents import JobRequirementsAgent, SkillMatcherAgent
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
    def test_shared_commercial_duration_is_grounded_from_source_clause(self):
        client = StaticClient(
            JobRequirementsOutput(
                job_requirements=[
                    {"capability": "React"},
                    {"capability": "Node.js", "minimum_commercial_years": 2.0},
                ]
            )
        )

        result = JobRequirementsAgent(client).run(
            JobListing(
                "Requirements\n"
                "• 2–5 years' commercial experience with React and Node.js."
            )
        )

        self.assertIsNotNone(result)
        self.assertEqual(
            [item.minimum_commercial_years for item in result.job_requirements],
            [2.0, 2.0],
        )

    def test_explicit_react_js_alias_matches_react_capability(self):
        client = StaticClient(
            lambda model: model(
                evaluations=[{"requirement_id": 0, "matched": False}]
            )
        )

        result = SkillMatcherAgent(client).run(
            [JobRequirement(capability="React")],
            CandidateCV("Frontend skills: React.js"),
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.matched_cv_skills, ["React"])
        self.assertEqual(result.missing_cv_skills, [])

    def test_capability_match_is_independent_from_commercial_tenure(self):
        requirements = [
            JobRequirement(
                capability="React",
                minimum_commercial_years=2.0,
            ),
            JobRequirement(
                capability="Node.js",
                minimum_commercial_years=2.0,
            ),
            JobRequirement(capability="REST APIs"),
        ]
        skills_result = SkillMatchResult(
            job_requirements=requirements,
            matched_cv_skills=["React", "Node.js", "REST APIs"],
            missing_cv_skills=[],
            rationale="All capabilities are present.",
        )
        commercial_tenure = SkillTenureOutput(
            skills=[
                {
                    "requirement_id": 0,
                    "target_years": 2.0,
                    "start_date": None,
                    "end_date": None,
                    "evidence": "React is listed without dated commercial evidence.",
                },
                {
                    "requirement_id": 1,
                    "target_years": 2.0,
                    "start_date": None,
                    "end_date": None,
                    "evidence": "Node.js is listed without dated commercial evidence.",
                },
            ]
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
            commercial_tenure,
            overall_experience,
        )

        self.assertEqual(skills_result.match_percentage, 100.0)
        self.assertEqual(scorecard.pillar_a.score, 100.0)
        self.assertEqual(scorecard.pillar_b.score, 0.0)
        self.assertEqual(scorecard.pillar_c.score, 12.5)
        self.assertEqual(scorecard.final_relevance, 47.5)

    def test_only_explicit_commercial_requirements_enter_tenure(self):
        requirements = [
            JobRequirement(capability="React", minimum_commercial_years=2.0),
            JobRequirement(capability="Node.js", minimum_commercial_years=2.0),
            JobRequirement(capability="REST APIs"),
        ]

        commercial_requirements = [
            requirement
            for requirement in requirements
            if requirement.minimum_commercial_years is not None
        ]

        self.assertEqual(
            [requirement.capability for requirement in commercial_requirements],
            ["React", "Node.js"],
        )


if __name__ == "__main__":
    unittest.main()