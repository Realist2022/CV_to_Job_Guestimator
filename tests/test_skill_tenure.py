import unittest

from src.schemas.experience import OverallExperienceOutput
from src.schemas.requirements import JobRequirement
from src.services.agents import SkillTenureAgent
from src.services.document_parser import CandidateCV


class TenureClient:
    def complete(self, system_prompt, user_prompt, response_model, max_retries=2):
        return response_model(
            skills=[
                {
                    "requirement_id": 0,
                    "role_ids": [0],
                    "evidence": "Built REST API endpoints in the Software Engineer role.",
                },
                {
                    "requirement_id": 1,
                    "role_ids": [],
                    "evidence": "Git is listed without dated role evidence.",
                },
            ]
        )


class SkillTenureAgentTest(unittest.TestCase):
    def test_derives_dates_from_referenced_roles(self):
        roles = OverallExperienceOutput(
            target_job_title="Full Stack Developer",
            target_overall_years=2.0,
            candidate_roles=[
                {
                    "role_title": "Software Engineer",
                    "start_date": "2025-07",
                    "end_date": "2025-10",
                    "match_rationale": "Relevant software role.",
                    "is_relevant": True,
                }
            ],
        )

        result = SkillTenureAgent(TenureClient()).run(
            job_requirements=[
                JobRequirement(
                    capability="REST APIs", minimum_commercial_years=1.0
                ),
                JobRequirement(capability="Git", minimum_commercial_years=1.0),
            ],
            overall_experience=roles,
            cv=CandidateCV("July 2025 - October 2025\nBuilt REST API endpoints."),
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.skills[0].start_date, "2025-07")
        self.assertEqual(result.skills[0].end_date, "2025-10")
        self.assertEqual(result.skills[0].target_years, 1.0)
        self.assertIsNone(result.skills[1].start_date)
        self.assertIsNone(result.skills[1].end_date)

    def test_rejects_role_link_without_capability_specific_evidence(self):
        client = TenureClient()
        client.complete = lambda *args, **kwargs: kwargs["response_model"](
            skills=[
                {
                    "requirement_id": 0,
                    "role_ids": [0],
                    "evidence": "Built REST API endpoints and data pipelines.",
                }
            ]
        )
        roles = OverallExperienceOutput(
            target_job_title="Full Stack Developer",
            target_overall_years=2.0,
            candidate_roles=[
                {
                    "role_title": "Software Engineer",
                    "start_date": "2025-07",
                    "end_date": "2025-10",
                    "match_rationale": "Relevant REST API and data pipeline role.",
                    "is_relevant": True,
                }
            ],
        )

        result = SkillTenureAgent(client).run(
            job_requirements=[
                JobRequirement(capability="React", minimum_commercial_years=2.0)
            ],
            overall_experience=roles,
            cv=CandidateCV("Skills: React.js"),
        )

        self.assertIsNotNone(result)
        self.assertIsNone(result.skills[0].start_date)
        self.assertIsNone(result.skills[0].end_date)

if __name__ == "__main__":
    unittest.main()