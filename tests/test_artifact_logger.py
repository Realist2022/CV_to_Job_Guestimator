import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest

from src.schemas.artifact import RunArtifact
from src.schemas.experience import OverallExperienceOutput, SkillTenureOutput
from src.schemas.pipeline import PipelineMetrics, PipelineResult
from src.schemas.pii import TextSpan
from src.schemas.requirements import SkillMatchResult
from src.schemas.scoring import Scorecard
from src.utils import ArtifactLogger


def make_pipeline_result() -> PipelineResult:
    return PipelineResult(
        engine="example/model:latest",
        pii_engine="local-pii:latest",
        execution_seconds=1.25,
        skills_eval=SkillMatchResult(
            job_requirements=[{"capability": "Python"}],
            matched_cv_skills=["Python"],
            missing_cv_skills=[],
            rationale="The requirement is present.",
        ),
        skill_tenure=SkillTenureOutput(
            skills=[
                {
                    "requirement_id": 0,
                    "target_years": 1.0,
                    "start_date": "2020-01",
                    "end_date": "2022-01",
                    "evidence": "Python developer",
                }
            ]
        ),
        overall_experience=OverallExperienceOutput(
            target_job_title="Python Developer",
            target_overall_years=1.0,
            candidate_roles=[],
        ),
        scorecard=Scorecard(
            final_relevance=80.0,
            pillar_a={"score": 100.0, "raw": "1/1 skills"},
            pillar_b={"score": 100.0, "raw": "One matched skill"},
            pillar_c={"score": 0.0, "raw": "No relevant roles"},
            counted_roles=[],
        ),
        metrics=PipelineMetrics(
            total_requirements=1,
            total_matched=1,
            match_percentage=100.0,
            final_relevance=80.0,
        ),
        redacted_cv="[PERSON_NAME]\nPython developer",
        pii_spans=[TextSpan(kind="person_name", text="Jane Doe")],
    )


class ArtifactLoggerTest(unittest.TestCase):
    def test_log_run_writes_a_versioned_valid_artifact_without_raw_pii(self):
        with tempfile.TemporaryDirectory() as output_dir:
            saved_path = Path(ArtifactLogger(output_dir).log_run(make_pipeline_result()))
            serialized = saved_path.read_text(encoding="utf-8")
            payload = json.loads(serialized)
            artifact = RunArtifact.model_validate_json(serialized)

            self.assertEqual(payload["schema_version"], "2.1")
            self.assertEqual(artifact.metadata.run_number, 1)
            self.assertEqual(artifact.metadata.engine, "example/model:latest")
            self.assertEqual(artifact.metadata.pii_engine, "local-pii:latest")
            self.assertEqual(artifact.skills_evaluation.matched_cv_skills, ["Python"])
            self.assertNotIn("Jane Doe", serialized)
            self.assertEqual(list(Path(output_dir).glob("*.tmp")), [])

    def test_repeated_runs_use_unique_safe_filenames(self):
        with tempfile.TemporaryDirectory() as output_dir:
            logger = ArtifactLogger(output_dir)

            first_path = Path(logger.log_run(make_pipeline_result()))
            second_path = Path(logger.log_run(make_pipeline_result()))

            self.assertNotEqual(first_path, second_path)
            self.assertNotIn("/", first_path.name)
            self.assertNotIn(":", first_path.name)
            self.assertTrue(first_path.exists())
            self.assertTrue(second_path.exists())

    def test_run_numbers_persist_across_logger_instances(self):
        with tempfile.TemporaryDirectory() as output_dir:
            first_path = Path(
                ArtifactLogger(output_dir).log_run(make_pipeline_result())
            )
            second_path = Path(
                ArtifactLogger(output_dir).log_run(make_pipeline_result())
            )
            first_artifact = RunArtifact.model_validate_json(
                first_path.read_text(encoding="utf-8")
            )
            second_artifact = RunArtifact.model_validate_json(
                second_path.read_text(encoding="utf-8")
            )

            self.assertTrue(first_path.name.startswith("run-000001_"))
            self.assertTrue(second_path.name.startswith("run-000002_"))
            self.assertEqual(first_artifact.metadata.run_number, 1)
            self.assertEqual(second_artifact.metadata.run_number, 2)

    def test_concurrent_runs_reserve_unique_numbers(self):
        with tempfile.TemporaryDirectory() as output_dir:
            def write_artifact(_: int) -> Path:
                return Path(
                    ArtifactLogger(output_dir).log_run(make_pipeline_result())
                )

            with ThreadPoolExecutor(max_workers=4) as executor:
                paths = list(executor.map(write_artifact, range(8)))

            run_numbers = sorted(
                RunArtifact.model_validate_json(path.read_text(encoding="utf-8"))
                .metadata.run_number
                for path in paths
            )

            self.assertEqual(run_numbers, list(range(1, 9)))
            self.assertEqual(len(set(paths)), 8)


if __name__ == "__main__":
    unittest.main()