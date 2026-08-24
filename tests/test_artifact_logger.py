import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.schemas.artifact import RunArtifact, RunConfig, RunModelConfig
from src.schemas.evaluation import CheckResult, EvaluationReport
from src.utils import ArtifactLogger
from tests.factories import build_pipeline_result


def make_pipeline_result():
    return build_pipeline_result(
        engine="example/model:latest",
        pii_engine="local-pii:latest",
        execution_seconds=1.25,
        skill_name="Python",
        target_job_title="Python Developer",
        target_overall_years=1.0,
        final_relevance=80.0,
    )


class ArtifactLoggerTest(unittest.TestCase):
    def test_log_run_writes_a_versioned_valid_artifact_without_raw_pii(self):
        result = make_pipeline_result()
        with tempfile.TemporaryDirectory() as output_dir:
            saved_path = Path(ArtifactLogger(output_dir).log_run(result))
            serialized = saved_path.read_text(encoding="utf-8")
            payload = json.loads(serialized)
            artifact = RunArtifact.model_validate_json(serialized)

            self.assertEqual(payload["schema_version"], "3.4")
            self.assertEqual(artifact.metadata.run_number, 1)
            self.assertEqual(artifact.metadata.engine, "example/model:latest")
            self.assertEqual(artifact.metadata.pii_engine, "local-pii:latest")
            self.assertEqual(artifact.skills_evaluation.matched_cv_skills, ["Python"])
            self.assertNotIn("Jane Doe", serialized)
            self.assertNotIn("redacted_cv\"", serialized)  # no embedded CV text, only a trace_id reference
            self.assertEqual(artifact.redacted_cv_trace_id, result.redacted_cv_trace_id)
            self.assertEqual(list(Path(output_dir).glob("*.tmp")), [])
            self.assertEqual(artifact.metadata.trace_id, result.trace_id)
            self.assertIn(str(result.trace_id)[-8:], saved_path.name)
            self.assertIsNone(artifact.config)

    def test_log_run_snapshots_the_config_that_produced_it(self):
        result = make_pipeline_result()
        config = RunConfig(
            task_name="cv_job_match",
            task_path="tasks/cv_job_match.yaml",
            pipeline="extraction",
            scoring_weights={"skills_match": 0.6, "work_experience": 0.4},
            pii_detectors=["regex", "model"],
            evaluation_model=RunModelConfig(
                name="gemini-flash", engine="gemini-3.1-flash-lite", temperature=0.0
            ),
            pii_model=RunModelConfig(
                name="local-llama", engine="llama3.2:latest", temperature=0.0
            ),
        )

        with tempfile.TemporaryDirectory() as output_dir:
            saved_path = Path(
                ArtifactLogger(output_dir).log_run(result, config=config)
            )
            artifact = RunArtifact.model_validate_json(
                saved_path.read_text(encoding="utf-8")
            )

            self.assertIsNotNone(artifact.config)
            self.assertEqual(artifact.config.task_name, "cv_job_match")
            self.assertEqual(artifact.config.scoring_weights["skills_match"], 0.6)
            self.assertEqual(artifact.config.evaluation_model.name, "gemini-flash")
            self.assertEqual(artifact.metadata.trace_id, result.trace_id)
            self.assertEqual(artifact.trace, result.trace)

    def test_log_run_can_include_evaluation_report(self):
        evaluation = EvaluationReport(
            passed=True,
            checks=[
                CheckResult(
                    name="min_final_relevance",
                    expected=">= 30",
                    actual="80.0",
                    passed=True,
                )
            ],
        )

        with tempfile.TemporaryDirectory() as output_dir:
            saved_path = Path(
                ArtifactLogger(output_dir).log_run(
                    make_pipeline_result(), evaluation=evaluation
                )
            )
            artifact = RunArtifact.model_validate_json(
                saved_path.read_text(encoding="utf-8")
            )

            self.assertIsNotNone(artifact.evaluation)
            self.assertTrue(artifact.evaluation.passed)
            self.assertEqual(artifact.evaluation.checks[0].name, "min_final_relevance")

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