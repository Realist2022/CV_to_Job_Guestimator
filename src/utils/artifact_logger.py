# src/utils/artifact_logger.py
import os
from typing import Dict, Any
from src.schemas.artifacts import PipelineRunArtifact, AgentStepTrace

class ArtifactLogger:
    def __init__(self, output_dir: str = "artifacts"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def log_run(
        self,
        engine: str,
        execution_time_seconds: float,
        pipeline_data: Dict[str, Any],
        report: Dict[str, Any]
    ) -> str:
        """
        Converts pipeline run outputs into a structured artifact and writes it to disk.
        Returns the saved file path.
        """
        # Convert Pydantic models from each agent step into dictionary payloads
        agent_traces = [
            AgentStepTrace(
                agent_id="Agent 1",
                action="Extract Job Requirements",
                output_payload=pipeline_data["job_requirements"].model_dump()
            ),
            AgentStepTrace(
                agent_id="Agent 2",
                action="Audit CV Technical Skills",
                output_payload=pipeline_data["matched_skills"].model_dump()
            ),
            AgentStepTrace(
                agent_id="Agent 3",
                action="Extract Career History & Relevance",
                output_payload=pipeline_data["overall_experience"].model_dump()
            ),
        ]

        # Build full artifact
        run_artifact = PipelineRunArtifact(
            engine=engine,
            execution_time_seconds=round(execution_time_seconds, 2),
            agent_traces=agent_traces,
            computed_report=report
        )

        # Format filename as artifacts/run_YYYYMMDD_HHMMSS.json
        timestamp_str = run_artifact.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"run_{timestamp_str}.json"
        filepath = os.path.join(self.output_dir, filename)

        # Write to JSON
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(run_artifact.model_dump_json(indent=2))

        return filepath