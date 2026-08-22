import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypeVar
from uuid import UUID

from pydantic import BaseModel

from src.schemas.artifact import (
    IngestionArtifact,
    IngestionRunConfig,
    RunArtifact,
    RunConfig,
)
from src.schemas.evaluation import EvaluationReport
from src.schemas.ingestion import IngestionResult
from src.schemas.pipeline import PipelineResult

_ArtifactT = TypeVar("_ArtifactT", bound=BaseModel)


class ArtifactLogger:
    _ARTIFACT_RUN_PATTERN = re.compile(r"^run-(\d+)_.*\.json$")
    _RESERVATION_RUN_PATTERN = re.compile(r"^\.run-(\d+)\.reserve$")

    def __init__(self, output_dir: str | Path = "artifacts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.last_run_number: int | None = None

    def log_run(
        self,
        result: PipelineResult,
        evaluation: EvaluationReport | None = None,
        config: RunConfig | None = None,
    ) -> str:
        return self._log(
            build=lambda run_number: RunArtifact.from_pipeline_result(
                result, run_number, evaluation, config
            ),
            engine_name=result.engine,
            trace_id=result.trace_id,
        )

    def log_ingestion_run(
        self,
        result: IngestionResult,
        evaluation: EvaluationReport | None = None,
        config: IngestionRunConfig | None = None,
    ) -> str:
        return self._log(
            build=lambda run_number: IngestionArtifact.from_ingestion_result(
                result, run_number, evaluation, config
            ),
            engine_name=result.pii_engine,
            trace_id=result.trace_id,
        )

    def _log(
        self,
        build: Callable[[int], _ArtifactT],
        engine_name: str,
        trace_id: UUID,
    ) -> str:
        run_number, reservation_path = self._reserve_run_number()
        try:
            artifact = build(run_number)
            out_path = self._write_artifact(
                artifact,
                run_number=run_number,
                engine_name=engine_name,
                trace_id=trace_id,
            )
            self.last_run_number = run_number
            return out_path
        finally:
            reservation_path.unlink(missing_ok=True)

    def _reserve_run_number(self) -> tuple[int, Path]:
        used_numbers = []
        for path in self.output_dir.iterdir():
            artifact_match = self._ARTIFACT_RUN_PATTERN.match(path.name)
            reservation_match = self._RESERVATION_RUN_PATTERN.match(path.name)
            match = artifact_match or reservation_match
            if match:
                used_numbers.append(int(match.group(1)))

        candidate = max(used_numbers, default=0) + 1
        while True:
            reservation_path = self.output_dir / f".run-{candidate:06d}.reserve"
            try:
                descriptor = os.open(
                    reservation_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError:
                candidate += 1
                continue
            os.close(descriptor)
            return candidate, reservation_path

    def _write_artifact(
        self,
        artifact: BaseModel,
        run_number: int,
        engine_name: str,
        trace_id: UUID,
    ) -> str:
        safe_engine_name = re.sub(r"[^A-Za-z0-9._-]+", "_", engine_name).strip("._") or "unknown-engine"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        # trace_id is a time-ordered UUIDv7, so its leading hex digits are
        # mostly timestamp and don't add much filename entropy; the tail is
        # its random portion, so use that for a short unique-looking suffix.
        filename = (
            f"run-{run_number:06d}_{safe_engine_name}_" f"{timestamp}_{str(trace_id)[-8:]}.json"
        )
        out_path = self.output_dir / filename
        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.output_dir,
                prefix=f".{filename}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(artifact.model_dump_json(indent=2))
                handle.write("\n")
                temporary_path = Path(handle.name)
            os.replace(temporary_path, out_path)
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise

        return str(out_path)
