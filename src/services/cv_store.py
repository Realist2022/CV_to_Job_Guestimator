"""Content-addressed persistence for RedactedCV artifacts.

Ingestion happens once per CV; matching can then run any number of times
against the same stored RedactedCV without the raw text being touched
again. Storage is content-addressed on RedactedCV.cv_id (sha256 of the
normalised raw text — see schemas/ingestion.py), so re-ingesting identical
CV text is idempotent: it just overwrites the same file, e.g. with a
fresher redaction if the PII policy changed since the last ingest.
"""

import os
import tempfile
from pathlib import Path

from src.schemas.ingestion import RedactedCV


class CVNotFoundError(KeyError):
    """Raised by load() when no RedactedCV has been stored for a cv_id."""


class CVIngestionStore:
    def __init__(self, output_dir: str | Path = "redacted_cvs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, redacted_cv: RedactedCV) -> str:
        out_path = self._path_for(redacted_cv.cv_id)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.output_dir,
                prefix=f".{redacted_cv.cv_id}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(redacted_cv.model_dump_json(indent=2))
                handle.write("\n")
                temporary_path = Path(handle.name)
            os.replace(temporary_path, out_path)
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
        return str(out_path)

    def load(self, cv_id: str) -> RedactedCV:
        path = self._path_for(cv_id)
        if not path.exists():
            raise CVNotFoundError(f"No ingested CV found for cv_id '{cv_id}'.")
        return RedactedCV.model_validate_json(path.read_text(encoding="utf-8"))

    def _path_for(self, cv_id: str) -> Path:
        return self.output_dir / f"{cv_id}.json"
