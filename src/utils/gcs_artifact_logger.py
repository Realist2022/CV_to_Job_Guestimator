"""Artifact logging to a Google Cloud Storage bucket.

Cloud Run's filesystem is read-only apart from /tmp, which is in-memory and
gone when the instance recycles -- so ArtifactLogger's local-disk writes
cannot work there, and writing to /tmp would mean the run history evaporates
along with the container. This backend puts the same artifacts in a bucket,
where they outlive the instance and can be read back later.

THE HARD PART IS THE RUN NUMBER, not the upload. ArtifactLogger reserves one
with an O_CREAT|O_EXCL file, which is atomic on one filesystem and has no
equivalent in object storage: GCS has no directories to scan cheaply and no
exclusive-create open. What it does have is generation preconditions --
`if_generation_match` makes a write conditional on the object still being
exactly the version you read. That is a compare-and-swap, so a single
counter object gives real sequential numbers across every instance:

    read counter (value + generation) -> write value+1 if generation matches

A second writer that got there first bumps the generation, the precondition
fails with 412, and we re-read and retry. The artifact upload itself then
uses `if_generation_match=0` ("only if this object does not exist"), so two
runs can never silently overwrite each other even if the counter were wrong.

The alternative -- dropping run numbers and naming artifacts by trace_id --
was tempting, since trace_id is already a time-ordered UUIDv7 and globally
unique. It is rejected because RunArtifact.run_number is a real field that
scripts/compare_runs.py and every existing artifact rely on, and inventing a
number that is not sequential would make it a lie rather than a simplification.
"""

import os
import random
import time
from typing import Callable, TypeVar
from uuid import UUID

from pydantic import BaseModel

from src.utils.artifact_logger import ArtifactLoggerBase, artifact_filename

_ArtifactT = TypeVar("_ArtifactT", bound=BaseModel)

BUCKET_ENV = "ARTIFACTS_BUCKET"
PREFIX_ENV = "ARTIFACTS_PREFIX"

#: Name of the compare-and-swap counter object, relative to the prefix. The
#: leading underscore keeps it out of the way of the run-*.json listing.
COUNTER_OBJECT = "_run_counter"

#: Contention here is between concurrent requests on one small service, not
#: a thundering herd; a handful of tries with backoff is plenty, and failing
#: loudly beats retrying forever inside a request handler.
MAX_RESERVE_ATTEMPTS = 8


class ArtifactUploadError(RuntimeError):
    """Raised when an artifact could not be written to the bucket."""


class GCSArtifactLogger(ArtifactLoggerBase):
    """ArtifactLogger's interface, backed by a GCS bucket.

    Duck-types the local logger exactly (see ArtifactLoggerBase), including
    `last_run_number`, so routes.py can hold either without knowing which.
    """

    def __init__(self, bucket_name: str, prefix: str = "artifacts", client=None):
        if not bucket_name:
            raise ValueError("GCSArtifactLogger requires a bucket name.")
        self.bucket_name = bucket_name
        # Normalised so "artifacts", "artifacts/" and "/artifacts" all name
        # the same place rather than creating three sibling trees.
        self.prefix = prefix.strip("/")
        self.last_run_number: int | None = None

        if client is None:
            # Imported here rather than at module scope so that importing
            # this module -- which src/utils/__init__.py does on every run,
            # including every local one -- never costs the google client's
            # import time or requires it to be installed to run the tests.
            from google.cloud import storage

            client = storage.Client()
        self._bucket = client.bucket(bucket_name)

    def _path(self, name: str) -> str:
        return f"{self.prefix}/{name}" if self.prefix else name

    def _log(
        self,
        build: Callable[[int], _ArtifactT],
        engine_name: str,
        trace_id: UUID,
    ) -> str:
        run_number = self._reserve_run_number()
        artifact = build(run_number)
        blob_path = self._path(artifact_filename(run_number, engine_name, trace_id))

        blob = self._bucket.blob(blob_path)
        try:
            blob.upload_from_string(
                artifact.model_dump_json(indent=2) + "\n",
                content_type="application/json",
                # Refuse to clobber: an existing object at this name means
                # the counter handed out a number twice, which is a bug
                # worth surfacing rather than quietly losing a run.
                if_generation_match=0,
            )
        except Exception as exc:
            raise ArtifactUploadError(
                f"Could not write gs://{self.bucket_name}/{blob_path}: {exc}"
            ) from exc

        self.last_run_number = run_number
        return f"gs://{self.bucket_name}/{blob_path}"

    def _reserve_run_number(self) -> int:
        """Claim the next run number, atomically, across all instances."""
        from google.api_core import exceptions as gcs_exceptions

        counter = self._bucket.blob(self._path(COUNTER_OBJECT))

        for attempt in range(MAX_RESERVE_ATTEMPTS):
            current, generation = self._read_counter(counter)
            candidate = current + 1
            try:
                counter.upload_from_string(
                    str(candidate),
                    content_type="text/plain",
                    # generation 0 means "only if absent", which is exactly
                    # what the first ever run needs.
                    if_generation_match=generation,
                )
            except gcs_exceptions.PreconditionFailed:
                # Someone else claimed this number first. Re-read and retry
                # with jitter so two racing instances don't lock step.
                time.sleep(min(0.05 * 2**attempt, 1.0) * (0.5 + random.random()))
                continue
            return candidate

        raise ArtifactUploadError(
            f"Could not reserve a run number after {MAX_RESERVE_ATTEMPTS} attempts "
            f"(gs://{self.bucket_name}/{self._path(COUNTER_OBJECT)} is heavily contended)."
        )

    def _read_counter(self, counter) -> tuple[int, int]:
        """Current value and the generation it was read at.

        Returns (0, 0) when the counter does not exist yet: generation 0 is
        GCS's "object must be absent" precondition, so a fresh bucket takes
        the same code path as a contended one.
        """
        from google.api_core import exceptions as gcs_exceptions

        try:
            counter.reload()
        except gcs_exceptions.NotFound:
            return 0, 0

        try:
            value = int(counter.download_as_bytes().decode("utf-8").strip())
        except (ValueError, UnicodeDecodeError) as exc:
            # Refuse to guess. Resetting to 0 here would start handing out
            # numbers that already name existing artifacts, and the upload
            # precondition would then fail on every single run.
            raise ArtifactUploadError(
                f"Run counter gs://{self.bucket_name}/{self._path(COUNTER_OBJECT)} "
                f"is not an integer: {exc}"
            ) from exc
        return value, counter.generation


def build_artifact_logger(output_dir: str = "artifacts"):
    """The artifact logger this deployment should use.

    Returns a GCSArtifactLogger when ARTIFACTS_BUCKET names a bucket, and
    the local-disk ArtifactLogger otherwise. Chosen by environment rather
    than by a constructor argument at each call site so that local runs, the
    CLI harness and the container all get the right one without any of them
    knowing which deployment they are in.
    """
    bucket = os.getenv(BUCKET_ENV, "").strip()
    if not bucket:
        from src.utils.artifact_logger import ArtifactLogger

        return ArtifactLogger(output_dir=output_dir)

    return GCSArtifactLogger(
        bucket_name=bucket,
        prefix=os.getenv(PREFIX_ENV, "").strip() or output_dir,
    )
