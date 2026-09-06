"""GCSArtifactLogger against a fake bucket that enforces GCS's preconditions.

The point of these tests is the run-number reservation, not the upload. On
local disk that reservation is an O_CREAT|O_EXCL file and obviously atomic;
in object storage it is a compare-and-swap on a counter object, and the
whole scheme is only correct if a losing writer actually retries. So the
fake below implements `if_generation_match` faithfully -- including raising
PreconditionFailed -- and one test drives a real race through it.
"""

import json

import pytest
from google.api_core import exceptions as gcs_exceptions

from src.utils.gcs_artifact_logger import (
    COUNTER_OBJECT,
    ArtifactUploadError,
    GCSArtifactLogger,
    build_artifact_logger,
)
from tests.factories import build_pipeline_result


class FakeBlob:
    def __init__(self, bucket: "FakeBucket", name: str):
        self._bucket = bucket
        self.name = name

    @property
    def generation(self) -> int:
        return self._bucket.generations.get(self.name, 0)

    def reload(self) -> None:
        if self.name not in self._bucket.objects:
            raise gcs_exceptions.NotFound(self.name)

    def download_as_bytes(self) -> bytes:
        return self._bucket.objects[self.name]

    def upload_from_string(self, data, content_type=None, if_generation_match=None):
        self._bucket.uploads.append(self.name)
        if self._bucket.on_upload is not None:
            self._bucket.on_upload(self.name)

        if if_generation_match is not None:
            expected = if_generation_match
            actual = self._bucket.generations.get(self.name, 0)
            if expected != actual:
                raise gcs_exceptions.PreconditionFailed(
                    f"generation {actual} != expected {expected}"
                )

        payload = data.encode("utf-8") if isinstance(data, str) else data
        self._bucket.objects[self.name] = payload
        self._bucket.generations[self.name] = self._bucket.generations.get(self.name, 0) + 1


class FakeBucket:
    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.generations: dict[str, int] = {}
        self.uploads: list[str] = []
        #: Hook to mutate the bucket mid-upload, to simulate another writer
        #: winning the race between our read and our write.
        self.on_upload = None

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(self, name)


class FakeClient:
    def __init__(self, bucket: FakeBucket):
        self._bucket = bucket

    def bucket(self, _name: str) -> FakeBucket:
        return self._bucket


@pytest.fixture
def bucket() -> FakeBucket:
    return FakeBucket()


@pytest.fixture
def logger(bucket: FakeBucket) -> GCSArtifactLogger:
    return GCSArtifactLogger("test-bucket", prefix="artifacts", client=FakeClient(bucket))


def test_first_run_starts_at_one(logger, bucket):
    path = logger.log_run(build_pipeline_result())

    assert logger.last_run_number == 1
    assert path.startswith("gs://test-bucket/artifacts/run-000001_")
    assert bucket.objects[f"artifacts/{COUNTER_OBJECT}"] == b"1"


def test_run_numbers_increment(logger):
    paths = [logger.log_run(build_pipeline_result()) for _ in range(3)]

    assert [p.split("/")[-1][:10] for p in paths] == ["run-000001", "run-000002", "run-000003"]
    assert logger.last_run_number == 3


def test_uploaded_artifact_is_the_run(logger, bucket):
    logger.log_run(build_pipeline_result(final_relevance=42.0))

    name = next(n for n in bucket.objects if n.endswith(".json"))
    payload = json.loads(bucket.objects[name])
    assert payload["metadata"]["run_number"] == 1
    assert payload["scorecard"]["final_relevance"] == 42.0


def test_a_losing_writer_retries_and_gets_the_next_number(logger, bucket):
    """The reservation's whole reason for existing.

    Another instance claims the number between our read and our write, so
    the precondition fails. The logger must re-read and take the next one --
    never hand back a number someone else already used.
    """
    counter = f"artifacts/{COUNTER_OBJECT}"

    def steal_once(name):
        if name == counter and bucket.on_upload is not None:
            bucket.on_upload = None  # only interfere with the first attempt
            bucket.objects[counter] = b"7"
            bucket.generations[counter] = 99

    bucket.on_upload = steal_once

    logger.log_run(build_pipeline_result())

    assert logger.last_run_number == 8, "should continue from the value the rival wrote"
    assert bucket.objects[counter] == b"8"


def test_gives_up_rather_than_spinning_forever(logger, bucket):
    """A permanently contended counter must fail the request, not hang it."""
    counter = f"artifacts/{COUNTER_OBJECT}"

    def always_steal(name):
        if name == counter:
            bucket.generations[counter] = bucket.generations.get(counter, 0) + 50

    bucket.on_upload = always_steal

    with pytest.raises(ArtifactUploadError, match="reserve a run number"):
        logger.log_run(build_pipeline_result())


def test_corrupt_counter_refuses_rather_than_resetting(logger, bucket):
    """Resetting to 0 would hand out numbers that already name artifacts,
    and every subsequent upload precondition would then fail."""
    counter = f"artifacts/{COUNTER_OBJECT}"
    bucket.objects[counter] = b"not-a-number"
    bucket.generations[counter] = 1

    with pytest.raises(ArtifactUploadError, match="not an integer"):
        logger.log_run(build_pipeline_result())


def test_never_overwrites_an_existing_artifact(logger, bucket, monkeypatch):
    """Belt and braces behind the counter: uploads use if_generation_match=0,
    so a duplicated number surfaces instead of silently losing a run."""
    monkeypatch.setattr(
        "src.utils.gcs_artifact_logger.artifact_filename",
        lambda *_args, **_kwargs: "collision.json",
    )

    logger.log_run(build_pipeline_result())
    with pytest.raises(ArtifactUploadError, match="Could not write"):
        logger.log_run(build_pipeline_result())


def test_ingestion_runs_share_the_same_counter(logger, bucket):
    from src.schemas.ingestion import IngestionResult, RedactedCV

    redacted = RedactedCV.from_raw_text(
        raw_text="Jane Doe",
        redacted_text="[PERSON_NAME]",
        pii_spans=[],
        pii_engine="presidio:test",
    )
    ingestion = IngestionResult(
        cv_id=redacted.cv_id,
        pii_engine="presidio:test",
        execution_seconds=0.01,
        pii_spans=[],
        redacted_cv=redacted,
    )

    logger.log_run(build_pipeline_result())
    logger.log_ingestion_run(ingestion)

    assert logger.last_run_number == 2, "one sequence for both artifact kinds"


@pytest.mark.parametrize("prefix", ["artifacts", "artifacts/", "/artifacts", "/artifacts/"])
def test_prefix_is_normalised(bucket, prefix):
    logger = GCSArtifactLogger("b", prefix=prefix, client=FakeClient(bucket))

    logger.log_run(build_pipeline_result())

    assert all(name.startswith("artifacts/") for name in bucket.objects)


def test_factory_returns_local_logger_without_a_bucket(monkeypatch, tmp_path):
    monkeypatch.delenv("ARTIFACTS_BUCKET", raising=False)

    logger = build_artifact_logger(output_dir=str(tmp_path))

    assert type(logger).__name__ == "ArtifactLogger"


def test_factory_returns_gcs_logger_when_a_bucket_is_set(monkeypatch):
    monkeypatch.setenv("ARTIFACTS_BUCKET", "some-bucket")
    monkeypatch.setenv("ARTIFACTS_PREFIX", "runs")
    monkeypatch.setattr(
        "google.cloud.storage.Client", lambda *a, **k: FakeClient(FakeBucket())
    )

    logger = build_artifact_logger()

    assert isinstance(logger, GCSArtifactLogger)
    assert logger.bucket_name == "some-bucket"
    assert logger.prefix == "runs"
