"""The overall-experience step running alongside requirements + matching.

Sequential order was arrangement, not a data dependency: the experience
agent reads only the listing and the CV, never the extracted requirements.
On the Modal deployment that step is ~40s of a ~53s warm run, so overlapping
it is most of the wall-clock win available without touching the model.

These pin the two things that could go wrong quietly: the steps not actually
overlapping, and a shared client corrupting the `attempts` count.
"""

import threading
import time

from src.prompts.templates import (
    JOB_REQUIREMENTS_SYSTEM_PROMPT,
    OVERALL_EXPERIENCE_SYSTEM_PROMPT,
    SKILL_MATCHER_SYSTEM_PROMPT,
)
from src.schemas.experience import OverallExperienceOutput
from src.schemas.ingestion import RedactedCV
from src.schemas.requirements import JobRequirementsOutput
from src.services.document_parser import JobListing
from src.services.matching_pipeline import MatchingPipeline
from tests.factories import RecordingClient

_RESPONSES = {
    JOB_REQUIREMENTS_SYSTEM_PROMPT: JobRequirementsOutput(
        job_requirements=[{"skill_name": "Python"}]
    ),
    SKILL_MATCHER_SYSTEM_PROMPT: lambda model: model(
        evaluations=[{"requirement_id": 0, "matched": True}]
    ),
    OVERALL_EXPERIENCE_SYSTEM_PROMPT: OverallExperienceOutput(
        target_job_title="Python Developer",
        target_overall_years=2.0,
        candidate_roles=[
            {
                "role_title": "Python Developer",
                "start_date": "2020-01",
                "end_date": "2022-01",
                "match_rationale": "Directly relevant development role.",
                "is_relevant": True,
            }
        ],
    ),
}


class SlowClient(RecordingClient):
    """RecordingClient that sleeps, so overlap shows up in wall-clock time,
    and that notices if it is ever called from two threads at once."""

    def __init__(self, model: str = "fake", delay: float = 0.3):
        super().__init__(model, _RESPONSES)
        self.delay = delay
        self._in_flight = 0
        self.max_in_flight = 0
        self._lock = threading.Lock()

    def complete(self, *args, **kwargs):
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            time.sleep(self.delay)
            return super().complete(*args, **kwargs)
        finally:
            with self._lock:
                self._in_flight -= 1


def _cv() -> RedactedCV:
    return RedactedCV.from_raw_text(
        raw_text="Jane Doe\nPython developer since 2020",
        redacted_text="[PERSON_NAME]\nPython developer since 2020",
        pii_spans=[],
        pii_engine="presidio:test",
    )


def _listing() -> JobListing:
    return JobListing("Requirements\nPython")


def test_sequential_by_default():
    """No second client means the previous behaviour, unchanged — which the
    CLI harness and the rest of the suite rely on."""
    client = SlowClient(delay=0.15)

    result = MatchingPipeline(client).run(_listing(), _cv(), verbose=False)

    assert [s.step for s in result.trace] == [
        "job_requirements_extraction",
        "skill_matching",
        "overall_experience_extraction",
        "scoring",
    ]
    assert client.max_in_flight == 1, "must not overlap calls on a single client"


def test_experience_overlaps_when_given_its_own_client():
    main = SlowClient(delay=0.3)
    experience = SlowClient(delay=0.3)

    result = MatchingPipeline(main, experience_client=experience).run(
        _listing(), _cv(), verbose=False
    )

    spans = {s.step: s for s in result.trace}
    started_apart = (
        spans["overall_experience_extraction"].started_at
        - spans["job_requirements_extraction"].started_at
    ).total_seconds()
    assert abs(started_apart) < 0.2, "the two should start together, not one after the other"

    # Wall clock shorter than the work performed is only possible if the
    # steps genuinely ran at the same time.
    assert result.execution_seconds < sum(s.duration_seconds for s in result.trace)


def test_neither_client_is_ever_called_concurrently():
    """Why a second client is required rather than reusing one:
    InstructorClient registers an attempt-counting hook around each call and
    documents itself as unsafe to call concurrently. Sharing one would
    corrupt `attempts` in every artifact — a bug that reads as model
    flakiness."""
    main = SlowClient(delay=0.2)
    experience = SlowClient(delay=0.2)

    MatchingPipeline(main, experience_client=experience).run(_listing(), _cv(), verbose=False)

    assert main.max_in_flight == 1
    assert experience.max_in_flight == 1
    assert len(experience.requests) == 1, "experience client serves only its own step"
    assert len(main.requests) == 2, "requirements + skill matching"


def test_attempts_survive_the_parallel_path():
    result = MatchingPipeline(
        SlowClient(delay=0.1), experience_client=SlowClient(delay=0.1)
    ).run(_listing(), _cv(), verbose=False)

    for span in result.trace:
        if span.step == "scoring":
            assert span.attempts is None, "no LLM call in scoring"
        else:
            assert span.attempts == 1, f"{span.step} lost its attempts count"


def test_trace_is_ordered_by_start_time():
    """The concurrent span is appended whenever the worker finishes, so
    without the sort an artifact would read as though experience ran last."""
    result = MatchingPipeline(
        SlowClient(delay=0.3), experience_client=SlowClient(delay=0.05)
    ).run(_listing(), _cv(), verbose=False)

    starts = [s.started_at for s in result.trace]
    assert starts == sorted(starts)
    assert result.trace[-1].step == "scoring", "scoring still runs last"


def test_both_paths_produce_the_same_result():
    """Overlapping changes when calls happen, never what is sent or scored."""
    sequential = MatchingPipeline(SlowClient(delay=0.05)).run(
        _listing(), _cv(), verbose=False
    )
    parallel = MatchingPipeline(
        SlowClient(delay=0.05), experience_client=SlowClient(delay=0.05)
    ).run(_listing(), _cv(), verbose=False)

    assert sequential.scorecard == parallel.scorecard
    assert sequential.metrics == parallel.metrics
    assert sequential.skills_eval == parallel.skills_eval
    assert sequential.overall_experience == parallel.overall_experience
