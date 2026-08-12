"""Structural interfaces the harness depends on.

The harness never imports business logic directly; it only requires that
resolved components satisfy these protocols.
"""

from typing import Protocol, runtime_checkable

from src.schemas.pipeline import PipelineResult
from src.services.document_parser import CandidateCV, JobListing


@runtime_checkable
class PipelineProtocol(Protocol):
    def run(
        self, listing: JobListing, cv: CandidateCV, *, verbose: bool = True
    ) -> PipelineResult: ...


@runtime_checkable
class EvaluatorProtocol(Protocol):
    def evaluate(self, result: PipelineResult): ...
