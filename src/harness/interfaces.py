"""Structural interface the `pipelines` registry enforces.

The harness never imports business logic directly; the `pipelines` registry
(see registry.py) checks every component it builds against this protocol at
create() time, so a factory registered under a valid name but returning
something that doesn't actually implement run() fails loudly right there
instead of surfacing as an AttributeError deep inside a task run.

There is no equivalent registry for evaluators — ThresholdEvaluator is the
only implementation and is constructed directly by the runner — so no
EvaluatorProtocol is declared here; add one if/when evaluators actually
become pluggable.
"""

from typing import Protocol, runtime_checkable

from src.schemas.pipeline import PipelineResult
from src.services.document_parser import CandidateCV, JobListing


@runtime_checkable
class PipelineProtocol(Protocol):
    def run(
        self, listing: JobListing, cv: CandidateCV, *, verbose: bool = True
    ) -> PipelineResult: ...
