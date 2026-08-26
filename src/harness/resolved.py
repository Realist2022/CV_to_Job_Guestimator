"""The shared contract between resolution (Layer 2) and orchestration (Layer 1).

A Resolved*Run is a fully-resolved unit of work: documents already parsed,
clients/pipelines already constructed, thresholds already chosen. It is the
only shape HarnessRunner's run_* methods accept, and both resolvers produce
it — task_resolver.py from a tasks/*.yaml TaskSpec (CLI), and
src/api/harness_adapter.py from an already-parsed HTTP request (web API).
Nothing here knows about YAML paths or UploadFiles; that's the point.

Lives in its own module (not runner.py) so both resolvers can import the
contract without importing each other: runner -> task_resolver -> resolved
stays acyclic.
"""

from dataclasses import dataclass, field

from src.harness.interfaces import PipelineProtocol
from src.harness.task_loader import EvaluationCriteria
from src.schemas.ingestion import RedactedCV
from src.services.document_parser import CandidateCV, JobListing
from src.services.ingestion_pipeline import IngestionPipeline
from src.services.llm_client import CompletionClient
from src.services.scoring_engine import RelevanceScoringEngine


@dataclass
class RunMetadata:
    """Where this run came from. Both None for API-triggered runs."""

    task_name: str | None = None
    task_path: str | None = None


@dataclass
class ResolvedExtractionRun:
    listing: JobListing
    cv: CandidateCV
    pipeline: PipelineProtocol
    eval_client: CompletionClient
    # Named config key (configs/llm.yaml) the client was built from — kept
    # separate from eval_client.model, which is the resolved provider string.
    eval_model_name: str
    scoring_engine: RelevanceScoringEngine
    pii_detector_names: list[str]
    # Empty criteria (no thresholds) is the "live call" case: the evaluator
    # returns checks=[], passed=True, and callers that only print/persist
    # non-empty check lists are unaffected.
    evaluation: EvaluationCriteria = field(default_factory=EvaluationCriteria)
    metadata: RunMetadata = field(default_factory=RunMetadata)
    verbose: bool = True


@dataclass
class ResolvedIngestionRun:
    cv: CandidateCV
    pipeline: IngestionPipeline
    pii_detector_names: list[str]
    evaluation: EvaluationCriteria = field(default_factory=EvaluationCriteria)
    metadata: RunMetadata = field(default_factory=RunMetadata)
    verbose: bool = True


@dataclass
class ResolvedMatchingRun:
    listing: JobListing
    redacted_cv: RedactedCV
    pipeline: PipelineProtocol
    eval_client: CompletionClient
    eval_model_name: str
    scoring_engine: RelevanceScoringEngine
    evaluation: EvaluationCriteria = field(default_factory=EvaluationCriteria)
    metadata: RunMetadata = field(default_factory=RunMetadata)
    verbose: bool = True


ResolvedRun = ResolvedExtractionRun | ResolvedIngestionRun | ResolvedMatchingRun
