"""One-shot compatibility wrapper: IngestionPipeline + MatchingPipeline.

Kept so every existing caller (the harness's "extraction" pipeline, the
web API's /api/compare, existing eval tasks) keeps working unchanged after
the split into standalone ingestion_pipeline.py / matching_pipeline.py.
New code that wants the actual benefit of the split — ingest a CV once,
match it against many listings without re-running PII detection — should
use those two pipelines directly (see CVIngestionStore in cv_store.py).

Trade-off vs. the pre-split implementation: PII redaction and job
requirements extraction used to run concurrently in a ThreadPoolExecutor
since they're independent. That optimization is gone here — ingestion now
runs to completion before matching starts, because that's the actual
point of the split (matching consumes a RedactedCV, which only exists once
ingestion has finished). For a single one-shot call this trades a small
amount of latency for the two halves being genuinely decoupled.
"""

from typing import Callable, Optional

from src.schemas.ingestion import IngestionResult
from src.schemas.pipeline import PipelineResult
from src.services.document_parser import CandidateCV, JobListing
from src.services.ingestion_pipeline import IngestionPipeline
from src.services.llm_client import CompletionClient
from src.services.matching_pipeline import MatchingPipeline
from src.services.pii_detector import PIIDetector
from src.services.scoring_engine import RelevanceScoringEngine


class ExtractionPipeline:
    def __init__(
        self,
        client: CompletionClient,
        pii_detector: Optional[PIIDetector] = None,
        pii_client: Optional[CompletionClient] = None,
        scoring_engine: Optional[RelevanceScoringEngine] = None,
    ):
        self.client = client
        self.ingestion = IngestionPipeline(pii_detector=pii_detector, pii_client=pii_client)
        self.matching = MatchingPipeline(client, scoring_engine=scoring_engine)
        # Preserved for existing callers that read these off an
        # ExtractionPipeline instance directly (e.g. harness/runner.py's
        # RunConfig construction reads pii_client.model/.temperature).
        self.pii_client = self.ingestion.pii_client
        self.pii_detector = self.ingestion.pii_detector
        self.scoring_engine = self.matching.scoring_engine

    def run(
        self,
        listing: JobListing,
        cv: CandidateCV,
        *,
        verbose: bool = True,
        on_ingested: Optional[Callable[[IngestionResult], object]] = None,
    ) -> PipelineResult:
        """Run ingestion then matching, returning one combined PipelineResult.

        `on_ingested`, if given, is called with the IngestionResult right
        after ingestion finishes and before matching starts. It exists so
        callers that persist/log runs (the harness, the web API) can save
        the RedactedCV and log its own IngestionArtifact the same way the
        standalone ingestion pipeline does — without this pipeline itself
        reaching for a CVIngestionStore/ArtifactLogger and picking I/O
        defaults (paths, config) that belong to the caller, not here. Kept
        optional and a no-op by default so this stays a pure result-in,
        result-out call for callers (tests included) that don't want any
        of that, e.g. see test_pipeline_privacy.py.
        """
        ingestion_result = self.ingestion.run(cv, verbose=verbose)
        if on_ingested is not None:
            on_ingested(ingestion_result)
        match_result = self.matching.run(
            listing, ingestion_result.redacted_cv, verbose=verbose
        )
        # Present as one run to callers built around the pre-split shape:
        # one trace_id/execution_seconds spanning both halves, trace spans
        # concatenated in the order they actually ran.
        return match_result.model_copy(
            update={
                "trace_id": ingestion_result.trace_id,
                "execution_seconds": round(
                    ingestion_result.execution_seconds + match_result.execution_seconds, 2
                ),
                "trace": [*ingestion_result.trace, *match_result.trace],
            }
        )
