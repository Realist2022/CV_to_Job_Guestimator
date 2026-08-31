# CV to Job Guestimator

CV to Job Guestimator is a local Python pipeline and web UI that compares a candidate CV against a job listing and produces a structured relevance score. Runs are driven by a task harness: declarative YAML tasks select the models, pipeline, inputs, and pass/fail criteria, while the pipeline extracts text from PDFs or TXT files, redacts candidate PII, sends the extracted content through a structured multi-agent LLM pipeline, calculates a weighted scorecard, and writes a JSON trace of each run.

The project is designed for local experimentation with CV/job matching logic. Candidate CVs, job listing PDFs, environment files, and generated artifacts should stay out of Git because they may contain personal or sensitive information.

## What It Does

The pipeline answers three questions:

1. What technical requirements does the job listing ask for?
2. Which of those requirements appear in the candidate CV?
3. How much relevant overall career experience does the CV show for the target role?

Those outputs are combined into a final relevance percentage using two weighted pillars:

| Pillar | Weight | Source |
| --- | ---: | --- |
| Skills match | 60% | Required skills found in the CV |
| Career match | 40% | Relevant career years against target experience |

The default weights live in `configs/scoring.yaml` and can be overridden per task via `scoring_weights`.

## Architecture

Every run goes through the same five-stage harness — task loading, component resolution, a pipeline stage, evaluation, and artifact logging — but which pipeline stage runs depends on the task's `pipeline:` field. There are three shapes, and they can be run standalone or chained:

```text
Task definition (tasks/*.yaml)
	|-- pipeline: extraction | ingestion | matching
	|-- model selection, input paths, pass/fail criteria
				|
				v
HarnessRunner (src/harness/runner.py)
	Loads configs/, resolves models and components from registries
				|
	  +---------+---------+---------+
	  |                   |         |
	  v                   v         v
pipeline: ingestion  pipeline: extraction  pipeline: matching
IngestionPipeline    (one-shot compat)     MatchingPipeline
  Detect + redact       = ingestion          Extract job requirements
  candidate CV PII       then matching       Match requirements vs.
  -> RedactedCV,                             the redacted CV
  saved to                                   Extract relevant career
  CVIngestionStore                           history -> scorecard
	  |                                          ^
	  `---- redacted_cv_id -------- CVIngestionStore ---------'
				|
				v
Pydantic schemas
	Validate agent outputs, pipeline results, scorecards, and artifacts
				|
				v
ThresholdEvaluator
	Checks the result against the task's evaluation criteria
				|
				v
ArtifactLogger
	Writes artifacts/run-000001_<engine>_<timestamp>_<run-id>.json
```

`ingestion` redacts a raw CV exactly once and persists the result (`redacted_cvs/<cv_id>.json`) via `CVIngestionStore`, so it never has to be redacted again. `matching` reads a previously-ingested `redacted_cv_id` and only ever sees redacted text — `src/services/matching_pipeline.py` doesn't import `CandidateCV` or a PII detector at all, so that boundary is enforced by the module's import graph, not just convention. `extraction` is the original one-shot path (raw CV in, full result out): it runs ingestion then matching back to back and persists the same `RedactedCV` either way, so a one-shot run gives the same on-disk guarantees as the two-task ingest-then-match flow.

The harness knows how to load a task, resolve components, run a pipeline stage, evaluate the result, and log an artifact. It contains no CV/job matching business logic; that stays in `src/services/` and `src/schemas/`.

The web UI wraps the same pipelines with a drag-and-drop upload page and `/api/compare`, `/api/ingest`, and `/api/match` endpoints (see [Running the Web UI](#running-the-web-ui)).

### Entry Point

`main.py` runs a harness task:

1. Loads the task file (default `tasks/cv_job_match.yaml`, or pass a path: `uv run main.py tasks/model_eval.yaml`).
2. `HarnessRunner` reads `configs/`, builds Instructor-backed clients from the named model configs, and composes the PII detectors.
3. Loads the job listing and/or CV from the first existing path listed in the task (an `ingestion` task only needs a CV; a `matching` task reads a `redacted_cv_id` instead of a raw CV).
4. Runs the pipeline stage the task's `pipeline:` field selects and computes the relevance report (or, for `ingestion`, the PII redaction report).
5. Evaluates the result against the task's criteria and prints PASS/FAIL checks.
6. Saves a JSON artifact under `artifacts/` and prints a readable score summary.

### Core Modules

| Module | Responsibility |
| --- | --- |
| `src/harness/runner.py` | Loads configs, resolves components, dispatches a task to the right pipeline shape, and logs the artifact. |
| `src/harness/task_loader.py` | Parses and validates task YAML into `TaskSpec` models (`pipeline: extraction \| ingestion \| matching`). |
| `src/harness/registry.py` | Name-to-factory registries for pipelines and PII detectors; enforces `PipelineProtocol` on anything registered as a pipeline. |
| `src/harness/evaluator.py` | Checks pipeline results against task pass/fail thresholds. |
| `src/harness/interfaces.py` | The structural protocol the `pipelines` registry enforces. |
| `src/model/providers.py` | Model providers (Ollama, OpenAI-compatible) that build LLM clients. |
| `src/model/model_registry.py` | Maps provider names in `configs/llm.yaml` to provider classes. |
| `src/model/adapters.py` | Turns a named model config into an `InstructorClient`, resolving API keys from the environment. |
| `src/config/loader.py` | Loads YAML-backed project configuration and `.env` values. |
| `src/services/document_parser.py` | Defines source document types and extracts text from PDF files using `pypdf`, PyMuPDF fallback, and OCR fallback. |
| `src/services/llm_client.py` | Wraps the OpenAI-compatible client with Instructor for structured Pydantic outputs. |
| `src/services/agents.py` | Implements the requirement extraction, skill matching, and overall experience agents. |
| `src/services/pii_detector.py` | Shared PII detector base/composite (`CompositePIIDetector`) and validation guards; composes detectors named in `configs/pii_policy.yaml`. |
| `src/services/presidio_detector.py` | The PII detector: local spaCy/Presidio NER + custom pattern recognizers — no LLM call, fully deterministic. |
| `src/services/ingestion_pipeline.py` | Standalone pipeline: redacts a raw CV once and produces a `RedactedCV`. |
| `src/services/matching_pipeline.py` | Standalone pipeline: matches a `RedactedCV` against a job listing. Never imports `CandidateCV` or a PII detector. |
| `src/services/extraction_pipeline.py` | One-shot compatibility wrapper: runs ingestion then matching and returns one combined result. |
| `src/services/cv_store.py` | Content-addressed persistence for `RedactedCV`s (`redacted_cvs/<cv_id>.json`), keyed by a hash of the normalised raw text. |
| `src/services/ingestion_persistence.py` | Shared "save the `RedactedCV`, log its `IngestionArtifact`" helper used by both the harness and the API. |
| `src/services/pipeline_tracing.py` | Shared step-timing context manager pipelines use to build their `TraceSpan` lists. |
| `src/services/scoring_engine.py` | Converts structured outputs into the weighted relevance scorecard. |
| `src/prompts/templates.py` | Stores the system prompts for each agent step. |
| `src/schemas/pii.py` | Defines PII span schemas. |
| `src/schemas/requirements.py` | Defines job requirement, skill evaluation, and skill match result schemas. |
| `src/schemas/experience.py` | Defines overall experience schemas. |
| `src/schemas/scoring.py` | Defines scorecard schemas. |
| `src/schemas/pipeline.py` | Defines pipeline result, metrics, and trace-span schemas. |
| `src/schemas/ingestion.py` | Defines `RedactedCV` and `IngestionResult` — the only CV representation `matching` is allowed to consume. |
| `src/schemas/artifact.py` | Defines the saved run/ingestion artifact formats and their config snapshots. |
| `src/schemas/evaluation.py` | Defines `CheckResult` / `EvaluationReport` — the harness's per-threshold PASS/FAIL output. |
| `src/utils/artifact_logger.py` | Serializes each run to a timestamped JSON file. |
| `src/api/app.py` | Builds the FastAPI app and serves the web UI. |
| `src/api/routes.py` | Implements the `/api/compare`, `/api/ingest`, and `/api/match` endpoints. |
| `src/api/schemas.py` | Typed response models for the API. |
| `src/web_app.py` | Backwards-compatible shim re-exporting the app from `src/api/app.py`. |
| `web/index.html` | Browser UI for uploading a job listing and CV, running comparison, and viewing results. |

## Harness Tasks and Configs

Runs are described declaratively:

- `configs/llm.yaml` defines named model configurations (provider, model, base URL, API key or `api_key_env`, temperature) for the `evaluation` role — the only LLM role left, since PII redaction has no LLM in the loop at all. It ships with `gemini-flash` (cloud), `local-llama` (stock Ollama), and two local fine-tuned builds: `cv-guestimator` — the promoted default, pinned to tag `cv-guestimator:v2` — and `cv-guestimator-v1-fixed`, the previous weights kept for rollback. See [Fine-Tuned Model Workflow](#fine-tuned-model-workflow).
- `configs/scoring.yaml` holds the default scoring weights.
- `configs/pii_policy.yaml` lists which PII detector(s) compose the composite detector — `presidio` (spaCy NER plus its own registry of custom pattern recognizers, including the NZ-specific IRD/driver's-licence/postcode formats) is the only one, fully local with no LLM call for PII at all.
- `configs/pipeline.yaml` holds pipeline runtime defaults such as verbosity and the default model selection, plus an optional `fallback_models` mapping (see below).
- `configs/deployment.yaml` documents ports and image names used by Docker.

Tasks under `tasks/` pick the pipeline shape, models, inputs, and evaluation thresholds. A task's `pipeline:` field is one of:

| Pipeline | Reads | Produces | Runs PII redaction? |
| --- | --- | --- | --- |
| `extraction` | a raw CV path + job listing | a full match result | yes |
| `ingestion` | a raw CV path | a `RedactedCV`, persisted via `CVIngestionStore` | yes |
| `matching` | a `redacted_cv_id` + job listing | a full match result | no — the CV already arrived redacted |

| Task | Pipeline | Purpose |
| --- | --- | --- |
| `tasks/cv_job_match.yaml` | `extraction` | The everyday full comparison, on the promoted `cv-guestimator` build. |
| `tasks/cv_ingest.yaml` | `ingestion` | Redacts a raw CV once and persists it, printing the `cv_id` to reuse. |
| `tasks/cv_match_from_redacted.yaml` | `matching` | Matches a job listing against a previously-ingested `redacted_cv_id`, with no PII detector call at all. |
| `tasks/model_eval.yaml` | `extraction` | Benchmarks `cv-guestimator` against real pass/fail thresholds (minimum score, maximum runtime). |
| `tasks/model_eval_v1_fixed.yaml` | `extraction` | The same benchmark on the previous weights — the rollback comparison. |

Every everyday task names the `cv-guestimator` *config key*, not a tag, so promoting a future build is a one-line change in `configs/llm.yaml` and no task file has to be touched. `model_eval_v1_fixed.yaml` is the deliberate exception: it pins the superseded build so a suspected regression in the default can be measured against it. Delete it once a rollback stops being plausible.

An `extraction` task looks like:

```yaml
name: cv_job_match
pipeline: extraction
models:
  evaluation: cv-guestimator   # any key from configs/llm.yaml
inputs:
  job_listing:
    - dataSet/tradeMeJobListing/Job_listing.txt
    - dataSet/tradeMeJobListing/Job_listing.pdf
  candidate_cv:
    - dataSet/tradeMeCV/<candidate-cv>.txt
    - dataSet/tradeMeCV/<candidate-cv>.pdf
evaluation:
  min_final_relevance: 0
```

### Evaluation criteria

Every key under a task's `evaluation:` is optional; each one present becomes one PASS/FAIL check in the run's `HARNESS EVALUATION` output and in the artifact's `evaluation` block.

| Criterion | Checks |
| --- | --- |
| `min_final_relevance` | `scorecard.final_relevance` >= value |
| `min_skills_match` | `metrics.match_percentage` >= value |
| `min_pii_spans` | number of detected PII spans >= value (`ingestion` tasks) |
| `max_execution_seconds` | wall-clock run time <= value |
| `max_attempts` | the most attempts any single LLM step needed <= value |

`max_attempts` is the sharpest quality signal of the five. An LLM step records `attempts: 2` or more when Instructor had to retry because the model returned output that failed schema validation, so `max_attempts: 1` fails a build that starts emitting malformed structured output. Steps with no LLM call (`pii_redaction` is entirely local) report `attempts: null` and are skipped rather than counted.

`configs/pipeline.yaml`'s `default_evaluation` block is the baseline every run inherits, from either entry point — `uv run main.py <task>` and the web API alike. A task's `evaluation:` layers on top **per key**, so a task that sets one criterion still inherits the rest, and setting a key to `null` in a task is how it opts out of an inherited default.

That is why the shipped tasks look thin: `cv_job_match.yaml` declares no thresholds at all and is judged purely on the inherited health checks, while `model_eval.yaml` adds only the two score gates on top. Before this, the everyday CLI task carried `min_final_relevance: 0` — a check that passes on any score, reporting PASS without meaning it — and the API applied entirely different criteria.

An inherited criterion that a result shape can't answer is dropped rather than raising: a global `min_final_relevance` must not crash an ingestion run that has no scorecard. A criterion written explicitly in a task is never dropped — it fails loudly against the wrong pipeline shape instead of being silently ignored while looking enforced.

Latency is the *weakest* of the five, and `max_execution_seconds` is deliberately loose because of it: identical runs of the same build vary by roughly 1.6x depending on GPU contention and thermal state, so a tight bound fails on hardware mood rather than on the model. Treat it as a breakage detector — it catches a hang, a silent fall back to the cloud model, or Ollama missing the tag and dropping to CPU. For judging whether one build is genuinely slower than another, compare two runs made back to back with `scripts/compare_runs.py` rather than tightening this threshold.

The first existing path in each input list wins, so TXT files take precedence over PDFs. There's no `models.pii` field at all — PII redaction runs entirely through presidio, with no LLM model to select for any pipeline shape. An `ingestion` task omits `models.evaluation` and `inputs.job_listing`; a `matching` task omits `inputs.candidate_cv`, setting `inputs.redacted_cv_id` instead (see `tasks/cv_ingest.yaml` / `tasks/cv_match_from_redacted.yaml` for both). `.vscode/task.schema.json` gives editor validation for all three shapes — regenerate it after changing `TaskSpec` with `uv run python scripts/gen_task_schema.py`.

## Model Configuration

The project uses the OpenAI Python SDK plus Instructor against OpenAI-compatible endpoints.

Harness runs (`main.py`) select models by name from `configs/llm.yaml`. The web API uses the default model names in `configs/pipeline.yaml` and resolves those names through the same `configs/llm.yaml` entries. Cloud entries reference API keys through `api_key_env`, which is resolved from the environment or a local `.env` file (for example `GOOGLE_API_KEY` for the Gemini OpenAI-compatible endpoint).

Default scoring weights live in `configs/scoring.yaml`. The web API upload form can still override those weights per request.

### Model fallback

`configs/pipeline.yaml` can name a fallback model per role:

```yaml
models:
  evaluation: cv-guestimator
fallback_models:
  evaluation: gemini-flash
```

That is the shipped configuration: there is no `pii` role to configure, because PII redaction never calls an LLM. With this set, `client_for_role("evaluation")` (used by the web API) returns a `FallbackInstructorClient` that calls the local `cv-guestimator` build first and only calls `gemini-flash` if the primary fails — the local Ollama server is unreachable or hasn't had the LoRA build `ollama create`d yet, or Instructor exhausts its retries without getting output that validates against the response schema. A role with no `fallback_models` entry behaves exactly as before (its plain configured client, no wrapping). Both the API responses and the logged run artifact (`RunModelConfig.fallback_used`) record whether a given run's evaluation actually fell back, so a run that silently used Gemini instead of the local SLM is visible after the fact, not just in logs.

This only applies where `client_for_role()` builds the client (`src/api/routes.py`). Harness tasks under `tasks/*.yaml` call `client_from_config()` directly with one pinned model each, so A/B evaluation runs stay reproducible and unaffected by fallback config.

Create a local `.env` file if you need to override these values:

```env
GOOGLE_API_KEY=<your-key>
```

Do not commit `.env`. It is intentionally ignored by Git.

## Fine-Tuned Model Workflow

The `cv-guestimator` entries in `configs/llm.yaml` are locally fine-tuned Llama-3.2 builds served by Ollama. Three pieces of the repo exist for that loop:

| Piece | Role |
| --- | --- |
| `training_data/examples/<domain>/<example_id>/{cv.txt,job.txt}` | Hand-curated input pairs, committed to Git (synthetic/public text only — never a real CV). |
| `scripts/build_training_dataset.py` | Turns those examples into an SFT dataset at `training_data/dataset.jsonl` (gitignored). |
| `docker/ollama/Modelfile` | Serving config for the exported GGUF: prompt template, stop tokens, sampling. |

### Building the dataset

```powershell
uv run python scripts/build_training_dataset.py
uv run python scripts/build_training_dataset.py --teacher gemini-flash --examples training_data/examples --out training_data/dataset.jsonl
```

For each example the script runs all three matching-pipeline requests (job requirements, skill matching, overall experience) against a *teacher* model — `gemini-flash` by default — and writes one JSONL row per `(example, stage)`.

The important detail is *how* it captures the prompts. It hooks Instructor's `completion:kwargs` event and records the literal outbound `messages`, rather than reconstructing them by hand, because Instructor's `Mode.JSON` appends a large JSON-schema block to the end of whatever system prompt is passed. That block is the last thing the model sees before the CV/JD content. A LoRA trained on hand-written prompt pairs is therefore trained on a different input distribution than the one it is served at inference time — which is what previously made prompt edits in `src/prompts/templates.py` have no observable effect on the fine-tuned model's output.

The script re-derives prompt/schema construction from the three agent classes rather than calling their `.run()` methods, since two of them post-process the raw LLM output. **If `src/services/agents.py` changes how it builds prompts or response models, mirror the change in the script**, or the dataset silently drifts from production.

### Serving a build

Export the trained adapter to GGUF, drop it in as `docker/ollama/cv-guestimator.gguf` (gitignored — it is large and machine-specific), then:

```powershell
ollama create cv-guestimator:v4 -f docker/ollama/Modelfile
```

Always build to a **new** tag. Re-running `ollama create` against an existing tag redefines what that name means, which silently invalidates every artifact already logged against it — and `cv-guestimator.gguf` is not necessarily the export the tag was originally built from.

`docker/ollama/Modelfile` mirrors the training project's canonical Modelfile, which is the source of truth for how these weights must be served: the Llama-3.2 `TEMPLATE` and all four stop tokens (`<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`, `<|eom_id|>`) must be present or the model runs on past its turn. It sets `temperature 0`, bakes in no `SYSTEM` prompt (the app supplies one per request), and raises `num_ctx` to 8192 so long CV + job-listing pairs fit in a single request.

### Which build is promoted

`cv-guestimator:v2` is the current default. It beat the Aug 24 build (`:latest` / `:v1-fixed`) and the later `:v3` export on the `model_eval` thresholds, and `configs/llm.yaml` maps the `cv-guestimator` key onto its tag:

| Tag | `configs/llm.yaml` name | Status |
| --- | --- | --- |
| `cv-guestimator:v2` | `cv-guestimator` | **Promoted default.** Every everyday task and the web API resolve here. |
| `cv-guestimator:v1-fixed` | `cv-guestimator-v1-fixed` | Previous weights, kept for rollback — `tasks/model_eval_v1_fixed.yaml`. |
| `cv-guestimator:latest` | — | The same Aug 24 weights as `:v1-fixed`, but served from the old Modelfile. No config entry; superseded. |
| `cv-guestimator:v3` | — | Discarded: it fragments roles worst and matches requirements too permissively. The Ollama tag is still built locally, so re-adding an entry is enough to revisit it. |

Config keys are mapped to **explicit version tags, never `:latest`**, so a run artifact's `engine` string always names the exact weights behind a score, and re-tagging in Ollama can't retroactively change what a past run meant.

To promote a future build: `ollama create cv-guestimator:v4 -f docker/ollama/Modelfile`, add an entry for it, benchmark it with a task pinned to that entry, and — if it wins — point the `cv-guestimator` key at the new tag and repoint the rollback entry at v2. Rolling back is the same one-line edit in reverse.

### Comparing two runs

`scripts/compare_runs.py` diffs two run artifacts so a promotion decision rests on the numbers rather than on reading JSON side by side:

```powershell
uv run python scripts/compare_runs.py --task model_eval        # last two runs of one task
uv run python scripts/compare_runs.py --latest                 # last two scored runs
uv run python scripts/compare_runs.py <baseline.json> <candidate.json>
```

It reports final relevance, skills match, requirements found, career pillar, roles counted, and runtime as a `baseline / candidate / delta` table, then names the individual skills that newly matched or newly went missing — which is usually the actual reason a score moved. It warns when either run was served by the fallback model, since that run measures `gemini-flash` rather than the local build.

It reads artifacts as plain JSON rather than validating them through `RunArtifact`, deliberately: the job is comparing a new build against runs recorded weeks ago under an older artifact shape, so it looks fields up defensively and prints `-` for anything a given artifact predates.

## Project Setup

This project uses Python 3.12 or newer and is configured with `pyproject.toml`.

Install dependencies with uv:

```powershell
uv sync
```

For scanned, image-only, or malformed PDFs, install Tesseract OCR and make sure `tesseract.exe` is available on `PATH`. On Windows, install Tesseract from the official UB Mannheim Windows builds or another trusted package source, then open a new terminal and verify:

```powershell
tesseract --version
```

The parser first tries embedded PDF text extraction. OCR is only used when text extraction returns nothing.

If you are using the default local Ollama setup, make sure Ollama is running and the configured model is available before executing the pipeline.

## Input Files

The default task expects local documents in these folders:

```text
dataSet/
	tradeMeJobListing/
		Job_listing.txt   # preferred when present
		Job_listing.pdf   # used when no TXT file exists
	tradeMeCV/
		<candidate-cv>.txt # preferred when present
		<candidate-cv>.pdf # used when no TXT file exists
```

Document paths are declared per task in `tasks/*.yaml`, so pointing a run at different documents means editing (or copying) a task file rather than code.

TXT files bypass PDF parsing and OCR entirely. This is useful for job listings from sites that export malformed, scanned, or otherwise non-selectable PDFs: copy the listing text into `Job_listing.txt` and run the CLI normally.

The `dataSet/` folder is ignored by Git because it can contain CVs, job descriptions, and other private source documents.

## Running the Pipeline

Run the default task from the repository root:

```powershell
uv run main.py
```

Or run a specific task:

```powershell
uv run main.py tasks/model_eval.yaml
```

A successful run prints a report like:

```text
SCORING ENGINE OUTPUT
Overall Match: <score>%
Skills Match:  <score>% (<matched>/<total> skills)
Career Match:  <score-or-N/A> (<candidate years> years vs <target years> years required)

HARNESS EVALUATION
  [PASS] min_final_relevance: expected >= 0, got 45.0
  Overall: PASS
```

A benchmark task declares more, so `uv run main.py tasks/model_eval.yaml` prints four:

```text
  [PASS] min_final_relevance: expected >= 30.0, got 51.2
  [PASS] min_skills_match: expected >= 30.0, got 76.92
  [PASS] max_execution_seconds: expected <= 60.0, got 7.6
  [PASS] max_attempts: expected <= 1, got 1
  Overall: PASS
```

Each run also writes a numbered, timestamped JSON file to `artifacts/`, such as `run-000001_llama3.2_latest_20260810T002103.083290Z_1a7c4409.json`. That folder is ignored by Git because artifacts may include extracted candidate/job data and model outputs.

## Running the Web UI

Start the local web server from the repository root:

```powershell
uv run uvicorn src.api.app:app --reload
```

(`uvicorn src.web_app:app --reload` still works via a compatibility shim.)

Open the printed local URL, usually:

```text
http://127.0.0.1:8000
```

The page accepts one job listing and one candidate CV. Each upload can be either:

- `.pdf` for normal PDF upload.
- `.txt` for pasted or pre-extracted text, which is useful when a job site produces a visually readable PDF that automated PDF libraries cannot extract.

Three endpoints mirror the harness's three pipeline shapes:

| Endpoint | Mirrors | Behavior |
| --- | --- | --- |
| `POST /api/compare` | `pipeline: extraction` | Job listing + raw CV in, full match result out. Also persists the redacted CV and its `IngestionArtifact` the same way `/api/ingest` does. |
| `POST /api/ingest` | `pipeline: ingestion` | Raw CV in, `cv_id` out. The response never carries PII spans or redacted text — only a count — since that's exactly what this endpoint exists to keep off the wire. |
| `POST /api/match` | `pipeline: matching` | Job listing + a previously-returned `cv_id` in, full match result out. No PII detector runs at all — the CV arrives already redacted. |

All three write the same artifact JSON files under `artifacts/` as the CLI.

### Evaluating API runs

An API request has no task file, so `/api/*` artifacts used to log `"evaluation": null`. They are now judged against the same `default_evaluation` baseline a task inherits (see [Evaluation criteria](#evaluation-criteria)), and the report comes back in both the JSON response and the artifact:

```json
"evaluation": {
  "passed": true,
  "checks": [
    { "name": "max_execution_seconds", "expected": "<= 60.0", "actual": "5.67", "passed": true },
    { "name": "max_attempts", "expected": "<= 1", "actual": "1", "passed": true }
  ]
}
```

There is no task to layer on top, so an endpoint applies the baseline alone — which is exactly what makes an `/api/compare` and an equivalent `uv run main.py` comparable. One block covers all three endpoints: the score thresholds skip `/api/ingest` (no scorecard), `min_pii_spans` skips `/api/match` (its CV arrived pre-redacted), and `/api/compare` is judged on all of them. Remove or empty `default_evaluation` to go back to `"evaluation": null`.

## Docker

The repository ships with a three-service Docker setup:

| Service | Image | Purpose |
| --- | --- | --- |
| `api` | `docker/Dockerfile.api` | FastAPI backend served by uvicorn on port 8000. |
| `web` | `docker/Dockerfile.web` | Vite-built UI served by nginx on port 5173, proxying `/api` to the backend. |
| `ollama` | `ollama/ollama` | Local model server with a named volume for model storage. |

Run everything with:

```powershell
docker compose up --build
```

`artifacts/` and `dataSet/` are mounted into the api container as volumes so private inputs and run traces stay on the host. `docker/ollama/Modelfile` builds a fine-tuned model from a local GGUF export — see [Fine-Tuned Model Workflow](#fine-tuned-model-workflow).

## Scoring Logic

The score is calculated in `RelevanceScoringEngine`:

1. Skills match score: matched required skills divided by total extracted job requirements.
2. Role de-duplication: roles repeating an exact `(role_title, start_date, end_date)` are dropped before any duration is summed.
3. Career match score: de-duplicated relevant career years divided by required overall years, capped at 100%.
4. Final relevance: weighted sum of the applicable pillar scores.

Date ranges are expected in `YYYY-MM` format. Values such as `Present`, `Current`, and `Now` are treated as the current date.

Step 2 exists because a weaker evaluation model can emit the same position more than once (a short and a long title variant, or the employer name as a second `role_title`), and summing every emitted role turns one three-month job into a year of experience. Because every `evaluation:` threshold is a *minimum*, that inflation reads as a passing run rather than a bug — so the guard sits in the scoring engine as well as in the prompt. The dedup key is exact-match, so differently-worded duplicates still slip through; merging overlapping date ranges is the more durable fix and is not implemented yet.

The complementary guard is in `src/prompts/templates.py`: the overall-experience prompt identifies a position by employer and date range rather than by wording, collapses title variants, and converts CV date wording to `YYYY-MM`. Each prompt there carries a version constant (`JOB_REQUIREMENTS_PROMPT_VERSION`, `SKILL_MATCHER_PROMPT_VERSION`, `OVERALL_EXPERIENCE_PROMPT_VERSION`); bump the version whenever you change the wording, since every run artifact records `config.prompt_versions` and that is what makes two runs comparable after the fact.

## Tests

The test suite covers artifact logging, pipeline privacy/redaction behavior, the ingestion/matching split (including that `matching_pipeline.py` never imports `CandidateCV` or a PII detector), requirement scoring, document parsing, the `presidio` PII detector, the model fallback client (in isolation and wired through the real agent chain), the web API, and the harness (task loading, registries, and threshold evaluation). Shared test doubles and builders (a fake LLM client, a schema-valid `PipelineResult` factory) live in `tests/factories.py`, a plain importable module rather than a pytest-fixture-only `conftest.py`, since the suite mixes `unittest.TestCase` classes with plain pytest functions.

```powershell
uv run pytest
```

### Linting

```powershell
uv run ruff check .
```

Configured narrowly for now (`F` for real mistakes, `I` for import order — see `[tool.ruff.lint]` in `pyproject.toml`); `mypy` is also available (`uv run mypy src`) but not yet wired into CI. `.github/workflows/ci.yml` runs both `ruff check` and `pytest` on every pull request.

## Privacy and Git Hygiene

The repository is configured to ignore:

- `.env` and `.env.*` for local configuration and possible secrets.
- `.venv/` for local Python environments.
- `dataSet/` for private PDFs and source documents.
- `artifacts/` for generated run traces.
- `redacted_cvs/` for persisted `RedactedCV`s from `ingestion` runs — redacted text only, never raw PII, but still local-only run output.
- `docker/ollama/*.gguf` for local fine-tuned model exports (large and machine-specific).
- `training_data/dataset.jsonl` for the generated SFT dataset — regenerate it with `scripts/build_training_dataset.py` rather than committing it. The curated `training_data/examples/` inputs *are* committed, so keep real CVs out of them.
- Python caches and test/coverage outputs.


## Repository Layout

### Top-level map

| Path | In Git? | What lives here |
| --- | --- | --- |
| `main.py` | yes | The CLI entry point. Loads a task, hands it to `HarnessRunner`, prints the report. Contains no matching logic. |
| `src/` | yes | All application code, split into layers (see [Inside `src/`](#inside-src)). |
| `configs/` | yes | Declarative *defaults*: models, scoring weights, PII policy, pipeline runtime, deployment. Changing behaviour usually starts here, not in code. |
| `tasks/` | yes | One YAML file per run recipe: which pipeline shape, which model, which inputs, which pass/fail thresholds. |
| `tests/` | yes | Pytest suite plus `factories.py`, the shared test doubles. |
| `scripts/` | yes | Developer utilities that are not part of the runtime (schema generation, training-data building). |
| `training_data/` | partly | `examples/` (curated, committed input pairs) and the generated `dataset.jsonl` (gitignored). |
| `web/` | yes | The browser UI — a static `index.html` plus a Vite project shell. |
| `docker/` | yes | Dockerfiles, nginx config, and the Ollama `Modelfile` for the fine-tuned build. |
| `.github/workflows/` | yes | CI: `ruff check` + `pytest` on every pull request. |
| `dataSet/` | **no** | Your private local CVs and job listings. Ignored because it holds real personal data. |
| `artifacts/` | **no** | Generated run traces, one numbered JSON per run. Ignored — they embed extracted CV/job content. |
| `redacted_cvs/` | **no** | Persisted `RedactedCV`s from `ingestion` runs, keyed by `cv_id`. Redacted text only, but still local run output. |
| `.venv/`, `.env` | **no** | Local environment and secrets. |

### Annotated tree

```text
.
|-- main.py                       # CLI entry point: task in, PASS/FAIL report + artifact out
|-- pyproject.toml                # deps, ruff/mypy config, Python >= 3.12
|-- uv.lock                       # locked dependency set (uv sync)
|-- docker-compose.yml            # api + web + ollama services
|-- README.md
|
|-- configs/                      # defaults, loaded by src/config/loader.py
|   |-- llm.yaml                  #   named model configs (provider, model, base_url, api_key_env)
|   |-- pipeline.yaml             #   default pipeline + default/fallback model per role
|   |-- scoring.yaml              #   default pillar weights (skills 60 / career 40)
|   |-- pii_policy.yaml           #   which detectors compose CompositePIIDetector
|   `-- deployment.yaml           #   ports and image names used by Docker
|
|-- tasks/                        # run recipes; `uv run main.py tasks/<file>.yaml`
|   |-- cv_job_match.yaml         #   extraction: the everyday full comparison
|   |-- cv_ingest.yaml            #   ingestion: redact a CV once, print its cv_id
|   |-- cv_match_from_redacted.yaml  # matching: score a job against a stored cv_id
|   |-- model_eval.yaml           #   extraction benchmark with real thresholds
|   `-- model_eval_v1_fixed.yaml  #   same benchmark, previous weights (rollback check)
|
|-- src/
|   |-- __init__.py
|   |-- web_app.py                # back-compat shim re-exporting src.api.app
|   |
|   |-- config/                   # LAYER: configuration
|   |   `-- loader.py             #   reads configs/*.yaml and .env
|   |
|   |-- model/                    # LAYER: model resolution (name -> client)
|   |   |-- providers.py          #   Ollama / OpenAI-compatible provider classes
|   |   |-- model_registry.py     #   provider name in llm.yaml -> provider class
|   |   `-- adapters.py           #   named config -> InstructorClient, resolves API keys
|   |
|   |-- harness/                  # LAYER: orchestration (no matching logic lives here)
|   |   |-- task_loader.py        #   task YAML -> validated TaskSpec
|   |   |-- registry.py           #   name -> factory for pipelines and PII detectors
|   |   |-- interfaces.py         #   PipelineProtocol that the registry enforces
|   |   |-- runner.py             #   loads configs, dispatches to a pipeline, logs artifact
|   |   `-- evaluator.py          #   result vs. the task's thresholds -> EvaluationReport
|   |
|   |-- services/                 # LAYER: the actual business logic
|   |   |-- document_parser.py    #   PDF/TXT -> text (pypdf -> PyMuPDF -> OCR fallbacks)
|   |   |-- pii_detector.py       #   detector base + CompositePIIDetector + guards
|   |   |-- presidio_detector.py  #   local spaCy/Presidio NER + NZ pattern recognizers
|   |   |-- ingestion_pipeline.py #   raw CV -> RedactedCV                 (PII stage)
|   |   |-- matching_pipeline.py  #   RedactedCV + job -> result           (no PII imports)
|   |   |-- extraction_pipeline.py#   one-shot wrapper: ingestion then matching
|   |   |-- cv_store.py           #   content-addressed RedactedCV persistence
|   |   |-- ingestion_persistence.py # shared "save RedactedCV + log artifact" helper
|   |   |-- agents.py             #   the three LLM agents (requirements/skills/experience)
|   |   |-- llm_client.py         #   Instructor wrapper + FallbackInstructorClient
|   |   |-- pipeline_tracing.py   #   step-timing context manager -> TraceSpan list
|   |   `-- scoring_engine.py     #   structured outputs -> weighted scorecard (+ role dedup)
|   |
|   |-- schemas/                  # LAYER: Pydantic data contracts between everything above
|   |   |-- pii.py                #   PII spans
|   |   |-- ingestion.py          #   RedactedCV / IngestionResult
|   |   |-- requirements.py       #   job requirements, skill evaluations, match result
|   |   |-- experience.py         #   overall/relevant career experience
|   |   |-- scoring.py            #   scorecard
|   |   |-- pipeline.py           #   pipeline result, metrics, trace spans
|   |   |-- evaluation.py         #   CheckResult / EvaluationReport
|   |   `-- artifact.py           #   on-disk run + ingestion artifact formats
|   |
|   |-- prompts/
|   |   `-- templates.py          #   system prompts + their version constants
|   |
|   |-- utils/
|   |   `-- artifact_logger.py    #   serializes a run to artifacts/run-NNNNNN_*.json
|   |
|   `-- api/                      # LAYER: HTTP, wrapping the same pipelines
|       |-- app.py                #   FastAPI app, serves web/index.html
|       |-- routes.py             #   /api/compare, /api/ingest, /api/match
|       `-- schemas.py            #   typed API response models
|
|-- web/
|   |-- index.html                # the actual UI: drag-and-drop upload + results
|   |-- package.json
|   |-- vite.config.ts
|   |-- public/
|   `-- src/
|
|-- scripts/
|   |-- gen_task_schema.py        # regenerates .vscode/task.schema.json from TaskSpec
|   |-- build_training_dataset.py # examples + teacher model -> training_data/dataset.jsonl
|   `-- compare_runs.py           # diffs two run artifacts (scores, skills, runtime)
|
|-- training_data/
|   |-- examples/                 # committed <domain>/<example_id>/{cv.txt,job.txt} pairs
|   `-- dataset.jsonl             # generated SFT dataset (gitignored)
|
|-- docker/
|   |-- Dockerfile.api            # uvicorn backend on :8000
|   |-- Dockerfile.web            # Vite build served by nginx on :5173
|   |-- nginx.conf                # proxies /api to the backend
|   `-- ollama/
|       |-- Modelfile             # serving config for the fine-tuned build
|       `-- cv-guestimator.gguf   # your exported weights (gitignored)
|
|-- tests/
|   |-- factories.py                    # fake LLM client, PipelineResult builder
|   |-- test_harness.py                 # task loading, registries, threshold evaluation
|   |-- test_ingestion_split.py         # the ingestion/matching boundary, incl. import graph
|   |-- test_pipeline_privacy.py        # redaction behaviour end to end
|   |-- test_presidio_detector.py       # detector recognizers
|   |-- test_requirement_scoring.py     # scoring engine maths
|   |-- test_document_parser.py         # PDF/TXT extraction and fallbacks
|   |-- test_artifact_logger.py         # artifact serialization
|   |-- test_fallback_client.py         # FallbackInstructorClient in isolation
|   |-- test_fallback_integration.py    # the same client through the real agent chain
|   `-- test_web_app.py                 # API routes
|
|-- .github/workflows/ci.yml      # ruff + pytest on every PR
|-- dataSet/                      # local only, ignored by Git
|-- redacted_cvs/                 # generated, ignored by Git
`-- artifacts/                    # generated, ignored by Git
```

### Inside `src/`

The package is layered, and the layers only depend downwards:

```text
main.py                 src/api/           <- two entry points, same pipelines
     \                    /
      v                  v
   src/harness/     (orchestration: load, dispatch, evaluate, log)
          |
          v
   src/services/    (business logic: parse, redact, prompt, score)
          |
          v
   src/schemas/  +  src/prompts/   (data contracts and prompt text - no logic)

   src/config/  +  src/model/      (leaves: read YAML, build clients)
```

- **`src/harness/`** is deliberately generic. It knows how to load a task, resolve components from a registry, run *a* pipeline, evaluate a result, and write an artifact. It contains no knowledge of CVs, skills, or scoring — swapping in a different pipeline means registering a new class, not editing the runner.
- **`src/services/`** holds everything domain-specific. The three pipeline modules are the important ones, and the split between them is a privacy boundary, not a stylistic one: `matching_pipeline.py` does not import `CandidateCV` or any PII detector, so a matching run *cannot* touch raw CV text even by accident. `test_ingestion_split.py` asserts that import graph, so the boundary fails the build if someone crosses it.
- **`src/schemas/`** is the contract layer. Every agent output, pipeline result, and on-disk artifact is a Pydantic model, which is what makes LLM output safe to score and old artifacts safe to re-read.
- **`src/model/`** and **`src/config/`** are leaves with no domain knowledge: given a name from `configs/llm.yaml`, hand back a working client.
- **`src/api/`** is a thin HTTP wrapper. Its three endpoints map one-to-one onto the three pipeline shapes, so the web UI and the CLI exercise the same code paths and write the same artifacts.

### Where to change what

| You want to... | Edit |
| --- | --- |
| Point a run at different documents | the `inputs:` block of a task in `tasks/` |
| Try a different model | `models.evaluation` in a task, or `configs/pipeline.yaml` for the web API |
| Add a model | a new entry in `configs/llm.yaml` |
| Reweight the score | `configs/scoring.yaml`, or `scoring_weights:` in one task |
| Change what the LLM is asked | `src/prompts/templates.py` — and bump that prompt's version constant |
| Change how the score is computed | `src/services/scoring_engine.py` |
| Change what counts as PII | `src/services/presidio_detector.py` and `configs/pii_policy.yaml` |
| Add a field to a task file | `TaskSpec` in `src/harness/task_loader.py`, then rerun `uv run python scripts/gen_task_schema.py` |
| Add a pipeline shape | a class in `src/services/`, registered in `src/harness/registry.py` |
| Change what a run records | `src/schemas/artifact.py` and `src/utils/artifact_logger.py` |
