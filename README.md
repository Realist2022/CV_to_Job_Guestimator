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

```text
Task definition (tasks/*.yaml)
	|-- pipeline name, model selection, input paths, pass/fail criteria
				|
				v
HarnessRunner (src/harness/runner.py)
	Loads configs/, resolves models and components from registries
				|
				v
SourceDocument.from_pdf / from_path
	Extracts PDF text with pypdf, falls back to PyMuPDF, then OCR
				|
				v
ExtractionPipeline
	Step 1: Detect and redact candidate CV PII
	Step 2: Extract authoritative job requirements
	Step 3: Match requirements against the redacted CV
	Step 4: Extract relevant career history
				|
				v
Pydantic schemas
	Validate agent outputs, pipeline results, scorecards, and artifacts
				|
				v
RelevanceScoringEngine
	Calculates weighted scorecard
				|
				v
ThresholdEvaluator
	Checks the result against the task's evaluation criteria
				|
				v
ArtifactLogger
	Writes artifacts/run-000001_<engine>_<timestamp>_<run-id>.json
```

The harness knows how to load a task, resolve components, run the pipeline, evaluate the result, and log an artifact. It contains no CV/job matching business logic; that stays in `src/services/` and `src/schemas/`.

The web UI wraps the same pipeline with a drag-and-drop upload page and a `/api/compare` endpoint.

### Entry Point

`main.py` runs a harness task:

1. Loads the task file (default `tasks/cv_job_match.yaml`, or pass a path: `uv run main.py tasks/model_eval.yaml`).
2. `HarnessRunner` reads `configs/`, builds Instructor-backed clients from the named model configs, and composes the PII detectors.
3. Loads the job listing and CV from the first existing path listed in the task.
4. Runs the four-step extraction pipeline and computes the relevance report.
5. Evaluates the result against the task's criteria and prints PASS/FAIL checks.
6. Saves a JSON artifact under `artifacts/` and prints a readable score summary.

### Core Modules

| Module | Responsibility |
| --- | --- |
| `src/harness/runner.py` | Loads configs, resolves components, runs a task end to end, and logs the artifact. |
| `src/harness/task_loader.py` | Parses and validates task YAML into `TaskSpec` models. |
| `src/harness/registry.py` | Name-to-factory registries for pipelines and PII detectors. |
| `src/harness/evaluator.py` | Checks pipeline results against task pass/fail thresholds. |
| `src/harness/interfaces.py` | Protocols the harness requires from resolved components. |
| `src/model/providers.py` | Model providers (Ollama, OpenAI-compatible) that build LLM clients. |
| `src/model/model_registry.py` | Maps provider names in `configs/llm.yaml` to provider classes. |
| `src/model/adapters.py` | Turns a named model config into an `InstructorClient`, resolving API keys from the environment. |
| `src/config_loader.py` | Loads YAML-backed project configuration and `.env` values. |
| `src/services/document_parser.py` | Defines source document types and extracts text from PDF files using `pypdf`, PyMuPDF fallback, and OCR fallback. |
| `src/services/llm_client.py` | Wraps the OpenAI-compatible client with Instructor for structured Pydantic outputs. |
| `src/services/agents.py` | Implements the PII, requirement extraction, skill matching, and overall experience agents. |
| `src/services/pii_detector.py` | Combines regex and model-based PII detection before CV text reaches evaluation agents. |
| `src/services/pipeline.py` | Orchestrates the end-to-end extraction, redaction, scoring, and pipeline result assembly. |
| `src/prompts/templates.py` | Stores the system prompts for each agent step. |
| `src/schemas/pii.py` | Defines PII span schemas. |
| `src/schemas/requirements.py` | Defines job requirement, skill evaluation, and skill match result schemas. |
| `src/schemas/experience.py` | Defines overall experience schemas. |
| `src/schemas/scoring.py` | Defines scorecard schemas. |
| `src/schemas/pipeline.py` | Defines pipeline result and metrics schemas. |
| `src/schemas/artifact.py` | Defines the saved run artifact format. |
| `src/services/scoring_engine.py` | Converts structured outputs into the weighted relevance scorecard. |
| `src/utils/artifact_logger.py` | Serializes each run to a timestamped JSON file. |
| `src/api/app.py` | Builds the FastAPI app and serves the web UI. |
| `src/api/routes.py` | Implements the `/api/compare` upload endpoint. |
| `src/api/schemas.py` | Typed response models for the API. |
| `src/web_app.py` | Backwards-compatible shim re-exporting the app from `src/api/app.py`. |
| `web/index.html` | Browser UI for uploading a job listing and CV, running comparison, and viewing results. |

## Harness Tasks and Configs

Runs are described declaratively:

- `configs/llm.yaml` defines named model configurations (provider, model, base URL, API key or `api_key_env`, temperature). It ships with `gemini-flash`, `local-llama`, and a placeholder for a fine-tuned `cv-guestimator-lora` build.
- `configs/scoring.yaml` holds the default scoring weights.
- `configs/pii_policy.yaml` lists which PII detectors compose the composite detector, in order.
- `configs/pipeline.yaml` holds pipeline runtime defaults such as verbosity.
- `configs/deployment.yaml` documents ports and image names used by Docker.

Tasks under `tasks/` pick the pipeline, models, inputs, and evaluation thresholds:

| Task | Purpose |
| --- | --- |
| `tasks/cv_job_match.yaml` | Full comparison with the cloud evaluation model. |
| `tasks/pii_redaction.yaml` | Fully local run that asserts PII spans were actually redacted. |
| `tasks/model_eval.yaml` | Benchmarks a candidate model against pass/fail thresholds. |

A task looks like:

```yaml
name: cv_job_match
pipeline: extraction
models:
  evaluation: gemini-flash
  pii: local-llama
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

The first existing path in each input list wins, so TXT files take precedence over PDFs.

## Model Configuration

The project uses the OpenAI Python SDK plus Instructor against OpenAI-compatible endpoints.

Harness runs (`main.py`) select models by name from `configs/llm.yaml`. The web API uses the default model names in `configs/pipeline.yaml` and resolves those names through the same `configs/llm.yaml` entries. Cloud entries reference API keys through `api_key_env`, which is resolved from the environment or a local `.env` file (for example `GOOGLE_API_KEY` for the Gemini OpenAI-compatible endpoint).

Default scoring weights live in `configs/scoring.yaml`. The web API upload form can still override those weights per request.

Create a local `.env` file if you need to override these values:

```env
GOOGLE_API_KEY=<your-key>
```

Do not commit `.env`. It is intentionally ignored by Git.

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

The compare endpoint writes the same artifact JSON files under `artifacts/` as the CLI.

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

`artifacts/` and `dataSet/` are mounted into the api container as volumes so private inputs and run traces stay on the host. `docker/ollama/Modelfile` is a placeholder for a fine-tuned LoRA build: once an adapter exists, point `ADAPTER` at it, run `ollama create cv-guestimator-lora -f docker/ollama/Modelfile`, and reference `cv-guestimator-lora` from `configs/llm.yaml` in a task.

## Scoring Logic

The score is calculated in `RelevanceScoringEngine`:

1. Skills match score: matched required skills divided by total extracted job requirements.
2. Career match score: relevant career years divided by required overall years, capped at 100%.
3. Final relevance: weighted sum of the applicable pillar scores.

Date ranges are expected in `YYYY-MM` format. Values such as `Present`, `Current`, and `Now` are treated as the current date.

## Tests

The test suite covers artifact logging, pipeline privacy/redaction behavior, requirement scoring, document parsing, the web API, and the harness (task loading, registries, and threshold evaluation). Run it from an environment that has `pytest` installed:

```powershell
uv run pytest
```

## Privacy and Git Hygiene

The repository is configured to ignore:

- `.env` and `.env.*` for local configuration and possible secrets.
- `.venv/` for local Python environments.
- `dataSet/` for private PDFs and source documents.
- `artifacts/` for generated run traces.
- Python caches and test/coverage outputs.


## Repository Layout

```text
.
|-- main.py
|-- pyproject.toml
|-- README.md
|-- docker-compose.yml
|-- configs/
|   |-- deployment.yaml
|   |-- llm.yaml
|   |-- pii_policy.yaml
|   |-- pipeline.yaml
|   `-- scoring.yaml
|-- tasks/
|   |-- cv_job_match.yaml
|   |-- model_eval.yaml
|   `-- pii_redaction.yaml
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- web_app.py        # compatibility shim for src.api.app
|   |-- api/
|   |   |-- __init__.py
|   |   |-- app.py
|   |   |-- routes.py
|   |   `-- schemas.py
|   |-- harness/
|   |   |-- __init__.py
|   |   |-- evaluator.py
|   |   |-- interfaces.py
|   |   |-- registry.py
|   |   |-- runner.py
|   |   `-- task_loader.py
|   |-- model/
|   |   |-- __init__.py
|   |   |-- adapters.py
|   |   |-- model_registry.py
|   |   `-- providers.py
|   |-- prompts/
|   |   |-- __init__.py
|   |   `-- templates.py
|   |-- schemas/
|   |   |-- __init__.py
|   |   |-- artifact.py
|   |   |-- experience.py
|   |   |-- pii.py
|   |   |-- pipeline.py
|   |   |-- requirements.py
|   |   `-- scoring.py
|   |-- services/
|   |   |-- __init__.py
|   |   |-- agents.py
|   |   |-- document_parser.py
|   |   |-- llm_client.py
|   |   |-- pii_detector.py
|   |   |-- pipeline.py
|   |   `-- scoring_engine.py
|   `-- utils/
|       |-- __init__.py
|       `-- artifact_logger.py
|-- web/
|   |-- index.html
|   |-- package.json
|   |-- vite.config.ts
|   |-- public/
|   `-- src/
|-- docker/
|   |-- Dockerfile.api
|   |-- Dockerfile.web
|   |-- nginx.conf
|   `-- ollama/
|       `-- Modelfile
|-- tests/
|   |-- test_artifact_logger.py
|   |-- test_document_parser.py
|   |-- test_harness.py
|   |-- test_pipeline_privacy.py
|   |-- test_requirement_scoring.py
|   `-- test_web_app.py
|-- dataSet/       # Local only, ignored by Git
`-- artifacts/     # Generated, ignored by Git
```
