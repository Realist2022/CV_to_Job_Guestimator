# CV to Job Guestimator

CV to Job Guestimator is a local Python pipeline that compares a candidate CV against a job listing and produces a structured relevance score. It extracts text from two PDFs, redacts candidate PII, sends the extracted content through a structured multi-agent LLM pipeline, calculates a weighted scorecard, and writes a JSON trace of each run.

The project is designed for local experimentation with CV/job matching logic. Candidate CVs, job listing PDFs, environment files, and generated artifacts should stay out of Git because they may contain personal or sensitive information.

## What It Does

The pipeline answers three questions:

1. What technical requirements does the job listing ask for?
2. Which of those requirements appear in the candidate CV, and for what time period?
3. How much relevant overall career experience does the CV show for the target role?

Those outputs are combined into a final relevance percentage using three weighted pillars:

| Pillar | Weight | Source |
| --- | ---: | --- |
| Skills match | 45% | Required skills found in the CV |
| Skill tenure | 35% | Candidate years against required years for matched skills |
| Overall tenure | 20% | Relevant career years against target experience |

The default weights live in `src/config.py`.

## Architecture

```text
PDF inputs
	|-- dataSet/tradeMeJobListing/Job_listing.pdf
	|-- dataSet/tradeMeCV/<candidate-cv>.pdf
				|
				v
SourceDocument.from_pdf
	Extracts PDF text with pypdf
				|
				v
ExtractionPipeline
	Step 1: Detect and redact candidate CV PII
	Step 2: Extract authoritative job requirements
	Step 3: Match requirements against the redacted CV
	Step 4: Extract relevant career history
	Step 5: Measure tenure for explicit commercial-year requirements
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
ArtifactLogger
	Writes artifacts/run-000001_<engine>_<timestamp>_<run-id>.json
```

### Entry Point

`main.py` orchestrates the full run:

1. Builds the expected local PDF paths.
2. Extracts text with `JobListing.from_pdf()` and `CandidateCV.from_pdf()`.
3. Creates separate Instructor-backed clients for PII detection and evaluation.
4. Runs the five-step extraction pipeline.
5. Computes the final relevance report.
6. Saves a JSON artifact under `artifacts/`.
7. Prints a readable score summary and agent outputs to the terminal.

### Core Modules

| Module | Responsibility |
| --- | --- |
| `src/config.py` | Loads model configuration and scoring weights. |
| `src/services/document_parser.py` | Defines source document types and extracts text from PDF files using `pypdf`. |
| `src/services/llm_client.py` | Wraps the OpenAI-compatible client with Instructor for structured Pydantic outputs. |
| `src/services/agents.py` | Implements the PII, requirement extraction, skill matching, overall experience, and skill tenure agents. |
| `src/services/pii_detector.py` | Combines regex and model-based PII detection before CV text reaches evaluation agents. |
| `src/services/pipeline.py` | Orchestrates the end-to-end extraction, redaction, scoring, and pipeline result assembly. |
| `src/prompts/templates.py` | Stores the system prompts for each agent step. |
| `src/schemas/pii.py` | Defines PII span schemas. |
| `src/schemas/requirements.py` | Defines job requirement, skill evaluation, and skill match result schemas. |
| `src/schemas/experience.py` | Defines overall experience and skill tenure schemas. |
| `src/schemas/scoring.py` | Defines scorecard schemas. |
| `src/schemas/pipeline.py` | Defines pipeline result and metrics schemas. |
| `src/schemas/artifact.py` | Defines the saved run artifact format. |
| `src/services/scoring_engine.py` | Converts structured outputs into the weighted relevance scorecard. |
| `src/utils/artifact_logger.py` | Serializes each run to a timestamped JSON file. |

## Model Configuration

The project uses the OpenAI Python SDK plus Instructor against an OpenAI-compatible endpoint. The implemented configuration currently points both the PII and evaluation clients at a local Ollama-compatible endpoint by default:

| Setting | Environment variable | Default |
| --- | --- | --- |
| PII/evaluation model name | `PII_MODEL_NAME` | `llama3.2:latest` |
| PII/evaluation base URL | `PII_MODEL_BASE_URL` | `http://localhost:11434/v1` |
| PII/evaluation API key | `PII_MODEL_API_KEY` | `ollama` |

Create a local `.env` file if you need to override these values:

```env
PII_MODEL_NAME=llama3.2:latest
PII_MODEL_BASE_URL=http://localhost:11434/v1
PII_MODEL_API_KEY=ollama
```

Do not commit `.env`. It is intentionally ignored by Git.

## Project Setup

This project uses Python 3.12 or newer and is configured with `pyproject.toml`.

Install dependencies with uv:

```powershell
uv sync
```

If you are using the default local Ollama setup, make sure Ollama is running and the configured model is available before executing the pipeline.

## Input Files

The current entry point expects local PDFs in these folders:

```text
dataSet/
	tradeMeJobListing/
		Job_listing.pdf
	tradeMeCV/
		<candidate-cv>.pdf
```

The exact CV filename is currently hardcoded in `main.py`. If you want to run the pipeline against different documents frequently, a future improvement should move these paths into command-line arguments or environment variables.

The `dataSet/` folder is ignored by Git because it can contain CVs, job descriptions, and other private source documents.

## Running the Pipeline

Run the project from the repository root:

```powershell
uv run main.py
```

A successful run prints a report like:

```text
SCORING ENGINE OUTPUT
Overall Match: <score>%
Skills Match:  <score>% (<matched>/<total> skills)
Skill Tenure:  <score-or-N/A> (...)
Career Match:  <score>% (<candidate years> years vs <target years> years required)
```

Each run also writes a numbered, timestamped JSON file to `artifacts/`, such as `run-000001_llama3.2_latest_20260810T002103.083290Z_1a7c4409.json`. That folder is ignored by Git because artifacts may include extracted candidate/job data and model outputs.

## Scoring Logic

The score is calculated in `RelevanceScoringEngine`:

1. Skills match score: matched required skills divided by total extracted job requirements.
2. Skill tenure score: average of candidate years divided by target years for each matched skill, capped at 100% per skill.
3. Overall tenure score: relevant career years divided by required overall years, capped at 100%.
4. Final relevance: weighted sum of the three pillar scores.

Date ranges are expected in `YYYY-MM` format. Values such as `Present`, `Current`, and `Now` are treated as the current date.

## Tests

The test suite covers artifact logging, pipeline privacy/redaction behavior, requirement scoring, and skill tenure calculations. Run it from an environment that has `pytest` installed:

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
|-- src/
|   |-- __init__.py
|   |-- config.py
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
|-- tests/
|   |-- test_artifact_logger.py
|   |-- test_pipeline_privacy.py
|   |-- test_requirement_scoring.py
|   `-- test_skill_tenure.py
|-- dataSet/       # Local only, ignored by Git
`-- artifacts/     # Generated, ignored by Git
```
