# CV to Job Guestimator

CV to Job Guestimator is a local Python pipeline that compares a candidate CV against a job listing and produces a structured relevance score. It extracts text from two PDFs, sends the extracted content through a three-step SLM pipeline Locally, calculates a weighted scorecard, and writes a JSON trace of each run.

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
DocumentParser
	Extracts text from PDFs with PyMuPDF
				|
				v
MultiAgentPipeline
	Agent 1: Extract job requirements
	Agent 2: Match CV skills against requirements
	Agent 3: Extract relevant career history
				|
				v
Pydantic output schemas
	Validate structured LLM responses
				|
				v
RelevanceScoringEngine
	Calculates weighted scorecard
				|
				v
ArtifactLogger
	Writes artifacts/run_YYYYMMDD_HHMMSS.json
```

### Entry Point

`main.py` orchestrates the full run:

1. Builds the expected local PDF paths.
2. Extracts text with `DocumentParser`.
3. Runs the multi-agent extraction pipeline.
4. Computes the final relevance report.
5. Saves a JSON artifact under `artifacts/`.
6. Prints a readable score summary to the terminal.

### Core Modules

| Module | Responsibility |
| --- | --- |
| `src/config.py` | Loads model configuration and scoring weights. |
| `src/services/document_parser.py` | Extracts text from PDF files using PyMuPDF. |
| `src/services/agent_pipeline.py` | Calls the OpenAI-compatible chat client and parses structured agent outputs. |
| `src/prompts/templates.py` | Stores the system prompts for each agent step. |
| `src/schemas/agent_outputs.py` | Defines Pydantic models for SLM outputs. |
| `src/services/scoring_engine.py` | Converts structured outputs into the weighted relevance scorecard. |
| `src/schemas/artifacts.py` | Defines the saved run artifact format. |
| `src/utils/artifact_logger.py` | Serializes each run to a timestamped JSON file. |

## Model Configuration

The project uses the OpenAI Python SDK against an OpenAI-compatible endpoint. By default it is configured for a local Ollama endpoint:

| Setting | Environment variable | Default |
| --- | --- | --- |
| Model name | `MODEL_NAME` | `llama3.2:latest` |
| Base URL | `MODEL_BASE_URL` | `http://localhost:11434/v1` |
| API key | `MODEL_API_KEY` | `ollama` |

Create a local `.env` file if you need to override these values:

```env
MODEL_NAME=llama3.2:latest
MODEL_BASE_URL=http://localhost:11434/v1
MODEL_API_KEY=ollama
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
MULTI-AGENT COMPUTED RELEVANCE REPORT
Overall Match Score: <score>%
Pillar A (Skills Match): <matched>/<total> skills
Pillar B (Skill Tenure): Avg tenure fit across <n> tools
Pillar C (Overall Tenure): <candidate years> yrs vs <target years> yrs required
```

Each run also writes a timestamped JSON file to `artifacts/`. That folder is ignored by Git because artifacts may include extracted candidate/job data and model outputs.

## Scoring Logic

The score is calculated in `RelevanceScoringEngine`:

1. Skills match score: matched required skills divided by total extracted job requirements.
2. Skill tenure score: average of candidate years divided by target years for each matched skill, capped at 100% per skill.
3. Overall tenure score: relevant career years divided by required overall years, capped at 100%.
4. Final relevance: weighted sum of the three pillar scores.

Date ranges are parsed with `python-dateutil`. Values such as `Present`, `Current`, and `Now` are treated as the current date.

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
|   |-- config.py
|   |-- prompts/
|   |   `-- templates.py
|   |-- schemas/
|   |   |-- agent_outputs.py
|   |   `-- artifacts.py
|   |-- services/
|   |   |-- agent_pipeline.py
|   |   |-- document_parser.py
|   |   `-- scoring_engine.py
|   `-- utils/
|       |-- artifact_logger.py
|       `-- date_utils.py
|-- dataSet/       # Local only, ignored by Git
`-- artifacts/     # Generated, ignored by Git
```
