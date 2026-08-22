"""Regenerates .vscode/task.schema.json from TaskSpec, the actual source of
truth for tasks/*.yaml (see src/harness/task_loader.py).

Run after changing TaskSpec/ModelSelection/TaskInputs/EvaluationCriteria so
the editor's schema never drifts from what load_task() actually accepts:

    python scripts/gen_task_schema.py
"""

import json
from pathlib import Path

from src.harness.task_loader import TaskSpec

OUT_PATH = Path(__file__).resolve().parents[1] / ".vscode" / "task.schema.json"


def main() -> None:
    schema = TaskSpec.model_json_schema()
    schema["title"] = "CV to Job Guestimator Task"
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", **schema}
    OUT_PATH.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
