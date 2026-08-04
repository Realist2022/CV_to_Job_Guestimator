# src/schemas/artifacts.py
from datetime import datetime, timezone
from typing import Any, Dict, List
from pydantic import BaseModel, Field

class AgentStepTrace(BaseModel):
    agent_id: str = Field(description="Identifier for the agent (e.g., 'Agent 1')")
    action: str = Field(description="Description of the task performed")
    output_payload: Dict[str, Any] = Field(description="Parsed output model converted to dict")

class PipelineRunArtifact(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    engine: str = Field(description="LLM engine model name used")
    execution_time_seconds: float = Field(description="Total pipeline runtime in seconds")
    agent_traces: List[AgentStepTrace] = Field(description="Step-by-step output from each agent")
    computed_report: Dict[str, Any] = Field(description="Final output from RelevanceScoringEngine")