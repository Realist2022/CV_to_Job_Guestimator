from pydantic import BaseModel, ConfigDict, Field


class ScorePillar(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=100.0)
    raw: str
    applicable: bool = True


class Scorecard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final_relevance: float = Field(ge=0.0, le=100.0)
    pillar_a: ScorePillar
    pillar_b: ScorePillar
    counted_roles: list[str]