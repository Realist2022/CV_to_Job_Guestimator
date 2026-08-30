from pydantic import Field

from src.schemas.base import StrictBaseModel


class ScorePillar(StrictBaseModel):
    score: float = Field(ge=0.0, le=100.0)
    raw: str
    applicable: bool = True


class Scorecard(StrictBaseModel):
    final_relevance: float = Field(ge=0.0, le=100.0)
    pillar_a: ScorePillar
    pillar_b: ScorePillar
    counted_roles: list[str]
