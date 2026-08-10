from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field


class PIIKind(str, Enum):
    person_name = "person_name"
    street_address = "street_address"
    referee = "referee"
    date_of_birth = "date_of_birth"
    nationality = "nationality"
    marital_or_family = "marital_or_family"
    other_identifier = "other_identifier"


class PIISpanModel(BaseModel):
    text: str = Field(
        description="The identifying text copied from the CV exactly as written."
    )
    kind: PIIKind = Field(description="What kind of identifying information this is.")


class PIIOutput(BaseModel):
    spans: list[PIISpanModel] = Field(
        description="Every piece of personally identifying text in the CV."
    )


@dataclass(frozen=True)
class TextSpan:
    kind: str
    text: str