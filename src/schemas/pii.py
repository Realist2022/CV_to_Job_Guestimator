from dataclasses import dataclass
from enum import Enum


class PIIKind(str, Enum):
    person_name = "person_name"
    street_address = "street_address"
    referee = "referee"
    date_of_birth = "date_of_birth"
    nationality = "nationality"
    marital_or_family = "marital_or_family"
    other_identifier = "other_identifier"


@dataclass(frozen=True)
class TextSpan:
    kind: str
    text: str