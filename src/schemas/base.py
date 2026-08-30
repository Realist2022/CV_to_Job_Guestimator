from pydantic import BaseModel, ConfigDict


class StrictBaseModel(BaseModel):
    """Shared base for schemas that reject unknown fields.

    A subclass that needs additional model_config options (e.g.
    arbitrary_types_allowed=True) can still set its own model_config —
    Pydantic v2 merges a subclass's model_config with its parent's rather
    than replacing it, so extra="forbid" stays in effect.
    """

    model_config = ConfigDict(extra="forbid")
