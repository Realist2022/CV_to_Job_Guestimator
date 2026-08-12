"""Name -> factory registries used to resolve components from task config."""

from typing import Any, Callable, Dict


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._factories: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, factory: Callable[..., Any]) -> None:
        if name in self._factories:
            raise ValueError(f"{self.kind} '{name}' is already registered.")
        self._factories[name] = factory

    def create(self, name: str, **kwargs: Any) -> Any:
        try:
            factory = self._factories[name]
        except KeyError:
            known = ", ".join(sorted(self._factories)) or "<none>"
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Registered: {known}"
            ) from None
        return factory(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._factories)


pipelines = Registry("pipeline")
pii_detectors = Registry("pii_detector")
