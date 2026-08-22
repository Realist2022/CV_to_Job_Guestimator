"""Name -> factory registries used to resolve components from task config."""

from typing import Any, Callable, Dict, Generic, Optional, TypeVar

from src.harness.interfaces import PipelineProtocol

T = TypeVar("T")


class Registry(Generic[T]):
    """Name -> factory registry, optionally enforcing a structural protocol.

    When `protocol` is given (must be `@runtime_checkable`), every component
    create() builds is checked against it with isinstance: a factory
    registered under a valid name but returning something that doesn't
    actually implement the contract fails loudly at the point it's built,
    not with an AttributeError three calls later inside a pipeline run.
    """

    def __init__(self, kind: str, protocol: Optional[type] = None):
        self.kind = kind
        self._protocol = protocol
        self._factories: Dict[str, Callable[..., T]] = {}

    def register(self, name: str, factory: Callable[..., T]) -> None:
        if name in self._factories:
            raise ValueError(f"{self.kind} '{name}' is already registered.")
        self._factories[name] = factory

    def create(self, name: str, **kwargs: Any) -> T:
        try:
            factory = self._factories[name]
        except KeyError:
            known = ", ".join(sorted(self._factories)) or "<none>"
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Registered: {known}"
            ) from None
        component = factory(**kwargs)
        if self._protocol is not None and not isinstance(component, self._protocol):
            raise TypeError(
                f"{self.kind} '{name}' ({type(component).__name__}) does not "
                f"satisfy {self._protocol.__name__}."
            )
        return component

    def names(self) -> list[str]:
        return sorted(self._factories)


pipelines: Registry[PipelineProtocol] = Registry("pipeline", protocol=PipelineProtocol)
pii_detectors: Registry[Any] = Registry("pii_detector")
