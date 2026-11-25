from __future__ import annotations

from collections.abc import Callable


class Registry:
    def __init__(self) -> None:
        self._builders: dict[str, Callable] = {}

    def register(self, name: str, builder: Callable) -> None:
        if name in self._builders:
            raise ValueError(f"Builder '{name}' already registered")
        self._builders[name] = builder

    def build(self, name: str, *args, **kwargs):
        if name not in self._builders:
            raise KeyError(f"Builder '{name}' not found")
        return self._builders[name](*args, **kwargs)

    def names(self) -> list[str]:
        return sorted(self._builders)
