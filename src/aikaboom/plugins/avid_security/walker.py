from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

ComponentKind = Literal["Model", "Dataset"]
ComponentScope = Literal["principal", "base", "dataset"]


@dataclass(frozen=True)
class Component:
    kind: ComponentKind
    hf_path: str
    developer: str | None
    base_models: tuple[str, ...]
    scope_in_bom: ComponentScope
    spdx_id: str

    @property
    def bare_name(self) -> str:
        """Lowercase last path segment of hf_path."""
        return self.hf_path.split("/")[-1].lower()
