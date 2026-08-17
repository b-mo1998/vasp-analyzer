"""Calculation metadata with explicit provenance.

Every field records *where* it came from. A value the app could not determine
stays ``None`` and is rendered as an em dash — the app never substitutes a
plausible default for a measured quantity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Provenance:
    """Where a metadata value came from, for display and for trust ordering."""

    source: str            # e.g. "DOSCAR", "OUTCAR", "INCAR"
    estimated: bool = False
    note: str = ""


@dataclass(frozen=True)
class Sourced(Generic[T]):
    """A value plus its provenance."""

    value: T
    provenance: Provenance

    @property
    def source(self) -> str:
        return self.provenance.source

    @property
    def estimated(self) -> bool:
        return self.provenance.estimated


@dataclass
class CalculationMetadata:
    """Aggregated facts about a calculation, assembled from several files."""

    system: Sourced[str] | None = None
    functional: Sourced[str] | None = None
    e_fermi: Sourced[float] | None = None
    encut: Sourced[float] | None = None
    kmesh: Sourced[str] | None = None
    n_bands: Sourced[int] | None = None
    n_electrons: Sourced[float] | None = None
    n_kpoints: Sourced[int] | None = None
    vasp_version: Sourced[str] | None = None
    spin_mode: Sourced[str] | None = None
    formula: Sourced[str] | None = None
    incar_tags: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def display(sourced: "Sourced[Any] | None", fmt: str = "{}") -> str:
        """Render a value for the UI, or an em dash when unavailable."""
        if sourced is None or sourced.value is None:
            return "—"
        text = fmt.format(sourced.value)
        return f"~{text}" if sourced.estimated else text
