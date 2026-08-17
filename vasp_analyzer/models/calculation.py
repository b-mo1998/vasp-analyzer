"""The top-level object representing one opened calculation directory."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..core.errors import Issue
from .capabilities import Capability, CapabilitySet
from .files import FileIndex
from .metadata import CalculationMetadata


@dataclass
class Calculation:
    """An indexed calculation directory.

    Created by the discovery service from a cheap scan: file presence, sizes and
    header peeks only. Heavy data (bands, DOS, structure) is loaded on demand by
    the loader service and cached there, not stored here.
    """

    directory: Path
    index: FileIndex
    capabilities: CapabilitySet = field(default_factory=CapabilitySet)
    metadata: CalculationMetadata = field(default_factory=CalculationMetadata)
    issues: list[Issue] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.directory.name

    @property
    def is_empty(self) -> bool:
        """True when nothing recognisable was found in the directory."""
        return not any(entry.exists for entry in self.index.entries.values())

    def can(self, cap: Capability) -> bool:
        return self.capabilities.has(cap)

    def why_not(self, cap: Capability) -> str:
        return self.capabilities.reason(cap)
