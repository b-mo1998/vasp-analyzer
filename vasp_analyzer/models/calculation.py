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
    """An indexed calculation, possibly spread across more than one directory.

    ``directory``/``index`` are the electronic-structure root — what
    Structure, Bands, Charge Density and Inputs read from. Molecular
    Dynamics and NEB can read from elsewhere: opening a project folder
    organised into ``band``/``md``/``neb`` subdirectories routes each page
    family to its own directory (see ``services.layout``), while opening an
    ordinary flat calculation directory leaves all three pointed at the same
    place, so ``md_directory``/``neb_directory`` default to ``directory``
    rather than being optional — every caller can use them directly with no
    fallback logic of its own.

    Created by the discovery service from a cheap scan: file presence, sizes
    and header peeks only. Heavy data (bands, DOS, structure, a trajectory)
    is loaded on demand by the loader service and cached there, not stored
    here.
    """

    directory: Path
    index: FileIndex
    capabilities: CapabilitySet = field(default_factory=CapabilitySet)
    metadata: CalculationMetadata = field(default_factory=CalculationMetadata)
    issues: list[Issue] = field(default_factory=list)

    # The folder the user actually opened or dropped, when it differs from
    # `directory` — e.g. a project folder whose electronic-structure files
    # live one level down in a "band" subdirectory. None means they're the
    # same: an ordinary flat calculation directory.
    opened_directory: Path | None = None

    md_directory: Path | None = None
    md_index: FileIndex | None = None
    neb_directory: Path | None = None
    neb_index: FileIndex | None = None

    # The volumetric files the Charge Density page may offer, decided at
    # discovery time rather than re-scanned by the page. That distinction
    # matters for a single dropped CHGCAR: the page used to list everything
    # in the containing folder, which would quietly offer the neighbours the
    # user did not drop — and, worse, files belonging to a different run.
    volumetric_sources: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.md_directory is None:
            self.md_directory = self.directory
        if self.md_index is None:
            self.md_index = self.index
        if self.neb_directory is None:
            self.neb_directory = self.directory
        if self.neb_index is None:
            self.neb_index = self.index

    @property
    def display_directory(self) -> Path:
        """The folder to show in the title bar / Reveal in Finder / Reload —
        the one the user actually opened, not wherever a page's data landed."""
        return self.opened_directory or self.directory

    @property
    def name(self) -> str:
        return self.display_directory.name

    @property
    def is_multi_root(self) -> bool:
        """True when Structure/Bands/Charge, MD and NEB read from more than
        one physical directory."""
        return self.md_directory != self.directory or self.neb_directory != self.directory

    @property
    def is_empty(self) -> bool:
        """True when nothing recognisable was found in the directory."""
        return not any(entry.exists for entry in self.index.entries.values())

    def can(self, cap: Capability) -> bool:
        return self.capabilities.has(cap)

    def why_not(self, cap: Capability) -> str:
        return self.capabilities.reason(cap)
