"""Capability model.

The GUI binds to capabilities, never to filenames. A capability that is
unavailable carries a short reason which the UI shows as a tooltip or inline
note — that is how "DOSCAR not found" reaches the user without a dialog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Capability(str, Enum):
    BANDS = "bands"
    DOS = "dos"
    DOS_PROJECTED = "dos_projected"
    STRUCTURE = "structure"
    SPIN_PROJECTION = "spin_projection"
    ELEMENT_PROJECTION = "element_projection"
    WANNIER_BANDS = "wannier_bands"
    KPATH_LABELS = "kpath_labels"
    MD_TRAJECTORY = "md_trajectory"
    MD_THERMODYNAMICS = "md_thermodynamics"
    NEB_PATH = "neb_path"
    NEB_ENERGIES = "neb_energies"
    CHARGE_DENSITY = "charge_density"
    INPUT_FILES = "input_files"
    METADATA = "metadata"


@dataclass
class CapabilitySet:
    """Which scientific features the opened directory can support."""

    available: set[Capability] = field(default_factory=set)
    reasons: dict[Capability, str] = field(default_factory=dict)

    def enable(self, cap: Capability) -> None:
        self.available.add(cap)
        self.reasons.pop(cap, None)

    def disable(self, cap: Capability, reason: str) -> None:
        self.available.discard(cap)
        self.reasons[cap] = reason

    def set(self, cap: Capability, ok: bool, reason: str = "") -> None:
        if ok:
            self.enable(cap)
        else:
            self.disable(cap, reason)

    def has(self, cap: Capability) -> bool:
        return cap in self.available

    def reason(self, cap: Capability) -> str:
        return self.reasons.get(cap, "")

    def __contains__(self, cap: object) -> bool:
        return cap in self.available
