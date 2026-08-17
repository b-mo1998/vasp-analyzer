"""Density-of-states data model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DosData:
    """Total density of states as read from DOSCAR.

    ``energies`` are absolute eV. ``total`` is ``(n_spin, n_points)``; for a
    spin-polarised calculation index 0 is spin up and index 1 spin down (stored
    positive — the plot mirrors it, the data does not).
    """

    energies: np.ndarray            # (n,)
    total: np.ndarray               # (nspin, n)
    integrated: np.ndarray          # (nspin, n)
    e_fermi: float | None = None
    n_ions: int = 0
    has_projected: bool = False
    source: str = "DOSCAR"

    @property
    def n_spin(self) -> int:
        return int(self.total.shape[0])

    @property
    def is_spin_polarized(self) -> bool:
        return self.n_spin > 1

    @property
    def n_points(self) -> int:
        return int(self.energies.size)

    @property
    def energy_range(self) -> tuple[float, float]:
        return float(self.energies.min()), float(self.energies.max())

    def max_in_window(self, e_min: float, e_max: float) -> float:
        """Peak DOS value within an absolute-energy window (for axis scaling)."""
        mask = (self.energies >= e_min) & (self.energies <= e_max)
        if not mask.any():
            return float(self.total.max()) if self.total.size else 1.0
        return float(self.total[:, mask].max())
