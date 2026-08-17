"""Crystal structure data model (kept independent of pymatgen)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np


@dataclass
class StructureData:
    """A periodic crystal structure read from POSCAR/CONTCAR."""

    lattice: np.ndarray                 # (3, 3) row vectors in Å
    species: list[str]                  # per-atom element symbols
    frac_coords: np.ndarray             # (n_atoms, 3)
    comment: str = ""
    source: str = "POSCAR"
    selective_dynamics: np.ndarray | None = None

    # Filled in lazily by the symmetry service (spglib via pymatgen-core)
    space_group_symbol: str | None = None
    space_group_number: int | None = None
    crystal_system: str | None = None

    @property
    def n_atoms(self) -> int:
        return len(self.species)

    @property
    def cart_coords(self) -> np.ndarray:
        return self.frac_coords @ self.lattice

    @property
    def lattice_parameters(self) -> tuple[float, float, float]:
        return tuple(float(np.linalg.norm(v)) for v in self.lattice)  # type: ignore[return-value]

    @property
    def lattice_angles(self) -> tuple[float, float, float]:
        """(alpha, beta, gamma) in degrees."""
        a, b, c = self.lattice
        def angle(u, v):
            cos = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
            return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        return angle(b, c), angle(a, c), angle(a, b)

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.lattice)))

    @property
    def composition(self) -> dict[str, int]:
        return dict(Counter(self.species))

    @property
    def reciprocal_lattice(self) -> np.ndarray:
        """Reciprocal lattice including the 2π factor, row vectors in Å⁻¹."""
        return 2.0 * np.pi * np.linalg.inv(self.lattice).T

    @property
    def formula(self) -> str:
        """Reduced formula, e.g. Si8 -> 'Si', Nb16O40 -> 'Nb2O5'."""
        counts = self.composition
        if not counts:
            return ""
        from math import gcd
        divisor = 0
        for n in counts.values():
            divisor = gcd(divisor, n)
        divisor = max(divisor, 1)
        parts = []
        for element, n in counts.items():
            reduced = n // divisor
            parts.append(element if reduced == 1 else f"{element}{reduced}")
        return "".join(parts)

    @property
    def full_formula(self) -> str:
        return "".join(f"{el}{n}" for el, n in self.composition.items())
