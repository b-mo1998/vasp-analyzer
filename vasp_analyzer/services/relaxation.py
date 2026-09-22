"""What a relaxation did: per-atom displacements from POSCAR to CONTCAR.

The one non-obvious step is the minimum-image convention. CONTCAR coordinates
are wrapped back into the cell, so an atom that relaxed a hundredth of an
ångström across a face is written 0.99 of a cell away from where it started.
Taking the raw difference would report that as an enormous move; wrapping the
fractional difference into [-0.5, 0.5) recovers the physical one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models.structure import StructureData


@dataclass(frozen=True)
class RelaxationResult:
    """Cartesian displacement of every atom, initial -> final, in Å."""

    vectors: np.ndarray        # (n_atoms, 3)
    initial: StructureData
    final: StructureData

    @property
    def magnitudes(self) -> np.ndarray:
        return np.linalg.norm(self.vectors, axis=1)

    @property
    def n_atoms(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def max_displacement(self) -> float:
        return float(self.magnitudes.max()) if self.n_atoms else 0.0

    @property
    def max_atom(self) -> int:
        return int(self.magnitudes.argmax()) if self.n_atoms else -1

    @property
    def mean_displacement(self) -> float:
        return float(self.magnitudes.mean()) if self.n_atoms else 0.0

    @property
    def rms_displacement(self) -> float:
        return float(np.sqrt(np.mean(self.magnitudes**2))) if self.n_atoms else 0.0

    def ranked(self, limit: int = 5) -> list[int]:
        """Atom indices, most-displaced first."""
        return [int(i) for i in np.argsort(-self.magnitudes)[:limit]]


def compare(initial: StructureData, final: StructureData) -> RelaxationResult | None:
    """Displacements between two structures of the same atoms.

    Returns None when the two do not describe the same set of atoms — a
    different count or species order means they are not before/after of one
    relaxation, and a per-atom difference would be meaningless.

    Displacements are measured in the *final* cell. A variable-cell relaxation
    (ISIF >= 3) changes the lattice too, so the same fractional shift is a
    different distance in Å depending on which lattice converts it; the
    relaxed cell is the one the final geometry actually lives in.
    """
    if initial.n_atoms != final.n_atoms:
        return None
    if list(initial.species) != list(final.species):
        return None

    delta_frac = np.asarray(final.frac_coords, dtype=float) - np.asarray(
        initial.frac_coords, dtype=float
    )
    # Minimum image: the physically shortest displacement, not the one that
    # happens to result from CONTCAR re-wrapping into the cell.
    delta_frac -= np.round(delta_frac)
    vectors = delta_frac @ np.asarray(final.lattice, dtype=float)
    return RelaxationResult(vectors=vectors, initial=initial, final=final)


def tile(vectors: np.ndarray, n_cells: int) -> np.ndarray:
    """Repeat per-atom vectors for a supercell.

    build_supercell flattens cell-major / atom-minor -- every base atom once
    per image cell, image cells in order -- so tiling the vector array the
    same way keeps index i of the result attached to atom i of the tiled
    structure.
    """
    return np.tile(np.asarray(vectors, dtype=float), (max(1, int(n_cells)), 1))
