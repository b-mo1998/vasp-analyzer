"""Periodic tiling of a StructureData into a supercell.

A pure geometric transform: replicate every atom across an na x nb x nc grid
of image cells, and scale the lattice to match. Symmetry, formula and lattice
parameters are properties of the underlying unit cell, not of any particular
tiling of it, so callers should keep referencing the original StructureData
for those and use the tiled result only for rendering and atom counts.
"""

from __future__ import annotations

import numpy as np

from ..models.structure import StructureData


def build_supercell(structure: StructureData, na: int, nb: int, nc: int) -> StructureData:
    """Return a new StructureData tiled na x nb x nc.

    Returns the original structure unchanged (not a copy) when no tiling is
    requested (1, 1, 1) — the common case, and worth avoiding the allocation
    for.
    """
    na, nb, nc = max(1, int(na)), max(1, int(nb)), max(1, int(nc))
    if (na, nb, nc) == (1, 1, 1):
        return structure

    repeats = np.array([na, nb, nc], dtype=float)
    offsets = np.array(
        [[i, j, k] for i in range(na) for j in range(nb) for k in range(nc)],
        dtype=float,
    )  # (n_cells, 3)

    # Every atom's fractional coordinate against every cell offset, then back
    # into [0, 1) of the *new*, larger cell. reshape(-1, 3) flattens this as
    # cell-major / atom-minor, so the repeated species list below has to match
    # that same order: all of the base cell's atoms, once per offset, in the
    # order the offsets were generated.
    tiled_frac = (structure.frac_coords[None, :, :] + offsets[:, None, :]) / repeats
    tiled_frac = tiled_frac.reshape(-1, 3)
    tiled_species = list(structure.species) * len(offsets)
    tiled_lattice = structure.lattice * repeats[:, None]

    # Constraints belong to the atom, so every copy of an atom inherits them.
    # Tiled the same way the species list is, to stay aligned with the
    # cell-major / atom-minor flattening above.
    tiled_selective = None
    if structure.selective_dynamics is not None:
        flags = np.asarray(structure.selective_dynamics, dtype=bool)
        if flags.ndim == 1:
            flags = np.repeat(flags[:, None], 3, axis=1)
        tiled_selective = np.tile(flags, (len(offsets), 1))

    return StructureData(
        lattice=tiled_lattice,
        species=tiled_species,
        frac_coords=tiled_frac,
        comment=structure.comment,
        source=structure.source,
        selective_dynamics=tiled_selective,
    )


def cell_count(na: int, nb: int, nc: int) -> int:
    return max(1, int(na)) * max(1, int(nb)) * max(1, int(nc))
