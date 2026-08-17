"""Bond inference and periodic-image expansion.

Bonds are not stored in a POSCAR, so they are inferred from interatomic
distance against the sum of covalent radii. Bonds that cross a cell face are
found by searching the 27 neighbouring images, and the image atom involved is
returned as a "ghost" so the bond does not end in mid-air.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from ..core.logging_setup import get_logger
from ..models.structure import StructureData
from ..resources.elements import covalent_radius, is_metal

log = get_logger("services.bonds")

# A bond is drawn when d < tolerance * (r_i + r_j) *and* d <= MAX_BOND_LENGTH.
#
# The absolute cap matters: universal covalent radii are generous for metals,
# and the multiplicative rule alone puts a "bond" between the Ti atoms of
# rutile at 3.57 Å, which is not a bond. 3.0 Å covers real bonds in the
# structures this application shows while excluding those artefacts.
BOND_TOLERANCE = 1.10
MAX_BOND_LENGTH = 3.0
MIN_BOND_LENGTH = 0.4    # guards against duplicate/overlapping sites
MAX_ATOMS_FOR_BONDS = 4000


@dataclass
class Site:
    """A drawable atom: either a real site or a periodic image of one."""

    index: int              # index into the original structure
    symbol: str
    position: np.ndarray    # cartesian, Å
    is_ghost: bool = False


@dataclass
class BondSet:
    sites: list[Site] = field(default_factory=list)
    bonds: list[tuple[int, int]] = field(default_factory=list)  # indices into sites

    @property
    def n_real(self) -> int:
        return sum(1 for site in self.sites if not site.is_ghost)


def build_sites_and_bonds(
    structure: StructureData, include_ghosts: bool = True
) -> BondSet:
    """Return drawable sites plus the bonds between them."""
    result = BondSet()

    wrapped = structure.frac_coords % 1.0
    cartesian = wrapped @ structure.lattice
    for index, symbol in enumerate(structure.species):
        result.sites.append(Site(index=index, symbol=symbol, position=cartesian[index]))

    n_atoms = structure.n_atoms
    if n_atoms > MAX_ATOMS_FOR_BONDS:
        log.info("Skipping bond search for %d atoms", n_atoms)
        return result

    radii = np.array([covalent_radius(s) for s in structure.species])
    metallic = np.array([is_metal(s) for s in structure.species])
    # In a compound, a short cation-cation contact is not a bond: rutile's
    # Ti atoms sit 2.96 Å apart along c and drawing that would misrepresent the
    # structure. In an elemental metal or an alloy those contacts are the
    # bonding, so the rule only applies when a non-metal is present.
    suppress_metal_metal = bool((~metallic).any())
    lattice = structure.lattice
    offsets = np.array(list(product((-1, 0, 1), repeat=3)), dtype=float)

    # Neighbour search over the 27 periodic images, via a KD-tree so the cost
    # stays near-linear instead of quadratic in the atom count.
    from scipy.spatial import cKDTree  # noqa: PLC0415 - optional heavy import

    shifts = offsets @ lattice
    images = np.concatenate([cartesian + shift for shift in shifts])
    tree = cKDTree(images)
    neighbourhoods = tree.query_ball_point(cartesian, r=MAX_BOND_LENGTH)

    ghost_lookup: dict[tuple[int, int, int, int], int] = {}

    for i, neighbours in enumerate(neighbourhoods):
        for flat in neighbours:
            image_index, j = divmod(flat, n_atoms)
            offset = offsets[image_index]
            in_cell = not offset.any()

            # Within the cell each pair is one bond; across a face, atom i and
            # atom j each get their own bond to an image of the other, which is
            # what a periodic picture should show.
            if in_cell and j <= i:
                continue
            if suppress_metal_metal and metallic[i] and metallic[j]:
                continue

            distance = float(np.linalg.norm(images[flat] - cartesian[i]))
            limit = min(BOND_TOLERANCE * (radii[i] + radii[j]), MAX_BOND_LENGTH)
            if distance < MIN_BOND_LENGTH or distance > limit:
                continue

            if in_cell:
                result.bonds.append((i, j))
                continue

            if not include_ghosts:
                continue

            ghost_key = (j, int(offset[0]), int(offset[1]), int(offset[2]))
            ghost_index = ghost_lookup.get(ghost_key)
            if ghost_index is None:
                ghost_index = len(result.sites)
                ghost_lookup[ghost_key] = ghost_index
                result.sites.append(
                    Site(
                        index=j,
                        symbol=structure.species[j],
                        position=images[flat],
                        is_ghost=True,
                    )
                )
            result.bonds.append((i, ghost_index))

    log.info(
        "Bond search: %d sites (%d ghosts), %d bonds",
        len(result.sites),
        len(result.sites) - n_atoms,
        len(result.bonds),
    )
    return result


def unit_cell_edges(lattice: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """The twelve edges of the parallelepiped, as cartesian point pairs."""
    corners = {}
    for bits in product((0, 1), repeat=3):
        corners[bits] = np.array(bits, dtype=float) @ lattice

    edges: list[tuple[np.ndarray, np.ndarray]] = []
    for bits in corners:
        for axis in range(3):
            if bits[axis] == 0:
                neighbour = list(bits)
                neighbour[axis] = 1
                edges.append((corners[bits], corners[tuple(neighbour)]))
    return edges
