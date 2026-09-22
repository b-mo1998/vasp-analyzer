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
    # Lattice translation this ghost is displaced by, in Å. Lets a caller move
    # ghosts with their parent atom (as trajectory playback does) without
    # repeating the neighbour search.
    shift: np.ndarray | None = None


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
    if n_atoms == 0:
        # Reachable from the editor, where deleting every atom is a legitimate
        # (and undoable) step. There is nothing to bond, and the neighbour
        # search below would fail on empty arrays rather than return nothing.
        return result

    radii = np.array([covalent_radius(s) for s in structure.species], dtype=float)
    metallic = np.array([is_metal(s) for s in structure.species], dtype=bool)
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
                        shift=offset @ lattice,
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


def nearest_neighbors(
    frac_coords: np.ndarray, lattice: np.ndarray, index: int, k: int = 3
) -> list[tuple[int, np.ndarray, float]]:
    """The ``k`` closest neighbouring *bonds* to ``index``, periodic images
    included.

    Returns ``(neighbour_row, unit_direction, distance)`` nearest first,
    where ``unit_direction`` points from ``index`` toward the neighbour in
    cartesian Å.

    Deliberately *not* a plain minimum-image search: that finds only the
    closest periodic copy of each other atom, which collapses to one result
    per row and misses real bonds. Graphene's two-atom cell is the clearest
    case — each carbon has three physical C-C bonds, all to periodic copies
    of the *same* other atom, not to three different atoms — so every one of
    the 27 neighbouring periodic images of every other atom is a candidate,
    and the same row can legitimately appear more than once with a different
    direction.

    Plain arrays rather than a model class, so this works the same for a
    read-only :class:`~vasp_analyzer.models.structure.StructureData` (the
    Structure page) and the mutable
    :class:`~vasp_analyzer.models.editable.EditableStructure` (the Editor's
    move gizmo) without either page depending on the other's model.
    """
    n = len(frac_coords)
    if n < 2 or not (0 <= index < n):
        return []
    origin = frac_coords[index] @ lattice
    offsets = np.array(
        [[i, j, kk] for i in (-1, 0, 1) for j in (-1, 0, 1) for kk in (-1, 0, 1)],
        dtype=float,
    )
    shifts = offsets @ lattice                              # (27, 3)
    base = frac_coords @ lattice                             # (n, 3)
    candidates = base[:, None, :] + shifts[None, :, :]        # (n, 27, 3)
    deltas = candidates - origin                               # (n, 27, 3)
    distances = np.linalg.norm(deltas, axis=2)                 # (n, 27)
    distances[index, :] = np.inf                               # never bond to self

    n_shifts = len(shifts)
    order = np.argsort(distances, axis=None)
    results: list[tuple[int, np.ndarray, float]] = []
    for flat_index in order:
        distance = float(distances.flat[flat_index])
        if not np.isfinite(distance) or distance < 1e-6:
            continue
        row = int(flat_index) // n_shifts
        delta = deltas.reshape(-1, 3)[flat_index]
        results.append((row, delta / distance, distance))
        if len(results) >= k:
            break
    return results


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
