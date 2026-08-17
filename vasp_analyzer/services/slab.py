"""Detecting a slab's vacuum gap.

A surface/slab calculation packs its atoms into a fraction of the cell along
one axis and leaves the rest empty, so the periodic images don't interact
through the vacuum. Framing the camera on the *full* cell (as a bulk structure
warrants) makes the atoms a tiny cluster in the middle of a mostly-empty view
— this looks for that pattern so the viewer can frame the atoms instead, while
still drawing the true, full periodic cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models.structure import StructureData

GAP_FRACTION_THRESHOLD = 0.25   # fraction of the cell length that counts as vacuum
MIN_VACUUM_ANGSTROM = 8.0       # and it has to be physically large, in absolute terms
FRAMING_MARGIN_ANGSTROM = 1.0   # minimum margin kept around the atoms when framing
FRAMING_MARGIN_FRACTION = 0.15  # relative margin, as a fraction of the atomic span


@dataclass(frozen=True)
class VacuumInfo:
    """The largest empty gap found along one lattice axis."""

    axis: int              # 0=a, 1=b, 2=c
    gap_fraction: float    # fraction of the cell length that is empty
    vacuum_length: float   # Å
    cell_length: float     # Å

    @property
    def axis_name(self) -> str:
        return "abc"[self.axis]


def detect_vacuum(
    structure: StructureData,
    gap_fraction_threshold: float = GAP_FRACTION_THRESHOLD,
    min_vacuum_angstrom: float = MIN_VACUUM_ANGSTROM,
) -> VacuumInfo | None:
    """Find the largest periodic gap (no atoms, wrap-around included) along
    each lattice axis, and report the biggest one if it looks like vacuum
    rather than ordinary periodic spacing.

    Both a relative and an absolute threshold matter: a sparse bulk cell (a
    2-atom diamond-silicon primitive cell, say) can easily have a >25%
    *fractional* gap along some axis purely from having few atoms to space
    out, without there being any real vacuum — the absolute cutoff is what
    tells the two apart, since genuine slab vacuum is built to be physically
    large enough (conventionally >=8-10 Å) to decouple periodic images, while
    a sparse bulk cell's largest gap is bounded by its ordinary lattice
    constant, typically a few Å.
    """
    if structure.n_atoms == 0:
        return None

    best: VacuumInfo | None = None
    for axis in range(3):
        fracs = np.sort(structure.frac_coords[:, axis] % 1.0)
        extended = np.concatenate([fracs, fracs[:1] + 1.0])
        gap_fraction = float(np.diff(extended).max())
        if gap_fraction < gap_fraction_threshold:
            continue

        axis_length = float(np.linalg.norm(structure.lattice[axis]))
        vacuum_length = gap_fraction * axis_length
        if vacuum_length < min_vacuum_angstrom:
            continue
        if best is not None and gap_fraction <= best.gap_fraction:
            continue

        best = VacuumInfo(
            axis=axis,
            gap_fraction=gap_fraction,
            vacuum_length=vacuum_length,
            cell_length=axis_length,
        )
    return best


def framing_corners(
    lattice: np.ndarray, vacuum: VacuumInfo | None, atom_cart_positions: np.ndarray
) -> np.ndarray:
    """Cartesian corners of the box the camera should frame.

    Normally the full periodic cell's 8 corners. When a vacuum axis is
    detected, that axis is clipped to the real atoms' own cartesian extent
    (plus a margin) instead of the full cell length, so an empty vacuum
    region doesn't dominate the initial framing.

    Deliberately works from already-computed cartesian atom positions rather
    than redoing fractional-coordinate arithmetic: those positions come from
    the same wrapping (``frac % 1.0``) the viewer actually renders, so this
    can't disagree with what's on screen the way re-deriving a periodic range
    from raw fractional coordinates could.
    """
    corners = np.array([np.array(bits, dtype=float) @ lattice for bits in np.ndindex(2, 2, 2)])
    if vacuum is None or atom_cart_positions.size == 0:
        return corners

    axis = vacuum.axis
    lo = float(atom_cart_positions[:, axis].min())
    hi = float(atom_cart_positions[:, axis].max())
    margin = max(FRAMING_MARGIN_ANGSTROM, FRAMING_MARGIN_FRACTION * (hi - lo))
    lo -= margin
    hi += margin

    clipped = corners.copy()
    # Split the 8 corners into the two groups that lie on either side of the
    # vacuum axis (by comparing to the cell's own midpoint there) and clamp
    # each group to the atomic range — robust to which bit-pattern happens to
    # correspond to "high" for an arbitrarily oriented lattice vector.
    midpoint = float(corners[:, axis].mean())
    high_side = clipped[:, axis] > midpoint
    clipped[high_side, axis] = hi
    clipped[~high_side, axis] = lo
    return clipped
