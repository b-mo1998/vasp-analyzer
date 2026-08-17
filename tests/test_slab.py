"""Slab/vacuum detection and vacuum-aware camera framing.

Two real bugs were caught building this, both against real or realistic data
rather than the first synthetic case tried:

1. A sparse bulk cell (a 2-atom diamond-silicon primitive cell) can easily
   have a >25% *fractional* gap along some axis just from having few atoms to
   space out, with no real vacuum at all — an absolute-length threshold is
   what actually distinguishes it from genuine slab vacuum, which is built
   physically large (conventionally >=8-10 A) on purpose.
2. The first framing-corner implementation reused an "unrolled" (possibly
   >1.0) fractional coordinate directly as if it were an absolute one,
   placing the camera-fit box entirely outside the real atoms on a
   wrap-around case — caught by comparing the computed box against the real
   graphene POSCAR's own atom positions, not just a self-consistent
   synthetic check.
"""

from __future__ import annotations

import numpy as np
import pytest

from vasp_analyzer.models.structure import StructureData
from vasp_analyzer.services.slab import detect_vacuum, framing_corners


def bulk_silicon_primitive() -> StructureData:
    """A 2-atom diamond-silicon primitive cell — no vacuum, but sparse."""
    return StructureData(
        lattice=np.eye(3) * 5.4667,
        species=["Si", "Si"],
        frac_coords=np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]),
    )


def bulk_silicon_conventional() -> StructureData:
    frac = np.array(
        [
            [0.00, 0.00, 0.00], [0.00, 0.50, 0.50],
            [0.50, 0.00, 0.50], [0.50, 0.50, 0.00],
            [0.25, 0.25, 0.25], [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75], [0.75, 0.75, 0.25],
        ]
    )
    return StructureData(lattice=np.eye(3) * 5.4667, species=["Si"] * 8, frac_coords=frac)


def thin_slab(cell_c: float = 20.0) -> StructureData:
    """Four atoms clustered at fractional c in [0.40, 0.50]; the rest of a
    20 A cell (18 A) is vacuum."""
    lattice = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, cell_c]])
    frac = np.array(
        [[0.1, 0.1, 0.40], [0.6, 0.1, 0.42], [0.1, 0.6, 0.48], [0.6, 0.6, 0.50]]
    )
    return StructureData(lattice=lattice, species=["Si"] * 4, frac_coords=frac)


def single_layer_2d(cell_c: float = 14.4682) -> StructureData:
    """Both atoms sitting on exactly the same c-plane — the graphene case:
    the vacuum consumes essentially the entire axis except one plane."""
    lattice = np.array([[2.4695, 0.0, 0.0], [-1.2347, 2.1386, 0.0], [0.0, 0.0, cell_c]])
    frac = np.array([[0.0, 0.0, 0.25], [2 / 3, 1 / 3, 0.25]])
    return StructureData(lattice=lattice, species=["C", "C"], frac_coords=frac)


# --- detection -----------------------------------------------------------------


def test_sparse_bulk_primitive_cell_is_not_flagged():
    assert detect_vacuum(bulk_silicon_primitive()) is None


def test_bulk_conventional_cell_is_not_flagged():
    assert detect_vacuum(bulk_silicon_conventional()) is None


def test_thin_slab_is_detected_on_the_right_axis():
    vacuum = detect_vacuum(thin_slab())
    assert vacuum is not None
    assert vacuum.axis == 2
    assert vacuum.axis_name == "c"
    assert vacuum.vacuum_length == pytest.approx(18.0, abs=0.1)


def test_single_layer_2d_material_is_detected():
    vacuum = detect_vacuum(single_layer_2d())
    assert vacuum is not None
    assert vacuum.axis == 2
    assert vacuum.vacuum_length > 14.0


def test_a_large_but_not_physically_large_gap_is_rejected():
    """A small cell (e.g. 6 A) with a 90% fractional gap still only has
    ~5.4 A of vacuum — below the physical decoupling threshold, so it should
    not be treated as a genuine slab."""
    structure = thin_slab(cell_c=6.0)
    assert detect_vacuum(structure) is None


def test_gap_just_under_the_fraction_threshold_is_rejected():
    lattice = np.eye(3) * 20.0
    # Atoms spread almost evenly around all three axes: no single gap on any
    # axis reaches 25%. (x and y must vary too, not just z — a constant
    # coordinate on an axis is itself a trivial 100% gap on that axis.)
    frac = np.array(
        [[0.1, 0.4, 0.0], [0.35, 0.65, 0.26], [0.6, 0.15, 0.51], [0.85, 0.9, 0.76]]
    )
    structure = StructureData(lattice=lattice, species=["Si"] * 4, frac_coords=frac)
    assert detect_vacuum(structure) is None


# --- framing ---------------------------------------------------------------------


def test_no_vacuum_returns_the_full_cell_corners():
    structure = bulk_silicon_conventional()
    corners = framing_corners(structure.lattice, None, structure.frac_coords @ structure.lattice)
    assert corners.shape == (8, 3)
    assert np.isclose(corners[:, 2].max(), structure.lattice[2, 2])
    assert np.isclose(corners[:, 2].min(), 0.0)


def test_vacuum_axis_is_clipped_to_the_atomic_extent():
    structure = thin_slab()
    vacuum = detect_vacuum(structure)
    real_cart = (structure.frac_coords % 1.0) @ structure.lattice

    corners = framing_corners(structure.lattice, vacuum, real_cart)

    # The clipped range must bracket the real atoms, with only a small
    # margin — not sprawl anywhere close to the full 20 A vacuum-inclusive
    # cell.
    assert corners[:, 2].min() <= real_cart[:, 2].min()
    assert corners[:, 2].max() >= real_cart[:, 2].max()
    assert corners[:, 2].max() - real_cart[:, 2].max() < 2.0
    assert corners[:, 2].max() < structure.lattice[2, 2] * 0.75
    # In-plane axes (no vacuum) are completely unaffected.
    assert np.isclose(corners[:, 0].max(), structure.lattice[0, 0])
    assert np.isclose(corners[:, 1].max(), structure.lattice[1, 1])


def test_framing_matches_the_real_graphene_structure():
    """The bug this guards against only showed up against real, non-trivial
    geometry: a first implementation placed the box entirely outside the
    real atoms here specifically, while passing on simpler synthetic cases."""
    structure = single_layer_2d()
    vacuum = detect_vacuum(structure)
    real_cart = (structure.frac_coords % 1.0) @ structure.lattice

    corners = framing_corners(structure.lattice, vacuum, real_cart)

    atom_z = real_cart[:, 2]
    assert corners[:, 2].min() <= atom_z.min()
    assert corners[:, 2].max() >= atom_z.max()
    # The framed box must be small (close to the atomic plane), not sprawled
    # across (or worse, entirely outside) the 14.5 A vacuum-inclusive cell.
    box_height = corners[:, 2].max() - corners[:, 2].min()
    assert box_height < 5.0


def test_framing_extent_is_meaningfully_tighter_with_vacuum_awareness():
    """A direct extent comparison against the naive full-cell fit, on real
    graphene geometry — the actual improvement this feature is for."""
    structure = single_layer_2d()
    real_cart = (structure.frac_coords % 1.0) @ structure.lattice

    naive = framing_corners(structure.lattice, None, real_cart)
    aware = framing_corners(structure.lattice, detect_vacuum(structure), real_cart)

    def extent(points: np.ndarray) -> float:
        centre = points.mean(axis=0)
        return float(np.linalg.norm(points - centre, axis=1).max())

    assert extent(aware) < extent(naive) * 0.5
