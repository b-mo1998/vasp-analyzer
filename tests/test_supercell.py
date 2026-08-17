"""Supercell tiling: pure geometry, checked against hand-computable cases and
a real graphene cell.
"""

from __future__ import annotations

import numpy as np
import pytest

from vasp_analyzer.models.structure import StructureData
from vasp_analyzer.services.supercell import build_supercell, cell_count


def graphene() -> StructureData:
    lattice = np.array(
        [[2.4694910408650514, 0.0, 0.0],
         [-1.2347455204325257, 2.1386419486320141, 0.0],
         [0.0, 0.0, 14.4682012468615859]]
    )
    frac = np.array([[0.0, 0.0, 0.25], [2 / 3, 1 / 3, 0.25]])
    return StructureData(lattice=lattice, species=["C", "C"], frac_coords=frac)


def test_1x1x1_returns_the_original_object_unchanged():
    structure = graphene()
    assert build_supercell(structure, 1, 1, 1) is structure


def test_single_atom_doubled_along_one_axis():
    structure = StructureData(
        lattice=np.eye(3) * 2.0, species=["C"], frac_coords=np.array([[0.0, 0.0, 0.0]])
    )
    tiled = build_supercell(structure, 2, 1, 1)

    assert tiled.n_atoms == 2
    assert tiled.species == ["C", "C"]
    assert np.allclose(tiled.lattice, np.diag([4.0, 2.0, 2.0]))
    assert np.allclose(sorted(tiled.cart_coords[:, 0]), [0.0, 2.0])


def test_atom_count_scales_by_the_cell_count():
    structure = graphene()
    for na, nb, nc in [(2, 1, 1), (2, 2, 1), (3, 3, 3)]:
        tiled = build_supercell(structure, na, nb, nc)
        assert tiled.n_atoms == structure.n_atoms * na * nb * nc


def test_volume_scales_by_the_cell_count():
    structure = graphene()
    tiled = build_supercell(structure, 2, 2, 1)
    assert tiled.volume == pytest.approx(structure.volume * 4)


def test_reduced_formula_is_unchanged_by_tiling():
    """A supercell is chemically the same substance — C stays C, not C8."""
    structure = graphene()
    tiled = build_supercell(structure, 2, 2, 2)
    assert tiled.formula == structure.formula == "C"


def test_lattice_vectors_scale_independently_per_axis():
    structure = graphene()
    tiled = build_supercell(structure, 3, 1, 2)
    assert np.allclose(tiled.lattice[0], structure.lattice[0] * 3)
    assert np.allclose(tiled.lattice[1], structure.lattice[1] * 1)
    assert np.allclose(tiled.lattice[2], structure.lattice[2] * 2)


def test_no_two_atoms_land_on_the_same_cartesian_position():
    structure = graphene()
    tiled = build_supercell(structure, 2, 2, 1)
    coords = tiled.cart_coords
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            assert np.linalg.norm(coords[i] - coords[j]) > 0.5


def test_negative_or_zero_repeat_is_clamped_to_one():
    structure = graphene()
    assert build_supercell(structure, 0, -3, 1).n_atoms == structure.n_atoms


def test_cell_count_helper():
    assert cell_count(2, 3, 1) == 6
    assert cell_count(0, 5, 1) == 5  # clamped to at least 1 per axis
