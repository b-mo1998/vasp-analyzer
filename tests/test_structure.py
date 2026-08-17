"""Structure model, bond inference, symmetry and projection maths."""

from __future__ import annotations

import numpy as np
import pytest

from vasp_analyzer.models.structure import StructureData
from vasp_analyzer.resources.elements import color, covalent_radius, is_known
from vasp_analyzer.services.bonds import build_sites_and_bonds, unit_cell_edges
from vasp_analyzer.services.symmetry import analyse, space_group_label
from vasp_analyzer.ui.pages.structure.camera import Camera, rotation_matrix


def silicon() -> StructureData:
    """Conventional diamond-structure silicon cell."""
    a = 5.4667335224494948
    frac = np.array(
        [
            [0.00, 0.00, 0.00], [0.00, 0.50, 0.50],
            [0.50, 0.00, 0.50], [0.50, 0.50, 0.00],
            [0.25, 0.25, 0.25], [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75], [0.75, 0.75, 0.25],
        ]
    )
    return StructureData(
        lattice=np.eye(3) * a, species=["Si"] * 8, frac_coords=frac, comment="Si8"
    )


def rutile() -> StructureData:
    lattice = np.diag([4.5941, 4.5941, 2.9589])
    frac = np.array(
        [
            [0.0, 0.0, 0.0], [0.5, 0.5, 0.5],
            [0.3053, 0.3053, 0.0], [0.6947, 0.6947, 0.0],
            [0.8053, 0.1947, 0.5], [0.1947, 0.8053, 0.5],
        ]
    )
    return StructureData(
        lattice=lattice, species=["Ti", "Ti", "O", "O", "O", "O"], frac_coords=frac
    )


# --- model ------------------------------------------------------------------


def test_lattice_geometry():
    structure = silicon()
    a, b, c = structure.lattice_parameters
    assert a == pytest.approx(5.4667335, abs=1e-6)
    assert structure.lattice_angles == pytest.approx((90.0, 90.0, 90.0))
    assert structure.volume == pytest.approx(a**3, rel=1e-9)


def test_reduced_formula():
    assert silicon().formula == "Si"
    assert rutile().formula == "TiO2"
    assert rutile().full_formula == "Ti2O4"


def test_reciprocal_lattice_is_consistent():
    structure = silicon()
    product = structure.lattice @ structure.reciprocal_lattice.T
    assert np.allclose(product, 2 * np.pi * np.eye(3), atol=1e-9)


def test_cartesian_conversion_round_trip():
    structure = rutile()
    back = structure.cart_coords @ np.linalg.inv(structure.lattice)
    assert np.allclose(back, structure.frac_coords, atol=1e-12)


# --- elements ---------------------------------------------------------------


def test_element_lookup_handles_labels():
    assert is_known("Si")
    assert covalent_radius("Si1") == covalent_radius("Si")
    assert color("O_s") == color("O")
    # An unknown label must fall back rather than raise.
    assert covalent_radius("Xx") > 0


# --- bonds ------------------------------------------------------------------


def test_silicon_is_four_coordinate():
    structure = silicon()
    result = build_sites_and_bonds(structure)

    assert result.n_real == 8
    counts = {index: 0 for index in range(8)}
    for i, j in result.bonds:
        for endpoint in (i, j):
            site = result.sites[endpoint]
            if not site.is_ghost:
                counts[site.index] += 1
    # Every silicon atom in diamond has exactly four neighbours.
    assert set(counts.values()) == {4}


def test_bonds_cross_periodic_boundaries():
    result = build_sites_and_bonds(silicon())
    assert any(site.is_ghost for site in result.sites), (
        "diamond silicon must bond across cell faces"
    )


def test_rutile_titanium_is_six_coordinate():
    result = build_sites_and_bonds(rutile())
    counts = {0: 0, 1: 0}
    for i, j in result.bonds:
        for endpoint in (i, j):
            site = result.sites[endpoint]
            if site.index in counts and site.symbol == "Ti" and not site.is_ghost:
                counts[site.index] += 1
    assert counts[0] == 6
    assert counts[1] == 6


def test_unit_cell_has_twelve_edges():
    edges = unit_cell_edges(silicon().lattice)
    assert len(edges) == 12


# --- symmetry ---------------------------------------------------------------


def test_silicon_symmetry_is_identified():
    structure = silicon()
    assert analyse(structure)
    assert structure.space_group_number == 227
    assert structure.crystal_system == "Cubic"
    assert "227" in space_group_label(structure)


def test_rutile_symmetry_is_identified():
    structure = rutile()
    assert analyse(structure)
    assert structure.space_group_number == 136
    assert structure.crystal_system == "Tetragonal"


def test_missing_symmetry_shows_an_em_dash():
    structure = silicon()
    structure.space_group_number = None
    structure.space_group_symbol = None
    assert space_group_label(structure) == "—"


# --- camera -----------------------------------------------------------------


def test_rotation_matrix_is_orthonormal():
    matrix = rotation_matrix(np.array([0.3, 1.0, -0.2]), 0.7)
    assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(matrix) == pytest.approx(1.0)


def test_projection_centres_the_structure():
    structure = silicon()
    camera = Camera()
    camera.reset()
    points = structure.cart_coords
    camera.fit(points)

    screen, depth, scale = camera.project(points, 800.0, 600.0)
    assert screen.shape == (8, 2)
    # The projected cloud is centred on the viewport.
    assert screen[:, 0].mean() == pytest.approx(400.0, abs=1.0)
    assert screen[:, 1].mean() == pytest.approx(300.0, abs=1.0)
    assert np.all(scale > 0)


def test_orthographic_projection_has_uniform_scale():
    structure = silicon()
    camera = Camera()
    camera.perspective = False
    camera.fit(structure.cart_coords)
    _, _, scale = camera.project(structure.cart_coords, 800.0, 600.0)
    assert np.allclose(scale, scale[0])


def test_perspective_projection_varies_with_depth():
    structure = silicon()
    camera = Camera()
    camera.perspective = True
    camera.fit(structure.cart_coords)
    _, depth, scale = camera.project(structure.cart_coords, 800.0, 600.0)
    assert scale.max() > scale.min(), "nearer atoms must project larger"


def test_zoom_is_clamped():
    camera = Camera()
    for _ in range(50):
        camera.scale(2.0)
    assert camera.zoom <= 12.0
    for _ in range(100):
        camera.scale(0.5)
    assert camera.zoom >= 0.15
