"""Structure model, bond inference, symmetry and projection maths."""

from __future__ import annotations

import numpy as np
import pytest

from vasp_analyzer.models.structure import StructureData
from vasp_analyzer.resources.elements import color, covalent_radius, is_known
from vasp_analyzer.services.bonds import (
    build_sites_and_bonds,
    nearest_neighbors,
    unit_cell_edges,
)
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


def test_nearest_neighbors_finds_silicons_four_tetrahedral_bonds():
    """Powers the Structure page's neighbour table — same physics
    test_silicon_is_four_coordinate already checks via the bond graph, here
    checked through the API the sidebar actually calls."""
    structure = silicon()
    results = nearest_neighbors(structure.frac_coords, structure.lattice, 0, k=4)

    assert len(results) == 4
    distances = [d for _row, _direction, d in results]
    assert all(d == pytest.approx(distances[0], abs=1e-6) for d in distances), (
        "diamond silicon's four nearest neighbours are equidistant"
    )
    # The textbook Si-Si bond length, not just "some positive number."
    assert distances[0] == pytest.approx(2.35, abs=0.05)


def test_nearest_neighbors_are_sorted_nearest_first():
    structure = rutile()
    results = nearest_neighbors(structure.frac_coords, structure.lattice, 0, k=6)
    distances = [d for _row, _direction, d in results]
    assert distances == sorted(distances)


def test_nearest_neighbors_direction_points_toward_the_neighbor():
    structure = silicon()
    results = nearest_neighbors(structure.frac_coords, structure.lattice, 0, k=1)
    row, direction, distance = results[0]

    origin = structure.frac_coords[0] @ structure.lattice
    # The nearest periodic image of the neighbour along that same direction,
    # at that same distance — direction and distance must describe one
    # mutually consistent bond, not two independently-computed numbers.
    expected = origin + direction * distance
    candidates = (
        structure.frac_coords[row] @ structure.lattice
        + np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
        @ structure.lattice
    )
    closest = candidates[np.argmin(np.linalg.norm(candidates - expected, axis=1))]
    np.testing.assert_allclose(closest, expected, atol=1e-6)


def test_nearest_neighbors_out_of_range_index_returns_nothing():
    structure = silicon()
    assert nearest_neighbors(structure.frac_coords, structure.lattice, 99, k=4) == []


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
    camera.set_lattice(structure.lattice)
    camera.reset()
    points = structure.cart_coords
    camera.fit(points)

    screen, depth, scale = camera.project(points, 800.0, 600.0)
    assert screen.shape == (8, 2)
    assert np.all(scale > 0)

    # What "centred" actually means: the point the camera was fitted to lands
    # on the middle of the viewport, exactly.
    centre_screen, _, _ = camera.project(camera.centre[None, :], 800.0, 600.0)
    assert centre_screen[0][0] == pytest.approx(400.0)
    assert centre_screen[0][1] == pytest.approx(300.0)

    # The *mean of the projected atoms* is a different quantity, and only
    # coincides with the above under orthographic projection — perspective
    # pushes near atoms outward, so an asymmetric cloud has a mean a few
    # pixels off centre while still being perfectly framed.
    camera.perspective = False
    flat, _, _ = camera.project(points, 800.0, 600.0)
    assert flat[:, 0].mean() == pytest.approx(400.0, abs=1e-6)
    assert flat[:, 1].mean() == pytest.approx(300.0, abs=1e-6)


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


# -- camera orientation presets ---------------------------------------------


def _screen_axes(camera: Camera, lattice: np.ndarray) -> np.ndarray:
    """Where the three cell axes point on screen, as unit-ish 2D vectors.

    Rows are a, b, c; ``x`` grows rightward and ``y`` grows *downward*, which
    is the Qt convention the axis triad is drawn in.
    """
    return camera.direction_screen(lattice / np.linalg.norm(lattice, axis=1, keepdims=True))


def test_default_view_puts_b_across_and_c_up():
    """The orientation every 3D page opens in, and that the default-view
    button returns to. Fixed deliberately: a structure that presents itself
    differently depending on how it was reached is one nobody can compare
    against a colleague's screen."""
    structure = silicon()
    camera = Camera()
    camera.set_lattice(structure.lattice)
    camera.reset()

    a, b, c = _screen_axes(camera, structure.lattice)
    assert b[0] > 0.9 and abs(b[1]) < 0.1, "b runs left-to-right"
    assert c[1] < -0.9, "c points up the screen"
    # a is nearly end-on — the small residual is the deliberate tilt, without
    # which every atom would hide exactly behind the one in front of it.
    assert np.linalg.norm(a) < 0.3
    assert np.linalg.norm(a) > 0.05


def test_orientation_presets_put_the_named_axis_at_the_viewer():
    structure = silicon()
    camera = Camera()
    camera.set_lattice(structure.lattice)
    for index, key in enumerate("abc"):
        camera.orient(key)
        axis = structure.lattice[index] / np.linalg.norm(structure.lattice[index])
        # Row 2 of the rotation is the toward-the-viewer direction.
        assert np.dot(camera.rotation[2], axis) == pytest.approx(1.0)


def test_reciprocal_presets_differ_from_real_ones_in_an_oblique_cell():
    """a* is the normal to the b-c plane, which is not a in a hexagonal cell.

    The distinction is the whole reason both sets of buttons exist: in
    graphene's lattice a and a* sit 30 degrees apart, so "down a" and "down
    a*" are genuinely different views of the same sheet.
    """
    lattice = np.array([[2.469, 0.0, 0.0], [-1.2345, 2.1382, 0.0], [0.0, 0.0, 14.468]])
    structure = StructureData(
        lattice=lattice,
        species=["C", "C"],
        frac_coords=np.array([[0.0, 0.0, 0.5], [1 / 3, 2 / 3, 0.5]]),
    )
    camera = Camera()
    camera.set_lattice(structure.lattice)

    real = camera.view_direction("a")
    reciprocal = camera.view_direction("a*")
    cosine = float(
        np.dot(real, reciprocal)
        / (np.linalg.norm(real) * np.linalg.norm(reciprocal))
    )
    assert np.degrees(np.arccos(cosine)) == pytest.approx(30.0, abs=0.5)

    # c is perpendicular to both in-plane axes here, so c and c* coincide —
    # the presets are not required to differ, only to be right.
    assert np.allclose(
        camera.view_direction("c") / np.linalg.norm(camera.view_direction("c")),
        camera.view_direction("c*") / np.linalg.norm(camera.view_direction("c*")),
    )


def test_orientation_preserves_zoom():
    """Switching orientation answers "which way", not "how close" — undoing
    the user's zoom would make the buttons unusable for looking closely at
    anything."""
    camera = Camera()
    camera.set_lattice(silicon().lattice)
    camera.scale(3.0)
    zoomed = camera.zoom
    camera.orient("b")
    assert camera.zoom == pytest.approx(zoomed)


def test_orientation_falls_back_to_cartesian_without_a_lattice():
    camera = Camera()
    camera.orient("c")
    assert np.allclose(camera.rotation @ camera.rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(camera.rotation) == pytest.approx(1.0)


# -- visualization style: Wireframe must actually differ from Ball-and-Stick --


def test_wireframe_draws_smaller_atoms_and_thinner_bonds_than_ball_and_stick(qtbot):
    """Wireframe and Ball-and-Stick used the same atom-radius scale and the
    same bond width, so selecting Wireframe changed nothing on screen. Both
    the atom radii and the bond width used to build the drawing must now
    actually differ between the two styles."""
    from vasp_analyzer.ui.pages.structure.viewer import StructureView, ViewOptions

    view = StructureView()
    qtbot.addWidget(view)
    view.set_structure(silicon())

    view.set_options(ViewOptions(style="Ball-and-Stick"))
    ball_radii = view._atom_radii()

    view.set_options(ViewOptions(style="Wireframe"))
    wire_radii = view._atom_radii()

    assert np.all(wire_radii < ball_radii)

    screen, depth, scale = view._project_sites()

    view.set_options(ViewOptions(style="Ball-and-Stick"))
    ball_bonds = view._bond_primitives(screen, depth, scale)

    view.set_options(ViewOptions(style="Wireframe"))
    wire_bonds = view._bond_primitives(screen, depth, scale)

    assert ball_bonds and wire_bonds
    # payload is (start, middle, colour, width) — width is index 3.
    assert wire_bonds[0][2][3] < ball_bonds[0][2][3]
