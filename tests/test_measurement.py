"""The Structure page's measurement tool: distance/angle math, and the
click-through interaction that builds a 2- or 3-atom chain.

Checked against two known physical values while building this: graphene's
C-C bond comes out to 1.4258 A (accepted value ~1.42 A) and silicon's
Si-Si-Si bond angle comes out to 109.47 degrees — the exact tetrahedral
angle diamond-cubic silicon's sp3 coordination requires.
"""

from __future__ import annotations

import numpy as np
import pytest

from vasp_analyzer.models.structure import StructureData
from vasp_analyzer.ui.pages.structure.viewer import (
    MAX_MEASURE_POINTS,
    Measurement,
    StructureView,
    _measure,
)


# --- pure distance/angle math --------------------------------------------------


def test_distance_between_two_points():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.5, 0.0, 0.0])
    result = _measure([a, b], ["C", "C"])
    assert result.kind == "distance"
    assert result.value == pytest.approx(1.5)
    assert result.symbols == ["C", "C"]


def test_graphene_bond_length_matches_the_known_value():
    # The two carbon sites from the real graphene POSCAR used throughout
    # this project's development.
    lattice = np.array(
        [[2.4694910408650514, 0.0, 0.0],
         [-1.2347455204325257, 2.1386419486320141, 0.0],
         [0.0, 0.0, 14.4682012468615859]]
    )
    frac = np.array([[0.0, 0.0, 0.25], [2 / 3, 1 / 3, 0.25]])
    cart = frac @ lattice
    result = _measure([cart[0], cart[1]], ["C", "C"])
    assert result.value == pytest.approx(1.4258, abs=5e-4)


def test_right_angle():
    a = np.array([1.0, 0.0, 0.0])
    vertex = np.array([0.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    result = _measure([a, vertex, c], ["O", "Si", "O"])
    assert result.kind == "angle"
    assert result.value == pytest.approx(90.0)


def test_tetrahedral_angle_matches_diamond_silicon():
    # Four directions to alternating corners of a cube from its centre —
    # the classic tetrahedral arrangement diamond-cubic silicon adopts.
    centre = np.array([0.0, 0.0, 0.0])
    d1 = np.array([1.0, 1.0, 1.0])
    d2 = np.array([1.0, -1.0, -1.0])
    result = _measure([d1, centre, d2], ["Si", "Si", "Si"])
    assert result.value == pytest.approx(109.4712, abs=1e-3)


def test_collinear_points_do_not_crash():
    """A degenerate angle (zero-length vector) must not raise or produce NaN."""
    a = np.array([0.0, 0.0, 0.0])
    result = _measure([a, a, a], ["C", "C", "C"])
    assert result is None


def test_single_point_yields_no_measurement():
    assert _measure([np.zeros(3)], ["C"]) is None


# --- StructureView interaction --------------------------------------------------


def silicon_structure() -> StructureData:
    a = 5.4667335224494948
    frac = np.array(
        [
            [0.00, 0.00, 0.00], [0.00, 0.50, 0.50],
            [0.50, 0.00, 0.50], [0.50, 0.50, 0.00],
            [0.25, 0.25, 0.25], [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75], [0.75, 0.75, 0.25],
        ]
    )
    return StructureData(lattice=np.eye(3) * a, species=["Si"] * 8, frac_coords=frac)


@pytest.fixture
def view(qapp, qtbot):
    v = StructureView()
    v.resize(400, 400)
    qtbot.addWidget(v)
    v.set_structure(silicon_structure())
    v.set_mode(StructureView.MODE_MEASURE)
    return v


def test_two_clicks_emit_a_distance(view, qtbot):
    results = []
    view.measurementChanged.connect(results.append)

    view._add_measure_point(0)
    assert results == [None] or results == []  # first point alone: no result yet
    view._add_measure_point(1)

    assert len(results) >= 1
    assert results[-1] is not None
    assert results[-1].kind == "distance"


def test_three_clicks_emit_an_angle(view, qtbot):
    results = []
    view.measurementChanged.connect(results.append)

    for site in (0, 1, 2):
        view._add_measure_point(site)

    assert results[-1].kind == "angle"


def test_a_fourth_click_restarts_the_chain(view):
    for site in range(MAX_MEASURE_POINTS + 1):
        view._add_measure_point(site)
    assert len(view._measure_points) == 1
    assert view._measure_points == [MAX_MEASURE_POINTS]


def test_clearing_emits_none(view):
    view._add_measure_point(0)
    view._add_measure_point(1)

    results = []
    view.measurementChanged.connect(results.append)
    view._clear_measurement()

    assert results == [None]
    assert view._measure_points == []


def test_switching_away_from_measure_mode_clears_the_chain(view):
    view._add_measure_point(0)
    view._add_measure_point(1)
    assert view._measure_points

    view.set_mode(StructureView.MODE_SELECT)
    assert view._measure_points == []


def test_loading_a_new_structure_clears_any_pending_measurement(view):
    view._add_measure_point(0)
    view._add_measure_point(1)

    view.set_structure(silicon_structure())
    assert view._measure_points == []
