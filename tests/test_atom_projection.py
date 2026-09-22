"""Per-atom band projection: ProcarData.atom_weight and its overlay builder.

The atom-ID counterpart to element/orbital projection — a Te dopant among a
sea of carbons is a specific ion row, not a species a filter can isolate.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.colors import to_rgb

from vasp_analyzer.models.bands import BandData
from vasp_analyzer.parsers.procar import ProcarData
from vasp_analyzer.ui.pages.bands.projection import (
    ATOM_PROJECTION_COLOR,
    NEUTRAL_RGB,
    build_atom_overlay,
)


def _procar_three_ions() -> ProcarData:
    """Three ions, one k-point, two bands: band 0 lives entirely on ion 1
    ("s"), band 1 entirely on ion 2 ("pz") — hand-checkable weights."""
    weights = np.zeros((1, 1, 2, 3, 2))  # (spin, kpt, band, ion, orbital)
    weights[0, 0, 0, 1, 0] = 0.8  # band 0: ion 1, orbital "s"
    weights[0, 0, 1, 2, 1] = 0.6  # band 1: ion 2, orbital "pz"
    return ProcarData(orbitals=["s", "pz"], ion_weights=weights)


def _bands(n_kpoints=1, n_bands=2, n_spin=1) -> BandData:
    energies = np.zeros((n_spin, n_kpoints, n_bands))
    return BandData(
        kpoints=np.zeros((n_kpoints, 3)),
        weights=np.ones(n_kpoints),
        energies=energies,
        occupations=np.zeros_like(energies),
        n_electrons=1.0,
    )


# --- ProcarData.atom_weight ---------------------------------------------------


def test_atom_weight_isolates_exactly_the_requested_ion():
    data = _procar_three_ions()
    weight = data.atom_weight([1])
    assert weight[0, 0] == pytest.approx(0.8)
    assert weight[0, 1] == pytest.approx(0.0)


def test_atom_weight_sums_multiple_ions():
    data = _procar_three_ions()
    weight = data.atom_weight([1, 2])
    assert weight[0, 0] == pytest.approx(0.8)
    assert weight[0, 1] == pytest.approx(0.6)


def test_atom_weight_respects_orbital_restriction():
    data = _procar_three_ions()
    # Ion 2 carries weight only in "pz"; restricting to "s" must zero it.
    weight = data.atom_weight([2], orbitals=["s"])
    assert weight[0, 1] == pytest.approx(0.0)
    weight_pz = data.atom_weight([2], orbitals=["pz"])
    assert weight_pz[0, 1] == pytest.approx(0.6)


def test_atom_weight_ignores_out_of_range_rows():
    data = _procar_three_ions()
    weight = data.atom_weight([1, 99])
    assert weight[0, 0] == pytest.approx(0.8)


def test_atom_weight_empty_selection_is_all_zero():
    data = _procar_three_ions()
    weight = data.atom_weight([])
    assert np.all(weight == 0.0)


# --- build_atom_overlay ---------------------------------------------------------


def test_atom_overlay_colours_the_band_carrying_the_selected_atom():
    bands = _bands()
    procar = _procar_three_ions()
    overlay = build_atom_overlay(bands, procar, rows=[1], orbitals=None)

    assert overlay is not None
    band0_rgb = overlay.colors[0][0, 0, :3]
    band1_rgb = overlay.colors[0][0, 1, :3]
    np.testing.assert_allclose(band0_rgb, to_rgb(ATOM_PROJECTION_COLOR), atol=1e-6)
    np.testing.assert_allclose(band1_rgb, NEUTRAL_RGB, atol=1e-6)


def test_atom_overlay_marker_size_scales_with_weight():
    bands = _bands()
    procar = _procar_three_ions()
    overlay = build_atom_overlay(bands, procar, rows=[1], orbitals=None)

    # Band 0 carries real weight on ion 1 (0.8), band 1 carries none.
    assert overlay.sizes[0][0, 0] > overlay.sizes[0][0, 1]


def test_atom_overlay_requires_a_non_empty_row_selection():
    bands = _bands()
    procar = _procar_three_ions()
    assert build_atom_overlay(bands, procar, rows=[], orbitals=None) is None


def test_atom_overlay_none_when_procar_shape_mismatches_bands():
    bands = _bands(n_kpoints=5)
    procar = _procar_three_ions()
    assert build_atom_overlay(bands, procar, rows=[1], orbitals=None) is None
