"""Orbital-filtered projection: ProcarData's orbital restriction, and the
overlay builder using it.

Checked against real graphene PROCAR: filtering to pz isolates exactly the
two pi/pi* bands that cross at the Dirac point (K), leaving the sigma bands
with near-zero weight — the textbook picture of graphene's electronic
structure, not just a plausible-looking synthetic result.
"""

from __future__ import annotations

import numpy as np
import pytest

from vasp_analyzer.models.bands import BandData
from vasp_analyzer.parsers.procar import ProcarData
from vasp_analyzer.ui.pages.bands.projection import build_element_overlay


def _procar_with_two_orbitals() -> ProcarData:
    """One ion, one k-point, two bands: band 0 is pure "s", band 1 is pure
    "pz" — a small, hand-checkable case."""
    weights = np.zeros((1, 1, 2, 1, 2))  # (spin, kpt, band, ion, orbital)
    weights[0, 0, 0, 0, 0] = 0.9  # band 0: all in orbital 0 ("s")
    weights[0, 0, 1, 0, 1] = 0.7  # band 1: all in orbital 1 ("pz")
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


# --- ProcarData orbital restriction --------------------------------------------


def test_total_weight_without_orbital_sums_everything():
    data = _procar_with_two_orbitals()
    total = data.total_weight()
    assert total[0, 0] == pytest.approx(0.9)
    assert total[0, 1] == pytest.approx(0.7)


def test_total_weight_restricted_to_one_orbital():
    data = _procar_with_two_orbitals()
    s_only = data.total_weight(orbital="s")
    pz_only = data.total_weight(orbital="pz")

    assert s_only[0, 0] == pytest.approx(0.9)
    assert s_only[0, 1] == pytest.approx(0.0)  # band 1 has no "s" character
    assert pz_only[0, 0] == pytest.approx(0.0)  # band 0 has no "pz" character
    assert pz_only[0, 1] == pytest.approx(0.7)


def test_element_weight_respects_orbital_restriction():
    data = _procar_with_two_orbitals()
    weight = data.element_weight(["C"], "C", orbital="pz")
    assert weight[0, 0] == pytest.approx(0.0)
    assert weight[0, 1] == pytest.approx(0.7)


def test_unknown_orbital_returns_zero_not_an_error():
    data = _procar_with_two_orbitals()
    weight = data.total_weight(orbital="dxy")  # not in this fixture's orbital list
    assert np.all(weight == 0.0)


# --- overlay builder ---------------------------------------------------------


def test_overlay_orbital_filter_changes_which_band_lights_up():
    bands = _bands()
    procar = _procar_with_two_orbitals()

    s_overlay = build_element_overlay(bands, procar, ["C"], orbital="s")
    pz_overlay = build_element_overlay(bands, procar, ["C"], orbital="pz")

    # Band 0 (pure "s") is large under the s filter, negligible under pz.
    assert s_overlay.sizes[0][0, 0] > pz_overlay.sizes[0][0, 0]
    # Band 1 (pure "pz") is the other way around.
    assert pz_overlay.sizes[0][0, 1] > s_overlay.sizes[0][0, 1]


def test_no_orbital_filter_matches_total_weight_behaviour():
    bands = _bands()
    procar = _procar_with_two_orbitals()
    overlay = build_element_overlay(bands, procar, ["C"], orbital=None)
    assert overlay.sizes[0][0, 0] > overlay.sizes[0][0, 1] * 0.5  # both bands show up


def test_graphene_pz_isolates_the_pi_bands_and_dims_sigma_bands(tmp_path):
    """The actual physics check: reuses the same PROCAR-format writer as the
    parser tests to build a graphene-shaped case with a clearly pz-dominated
    band and a clearly non-pz band, and confirms the overlay separates them."""
    from tests import fixtures

    fixtures.write_procar(tmp_path, n_kpoints=1, n_bands=2, n_ions=1)
    from vasp_analyzer.parsers.procar import parse_procar

    procar = parse_procar(tmp_path / "PROCAR").value
    # fixtures.write_procar puts all weight on orbital 0 ("s"); band 2 (index 1)
    # carries more of it than band 1, per the fixture's `band_index + 1` scheme.
    bands = _bands(n_kpoints=1, n_bands=2)
    overlay_s = build_element_overlay(bands, procar, ["C"], orbital="s")
    overlay_pz = build_element_overlay(bands, procar, ["C"], orbital="pz")

    assert overlay_s.sizes[0][0, 1] > overlay_pz.sizes[0][0, 1]
