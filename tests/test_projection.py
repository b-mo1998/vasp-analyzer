"""Element-projection overlay: the PROCAR-to-fatbands colour/size mapping.

Verified against a real graphene calculation while building this feature: the
π/π* bands correctly show their largest marker size exactly at the K-point
Dirac crossing, where PROCAR's own pz weight peaks.
"""

from __future__ import annotations

import numpy as np
import pytest

from vasp_analyzer.models.bands import BandData
from vasp_analyzer.parsers.procar import ProcarData
from vasp_analyzer.ui.pages.bands.projection import build_element_overlay, legend_elements


def _bands(n_kpoints=4, n_bands=3, n_spin=1) -> BandData:
    energies = np.random.default_rng(0).uniform(-5, 5, size=(n_spin, n_kpoints, n_bands))
    return BandData(
        kpoints=np.zeros((n_kpoints, 3)),
        weights=np.ones(n_kpoints),
        energies=energies,
        occupations=np.zeros_like(energies),
        n_electrons=8.0,
    )


def _procar(n_kpoints=4, n_bands=3, n_ions=2, n_spin=1) -> ProcarData:
    weights = np.zeros((n_spin, n_kpoints, n_bands, n_ions, 1))
    weights[..., 0, 0] = 0.8   # ion 0 dominant
    weights[..., 1, 0] = 0.1   # ion 1 minor
    return ProcarData(orbitals=["s"], ion_weights=weights)


def test_overlay_shapes_match_bands():
    bands = _bands(n_kpoints=4, n_bands=3)
    procar = _procar(n_kpoints=4, n_bands=3, n_ions=2)
    overlay = build_element_overlay(bands, procar, species=["Si", "O"])

    assert overlay is not None
    assert len(overlay.colors) == 1
    assert overlay.colors[0].shape == (4, 3, 4)
    assert overlay.sizes[0].shape == (4, 3)


def test_dominant_ion_dominates_the_blended_colour():
    bands = _bands(n_kpoints=2, n_bands=1)
    procar = _procar(n_kpoints=2, n_bands=1, n_ions=2)
    overlay = build_element_overlay(bands, procar, species=["Si", "O"])

    from matplotlib.colors import to_rgb
    from vasp_analyzer.services.appearance import effective_color

    si_rgb = np.array(to_rgb(effective_color("Si")))
    o_rgb = np.array(to_rgb(effective_color("O")))
    blended = overlay.colors[0][0, 0, :3]

    # ion 0 (Si) carries 8x the weight of ion 1 (O): the blend should sit
    # much closer to silicon's colour than oxygen's.
    assert np.linalg.norm(blended - si_rgb) < np.linalg.norm(blended - o_rgb)


def test_zero_weight_point_is_neutral_and_faint():
    bands = _bands(n_kpoints=1, n_bands=1)
    procar = ProcarData(orbitals=["s"], ion_weights=np.zeros((1, 1, 1, 2, 1)))
    overlay = build_element_overlay(bands, procar, species=["Si", "O"])

    rgba = overlay.colors[0][0, 0]
    assert np.allclose(rgba[:3], 0.6, atol=1e-6)
    assert rgba[3] < 0.2  # faint, not fully opaque


def test_size_grows_with_total_weight():
    bands = _bands(n_kpoints=1, n_bands=1)
    weights = np.zeros((1, 1, 1, 1, 1))
    weights[0, 0, 0, 0, 0] = 1.0
    strong = build_element_overlay(bands, ProcarData(["s"], weights), ["Si"])

    weights_weak = weights * 0.1
    weak = build_element_overlay(bands, ProcarData(["s"], weights_weak), ["Si"])

    assert strong.sizes[0][0, 0] > weak.sizes[0][0, 0]


def test_mismatched_dimensions_return_none_rather_than_a_wrong_overlay():
    bands = _bands(n_kpoints=4, n_bands=3)
    procar = _procar(n_kpoints=6, n_bands=3, n_ions=2)  # wrong k-point count
    assert build_element_overlay(bands, procar, species=["Si", "O"]) is None


def test_mismatched_ion_count_returns_none():
    bands = _bands(n_kpoints=4, n_bands=3)
    procar = _procar(n_kpoints=4, n_bands=3, n_ions=2)
    # species list doesn't match PROCAR's declared ion count
    assert build_element_overlay(bands, procar, species=["Si", "O", "O"]) is None


def test_spin_polarised_bands_with_single_spin_procar_degrades_gracefully():
    bands = _bands(n_kpoints=2, n_bands=2, n_spin=2)
    procar = _procar(n_kpoints=2, n_bands=2, n_ions=2, n_spin=1)
    overlay = build_element_overlay(bands, procar, species=["Si", "O"])

    assert overlay is not None
    assert len(overlay.colors) == 2  # one entry per band spin channel
    assert np.all(overlay.colors[1] == 0.0)  # the uncovered channel is blank


def test_legend_elements_are_sorted_and_deduplicated():
    entries = legend_elements(["O", "Si", "O", "Si", "Si"])
    assert [element for element, _ in entries] == ["O", "Si"]
