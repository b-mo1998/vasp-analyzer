"""Spin-projected band overlay: build_spin_overlay's colour mapping for both
a non-collinear (mx/my/mz) PROCAR and a collinear ISPIN=2 one.

Checked against the real noncollinear graphene PROCAR: the lowest
spin-orbit-split Kramers pair at Gamma (bands 1/2) comes back as two visually
opposite colours, not the same flat colour the old element-only overlay gave
every point in this single-element system.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vasp_analyzer.models.bands import BandData
from vasp_analyzer.parsers.procar import ProcarData, parse_procar
from vasp_analyzer.ui.pages.bands.projection import SPIN_CMAP, build_spin_overlay

from . import fixtures


def _bands(n_kpoints=1, n_bands=4, n_spin=1) -> BandData:
    energies = np.zeros((n_spin, n_kpoints, n_bands))
    return BandData(
        kpoints=np.zeros((n_kpoints, 3)),
        weights=np.ones(n_kpoints),
        energies=energies,
        occupations=np.zeros_like(energies),
        n_electrons=1.0,
    )


# --- non-collinear (mx/my/mz) -------------------------------------------------


def test_spin_overlay_colours_opposite_bands_oppositely(tmp_path):
    fixtures.write_procar_noncollinear(tmp_path, n_kpoints=1, n_bands=4, n_ions=2)
    procar = parse_procar(tmp_path / "PROCAR").value
    bands = _bands(n_kpoints=1, n_bands=4)

    result = build_spin_overlay(bands, procar, axis=2)
    assert result is not None
    overlay, vmax = result

    assert vmax == pytest.approx(0.5)
    # Band 0 has mz=+0.5 (the fixture's even-band sign), band 1 has mz=-0.5.
    color0 = overlay.colors[0][0, 0]
    color1 = overlay.colors[0][0, 1]
    np.testing.assert_allclose(color0, SPIN_CMAP(1.0), atol=1e-6)  # fully "up" -> blue end
    np.testing.assert_allclose(color1, SPIN_CMAP(0.0), atol=1e-6)  # fully "down" -> red end


def test_spin_overlay_zero_moment_is_the_colormap_midpoint(tmp_path):
    fixtures.write_procar_noncollinear(tmp_path, n_kpoints=1, n_bands=4, n_ions=2)
    procar = parse_procar(tmp_path / "PROCAR").value
    bands = _bands(n_kpoints=1, n_bands=4)

    # mx is zero everywhere in the fixture: colouring by x should sit exactly
    # at the colormap's midpoint (white) for every band.
    result = build_spin_overlay(bands, procar, axis=0)
    assert result is not None
    overlay, _vmax = result
    np.testing.assert_allclose(overlay.colors[0][0, 0], SPIN_CMAP(0.5), atol=1e-6)


def test_spin_overlay_none_for_dimension_mismatch(tmp_path):
    fixtures.write_procar_noncollinear(tmp_path, n_kpoints=1, n_bands=4, n_ions=2)
    procar = parse_procar(tmp_path / "PROCAR").value
    mismatched_bands = _bands(n_kpoints=1, n_bands=99)
    assert build_spin_overlay(mismatched_bands, procar) is None


def test_spin_overlay_none_without_spin_data(tmp_path):
    fixtures.write_procar(tmp_path, n_kpoints=1, n_bands=2, n_ions=2)
    procar = parse_procar(tmp_path / "PROCAR").value
    bands = _bands(n_kpoints=1, n_bands=2)
    assert build_spin_overlay(bands, procar) is None


# --- collinear ISPIN=2 (up minus down) -----------------------------------------


def test_spin_overlay_collinear_uses_up_minus_down():
    # Two spin channels, one ion, one orbital: band 0 fully spin-up
    # (channel 0 carries all the weight), band 1 fully spin-down.
    weights = np.zeros((2, 1, 2, 1, 1))  # (spin, kpt, band, ion, orbital)
    weights[0, 0, 0, 0, 0] = 1.0  # band 0: up only
    weights[1, 0, 1, 0, 0] = 1.0  # band 1: down only
    procar = ProcarData(orbitals=["s"], ion_weights=weights)
    bands = _bands(n_kpoints=1, n_bands=2, n_spin=2)

    result = build_spin_overlay(bands, procar)
    assert result is not None
    overlay, vmax = result
    assert vmax == pytest.approx(1.0)
    np.testing.assert_allclose(overlay.colors[0][0, 0], SPIN_CMAP(1.0), atol=1e-6)
    np.testing.assert_allclose(overlay.colors[0][0, 1], SPIN_CMAP(0.0), atol=1e-6)


# --- real data ------------------------------------------------------------------


def test_real_graphene_kramers_pair_gets_opposite_colours():
    real = Path("Graphene/band/PROCAR")
    real_eigenval = Path("Graphene/band/EIGENVAL")
    if not real.exists() or not real_eigenval.exists():
        pytest.skip("requires the local Graphene/ data")

    from vasp_analyzer.parsers.eigenval import parse_eigenval

    procar = parse_procar(real).value
    bands = parse_eigenval(real_eigenval).value
    assert procar.is_noncollinear

    result = build_spin_overlay(bands, procar, axis=2)
    assert result is not None
    overlay, vmax = result
    assert vmax > 0.1

    color0 = overlay.colors[0][0, 0]
    color1 = overlay.colors[0][0, 1]
    # The two Kramers partners land on visually distinct halves of the scale.
    assert (color0[0] > color0[2]) != (color1[0] > color1[2])  # one redder, one bluer
