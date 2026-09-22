"""BandData.band_gap(): value and direct/indirect classification.

Checked against two real calculations: silicon comes back indirect (its
textbook electronic structure — VBM at Gamma, CBM elsewhere), graphene comes
back with a near-zero gap (the Dirac point is never sampled exactly on a
discrete k-point mesh, so a tiny but nonzero value is the physically correct
thing for a finite mesh to report, not a bug).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vasp_analyzer.models.bands import BandData
from vasp_analyzer.parsers.doscar import parse_doscar
from vasp_analyzer.parsers.eigenval import parse_eigenval

SILICON_DIR = Path("Silicon Files")
GRAPHENE_DIR = Path("Graphene/band")

# These two real calculation directories are the user's own data — gitignored,
# not part of the repository, and won't exist for anyone else who clones it or
# in CI. The tests that use them are real-physics validations worth keeping
# (they're what actually caught that the direct/indirect logic behaves
# correctly on non-synthetic data), but must skip cleanly rather than fail
# when that data isn't present, the way every other test in this suite — all
# built on synthetic fixtures — does not need to.
needs_silicon = pytest.mark.skipif(
    not (SILICON_DIR / "EIGENVAL").exists(), reason="requires the local Silicon Files/ data"
)
needs_graphene = pytest.mark.skipif(
    not (GRAPHENE_DIR / "EIGENVAL").exists(), reason="requires the local Graphene/ data"
)


def _bands(energies: np.ndarray) -> BandData:
    n_spin, n_kpoints, n_bands = energies.shape
    return BandData(
        kpoints=np.zeros((n_kpoints, 3)),
        weights=np.ones(n_kpoints),
        energies=energies,
        occupations=np.zeros_like(energies),
        n_electrons=1.0,
    )


# --- synthetic, hand-checkable cases ---------------------------------------------


def test_direct_gap_at_a_single_kpoint():
    # One k-point, VBM and CBM both there by construction: direct.
    energies = np.array([[[-1.0, 1.0]]])  # (spin=1, kpt=1, bands=2)
    result = _bands(energies).band_gap(e_fermi=0.0)
    assert result is not None
    assert result.gap == pytest.approx(2.0)
    assert result.is_direct


def test_indirect_gap_across_two_kpoints():
    # k-point 0 has the higher occupied state (VBM); k-point 1 has the lower
    # unoccupied state (CBM) — different k-points, so indirect.
    energies = np.array(
        [[[-0.5, 3.0], [-2.0, 1.0]]]  # (spin=1, kpt=2, bands=2)
    )
    result = _bands(energies).band_gap(e_fermi=0.0)
    assert result is not None
    assert not result.is_direct
    assert result.vbm_kpoint == 0
    assert result.cbm_kpoint == 1
    assert result.gap == pytest.approx(1.5)  # cbm(1.0) - vbm(-0.5)


def test_metallic_when_vbm_and_cbm_both_sit_at_the_fermi_level():
    # By construction, the occupied max is always <= e_fermi < the unoccupied
    # min, so the gap can never come out negative here — a metal can only
    # show up as a *near-zero* gap, both extrema pinned right at E_F, which is
    # exactly what a real metal's Fermi surface looks like. The two values
    # are kept comfortably outside band_gap()'s own e_fermi +/- 1e-6
    # occupancy tolerance, so the test exercises the *gap* threshold
    # specifically, not that unrelated fudge factor.
    energies = np.array([[[-2e-5, 3.0], [-5.0, 2e-5]]])
    result = _bands(energies).band_gap(e_fermi=0.0)
    assert result is None


def test_fully_occupied_returns_none():
    energies = np.array([[[-2.0, -1.0]]])
    assert _bands(energies).band_gap(e_fermi=10.0) is None


def test_fully_unoccupied_returns_none():
    energies = np.array([[[1.0, 2.0]]])
    assert _bands(energies).band_gap(e_fermi=-10.0) is None


# --- real data --------------------------------------------------------------------


@needs_silicon
def test_silicon_gap_is_indirect():
    bands = parse_eigenval(SILICON_DIR / "EIGENVAL").value
    dos = parse_doscar(SILICON_DIR / "DOSCAR").value
    result = bands.band_gap(dos.e_fermi)

    assert result is not None
    assert not result.is_direct, "silicon's gap is textbook indirect"
    assert 0.0 < result.gap < 3.0  # a physically sane range, not a specific DFT value


@needs_graphene
def test_graphene_gap_is_near_zero():
    bands = parse_eigenval(GRAPHENE_DIR / "EIGENVAL").value
    dos = parse_doscar(GRAPHENE_DIR / "DOSCAR").value
    result = bands.band_gap(dos.e_fermi)

    # A finite k-point mesh essentially never samples the Dirac point exactly,
    # so a small nonzero gap is the physically correct result here, not a bug
    # — but it must be small, reflecting the near-degeneracy at K.
    assert result is not None
    assert result.gap < 0.1
