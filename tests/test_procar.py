"""PROCAR parsing: dimensions, element/orbital weight extraction, and the
ragged-array bug the real graphene PROCAR exposed (the first table's rows
weren't truncated to the orbital count before orbital names were known)."""

from __future__ import annotations

import numpy as np
import pytest

from vasp_analyzer.parsers.procar import parse_procar, peek_procar_header

from . import fixtures


def test_dimensions_and_orbitals(tmp_path):
    fixtures.write_procar(tmp_path, n_kpoints=5, n_bands=4, n_ions=2)
    result = parse_procar(tmp_path / "PROCAR")

    assert result.ok
    data = result.value
    assert data.n_kpoints == 5
    assert data.n_bands == 4
    assert data.n_ions == 2
    assert data.n_spin == 1
    assert data.orbitals == ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "x2-y2"]
    assert data.ion_weights.shape == (1, 5, 4, 2, 9)


def test_first_table_is_not_ragged_against_later_ones(tmp_path):
    """Regression: the very first ion table (read before orbital names are
    known) used to keep its trailing per-ion 'tot' column, one longer than
    every subsequent table, which numpy refused to stack into an array."""
    fixtures.write_procar(tmp_path, n_kpoints=3, n_bands=2, n_ions=2)
    result = parse_procar(tmp_path / "PROCAR")
    assert result.ok
    # A regular, non-ragged shape is itself the assertion: parse_procar would
    # have failed with a ValueError from numpy otherwise.
    assert result.value.ion_weights.shape == (1, 3, 2, 2, 9)


def test_weight_pattern_matches_the_fixture(tmp_path):
    # fixtures.write_procar puts `band_index + 1` on ion 1's first orbital and
    # zero everywhere else, so the parsed array is exactly checkable.
    fixtures.write_procar(tmp_path, n_kpoints=2, n_bands=3, n_ions=2)
    data = parse_procar(tmp_path / "PROCAR").value

    for band in range(3):
        assert data.ion_weights[0, 0, band, 0, 0] == pytest.approx(band + 1)
        assert data.ion_weights[0, 0, band, 1, 0] == 0.0
        assert data.ion_weights[0, 0, band, 0, 1:].sum() == 0.0


def test_element_weight_sums_the_right_ions(tmp_path):
    fixtures.write_procar(tmp_path, n_kpoints=2, n_bands=3, n_ions=2)
    data = parse_procar(tmp_path / "PROCAR").value

    # Ion 1 (index 0) is carbon, ion 2 (index 1) is oxygen; only ion 0 carries
    # any weight in this fixture.
    carbon = data.element_weight(["C", "O"], "C")
    oxygen = data.element_weight(["C", "O"], "O")
    assert np.allclose(oxygen, 0.0)
    assert carbon[0, 2] == pytest.approx(3.0)  # band index 2 -> weight 3


def test_spin_polarised_procar_doubles_the_kpoint_blocks(tmp_path):
    fixtures.write_procar(tmp_path, n_kpoints=4, n_bands=2, n_ions=1, n_spin=2)
    result = parse_procar(tmp_path / "PROCAR")

    assert result.ok
    assert result.value.n_spin == 2
    assert result.value.ion_weights.shape == (2, 4, 2, 1, 9)


def test_truncated_procar_yields_a_warning_not_a_crash(tmp_path):
    fixtures.write_procar(tmp_path, n_kpoints=10, n_bands=4, truncate_after_band=7)
    result = parse_procar(tmp_path / "PROCAR")

    assert result.ok, "a truncated PROCAR must still yield what it contains"
    assert any(issue.code == "procar_truncated" for issue in result.issues)


def test_garbage_file_reports_error_without_raising(tmp_path):
    (tmp_path / "PROCAR").write_text("not a PROCAR file\n")
    result = parse_procar(tmp_path / "PROCAR")
    assert result.failed
    assert result.errors


def test_missing_file_reports_error(tmp_path):
    result = parse_procar(tmp_path / "PROCAR")
    assert result.failed
    assert result.errors[0].code == "procar_missing"


def test_peek_header_matches_full_parse(tmp_path):
    fixtures.write_procar(tmp_path, n_kpoints=6, n_bands=5, n_ions=3)
    header = peek_procar_header(tmp_path / "PROCAR")
    full = parse_procar(tmp_path / "PROCAR").value

    assert header.n_kpoints == full.n_kpoints
    assert header.n_bands == full.n_bands
    assert header.n_ions == full.n_ions


# --- non-collinear (LSORBIT=T) spin texture ---------------------------------------
#
# Checked against a real graphene PROCAR run with LSORBIT=.TRUE.: bands 1/2
# (a near-degenerate spin-orbit-split pair at Gamma) came back with mz values
# of -0.570 and +0.570 — equal and opposite, exactly the textbook signature
# of a Kramers pair. The synthetic fixture mirrors that layout: one "ion..."
# header per band, four stacked ion tables (charge, mx, my, mz) with no
# repeated header and no blank line between them.


def test_noncollinear_procar_is_detected(tmp_path):
    fixtures.write_procar_noncollinear(tmp_path, n_kpoints=3, n_bands=4, n_ions=2)
    result = parse_procar(tmp_path / "PROCAR")

    assert result.ok
    data = result.value
    assert data.is_noncollinear
    assert data.spin_texture is not None
    assert data.spin_texture.shape == (3, 4, 2, 9, 3)
    # The charge (total-character) table still parses exactly as before —
    # capturing mx/my/mz must not disturb the existing ion_weights path.
    assert data.ion_weights.shape == (1, 3, 4, 2, 9)
    assert data.n_spin == 1


def test_noncollinear_charge_weights_match_the_fixture(tmp_path):
    fixtures.write_procar_noncollinear(tmp_path, n_kpoints=2, n_bands=3, n_ions=2)
    data = parse_procar(tmp_path / "PROCAR").value

    for band in range(3):
        assert data.ion_weights[0, 0, band, 0, 0] == pytest.approx(band + 1)


def test_spin_projection_alternates_sign_by_band_as_the_fixture_defines(tmp_path):
    fixtures.write_procar_noncollinear(tmp_path, n_kpoints=2, n_bands=4, n_ions=2)
    data = parse_procar(tmp_path / "PROCAR").value

    sz = data.spin_projection(axis=2)
    assert sz.shape == (2, 4)
    assert sz[0, 0] == pytest.approx(0.5)
    assert sz[0, 1] == pytest.approx(-0.5)
    assert sz[0, 2] == pytest.approx(0.5)
    assert sz[0, 3] == pytest.approx(-0.5)

    # mx and my are zero throughout the fixture.
    assert np.allclose(data.spin_projection(axis=0), 0.0)
    assert np.allclose(data.spin_projection(axis=1), 0.0)


def test_spin_projection_respects_orbital_restriction(tmp_path):
    # All of the fixture's spin weight sits on ion 1's orbital 0 ("s"); a
    # filter that excludes "s" should zero it out.
    fixtures.write_procar_noncollinear(tmp_path, n_kpoints=1, n_bands=2, n_ions=2)
    data = parse_procar(tmp_path / "PROCAR").value

    assert data.spin_projection(axis=2, orbitals=["s"])[0, 0] == pytest.approx(0.5)
    assert data.spin_projection(axis=2, orbitals=["pz"])[0, 0] == pytest.approx(0.0)


def test_collinear_procar_has_no_spin_texture(tmp_path):
    fixtures.write_procar(tmp_path, n_kpoints=3, n_bands=2, n_ions=2)
    data = parse_procar(tmp_path / "PROCAR").value

    assert not data.is_noncollinear
    assert data.spin_texture is None
    assert data.spin_projection() is None


def test_real_graphene_procar_has_consistent_kramers_pair_spin(tmp_path):
    """The actual physics check against real data: bands 1 and 2 at k-point 1
    (Gamma) are the lowest spin-orbit-split Kramers pair — near-degenerate in
    energy, equal and opposite in z-spin."""
    from pathlib import Path

    real = Path("Graphene/band/PROCAR")
    if not real.exists():
        pytest.skip("requires the local Graphene/ data")

    data = parse_procar(real).value
    assert data.is_noncollinear
    sz = data.spin_projection(axis=2)
    assert sz[0, 0] == pytest.approx(-sz[0, 1], abs=1e-6)
    assert abs(sz[0, 0]) > 0.1  # a real, non-trivial spin splitting
