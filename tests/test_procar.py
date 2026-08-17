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
