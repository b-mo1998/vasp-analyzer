"""Parser behaviour, including the ways real files go wrong."""

from __future__ import annotations

import numpy as np
import pytest

from vasp_analyzer.parsers.doscar import parse_doscar, peek_doscar_header
from vasp_analyzer.parsers.eigenval import parse_eigenval, peek_eigenval_header
from vasp_analyzer.parsers.incar import (
    as_bool,
    describe_functional,
    describe_spin,
    parse_incar,
)
from vasp_analyzer.parsers.kpoints import parse_kpoints
from vasp_analyzer.parsers.poscar import parse_poscar

from . import fixtures


# --- EIGENVAL -------------------------------------------------------------------


def test_eigenval_shapes(tmp_path):
    fixtures.write_eigenval(tmp_path, n_kpoints=10, n_bands=5)
    result = parse_eigenval(tmp_path / "EIGENVAL")

    assert result.ok
    bands = result.value
    assert bands.energies.shape == (1, 10, 5)
    assert bands.n_electrons == 8.0
    assert not bands.is_spin_polarized


def test_eigenval_spin_polarised_layout(tmp_path):
    fixtures.write_eigenval(tmp_path, n_kpoints=6, n_bands=4, n_spin=2)
    result = parse_eigenval(tmp_path / "EIGENVAL")

    assert result.ok
    bands = result.value
    assert bands.n_spin == 2
    assert bands.is_spin_polarized
    # Spin-down was written 0.15 eV above spin-up.
    assert np.allclose(bands.energies[1] - bands.energies[0], 0.15, atol=1e-6)


def test_eigenval_truncated_file_yields_partial_data_and_a_warning(tmp_path):
    fixtures.write_eigenval(tmp_path, n_kpoints=12, n_bands=5, truncate_after=7)
    result = parse_eigenval(tmp_path / "EIGENVAL")

    assert result.ok, "a truncated file must still yield what it contains"
    assert result.value.n_kpoints == 7
    assert any(issue.code == "eigenval_truncated" for issue in result.issues)


def test_eigenval_garbage_reports_error_without_raising(tmp_path):
    (tmp_path / "EIGENVAL").write_text("not an eigenval file\n")
    result = parse_eigenval(tmp_path / "EIGENVAL")

    assert result.failed
    assert result.errors


def test_eigenval_missing_file_reports_error(tmp_path):
    result = parse_eigenval(tmp_path / "EIGENVAL")
    assert result.failed
    assert result.errors[0].code == "eigenval_missing"


def test_peek_header_matches_full_parse(tmp_path):
    fixtures.write_eigenval(tmp_path, n_kpoints=9, n_bands=7, n_electrons=14)
    header = peek_eigenval_header(tmp_path / "EIGENVAL")
    full = parse_eigenval(tmp_path / "EIGENVAL").value

    assert header.n_kpoints == full.n_kpoints
    assert header.n_bands == full.n_bands
    assert header.n_electrons == full.n_electrons


# --- DOSCAR ---------------------------------------------------------------------


def test_doscar_total_dos(tmp_path):
    fixtures.write_doscar(tmp_path, n_points=40, e_fermi=5.3068)
    result = parse_doscar(tmp_path / "DOSCAR")

    assert result.ok
    dos = result.value
    assert dos.n_points == 40
    assert dos.e_fermi == pytest.approx(5.3068)
    assert dos.total.shape == (1, 40)


def test_doscar_spin_polarised(tmp_path):
    fixtures.write_doscar(tmp_path, n_points=30, n_spin=2)
    result = parse_doscar(tmp_path / "DOSCAR")

    assert result.ok
    assert result.value.is_spin_polarized
    assert result.value.total.shape == (2, 30)


def test_malformed_doscar_fails_cleanly(tmp_path):
    fixtures.write_doscar(tmp_path, malformed=True)
    result = parse_doscar(tmp_path / "DOSCAR")

    assert result.failed
    assert result.errors
    assert peek_doscar_header(tmp_path / "DOSCAR") is None


# --- POSCAR ---------------------------------------------------------------------


def test_poscar_geometry(tmp_path):
    fixtures.write_poscar(tmp_path, atoms=4)
    result = parse_poscar(tmp_path / "POSCAR")

    assert result.ok
    structure = result.value
    assert structure.n_atoms == 4
    assert structure.formula == "Si"
    assert structure.lattice_parameters[0] == pytest.approx(5.4667335, abs=1e-6)
    assert structure.volume == pytest.approx(5.4667335**3, rel=1e-6)


def test_poscar_vasp4_without_species_warns_but_parses(tmp_path):
    (tmp_path / "POSCAR").write_text(
        "old style\n1.0\n"
        " 4.0 0.0 0.0\n 0.0 4.0 0.0\n 0.0 0.0 4.0\n"
        " 2\nDirect\n 0.0 0.0 0.0\n 0.5 0.5 0.5\n"
    )
    result = parse_poscar(tmp_path / "POSCAR")

    assert result.ok
    assert result.value.n_atoms == 2
    assert any(issue.code == "poscar_no_species" for issue in result.issues)


def test_poscar_cartesian_coordinates_are_converted(tmp_path):
    (tmp_path / "POSCAR").write_text(
        "cartesian\n1.0\n"
        " 4.0 0.0 0.0\n 0.0 4.0 0.0\n 0.0 0.0 4.0\n"
        " Si\n 1\nCartesian\n 2.0 2.0 2.0\n"
    )
    result = parse_poscar(tmp_path / "POSCAR")

    assert result.ok
    assert np.allclose(result.value.frac_coords[0], [0.5, 0.5, 0.5])


def test_poscar_truncated_coordinates_reports_error(tmp_path):
    (tmp_path / "POSCAR").write_text(
        "short\n1.0\n"
        " 4.0 0.0 0.0\n 0.0 4.0 0.0\n 0.0 0.0 4.0\n"
        " Si\n 3\nDirect\n 0.0 0.0 0.0\n"
    )
    result = parse_poscar(tmp_path / "POSCAR")
    assert result.failed


# --- INCAR / KPOINTS ---------------------------------------------------------------


def test_incar_tags_and_comments(tmp_path):
    (tmp_path / "INCAR").write_text(
        "SYSTEM = Si\nISTART=1 ; ICHARG=11  # read from CHGCAR\n"
        "METAGGA = SCAN   ! comment\nLSORBIT = .TRUE.\n"
    )
    result = parse_incar(tmp_path / "INCAR")

    assert result.ok
    tags = result.value
    assert tags["SYSTEM"] == "Si"
    assert tags["ISTART"] == "1"
    assert tags["ICHARG"] == "11"
    assert tags["METAGGA"] == "SCAN"
    assert as_bool(tags["LSORBIT"]) is True


def test_functional_description_never_guesses():
    assert describe_functional({"METAGGA": "SCAN"}) == "SCAN"
    assert describe_functional({"GGA": "PE"}) == "PBE"
    assert describe_functional({"GGA": "PE", "LSORBIT": ".TRUE."}) == "PBE + SOC"
    # No exchange-correlation tag at all: report nothing rather than "PBE".
    assert describe_functional({"ENCUT": "520"}) is None


def test_spin_description():
    assert describe_spin({"ISPIN": "2"}) == "Spin-polarised"
    assert describe_spin({"LSORBIT": ".TRUE."}) == "Non-collinear"
    assert describe_spin({}) is None


def test_kpoints_mesh(tmp_path):
    fixtures.write_kpoints_mesh(tmp_path, mesh=(8, 8, 6))
    result = parse_kpoints(tmp_path / "KPOINTS")

    assert result.ok
    assert result.value.mesh_label == "8×8×6"
    assert not result.value.is_line_mode


def test_kpoints_line_mode_labels(tmp_path):
    fixtures.write_kpoints_line(tmp_path, divisions=4)
    result = parse_kpoints(tmp_path / "KPOINTS")

    assert result.ok
    spec = result.value
    assert spec.is_line_mode
    assert spec.n_divisions == 4
    assert spec.path_labels[0] == "Γ"
    assert "X" in spec.path_labels
