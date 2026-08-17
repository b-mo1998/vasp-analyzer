"""Capability detection across the specified partial-calculation scenarios.

These are the cases the application must never crash on. Each one asserts both
what works and — just as importantly — that the unavailable features report a
reason instead of failing.
"""

from __future__ import annotations

import pytest

from vasp_analyzer.models.capabilities import Capability
from vasp_analyzer.models.files import FileStatus
from vasp_analyzer.services.discovery import discover

from . import fixtures


@pytest.fixture
def calc_dir(tmp_path):
    return tmp_path


def test_case_a_everything_available(calc_dir):
    fixtures.case_a_complete(calc_dir)
    calc = discover(calc_dir)

    assert calc.can(Capability.BANDS)
    assert calc.can(Capability.DOS)
    assert calc.can(Capability.STRUCTURE)
    assert calc.can(Capability.METADATA)
    assert calc.metadata.e_fermi is not None
    assert calc.metadata.e_fermi.source == "DOSCAR"
    assert calc.metadata.encut is not None


def test_case_b_bands_without_dos(calc_dir):
    fixtures.case_b_bands_only(calc_dir)
    calc = discover(calc_dir)

    assert calc.can(Capability.BANDS)
    assert not calc.can(Capability.DOS)
    assert "DOSCAR" in calc.why_not(Capability.DOS)
    assert calc.can(Capability.STRUCTURE)
    # No Fermi source exists, and none may be invented.
    assert calc.metadata.e_fermi is None


def test_case_c_dos_without_bands(calc_dir):
    fixtures.case_c_no_eigenval(calc_dir)
    calc = discover(calc_dir)

    assert not calc.can(Capability.BANDS)
    assert "EIGENVAL" in calc.why_not(Capability.BANDS)
    assert calc.can(Capability.DOS)
    assert calc.can(Capability.STRUCTURE)


def test_case_d_structure_and_inputs_only(calc_dir):
    fixtures.case_d_structure_only(calc_dir)
    calc = discover(calc_dir)

    assert calc.can(Capability.STRUCTURE)
    assert calc.can(Capability.INPUT_FILES)
    assert not calc.can(Capability.BANDS)
    assert not calc.can(Capability.DOS)


def test_case_e_partial_wannier(calc_dir):
    fixtures.case_e_partial_wannier(calc_dir)
    calc = discover(calc_dir)

    assert calc.can(Capability.BANDS)
    # .eig + .win is enough to offer Wannier bands.
    assert calc.can(Capability.WANNIER_BANDS)


def test_case_f_malformed_doscar_does_not_break_the_rest(calc_dir):
    fixtures.case_f_malformed_doscar(calc_dir)
    calc = discover(calc_dir)

    assert calc.can(Capability.BANDS)
    assert calc.can(Capability.STRUCTURE)
    assert not calc.can(Capability.DOS)
    assert calc.index.entries["doscar"].status is FileStatus.MALFORMED
    assert any("DOSCAR" in str(issue) for issue in calc.issues)


def test_case_g_empty_directory(calc_dir):
    fixtures.case_g_empty(calc_dir)
    calc = discover(calc_dir)

    assert calc.is_empty
    assert not calc.can(Capability.BANDS)
    assert not calc.can(Capability.DOS)
    assert not calc.can(Capability.STRUCTURE)
    assert not calc.can(Capability.INPUT_FILES)


def test_missing_directory_is_reported_not_raised(tmp_path):
    calc = discover(tmp_path / "does-not-exist")
    assert calc.is_empty
    assert any(issue.code == "directory_missing" for issue in calc.issues)


def test_contcar_used_when_poscar_absent(calc_dir):
    fixtures.write_eigenval(calc_dir)
    fixtures.write_poscar(calc_dir, name="CONTCAR")
    calc = discover(calc_dir)

    assert calc.can(Capability.STRUCTURE)
    assert calc.metadata.formula is not None
    assert calc.metadata.formula.source == "CONTCAR"


def test_line_mode_kpoints_enables_path_labels(calc_dir):
    fixtures.write_eigenval(calc_dir)
    fixtures.write_kpoints_line(calc_dir)
    calc = discover(calc_dir)

    assert calc.can(Capability.KPATH_LABELS)


def test_mesh_kpoints_disables_path_labels(calc_dir):
    fixtures.write_eigenval(calc_dir)
    fixtures.write_kpoints_mesh(calc_dir)
    calc = discover(calc_dir)

    assert not calc.can(Capability.KPATH_LABELS)
    assert "mesh" in calc.why_not(Capability.KPATH_LABELS)
