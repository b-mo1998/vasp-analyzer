"""Cross-file consistency: stale files must be detected, not trusted.

Motivated by a real directory in which an OUTCAR from an unrelated run had been
copied alongside a silicon calculation. Trusting it would have shown a Fermi
level 7 eV away from the true one.
"""

from __future__ import annotations

import pytest

from vasp_analyzer.models.capabilities import Capability
from vasp_analyzer.models.files import FileStatus
from vasp_analyzer.services.discovery import discover

from . import fixtures


def test_matching_outcar_is_trusted(tmp_path):
    fixtures.write_eigenval(tmp_path, n_kpoints=12, n_bands=6, n_electrons=8, n_ions=2)
    fixtures.write_poscar(tmp_path, atoms=2)
    fixtures.write_outcar(
        tmp_path, n_bands=6, n_kpoints=12, n_electrons=8.0, n_ions=2, species="Si"
    )
    calc = discover(tmp_path)

    assert calc.index.entries["outcar"].status is FileStatus.FOUND
    assert calc.metadata.vasp_version is not None
    assert calc.metadata.encut is not None
    assert calc.metadata.encut.source == "OUTCAR"


def test_outcar_from_a_different_run_is_rejected(tmp_path):
    fixtures.write_eigenval(tmp_path, n_kpoints=12, n_bands=6, n_electrons=8, n_ions=2)
    fixtures.write_poscar(tmp_path, atoms=2)
    # A leftover from an entirely different calculation.
    fixtures.write_outcar(
        tmp_path, n_bands=800, n_kpoints=16, n_electrons=484.0, n_ions=118, species="Te"
    )
    calc = discover(tmp_path)

    assert calc.index.entries["outcar"].status is FileStatus.INCONSISTENT
    assert any(issue.code == "outcar_inconsistent" for issue in calc.issues)
    # None of its metadata may leak through.
    assert calc.metadata.vasp_version is None
    assert calc.metadata.e_fermi is None


def test_doscar_fermi_wins_over_stale_outcar(tmp_path):
    fixtures.write_eigenval(tmp_path, n_kpoints=12, n_bands=6, n_electrons=8, n_ions=2)
    fixtures.write_poscar(tmp_path, atoms=2)
    fixtures.write_doscar(tmp_path, e_fermi=5.3068, n_ions=2)
    fixtures.write_outcar(
        tmp_path, n_bands=800, n_kpoints=16, n_electrons=484.0, n_ions=118, e_fermi=-1.67
    )
    calc = discover(tmp_path)

    assert calc.metadata.e_fermi is not None
    assert calc.metadata.e_fermi.value == pytest.approx(5.3068)
    assert calc.metadata.e_fermi.source == "DOSCAR"


def test_disagreeing_fermi_levels_are_reported(tmp_path):
    fixtures.write_eigenval(tmp_path, n_kpoints=12, n_bands=6, n_electrons=8, n_ions=2)
    fixtures.write_poscar(tmp_path, atoms=2)
    fixtures.write_doscar(tmp_path, e_fermi=5.30, n_ions=2)
    # Consistent with the calculation, but its Fermi level differs.
    fixtures.write_outcar(
        tmp_path, n_bands=6, n_kpoints=12, n_electrons=8.0, n_ions=2, e_fermi=4.10
    )
    calc = discover(tmp_path)

    assert calc.metadata.e_fermi.source == "DOSCAR"
    assert any(issue.code == "fermi_disagreement" for issue in calc.issues)


def test_doscar_for_a_different_structure_is_rejected(tmp_path):
    fixtures.write_eigenval(tmp_path, n_kpoints=12, n_bands=6, n_ions=2)
    fixtures.write_poscar(tmp_path, atoms=2)
    fixtures.write_doscar(tmp_path, n_ions=48, e_fermi=1.23)
    calc = discover(tmp_path)

    assert calc.index.entries["doscar"].status is FileStatus.INCONSISTENT
    assert calc.metadata.e_fermi is None


def test_matching_procar_is_trusted(tmp_path):
    fixtures.write_eigenval(tmp_path, n_kpoints=6, n_bands=4, n_electrons=8, n_ions=2)
    fixtures.write_poscar(tmp_path, atoms=2)
    fixtures.write_procar(tmp_path, n_kpoints=6, n_bands=4, n_ions=2)
    calc = discover(tmp_path)

    assert calc.index.entries["procar"].status is FileStatus.FOUND
    assert calc.can(Capability.ELEMENT_PROJECTION)


def test_procar_from_a_different_run_is_rejected(tmp_path):
    fixtures.write_eigenval(tmp_path, n_kpoints=6, n_bands=4, n_electrons=8, n_ions=2)
    fixtures.write_poscar(tmp_path, atoms=2)
    fixtures.write_procar(tmp_path, n_kpoints=20, n_bands=40, n_ions=10)
    calc = discover(tmp_path)

    assert calc.index.entries["procar"].status is FileStatus.INCONSISTENT
    assert any(issue.code == "procar_inconsistent" for issue in calc.issues)
