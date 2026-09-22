"""NEB discovery, energy extraction, reaction coordinate and barriers.

The most valuable checks here are the ones that refuse to answer: a band with
unrun intermediate images must not report a barrier or claim convergence, since
the true maximum may be exactly the image that is missing.

Real-data checks run against the user's own CI-NEB attempt, which is genuinely
unconverged — that makes it a better test than a clean run would be.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vasp_analyzer.models.capabilities import Capability
from vasp_analyzer.parsers.outcar import parse_ionic_state
from vasp_analyzer.parsers.vtst import merge_profiles, parse_neb_dat, parse_nebef_dat
from vasp_analyzer.services.discovery import discover
from vasp_analyzer.services.neb import (
    find_image_directories,
    load_neb_path,
    looks_like_neb,
    reaction_coordinate,
)

from . import fixtures

ARCHIVE = Path("Graphene/neb/archive/attempt-01-ci-lbfgs")
PREPARED = Path("Graphene/neb")
needs_neb = pytest.mark.skipif(
    not (ARCHIVE / "neb.dat").exists(), reason="requires the local Graphene NEB data"
)
needs_prepared = pytest.mark.skipif(
    not (PREPARED / "00" / "POSCAR").exists(),
    reason="requires the local Graphene NEB data",
)


# --- discovery -------------------------------------------------------------------


def test_image_directories_sort_numerically(tmp_path):
    # Lexicographic order would give 00, 01, 02, 10, 11 as 00, 01, 02, 10, 11
    # only by luck; with 2 and 10 present it breaks outright.
    for index in (0, 1, 2, 10, 11, 3):
        fixtures.write_neb_image(tmp_path, index)
    found = find_image_directories(tmp_path)
    assert [index for index, _ in found] == [0, 1, 2, 3, 10, 11]


def test_looks_like_neb_accepts_a_real_band(tmp_path):
    fixtures.case_neb(tmp_path)
    assert looks_like_neb(tmp_path)


def test_looks_like_neb_rejects_two_stray_numeric_folders(tmp_path):
    (tmp_path / "00").mkdir()
    (tmp_path / "01").mkdir()
    assert not looks_like_neb(tmp_path)


def test_looks_like_neb_rejects_folders_without_structures(tmp_path):
    for index in range(4):
        (tmp_path / f"{index:02d}").mkdir()
    assert not looks_like_neb(tmp_path)


def test_neb_capability_is_set(tmp_path):
    fixtures.case_neb(tmp_path)
    calculation = discover(tmp_path)
    assert calculation.can(Capability.NEB_PATH)
    assert calculation.can(Capability.NEB_ENERGIES)


def test_neb_capability_absent_in_an_ordinary_directory(tmp_path):
    fixtures.case_a_complete(tmp_path)
    calculation = discover(tmp_path)
    assert not calculation.can(Capability.NEB_PATH)
    assert "image folders" in calculation.why_not(Capability.NEB_PATH)


# --- reaction coordinate ---------------------------------------------------------------


def test_reaction_coordinate_is_cumulative_and_monotonic(tmp_path):
    fixtures.case_neb(tmp_path)
    path = load_neb_path(tmp_path).value
    coordinate = path.reaction_coordinate

    assert coordinate is not None
    assert coordinate[0] == pytest.approx(0.0)
    assert np.all(np.diff(coordinate) > 0), "path length can only increase"


def test_reaction_coordinate_uses_minimum_image(tmp_path):
    """An atom that crosses a cell face between two images has moved a little,
    not almost a whole lattice vector; without the minimum-image correction
    the path length would jump by the cell size."""
    from vasp_analyzer.models.structure import StructureData

    lattice = np.eye(3) * 10.0
    first = StructureData(
        lattice=lattice, species=["C"], frac_coords=np.array([[0.95, 0.0, 0.0]])
    )
    second = StructureData(
        lattice=lattice, species=["C"], frac_coords=np.array([[0.05, 0.0, 0.0]])
    )
    coordinate = reaction_coordinate([first, second])

    assert coordinate is not None
    # 0.1 of a 10 A cell = 1.0 A, not 9.0 A.
    assert coordinate[1] == pytest.approx(1.0, abs=1e-9)


def test_reaction_coordinate_none_when_an_image_is_missing():
    assert reaction_coordinate([None, None]) is None


# --- energies and barriers ----------------------------------------------------------------


def test_barrier_from_a_complete_band(tmp_path):
    fixtures.case_neb(tmp_path, energies=(-10.0, -9.0, -8.0, -9.2, -9.8))
    path = load_neb_path(tmp_path).value

    assert path.complete_energies
    assert path.peak_index == 2
    assert path.forward_barrier == pytest.approx(2.0)   # -8.0 - (-10.0)
    assert path.reverse_barrier == pytest.approx(1.8)   # -8.0 - (-9.8)
    assert path.reaction_energy == pytest.approx(0.2)   # -9.8 - (-10.0)


def test_relative_energies_are_measured_from_the_initial_image(tmp_path):
    fixtures.case_neb(tmp_path, energies=(-10.0, -9.0, -8.0, -9.2, -9.8))
    path = load_neb_path(tmp_path).value
    assert path.relative_energies()[0] == pytest.approx(0.0)
    assert path.relative_energies()[2] == pytest.approx(2.0)


def test_incomplete_band_refuses_to_report_a_barrier(tmp_path):
    """The whole point: with image 2 unrun, the highest *known* energy is not
    the barrier, and reporting it would understate the real one."""
    fixtures.write_neb_incar(tmp_path)
    for index, energy in enumerate([-10.0, -9.0, None, -9.2, -9.8]):
        fixtures.write_neb_image(
            tmp_path, index, shift=0.05 * index, energy=energy, max_force=0.01
        )
    result = load_neb_path(tmp_path)
    path = result.value

    assert not path.complete_energies
    assert path.forward_barrier is None
    assert path.reverse_barrier is None
    assert any(issue.code == "neb_partial_energies" for issue in result.issues)


def test_incomplete_forces_leave_convergence_unknown(tmp_path):
    fixtures.write_neb_incar(tmp_path)
    # Endpoints report a force; the movable images never ran.
    for index in range(5):
        endpoint = index in (0, 4)
        fixtures.write_neb_image(
            tmp_path,
            index,
            shift=0.05 * index,
            energy=-10.0 + index if endpoint else None,
            max_force=0.001 if endpoint else None,
        )
    path = load_neb_path(tmp_path).value

    assert not path.complete_forces
    assert path.converged is None, "two fixed endpoints cannot certify a band"


def test_converged_band_reports_convergence(tmp_path):
    fixtures.case_neb(tmp_path, forces=(0.001, 0.01, 0.02, 0.01, 0.001))
    path = load_neb_path(tmp_path).value
    assert path.force_threshold == pytest.approx(0.05)
    assert path.converged is True


def test_unconverged_band_reports_not_converged(tmp_path):
    fixtures.case_neb(tmp_path, forces=(0.001, 0.5, 1.2, 0.4, 0.001))
    path = load_neb_path(tmp_path).value
    assert path.converged is False
    assert path.max_force == pytest.approx(1.2)


def test_convergence_unknown_without_a_force_criterion(tmp_path):
    # A positive EDIFFG is an energy criterion and says nothing about forces.
    fixtures.write_neb_incar(tmp_path, EDIFFG="1E-4")
    for index in range(5):
        fixtures.write_neb_image(
            tmp_path, index, shift=0.05 * index, energy=-10.0 + index, max_force=0.01
        )
    path = load_neb_path(tmp_path).value
    assert path.force_threshold is None
    assert path.converged is None


# --- climbing image ----------------------------------------------------------------------


def test_climbing_image_is_inferred_and_labelled_as_such(tmp_path):
    fixtures.case_neb(tmp_path, energies=(-10.0, -9.0, -8.0, -9.2, -9.8), climbing=True)
    path = load_neb_path(tmp_path).value

    assert path.is_climbing
    assert path.climbing_image == 2
    assert path.climbing_image_origin == "inferred"


def test_no_climbing_image_for_a_plain_neb(tmp_path):
    fixtures.case_neb(tmp_path, climbing=False)
    path = load_neb_path(tmp_path).value

    assert not path.is_climbing
    assert path.method == "NEB"
    assert path.climbing_image is None


def test_climbing_image_is_never_an_endpoint(tmp_path):
    # The endpoints are fixed and cannot climb, even when one is highest.
    fixtures.case_neb(tmp_path, energies=(-10.0, -9.5, -9.4, -9.3, -1.0))
    path = load_neb_path(tmp_path).value
    assert path.climbing_image not in (0, path.n_images - 1)


# --- structure selection -------------------------------------------------------------------


def test_contcar_is_preferred_over_poscar(tmp_path):
    fixtures.write_neb_incar(tmp_path)
    for index in range(5):
        fixtures.write_neb_image(
            tmp_path, index, shift=0.05 * index, energy=-10.0, contcar=0 < index < 4
        )
    path = load_neb_path(tmp_path).value

    assert "01/CONTCAR" in path.images[1].source_files
    assert "00/POSCAR" in path.images[0].source_files


def test_empty_contcar_falls_back_to_poscar(tmp_path):
    fixtures.write_neb_incar(tmp_path)
    for index in range(5):
        folder = fixtures.write_neb_image(tmp_path, index, energy=-10.0, contcar=True)
        if index == 2:
            (folder / "CONTCAR").write_text("")   # a job killed mid-write
    path = load_neb_path(tmp_path).value

    assert path.images[2].structure is not None
    assert "02/POSCAR" in path.images[2].source_files


def test_mismatched_atom_count_is_reported_not_fatal(tmp_path):
    fixtures.write_neb_incar(tmp_path)
    for index in range(5):
        fixtures.write_neb_image(
            tmp_path, index, n_atoms=4 if index == 3 else 2, energy=-10.0
        )
    result = load_neb_path(tmp_path)

    assert result.ok, "one bad image must not lose the whole band"
    assert any(
        issue.code == "neb_image_atom_mismatch" for issue in result.issues
    )
    assert result.value.images[3].note


# --- VTST files -------------------------------------------------------------------------------


def test_nebef_force_magnitude_wins_over_neb_dat_tangential(tmp_path):
    """neb.dat's force column is a signed tangential component; nebef.dat's is
    a magnitude. Comparing the signed one against EDIFFG would call a wildly
    unconverged band converged."""
    (tmp_path / "neb.dat").write_text(
        "  0 0.0 0.0  0.0 0\n  1 0.5 1.0 -3.3 1\n  2 1.0 0.5  0.4 2\n"
    )
    (tmp_path / "nebef.dat").write_text(
        "  0 0.001 -10.0 0.0\n  1 8.010 -9.0 1.0\n  2 0.400 -9.5 0.5\n"
    )
    neb = parse_neb_dat(tmp_path / "neb.dat").value
    nebef = parse_nebef_dat(tmp_path / "nebef.dat").value
    merged = merge_profiles(neb, nebef)

    assert merged.force_is_magnitude
    assert merged.force[1] == pytest.approx(8.010)
    assert merged.reaction_coordinate == [0.0, 0.5, 1.0]
    assert merged.absolute_energy[0] == pytest.approx(-10.0)


def test_vtst_tables_ignored_when_the_image_count_disagrees(tmp_path):
    fixtures.case_neb(tmp_path)   # five images
    (tmp_path / "neb.dat").write_text("  0 0.0 0.0 0.0 0\n  1 0.5 1.0 0.1 1\n")
    result = load_neb_path(tmp_path)

    assert any(
        issue.code == "neb_vtst_image_mismatch" for issue in result.issues
    )


# --- OUTCAR ionic state -------------------------------------------------------------------------


def test_ionic_state_prefers_sigma_zero_energy(tmp_path):
    (tmp_path / "OUTCAR").write_text(
        "  free  energy   TOTEN  =       -73.58119959 eV\n\n"
        "  energy  without entropy=      -73.56866204  "
        "energy(sigma->0) =      -73.57493082\n\n"
        " POSITION                                       TOTAL-FORCE (eV/Angst)\n"
        " ------------------------------------------------------------\n"
        "   0.0 0.0 0.0   0.001 0.002 0.002\n"
        " ------------------------------------------------------------\n"
    )
    state = parse_ionic_state(tmp_path / "OUTCAR").value

    # VTST reports energy(sigma->0), so matching it keeps the two consistent.
    assert state.energy == pytest.approx(-73.57493082)
    assert state.free_energy == pytest.approx(-73.58119959)
    assert state.max_force == pytest.approx(0.003, abs=1e-6)


# --- real data ---------------------------------------------------------------------------------


@needs_neb
def test_real_ci_neb_matches_vtst_exactly():
    """Against the user's own CI-NEB: our reaction coordinate and energies must
    reproduce VTST's neb.dat/nebef.dat, which were produced independently."""
    path = load_neb_path(ARCHIVE).value

    assert path.n_images == 5
    assert path.method == "NEB (Climbing Image)"
    assert path.complete_energies

    expected_coordinate = [0.0, 0.279083, 0.751454, 1.332791, 1.752757]
    np.testing.assert_allclose(
        path.reaction_coordinate, expected_coordinate, atol=1e-6
    )

    expected_relative = [0.0, 1.790489, 8.778404, 9.888316, 9.581311]
    for actual, expected in zip(path.relative_energies(), expected_relative):
        assert actual == pytest.approx(expected, abs=1e-3)


@needs_neb
def test_real_ci_neb_is_reported_unconverged():
    """This run was soft-stopped with oscillatory forces. Reporting it as
    converged — or quietly publishing its 9.9 eV barrier without that caveat —
    would be the single most misleading thing this page could do."""
    path = load_neb_path(ARCHIVE).value

    assert path.force_threshold == pytest.approx(0.03)   # |EDIFFG|
    assert path.max_force == pytest.approx(12.170658, abs=1e-4)
    assert path.converged is False


@needs_neb
def test_real_ci_neb_climbing_image_matches_the_spline_extremum():
    """VTST's exts.dat puts the extremum near image 2.93; the highest computed
    image is 03, which is what the inference must land on."""
    path = load_neb_path(ARCHIVE).value
    assert path.climbing_image == 3
    assert path.climbing_image_origin == "inferred"


@needs_neb
def test_real_ci_neb_reads_pseudopotentials_from_an_image_outcar():
    path = load_neb_path(ARCHIVE).value
    assert path.potcar_titles
    assert path.potcar_titles[0].startswith("PAW_PBE")
    assert path.functional == "PBE (PAW)"


@needs_prepared
def test_real_prepared_neb_reports_a_barrier_only_when_the_band_is_complete():
    """The top-level directory's five images always parse; whether they carry
    energies depends on how far that calculation has actually been run.

    Deliberately written against *either* state rather than asserting the
    band is unrun: this points at a live calculation directory, and finishing
    the run is a normal thing to do to it. What must always hold is that the
    barrier and the completeness of the energies agree with each other —
    a barrier quoted from a partial band would be the real bug.
    """
    result = load_neb_path(PREPARED)
    path = result.value

    assert path.n_images == 5
    assert all(image.structure is not None for image in path.images)

    if path.complete_energies:
        assert path.forward_barrier is not None
        assert not any(
            issue.code == "neb_partial_energies" for issue in result.issues
        )
    else:
        assert path.forward_barrier is None
        assert path.converged is None
        assert any(issue.code == "neb_partial_energies" for issue in result.issues)
