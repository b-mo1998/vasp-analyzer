"""MD trajectory: XDATCAR/OSZICAR parsing, unwrapping, MSD and assembly.

The unwrapping and MSD tests are built on a fixture whose atom crosses the
periodic boundary by construction, because that is the case naive code gets
wrong: the wrapped coordinates make the atom appear to jump back across the
cell, and any displacement measured from them is meaningless.

Real-data checks against the user's own graphene run are marked to skip when
that directory isn't present, the way the band-gap tests already are.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vasp_analyzer.models.trajectory import MDTrajectory
from vasp_analyzer.parsers.oszicar import parse_oszicar, peek_oszicar_steps
from vasp_analyzer.parsers.xdatcar import parse_xdatcar, peek_xdatcar_frames
from vasp_analyzer.services.discovery import discover
from vasp_analyzer.services.md import (
    describe_ensemble,
    frame_for_time,
    load_trajectory,
    target_temperature,
)
from vasp_analyzer.services.trajectory import (
    mean_square_displacement,
    unwrap_fractional,
    unwrapped_cartesian,
)

from . import fixtures

MD_DIR = Path("Graphene/md")
needs_md = pytest.mark.skipif(
    not (MD_DIR / "XDATCAR").exists(), reason="requires the local Graphene MD data"
)


# --- XDATCAR ----------------------------------------------------------------------


def test_fixed_cell_xdatcar_dimensions(tmp_path):
    fixtures.write_xdatcar(tmp_path, n_frames=7, n_atoms=4)
    result = parse_xdatcar(tmp_path / "XDATCAR")

    assert result.ok
    trajectory = result.value
    assert trajectory.n_frames == 7
    assert trajectory.n_atoms == 4
    assert trajectory.frac_coords.shape == (7, 4, 3)
    assert trajectory.lattices.shape == (7, 3, 3)
    assert not trajectory.is_variable_cell


def test_variable_cell_xdatcar_tracks_each_frames_lattice(tmp_path):
    fixtures.write_xdatcar(tmp_path, n_frames=5, n_atoms=2, variable_cell=True)
    trajectory = parse_xdatcar(tmp_path / "XDATCAR").value

    assert trajectory.n_frames == 5
    assert trajectory.is_variable_cell
    # The fixture grows the cell by 1% per frame.
    assert trajectory.lattices[-1][0, 0] > trajectory.lattices[0][0, 0]


def test_species_are_expanded_per_atom(tmp_path):
    fixtures.write_xdatcar(tmp_path, n_frames=3, n_atoms=4, species=("C", "H"))
    trajectory = parse_xdatcar(tmp_path / "XDATCAR").value
    assert len(trajectory.species) == 4
    assert set(trajectory.species) == {"C", "H"}
    assert trajectory.elements == ["C", "H"]  # first-appearance order


def test_truncated_xdatcar_keeps_complete_frames(tmp_path):
    fixtures.write_xdatcar(tmp_path, n_frames=6, n_atoms=4, truncate_last=True)
    result = parse_xdatcar(tmp_path / "XDATCAR")

    assert result.ok, "a partial final frame must not reject the whole file"
    assert result.value.n_frames == 5
    assert any(issue.code == "xdatcar_truncated" for issue in result.issues)


def test_garbage_xdatcar_reports_error(tmp_path):
    (tmp_path / "XDATCAR").write_text("not an XDATCAR\n" * 5)
    assert parse_xdatcar(tmp_path / "XDATCAR").failed


def test_peek_counts_frames_without_parsing(tmp_path):
    fixtures.write_xdatcar(tmp_path, n_frames=12, n_atoms=3)
    assert peek_xdatcar_frames(tmp_path / "XDATCAR") == 12


# --- OSZICAR ------------------------------------------------------------------------


def test_oszicar_md_fields(tmp_path):
    fixtures.write_oszicar_md(tmp_path, n_steps=8)
    result = parse_oszicar(tmp_path / "OSZICAR")

    assert result.ok
    data = result.value
    assert data.n_steps == 8
    assert data.is_molecular_dynamics
    assert data.array("temperature").shape == (8,)


def test_oszicar_energy_identity_holds(tmp_path):
    """VASP's own relation E = F + EK + SP + SK — proof the columns are not
    being confused with one another."""
    fixtures.write_oszicar_md(tmp_path, n_steps=5)
    data = parse_oszicar(tmp_path / "OSZICAR").value

    total = data.array("total_energy")
    potential = data.array("potential_energy")
    kinetic = data.array("kinetic_energy")
    np.testing.assert_allclose(total, potential + kinetic, atol=1e-6)


def test_oszicar_ignores_electronic_iteration_lines(tmp_path):
    # The fixture writes a DAV: line before every ionic line; counting those
    # would double the step count.
    fixtures.write_oszicar_md(tmp_path, n_steps=6)
    assert peek_oszicar_steps(tmp_path / "OSZICAR") == 6


def test_empty_oszicar_reports_error(tmp_path):
    (tmp_path / "OSZICAR").write_text("DAV:   1  -0.1E+03\n")
    assert parse_oszicar(tmp_path / "OSZICAR").failed


# --- unwrapping -----------------------------------------------------------------------


def test_unwrap_removes_the_boundary_jump():
    # An atom stepping +0.3 per frame from 0.8 wraps to 0.1 at the next frame;
    # the wrapped difference is -0.7, the true one is +0.3.
    frac = np.array([[[0.8, 0.0, 0.0]], [[0.1, 0.0, 0.0]], [[0.4, 0.0, 0.0]]])
    unwrapped = unwrap_fractional(frac)

    np.testing.assert_allclose(unwrapped[:, 0, 0], [0.8, 1.1, 1.4], atol=1e-12)


def test_unwrap_leaves_a_non_crossing_trajectory_alone():
    frac = np.array([[[0.1, 0.0, 0.0]], [[0.2, 0.0, 0.0]], [[0.3, 0.0, 0.0]]])
    np.testing.assert_allclose(unwrap_fractional(frac), frac, atol=1e-12)


def test_unwrap_handles_a_single_frame():
    frac = np.array([[[0.5, 0.5, 0.5]]])
    np.testing.assert_allclose(unwrap_fractional(frac), frac)


def test_unwrapped_cartesian_measures_true_displacement(tmp_path):
    # Atom 0 advances 0.3 fractional units per frame across the boundary; over
    # four steps that is 1.2 cells = 6.0 Å in a 5 Å cell.
    fixtures.write_xdatcar(tmp_path, n_frames=5, n_atoms=2, lattice=5.0, drift=0.3)
    trajectory = parse_xdatcar(tmp_path / "XDATCAR").value

    cartesian = unwrapped_cartesian(trajectory)
    travelled = cartesian[-1, 0, 0] - cartesian[0, 0, 0]
    assert travelled == pytest.approx(6.0, abs=1e-6)


def test_variable_cell_unwrapping_uses_each_frames_lattice(tmp_path):
    fixtures.write_xdatcar(
        tmp_path, n_frames=4, n_atoms=2, lattice=5.0, variable_cell=True
    )
    trajectory = parse_xdatcar(tmp_path / "XDATCAR").value
    cartesian = unwrapped_cartesian(trajectory)
    assert cartesian.shape == (4, 2, 3)
    assert np.isfinite(cartesian).all()


# --- MSD --------------------------------------------------------------------------------


def test_msd_starts_at_zero(tmp_path):
    fixtures.write_xdatcar(tmp_path, n_frames=6, n_atoms=4, drift=0.05)
    trajectory = parse_xdatcar(tmp_path / "XDATCAR").value
    curves = mean_square_displacement(trajectory)

    for values in curves.values():
        assert values[0] == pytest.approx(0.0, abs=1e-12)


def test_msd_matches_a_hand_computed_displacement(tmp_path):
    # One of four atoms moves 0.1 * 5 A = 0.5 A per frame; after two frames its
    # squared displacement is 1.0 A^2 and the element mean is 1.0 / 4.
    fixtures.write_xdatcar(tmp_path, n_frames=3, n_atoms=4, lattice=5.0, drift=0.1)
    trajectory = parse_xdatcar(tmp_path / "XDATCAR").value
    curves = mean_square_displacement(trajectory)

    assert curves["C"][2] == pytest.approx(1.0 / 4, abs=1e-9)


def test_msd_has_one_curve_per_element_plus_total(tmp_path):
    fixtures.write_xdatcar(tmp_path, n_frames=4, n_atoms=4, species=("C", "H"))
    trajectory = parse_xdatcar(tmp_path / "XDATCAR").value
    curves = mean_square_displacement(trajectory)

    assert set(curves) == {"C", "H", "total"}


def test_msd_of_a_single_frame_is_empty(tmp_path):
    fixtures.write_xdatcar(tmp_path, n_frames=1, n_atoms=2)
    trajectory = parse_xdatcar(tmp_path / "XDATCAR").value
    assert mean_square_displacement(trajectory) == {}


# --- ensemble and timing -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        ({"IBRION": "0", "MDALGO": "2", "SMASS": "1.0", "ISIF": "2"}, "NVT"),
        ({"IBRION": "0", "MDALGO": "0", "SMASS": "-3", "ISIF": "2"}, "NVE"),
        ({"IBRION": "0", "MDALGO": "3", "ISIF": "3"}, "NPT"),
        ({"IBRION": "2"}, None),          # a relaxation, not MD
        ({}, None),
    ],
)
def test_ensemble_inference(tags, expected):
    assert describe_ensemble(tags) == expected


def test_target_temperature_defaults_teend_to_tebeg():
    assert target_temperature({"TEBEG": "300"}) == (300.0, 300.0)
    assert target_temperature({"TEBEG": "300", "TEEND": "600"}) == (300.0, 600.0)
    assert target_temperature({}) is None


def test_times_use_potim_and_stride():
    trajectory = MDTrajectory(
        species=["C"],
        frac_coords=np.zeros((4, 1, 3)),
        lattices=np.tile(np.eye(3), (4, 1, 1)),
        timestep=2.0,
        frame_stride=5,
    )
    # 2 fs per step, one frame every 5 steps -> 10 fs = 0.01 ps per frame.
    np.testing.assert_allclose(trajectory.times(), [0.0, 0.01, 0.02, 0.03])
    assert trajectory.duration_ps == pytest.approx(0.04)


def test_times_fall_back_to_frame_index_without_potim():
    trajectory = MDTrajectory(
        species=["C"],
        frac_coords=np.zeros((3, 1, 3)),
        lattices=np.tile(np.eye(3), (3, 1, 1)),
    )
    assert not trajectory.has_time_axis
    np.testing.assert_allclose(trajectory.times(), [0, 1, 2])
    assert trajectory.duration_ps is None


# --- assembly -------------------------------------------------------------------------------


def test_load_trajectory_combines_every_source(tmp_path):
    fixtures.case_md(tmp_path, n_frames=10, n_atoms=4)
    calculation = discover(tmp_path)
    result = load_trajectory(calculation.index)

    assert result.ok
    trajectory = result.value
    assert trajectory.n_frames == 10
    assert trajectory.timestep == pytest.approx(1.0)
    assert trajectory.ensemble == "NVT"
    assert trajectory.target_temperature == (300.0, 300.0)
    assert trajectory.has("temperature")
    assert trajectory.has("total_energy")
    assert "XDATCAR" in trajectory.source_files
    assert "OSZICAR" in trajectory.source_files


def test_missing_oszicar_still_yields_a_playable_trajectory(tmp_path):
    fixtures.write_xdatcar(tmp_path, n_frames=5, n_atoms=3)
    fixtures.write_incar_md(tmp_path)
    calculation = discover(tmp_path)
    result = load_trajectory(calculation.index)

    assert result.ok
    assert result.value.n_frames == 5
    assert not result.value.has("temperature")


def test_missing_incar_leaves_the_axis_in_frames(tmp_path):
    fixtures.write_xdatcar(tmp_path, n_frames=5, n_atoms=3)
    calculation = discover(tmp_path)
    result = load_trajectory(calculation.index)

    assert result.ok
    assert result.value.timestep is None
    assert any(issue.code == "md_no_timestep" for issue in result.issues)


def test_misaligned_oszicar_is_dropped_not_misplotted(tmp_path):
    # 7 ionic steps against 5 frames is not a whole-number stride, so the
    # series cannot be aligned and must be omitted rather than mismatched.
    fixtures.write_xdatcar(tmp_path, n_frames=5, n_atoms=2)
    fixtures.write_oszicar_md(tmp_path, n_steps=7)
    fixtures.write_incar_md(tmp_path)
    result = load_trajectory(discover(tmp_path).index)

    assert result.ok
    assert not result.value.has("temperature")
    assert any(issue.code == "md_series_misaligned" for issue in result.issues)


def test_nblock_stride_subsamples_the_series(tmp_path):
    # NBLOCK=2 means XDATCAR holds every second step, so a 10-step OSZICAR
    # aligns onto 5 frames.
    fixtures.write_xdatcar(tmp_path, n_frames=5, n_atoms=2)
    fixtures.write_oszicar_md(tmp_path, n_steps=10)
    fixtures.write_incar_md(tmp_path, NBLOCK="2")
    result = load_trajectory(discover(tmp_path).index)

    assert result.ok
    trajectory = result.value
    assert trajectory.has("temperature")
    assert len(trajectory.series["temperature"]) == 5
    assert trajectory.frame_stride == 2


def test_frame_for_time_finds_the_nearest_frame(tmp_path):
    fixtures.case_md(tmp_path, n_frames=10, n_atoms=2)
    trajectory = load_trajectory(discover(tmp_path).index).value
    # 1 fs steps -> 0.001 ps per frame.
    assert frame_for_time(trajectory, 0.0) == 0
    assert frame_for_time(trajectory, 0.0049) == 5
    assert frame_for_time(trajectory, 99.0) == trajectory.n_frames - 1


# --- capabilities ---------------------------------------------------------------------------


def test_md_capability_requires_more_than_one_frame(tmp_path):
    from vasp_analyzer.models.capabilities import Capability

    fixtures.write_xdatcar(tmp_path, n_frames=1, n_atoms=2)
    calculation = discover(tmp_path)
    assert not calculation.can(Capability.MD_TRAJECTORY)
    assert "single configuration" in calculation.why_not(Capability.MD_TRAJECTORY)


def test_md_capability_without_xdatcar(tmp_path):
    from vasp_analyzer.models.capabilities import Capability

    fixtures.write_poscar(tmp_path)
    calculation = discover(tmp_path)
    assert not calculation.can(Capability.MD_TRAJECTORY)
    assert not calculation.can(Capability.MD_THERMODYNAMICS)


def test_thermodynamics_needs_oszicar(tmp_path):
    from vasp_analyzer.models.capabilities import Capability

    fixtures.write_xdatcar(tmp_path, n_frames=4, n_atoms=2)
    calculation = discover(tmp_path)
    assert calculation.can(Capability.MD_TRAJECTORY)
    assert not calculation.can(Capability.MD_THERMODYNAMICS)


# --- real data ------------------------------------------------------------------------------


@needs_md
def test_real_graphene_md_run():
    """Against the user's own 5x5 graphene NVT run: 1000 frames of 50 carbons
    at 1 fs, with the NVT ensemble inferred from IBRION/MDALGO/SMASS."""
    calculation = discover(MD_DIR)
    trajectory = load_trajectory(calculation.index).value

    assert trajectory.n_frames == 1000
    assert trajectory.n_atoms == 50
    assert trajectory.elements == ["C"]
    assert trajectory.timestep == pytest.approx(1.0)
    assert trajectory.ensemble == "NVT"
    assert trajectory.duration_ps == pytest.approx(1.0)
    assert not trajectory.is_variable_cell


@needs_md
def test_real_graphene_msd_is_physically_sensible():
    """Graphene at 300 K vibrates but does not diffuse, so the MSD must start
    at zero and stay small — a value of many Å² would mean the unwrapping had
    manufactured displacement that never happened."""
    calculation = discover(MD_DIR)
    trajectory = load_trajectory(calculation.index).value
    curves = mean_square_displacement(trajectory)

    assert curves["C"][0] == pytest.approx(0.0, abs=1e-12)
    assert 0.0 < float(np.max(curves["C"])) < 1.0


@needs_md
def test_real_graphene_temperature_matches_the_thermostat():
    calculation = discover(MD_DIR)
    trajectory = load_trajectory(calculation.index).value
    temperature = trajectory.series_for("temperature")

    assert temperature is not None
    assert len(temperature) == trajectory.n_frames
    # A 1 ps NVT pilot fluctuates widely but must average near its setpoint.
    assert 200.0 < float(np.mean(temperature.values)) < 400.0
