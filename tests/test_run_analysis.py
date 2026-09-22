"""The analysis added alongside the fixes: VASPKIT k-point labels, VTST's own
spline, MD run statistics, and the range slider they are read through."""

from __future__ import annotations

import numpy as np
import pytest

from vasp_analyzer.models.trajectory import MDSeries, MDTrajectory
from vasp_analyzer.parsers.kpoints import parse_kpoints
from vasp_analyzer.parsers.vtst import parse_spline_dat
from vasp_analyzer.services import md_stats
from vasp_analyzer.ui.pages.bands.projection import _max_marker_size

# --- KPOINTS labels ---------------------------------------------------------


def _write_line_kpoints(path, body: str) -> None:
    path.write_text("K-Path\n  20\nLine-Mode\nReciprocal\n" + body)


def test_labels_after_a_comment_marker_are_read(tmp_path):
    path = tmp_path / "KPOINTS"
    _write_line_kpoints(
        path,
        " 0.000 0.000 0.000 ! Γ\n-0.333 0.666 0.000 ! K\n",
    )
    assert parse_kpoints(path).value.path_labels == ["Γ", "K"]


def test_bare_labels_are_read(tmp_path):
    """Regression: VASPKIT writes the label with no ``!`` marker at all, and
    the whole path came back unlabelled — which is what left a band structure
    showing a bare "Wave vector" axis instead of its high-symmetry points."""
    path = tmp_path / "KPOINTS"
    _write_line_kpoints(
        path,
        "   0.0000000000   0.0000000000  -0.5000000000    -Z\n"
        "   0.0000000000   0.0000000000   0.0000000000     G\n",
    )
    assert parse_kpoints(path).value.path_labels == ["-Z", "Γ"]


def test_a_comment_marker_still_wins_over_a_trailing_token(tmp_path):
    path = tmp_path / "KPOINTS"
    _write_line_kpoints(path, " 0.0 0.0 0.0 ignored ! L\n")
    assert parse_kpoints(path).value.path_labels == ["L"]


def test_coordinates_without_any_label_stay_empty(tmp_path):
    path = tmp_path / "KPOINTS"
    _write_line_kpoints(path, " 0.0 0.0 0.0\n 0.5 0.5 0.5\n")
    assert parse_kpoints(path).value.path_labels == ["", ""]


# --- spline.dat -------------------------------------------------------------


def test_spline_dat_reads_coordinate_and_energy(tmp_path):
    path = tmp_path / "spline.dat"
    path.write_text(
        "0\t0.000000\t0.000000\t0.000000\n"
        "0.05\t0.0440958\t0.0100000\t0.005\n"
        "0.10\t0.0881916\t0.0250000\t0.009\n"
    )
    curve = parse_spline_dat(path).value

    assert curve.n_points == 3
    # Column 1 is the reaction coordinate, column 2 the relative energy.
    assert curve.reaction_coordinate == pytest.approx([0.0, 0.0440958, 0.0881916])
    assert curve.relative_energy == pytest.approx([0.0, 0.01, 0.025])
    assert curve.barrier == pytest.approx(0.025)


def test_empty_spline_dat_fails_rather_than_returning_a_flat_line(tmp_path):
    path = tmp_path / "spline.dat"
    path.write_text("# nothing numeric here\n")
    assert parse_spline_dat(path).failed


# --- MD statistics ----------------------------------------------------------


def _trajectory(
    temperature=None, energy=None, target=(300.0, 300.0), n_atoms=2
) -> MDTrajectory:
    n_frames = len(temperature) if temperature is not None else len(energy)
    trajectory = MDTrajectory(
        species=["C"] * n_atoms,
        frac_coords=np.zeros((n_frames, n_atoms, 3)),
        lattices=np.tile(np.eye(3), (n_frames, 1, 1)),
        target_temperature=target,
    )
    if temperature is not None:
        trajectory.series["temperature"] = MDSeries(
            "temperature", np.asarray(temperature, float), "K"
        )
    if energy is not None:
        trajectory.series["total_energy"] = MDSeries(
            "total_energy", np.asarray(energy, float), "eV"
        )
    return trajectory


def test_statistics_skip_the_equilibration_transient():
    # A run that starts cold and settles at 300 K: including the ramp would
    # drag the reported mean well below the temperature actually held.
    values = np.concatenate([np.full(20, 50.0), np.full(80, 300.0)])
    stats = md_stats.analyse(_trajectory(temperature=values))

    assert stats.start_index == 20
    assert stats.temperature.mean == pytest.approx(300.0)
    assert stats.temperature.n_samples == 80


def test_temperature_offset_measures_the_thermostat():
    stats = md_stats.analyse(_trajectory(temperature=np.full(50, 290.0)))
    assert stats.target_temperature == pytest.approx(300.0)
    assert stats.temperature_offset == pytest.approx(-10.0)


def test_offset_is_unknown_without_a_target():
    stats = md_stats.analyse(_trajectory(temperature=np.full(50, 290.0), target=None))
    assert stats.temperature_offset is None


def test_drift_is_a_trend_not_the_last_step():
    # A clean ramp of +0.01 per step over the analysed window.
    energy = np.arange(100, dtype=float) * 0.01
    stats = md_stats.analyse(_trajectory(energy=energy))

    window = 100 - stats.start_index
    assert stats.total_energy.drift == pytest.approx(0.01 * (window - 1), rel=1e-6)


def test_a_single_noisy_final_step_does_not_dominate_the_drift():
    energy = np.zeros(100)
    energy[-1] = 5.0
    stats = md_stats.analyse(_trajectory(energy=energy))
    # (last - first) would report 5.0; a fitted trend reports far less.
    assert abs(stats.total_energy.drift) < 1.0


def test_no_telemetry_yields_no_statistics():
    """An XDATCAR with no OSZICAR beside it has geometry but nothing to
    average — the card is hidden rather than filled with zeros."""
    assert md_stats.analyse(_trajectory(temperature=None, energy=[])) is None


def test_a_very_short_run_is_still_summarised():
    stats = md_stats.analyse(_trajectory(temperature=[300.0, 302.0]))
    assert stats is not None
    # Skipping 20% of two samples would leave nothing to average.
    assert stats.start_index == 0


def test_format_value_avoids_scientific_notation():
    assert md_stats.format_value(136.4) == "136"
    assert md_stats.format_value(-458.87) == "-459"
    assert md_stats.format_value(12.34) == "12.3"
    assert md_stats.format_value(0.5246) == "0.525"
    assert "e" not in md_stats.format_value(140.0)


# --- fatband marker sizing --------------------------------------------------


def test_marker_size_shrinks_as_the_plot_gets_denser():
    """A 118-atom nanotube plots 40,000 points; at the size tuned for a small
    cell every marker overlaps and the axes paint solid black."""
    small = _max_marker_size(400)
    large = _max_marker_size(40_000)

    assert small > large
    assert large >= 6.0          # never so small it vanishes
    assert small <= 90.0         # never larger than the tuned maximum


def test_marker_size_is_defined_for_an_empty_plot():
    assert _max_marker_size(0) == 90.0
