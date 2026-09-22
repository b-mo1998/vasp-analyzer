"""Phonon parsing, physics and page behaviour.

The fixtures solve a real spring-model lattice-dynamics problem (see
``phonon_fixtures``), so these assert genuine invariants — acoustic branches
reaching zero at Γ, 3N branches, orthonormal eigenvectors, and a wave that
actually alternates between cells at the zone boundary — rather than checking
that invented numbers survived a round trip.
"""

from __future__ import annotations

import numpy as np
import pytest

from vasp_analyzer.models.phonon import IMAGINARY_TOLERANCE_THZ, PhononDataset
from vasp_analyzer.parsers.phonopy import parse_band_yaml, parse_dos_dat
from vasp_analyzer.resources.elements import atomic_mass
from vasp_analyzer.services import phonon as phonon_service

from . import phonon_fixtures as pf


@pytest.fixture
def dataset(tmp_path) -> PhononDataset:
    pf.complete_phonon_calculation(tmp_path)
    result = phonon_service.load_phonons(tmp_path)
    assert result.ok
    return result.value


# --- parsing -------------------------------------------------------------------


def test_band_yaml_dimensions_and_labels(tmp_path):
    pf.write_band_yaml(tmp_path, n_qpoints=11)
    band_file = parse_band_yaml(tmp_path / "band.yaml").value

    assert len(band_file.bands) == 11
    assert band_file.n_atoms == 2
    assert band_file.species == ["Si", "Si"]
    assert band_file.supercell == (2, 2, 1)
    # TeX-style labels are normalised for display.
    assert band_file.bands[0].label == "Γ"
    assert band_file.bands[-1].label == "X"


def test_branch_count_is_three_per_atom(tmp_path):
    pf.write_band_yaml(tmp_path, species=["Si", "Si", "Si"])
    band_file = parse_band_yaml(tmp_path / "band.yaml").value
    assert band_file.bands[0].frequencies.size == 3 * 3


def test_eigenvectors_are_de_mass_weighted(tmp_path):
    """Phonopy's eigenvectors are mass-weighted; the parser divides by
    sqrt(mass) so what the viewer draws is a displacement. Re-applying the
    weight must recover an orthonormal set."""
    pf.write_band_yaml(tmp_path, species=["Si", "O"])
    band_file = parse_band_yaml(tmp_path / "band.yaml").value

    eigen = band_file.bands[3].eigenvectors
    assert eigen is not None
    masses = np.array([atomic_mass(s) for s in band_file.species])
    reweighted = (eigen * np.sqrt(masses)[None, :, None]).reshape(eigen.shape[0], -1)
    gram = reweighted.conj() @ reweighted.T
    np.testing.assert_allclose(gram, np.eye(len(reweighted)), atol=1e-6)


def test_eigenvectors_can_be_absent(tmp_path):
    pf.write_band_yaml(tmp_path, with_eigenvectors=False)
    band_file = parse_band_yaml(tmp_path / "band.yaml").value
    assert band_file.bands[0].eigenvectors is None
    assert band_file.bands[0].frequencies.size == 6


def test_a_file_with_no_qpoints_fails(tmp_path):
    path = tmp_path / "band.yaml"
    path.write_text("nqpoint: 0\nnatom: 2\nphonon:\n")
    assert parse_band_yaml(path).failed


def test_dos_dat_reads_frequency_and_values(tmp_path):
    pf.write_total_dos(tmp_path, n_points=25)
    dos = parse_dos_dat(tmp_path / "total_dos.dat").value
    assert dos.frequencies.size == 25
    assert dos.total.size == 25
    assert not dos.has_projection


# --- physics --------------------------------------------------------------------


def test_acoustic_branches_reach_zero_at_gamma(dataset):
    """The acoustic sum rule made real: three branches must vanish at q = 0."""
    gamma = dataset.bands[0]
    assert np.allclose(gamma.qpoint, 0.0)
    lowest = np.sort(gamma.frequencies)[:3]
    np.testing.assert_allclose(lowest, 0.0, atol=1e-6)


def test_a_stable_model_reports_no_imaginary_modes(dataset):
    assert dataset.imaginary_count() == 0
    assert dataset.is_stable


def test_a_negative_force_constant_produces_imaginary_modes(tmp_path):
    pf.unstable_phonon_calculation(tmp_path)
    unstable = phonon_service.load_phonons(tmp_path).value
    assert unstable.imaginary_count() > 0
    assert not unstable.is_stable


def test_near_zero_negatives_are_not_counted_as_unstable(dataset):
    """Acoustic modes at Γ come back at -1e-9 and similar. Counting those
    would report every converged calculation as unstable."""
    matrix = dataset.frequency_matrix
    assert (matrix < 0).any(), "the fixture should contain small negatives"
    assert (matrix < -IMAGINARY_TOLERANCE_THZ).sum() == 0
    assert dataset.imaginary_count() == 0


def test_mode_type_is_only_claimed_at_gamma(dataset):
    assert dataset.mode_type(0, 0) == "Acoustic"
    assert dataset.mode_type(0, 5) == "Optical"
    # Away from Γ branches mix, so no claim is made.
    assert dataset.mode_type(dataset.n_qpoints - 1, 0) == "—"


def test_degeneracy_counts_coincident_branches(dataset):
    gamma = 0
    # The three acoustic modes are degenerate at Γ.
    assert dataset.degeneracy(gamma, 0) >= 3


def test_group_velocity_is_none_at_a_path_discontinuity():
    """Two segments meeting at one point have zero separation, so the finite
    difference has no defined slope."""
    from vasp_analyzer.models.phonon import PhononBand

    # Three q-points stacked at the same path distance: the neighbours of the
    # middle one are zero apart, so dw/dq has no value there.
    dataset = PhononDataset(
        bands=[
            PhononBand(np.zeros(3), 0.5, np.array([1.0])),
            PhononBand(np.zeros(3), 0.5, np.array([2.0])),
            PhononBand(np.zeros(3), 0.5, np.array([3.0])),
        ]
    )
    assert dataset.group_velocity(1, 0) is None


def test_group_velocity_is_positive_on_a_real_branch(dataset):
    velocity = dataset.group_velocity(3, 0)
    assert velocity is not None and velocity >= 0.0


# --- DOS -------------------------------------------------------------------------


def test_gaussian_dos_integrates_to_the_mode_count():
    frequencies = np.array([1.0, 2.0, 3.0])
    dos = phonon_service.gaussian_dos(frequencies, smearing=0.05, n_points=4000)
    area = np.trapezoid(dos.total, dos.frequencies)
    assert area == pytest.approx(len(frequencies), rel=0.02)


def test_wider_smearing_broadens_without_changing_the_area():
    frequencies = np.array([2.0, 2.4])
    narrow = phonon_service.gaussian_dos(frequencies, 0.05, n_points=4000)
    wide = phonon_service.gaussian_dos(frequencies, 0.30, n_points=4000)

    assert wide.total.max() < narrow.total.max()
    assert np.trapezoid(wide.total, wide.frequencies) == pytest.approx(
        np.trapezoid(narrow.total, narrow.frequencies), rel=0.02
    )


def test_element_projection_sums_to_the_total(tmp_path):
    pf.write_band_yaml(tmp_path, species=["Si", "O"])
    pf.write_phonopy_yaml(tmp_path)
    dataset = phonon_service.load_phonons(tmp_path).value

    dos = phonon_service.project_dos_by_element(dataset, smearing=0.1, n_points=300)
    assert dos is not None
    assert set(dos.projected) == {"Si", "O"}
    stacked = sum(dos.projected.values())
    np.testing.assert_allclose(stacked, dos.total, rtol=1e-6, atol=1e-8)


def test_projection_needs_eigenvectors(tmp_path):
    pf.write_band_yaml(tmp_path, with_eigenvectors=False)
    dataset = phonon_service.load_phonons(tmp_path).value
    assert phonon_service.project_dos_by_element(dataset, 0.1) is None


# --- mode animation ---------------------------------------------------------------


def test_gamma_mode_moves_every_cell_identically(dataset):
    phases = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    frames = phonon_service.mode_displacements(
        dataset, 0, 3, phases, amplitude=1.0, supercell=(2, 1, 1)
    )
    assert frames is not None
    base = _supercell_positions(dataset, [(0, 0, 0), (1, 0, 0)])
    displacement = frames[1] - base
    n = dataset.n_atoms
    np.testing.assert_allclose(displacement[:n], displacement[n:], atol=1e-9)


def test_zone_boundary_mode_alternates_between_cells(dataset):
    """The q·R phase is what makes a phonon a wave. Without it a zone-boundary
    mode would look identical to a Γ mode."""
    phases = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    last = dataset.n_qpoints - 1
    assert dataset.bands[last].qpoint[0] == pytest.approx(0.5)

    frames = phonon_service.mode_displacements(
        dataset, last, 3, phases, amplitude=1.0, supercell=(2, 1, 1)
    )
    base = _supercell_positions(dataset, [(0, 0, 0), (1, 0, 0)])
    displacement = frames[1] - base
    n = dataset.n_atoms
    np.testing.assert_allclose(displacement[:n], -displacement[n:], atol=1e-9)


def test_amplitude_scales_the_displacement_linearly(dataset):
    phases = np.array([0.3])
    base = dataset.structure.cart_coords
    one = phonon_service.mode_displacements(dataset, 4, 2, phases, amplitude=1.0)
    two = phonon_service.mode_displacements(dataset, 4, 2, phases, amplitude=2.0)
    np.testing.assert_allclose(two - base, 2.0 * (one - base), atol=1e-9)


def test_animation_supercell_follows_the_wavevector(dataset):
    # q = 0 repeats every cell; q = 1/2 needs two to show the alternation.
    assert phonon_service.animation_supercell(dataset, 0) == (1, 1, 1)
    assert phonon_service.animation_supercell(dataset, dataset.n_qpoints - 1)[0] == 2


def test_animation_supercell_is_capped(dataset):
    """A long-wavelength mode must not ask for a hundred cells."""
    from vasp_analyzer.models.phonon import PhononBand

    dataset.bands.append(
        PhononBand(np.array([0.01, 0.0, 0.0]), 1.0, dataset.bands[0].frequencies)
    )
    repeats = phonon_service.animation_supercell(dataset, dataset.n_qpoints - 1)
    assert max(repeats) <= 4


def test_no_animation_without_eigenvectors(tmp_path):
    pf.write_band_yaml(tmp_path, with_eigenvectors=False)
    dataset = phonon_service.load_phonons(tmp_path).value
    phases = np.array([0.0])
    assert phonon_service.mode_displacements(dataset, 0, 0, phases) is None


def _supercell_positions(dataset, offsets) -> np.ndarray:
    structure = dataset.structure
    return np.concatenate(
        [(structure.frac_coords + np.array(o)) @ structure.lattice for o in offsets]
    )


# --- discovery --------------------------------------------------------------------


def test_phonon_root_finds_a_subdirectory(tmp_path):
    nested = tmp_path / "phonon"
    pf.complete_phonon_calculation(nested)
    assert phonon_service.phonon_root(tmp_path) == nested


def test_phonon_root_prefers_the_directory_itself(tmp_path):
    pf.complete_phonon_calculation(tmp_path)
    pf.complete_phonon_calculation(tmp_path / "phonon")
    assert phonon_service.phonon_root(tmp_path) == tmp_path


def test_a_directory_without_phonons_has_no_root(tmp_path):
    (tmp_path / "POSCAR").write_text("nothing\n")
    assert phonon_service.phonon_root(tmp_path) is None


def test_source_files_lists_only_what_exists(tmp_path):
    pf.write_band_yaml(tmp_path)
    pf.write_phonopy_yaml(tmp_path)
    sources = phonon_service.describe_sources(tmp_path)
    assert "band.yaml" in sources and "phonopy.yaml" in sources
    assert "total_dos.dat" not in sources


def test_missing_dos_removes_only_the_dos(tmp_path):
    pf.write_band_yaml(tmp_path)
    pf.write_phonopy_yaml(tmp_path)
    dataset = phonon_service.load_phonons(tmp_path).value

    assert dataset.dos is None
    assert dataset.has_dispersion          # dispersion still works
    assert dataset.has_eigenvectors        # so does animation


# --- real phonopy format compatibility -------------------------------------------
#
# The fixtures above write an inline `label:` per q-point, which is not what
# real phonopy output does — a real band.yaml (checked against the user's
# own Graphene calculation) names high-symmetry points once per path segment,
# in a header `labels:`/`segment_nqpoint:` block instead.


def test_segment_labels_are_read_from_the_header_block(tmp_path):
    pf.write_band_yaml_segmented(tmp_path, points_per_segment=6)
    band_file = parse_band_yaml(tmp_path / "band.yaml").value

    # Two 6-point segments sharing a boundary point: index 5 and 6 are both
    # "M", the point where the path continues without a direction change.
    labelled = [(i, b.label) for i, b in enumerate(band_file.bands) if b.label]
    assert labelled == [(0, "Γ"), (5, "M"), (6, "M"), (11, "K")]


def test_a_straight_through_point_merges_to_one_tick_not_a_pipe(tmp_path):
    """Regression: Γ-M-K's M is labelled from both the segment ending there
    and the one starting there. Both say "M", so the dispersion plot must
    show one "M" tick, not the literal duplicate "M|M"."""
    from vasp_analyzer.ui.pages.phonons.plots import DispersionPlot

    pf.write_band_yaml_segmented(tmp_path, points_per_segment=6)
    from vasp_analyzer.services.phonon import load_phonons

    dataset = load_phonons(tmp_path).value

    plot = DispersionPlot()
    plot.set_dataset(dataset)
    labels = [t.get_text() for t in plot.axes.get_xticklabels()]
    assert "M" in labels
    assert "M|M" not in labels


def test_genuinely_different_adjacent_labels_still_pipe(tmp_path):
    """A real direction change at a shared boundary (not a straight-through
    point) is the case the "X|M" merge exists for, and must still work."""
    from vasp_analyzer.services.phonon import load_phonons
    from vasp_analyzer.ui.pages.phonons.plots import DispersionPlot

    # Distinct end/start labels at the shared boundary this time.
    pf.write_band_yaml_segmented(
        tmp_path, points_per_segment=6, path_labels=(r"$\Gamma$", "M", "K")
    )
    band_file = parse_band_yaml(tmp_path / "band.yaml").value
    # Force a direction-change case by relabelling the boundary asymmetrically.
    band_file.bands[5].label = "X"
    band_file.bands[6].label = "M"

    dataset = load_phonons(tmp_path).value
    dataset.bands[5].label = "X"
    dataset.bands[6].label = "M"

    plot = DispersionPlot()
    plot.set_dataset(dataset)
    labels = [t.get_text() for t in plot.axes.get_xticklabels()]
    assert "X|M" in labels


def test_inline_labels_still_take_priority_over_the_header_block(tmp_path):
    """A file carrying the older inline `label:` form must not be overridden
    by a coincidentally-present (but irrelevant) header block."""
    pf.write_band_yaml(tmp_path, n_qpoints=8, labels=("Γ", "X"))
    band_file = parse_band_yaml(tmp_path / "band.yaml").value
    assert band_file.bands[0].label == "Γ"
    assert band_file.bands[-1].label == "X"


# --- DOS without a separate total_dos.dat -----------------------------------------


def test_dos_falls_back_to_projected_dos_dat_alone(tmp_path):
    """phonopy's --pdos runs write projected_dos.dat without a separate
    total_dos.dat. Regression: this used to mean no DOS at all, even though
    the total is recoverable by summing the per-atom columns."""
    pf.write_band_yaml(tmp_path, species=["Si", "Si"])
    pf.write_projected_dos(tmp_path, n_atoms=2)
    pf.write_phonopy_yaml(tmp_path)
    assert not (tmp_path / "total_dos.dat").exists()

    dataset = phonon_service.load_phonons(tmp_path).value
    assert dataset.dos is not None
    assert dataset.dos.has_projection
    assert dataset.dos.total.sum() > 0


def test_total_dos_dat_is_preferred_when_both_exist(tmp_path):
    pf.write_band_yaml(tmp_path, species=["Si", "Si"])
    pf.write_total_dos(tmp_path)
    pf.write_projected_dos(tmp_path, n_atoms=2)
    dataset = phonon_service.load_phonons(tmp_path).value
    assert dataset.dos is not None
    assert dataset.dos.has_projection


# --- nested discovery --------------------------------------------------------------


def test_phonon_root_finds_a_folder_nested_two_levels_deep(tmp_path):
    """Mirrors a real layout: the raw DFPT run lives directly under a
    "phonon" folder, and Phonopy's post-processed output (what actually has
    band.yaml) lives in its own differently-named child folder — "dispersion"
    here, but the name itself must not matter."""
    dfpt_run = tmp_path / "phonon"
    dfpt_run.mkdir()
    (dfpt_run / "POSCAR").write_text("not phonon output\n")
    nested = dfpt_run / "dispersion"
    pf.complete_phonon_calculation(nested)

    assert phonon_service.phonon_root(tmp_path) == nested


def test_phonon_root_does_not_search_past_its_depth_limit(tmp_path):
    too_deep = tmp_path / "a" / "b" / "c"
    pf.complete_phonon_calculation(too_deep)
    assert phonon_service.phonon_root(tmp_path) is None
