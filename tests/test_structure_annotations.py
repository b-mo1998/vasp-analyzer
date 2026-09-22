"""Per-atom annotations on the Structure page, and selection geometry on MD.

Four features that share one shape -- data the app already had access to but
threw away, now surfaced per atom:

* magnetic moments from OUTCAR's magnetization table,
* relaxation displacements from POSCAR -> CONTCAR,
* symmetry-equivalent atoms and Wyckoff positions from spglib,
* a distance or angle over time from the MD multi-selection.
"""

from __future__ import annotations

import numpy as np
import pytest

from vasp_analyzer.models.structure import StructureData
from vasp_analyzer.models.trajectory import MDTrajectory
from vasp_analyzer.parsers.outcar import parse_magnetic_moments
from vasp_analyzer.services import relaxation
from vasp_analyzer.services.discovery import discover
from vasp_analyzer.services.loader import DataLoader
from vasp_analyzer.services.symmetry import equivalence
from vasp_analyzer.services.trajectory import angle_series, pair_distance_series
from vasp_analyzer.ui.pages.structure.page import StructurePage
from vasp_analyzer.ui.pages.structure.viewer import StructureView, VectorOverlay

from . import fixtures


# -- OUTCAR magnetization -------------------------------------------------

MAG_TABLE = """
# of ion       s       p       d       tot
------------------------------------------
    1        0.012   0.034   2.101   {a}
    2        0.012  -0.034  -2.101   {b}
------------------------------------------
tot          0.024   0.000   0.000   0.024
"""


def _write_outcar(path, blocks: list[tuple[str, float, float]]) -> None:
    text = "header\n E-fermi :   1.0\n"
    for axis, a, b in blocks:
        text += f"\n magnetization ({axis})\n" + MAG_TABLE.format(a=a, b=b)
    path.write_text(text)


def test_collinear_moments_come_from_the_last_table(tmp_path):
    """Earlier tables belong to unconverged ionic steps; only the final one
    describes the result."""
    outcar = tmp_path / "OUTCAR"
    _write_outcar(outcar, [("x", 1.0, -1.0), ("x", 2.147, -2.123)])
    result = parse_magnetic_moments(outcar)

    assert result.ok
    moments = result.value
    assert moments.collinear
    assert moments.scalar.tolist() == pytest.approx([2.147, -2.123])
    # Collinear moments are quantised along z by convention.
    assert moments.vectors[0].tolist() == pytest.approx([0.0, 0.0, 2.147])
    assert moments.total == pytest.approx(0.024)


def test_noncollinear_moments_assemble_a_vector(tmp_path):
    outcar = tmp_path / "OUTCAR"
    _write_outcar(outcar, [("x", 1.0, -1.0), ("y", 0.5, -0.5), ("z", 0.25, -0.25)])
    result = parse_magnetic_moments(outcar)

    assert result.ok
    assert not result.value.collinear
    assert result.value.vectors[0].tolist() == pytest.approx([1.0, 0.5, 0.25])


def test_an_outcar_without_magnetization_fails_cleanly(tmp_path):
    outcar = tmp_path / "OUTCAR"
    outcar.write_text("E-fermi : 1.0\n no tables here\n")
    result = parse_magnetic_moments(outcar)

    assert result.failed
    assert "magnetization" in result.issues[0].message


# -- relaxation displacements -----------------------------------------------


def _cell(coords, lattice=None, species=None) -> StructureData:
    coords = np.asarray(coords, dtype=float)
    return StructureData(
        lattice=np.eye(3) * 5.0 if lattice is None else np.asarray(lattice),
        species=species or ["Si"] * len(coords),
        frac_coords=coords,
        source="POSCAR",
    )


def test_displacement_uses_the_minimum_image():
    """CONTCAR re-wraps into the cell, so an atom that crossed a face by 0.02
    is written 0.98 of a cell away. Without the minimum image that reads as
    a 4.9 Å move; the physical move is 0.1 Å."""
    initial = _cell([[0.01, 0.5, 0.5]])
    final = _cell([[0.99, 0.5, 0.5]])
    result = relaxation.compare(initial, final)

    assert result is not None
    assert result.magnitudes[0] == pytest.approx(0.1, abs=1e-9)
    assert result.vectors[0][0] == pytest.approx(-0.1, abs=1e-9)


def test_displacement_is_measured_in_the_final_cell():
    """A variable-cell relaxation changes the lattice; the relaxed geometry
    lives in the final cell, so that is the one that converts fractional
    shifts to ångström."""
    initial = _cell([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], lattice=np.eye(3) * 4.0)
    final = _cell([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0]], lattice=np.eye(3) * 5.0)
    result = relaxation.compare(initial, final)

    assert result.magnitudes[1] == pytest.approx(0.5)   # 0.1 * 5 Å, not * 4 Å


def test_different_atoms_are_not_compared():
    assert relaxation.compare(_cell([[0, 0, 0]]), _cell([[0, 0, 0], [0.5, 0.5, 0.5]])) is None
    assert relaxation.compare(
        _cell([[0, 0, 0]], species=["Si"]), _cell([[0, 0, 0]], species=["C"])
    ) is None


def test_ranked_lists_the_most_displaced_first():
    initial = _cell([[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]])
    final = _cell([[0.01, 0, 0], [0.5, 0.06, 0], [0, 0.5, 0.03]])
    result = relaxation.compare(initial, final)

    assert result.ranked(3) == [1, 2, 0]
    assert result.max_atom == 1


def test_tile_repeats_cell_major_atom_minor():
    """build_supercell flattens every base atom once per image cell, image
    cells in order; the tiled vectors must interleave identically or arrow i
    lands on the wrong atom."""
    vectors = np.array([[1.0, 0, 0], [2.0, 0, 0]])
    tiled = relaxation.tile(vectors, 3)

    assert tiled.shape == (6, 3)
    assert tiled[:, 0].tolist() == [1, 2, 1, 2, 1, 2]


# -- symmetry equivalence -----------------------------------------------------


def _diamond_si() -> StructureData:
    return StructureData(
        lattice=np.eye(3) * 5.43,
        species=["Si"] * 8,
        frac_coords=np.array([
            [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
            [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75],
        ]),
        source="POSCAR",
    )


def test_all_diamond_silicon_atoms_are_one_orbit():
    orbits = equivalence(_diamond_si())

    assert orbits is not None
    assert orbits.multiplicity(0) == 8
    assert sorted(orbits.equivalents_of(3)) == list(range(8))
    # The letter only: the "8" in "8a" is the conventional-cell multiplicity,
    # which is not what an orbit count in an arbitrary supercell measures.
    assert orbits.wyckoff_label(0) == "a"
    assert orbits.site_symmetry[0] == "-43m"


def test_a_lone_dopant_is_its_own_orbit():
    structure = _diamond_si()
    structure.species[4] = "Ge"
    orbits = equivalence(structure)

    assert orbits is not None
    assert orbits.equivalents_of(4) == [4]
    assert 4 not in orbits.equivalents_of(0)


# -- viewer overlays and highlight ---------------------------------------


@pytest.fixture()
def view(qtbot):
    widget = StructureView()
    qtbot.addWidget(widget)
    widget.resize(480, 400)
    widget.set_structure(_diamond_si())
    widget.show()
    widget.grab()
    return widget


def test_an_overlay_changes_the_render(view):
    before = view.grab().toImage()
    vectors = np.tile([[0.0, 0.0, 1.0]], (8, 1))
    view.set_vector_overlay("m", VectorOverlay(vectors, "#e11d48", scale=1.0))

    assert view.grab().toImage() != before


def test_highlighted_atoms_ring_but_do_not_select(view):
    view.set_highlighted_atoms({1, 2, 3})

    assert view.highlighted_atoms() == {1, 2, 3}
    assert view.selected_index() is None


def test_a_new_structure_drops_overlays_and_highlights(view):
    view.set_vector_overlay("m", VectorOverlay(np.zeros((8, 3)), "#000"))
    view.set_highlighted_atoms({0})
    view.set_structure(_cell([[0, 0, 0]]))

    assert view.vector_overlay("m") is None
    assert view.highlighted_atoms() == set()


# -- the Structure page ---------------------------------------------------


def _relaxed_calculation(tmp_path):
    """POSCAR, a CONTCAR displaced from it, and an OUTCAR with moments."""
    fixtures.write_poscar(tmp_path, atoms=2)
    poscar = (tmp_path / "POSCAR").read_text()
    # CONTCAR: move the second atom by +0.1 fractional along x.
    contcar_lines = poscar.splitlines()
    contcar_lines[-1] = "  0.60000000  0.50000000  0.50000000"
    (tmp_path / "CONTCAR").write_text("\n".join(contcar_lines) + "\n")
    _write_outcar(tmp_path / "OUTCAR", [("x", 1.5, -1.5)])
    return discover(tmp_path)


@pytest.fixture()
def page(qtbot, tmp_path):
    loader = DataLoader()
    widget = StructurePage(loader)
    qtbot.addWidget(widget)
    calculation = _relaxed_calculation(tmp_path)
    loader.set_calculation(calculation)
    widget.set_calculation(calculation)
    qtbot.waitUntil(lambda: widget._structure is not None, timeout=5000)
    return widget


def test_the_page_loads_moments_and_displacements(page):
    assert page._moments is not None
    assert page._sidebar.show_moments.isEnabled()
    assert page._relaxation is not None
    assert page._sidebar.show_displacements.isEnabled()
    assert not page._info._mag_card.isHidden()
    assert not page._info._relax_card.isHidden()


def test_toggling_moments_installs_the_overlay(page, qtbot):
    assert page._view.vector_overlay("moments") is None
    page._sidebar.show_moments.setChecked(True)
    qtbot.wait(20)
    overlay = page._view.vector_overlay("moments")

    assert overlay is not None
    assert overlay.vectors.shape == (2, 3)
    # Down-moments get the signed colour so antiferromagnetic order reads
    # as alternating colour, not just alternating direction.
    assert overlay.signed_color is not None


def test_displacement_overlay_follows_the_exaggeration_slider(page, qtbot):
    page._sidebar.show_displacements.setChecked(True)
    page._sidebar.displacement_scale.setValue(25.0)
    qtbot.wait(20)

    assert page._view.vector_overlay("displacements").scale == pytest.approx(25.0)


def test_overlays_are_tiled_with_the_supercell(page, qtbot):
    page._sidebar.show_moments.setChecked(True)
    page._on_supercell_changed(2, 2, 1)
    qtbot.wait(50)
    overlay = page._view.vector_overlay("moments")

    assert page._structure.n_atoms == 8
    assert overlay is not None and overlay.vectors.shape[0] == 8


def test_selecting_an_atom_reports_its_annotations(page, qtbot):
    page._on_atom_selected(1)
    qtbot.wait(20)
    rows = page._info._rows

    assert not rows["atom_moment"].isHidden()
    assert not rows["atom_displacement"].isHidden()
    assert not rows["atom_wyckoff"].isHidden()
    # Atom 2 moved 0.1 fractional along x in a 5.4667 Å cell.
    assert page._relaxation.magnitudes[1] == pytest.approx(0.54667, abs=1e-4)
    assert page._moments.scalar[1] == pytest.approx(-1.5)


def test_selection_highlights_the_equivalent_atoms(page, qtbot):
    page._on_atom_selected(0)
    qtbot.wait(20)
    # Two Si on a body-centred pair: symmetry-equivalent to each other.
    assert page._view.highlighted_atoms() == {1}

    page._sidebar.highlight_equivalent.setChecked(False)
    qtbot.wait(20)
    assert page._view.highlighted_atoms() == set()


def test_a_calculation_without_contcar_disables_the_toggle(qtbot, tmp_path):
    fixtures.write_poscar(tmp_path, atoms=2)
    loader = DataLoader()
    page = StructurePage(loader)
    qtbot.addWidget(page)
    calculation = discover(tmp_path)
    loader.set_calculation(calculation)
    page.set_calculation(calculation)
    qtbot.waitUntil(lambda: page._structure is not None, timeout=5000)

    assert page._relaxation is None
    assert not page._sidebar.show_displacements.isEnabled()
    assert page._moments is None
    assert not page._sidebar.show_moments.isEnabled()
    assert page._info._relax_card.isHidden()


# -- MD selection geometry -------------------------------------------------


def _trajectory() -> MDTrajectory:
    lattices = np.tile(np.eye(3) * 10.0, (4, 1, 1))
    frac = np.array([
        [[0.0, 0, 0], [0.10, 0, 0], [0.10, 0.10, 0]],
        [[0.0, 0, 0], [0.12, 0, 0], [0.12, 0.12, 0]],
        [[0.0, 0, 0], [0.14, 0, 0], [0.14, 0.14, 0]],
        [[0.98, 0, 0], [0.14, 0, 0], [0.14, 0.14, 0]],   # atom 0 wraps across a face
    ])
    return MDTrajectory(species=["C", "C", "C"], frac_coords=frac, lattices=lattices)


def test_pair_distance_series_tracks_a_stretching_bond():
    d = pair_distance_series(_trajectory(), 0, 1)
    assert d[:3].tolist() == pytest.approx([1.0, 1.2, 1.4])


def test_pair_distance_uses_the_minimum_image_per_frame():
    """Frame 4 has atom 0 wrapped to x=0.98: naively that is 8.4 Å from atom
    1, physically it is 1.6 Å."""
    d = pair_distance_series(_trajectory(), 0, 1)
    assert d[3] == pytest.approx(1.6)


def test_angle_series_at_the_vertex():
    a = angle_series(_trajectory(), 0, 1, 2)
    assert a[0] == pytest.approx(90.0)


# -- overlays that would draw nothing are withheld, with the reason -----------


def test_a_run_that_converged_non_magnetic_disables_the_moments_toggle(
    qtbot, tmp_path
):
    """ISPIN=2 with MAGMOM=0 everywhere still writes magnetization tables --
    full of zeros. The card reports the zero; an enabled toggle that drew
    nothing read as broken instead."""
    fixtures.write_poscar(tmp_path, atoms=2)
    _write_outcar(tmp_path / "OUTCAR", [("x", 0.0, -0.0)])
    loader = DataLoader()
    page = StructurePage(loader)
    qtbot.addWidget(page)
    calculation = discover(tmp_path)
    loader.set_calculation(calculation)
    page.set_calculation(calculation)
    qtbot.waitUntil(lambda: page._structure is not None, timeout=5000)

    assert page._moments is not None, "the zeros are still parsed and reported"
    assert not page._info._mag_card.isHidden()
    assert not page._sidebar.show_moments.isEnabled()
    assert "zero" in page._sidebar.show_moments.toolTip()


def test_an_unmoved_contcar_disables_the_displacement_toggle(qtbot, tmp_path):
    """A static run (SCF, bands, DOS) copies POSCAR to CONTCAR unchanged, so
    "what did the relaxation do" has no answer there. Gated on the data, not
    the folder name, so a real relaxation anywhere still qualifies."""
    fixtures.write_poscar(tmp_path, atoms=2)
    (tmp_path / "CONTCAR").write_text((tmp_path / "POSCAR").read_text())
    loader = DataLoader()
    page = StructurePage(loader)
    qtbot.addWidget(page)
    calculation = discover(tmp_path)
    loader.set_calculation(calculation)
    page.set_calculation(calculation)
    qtbot.waitUntil(lambda: page._structure is not None, timeout=5000)

    assert page._relaxation is None
    assert not page._sidebar.show_displacements.isEnabled()
    assert "identical" in page._sidebar.show_displacements.toolTip()
    assert page._info._relax_card.isHidden()


def test_the_relaxation_card_has_no_largest_move_row(page):
    assert "relax_max" not in page._info._rows
    assert "relax_mean" in page._info._rows and "relax_rms" in page._info._rows


def test_long_values_wrap_instead_of_being_cut_off(qtbot):
    """The panel is fixed-width, so a truncated value could never be revealed
    by enlarging the window. Nothing may be truncated: a long value wraps and
    the row grows taller."""
    from vasp_analyzer.ui.pages.structure.info_panel import InfoRow

    row = InfoRow("System")
    qtbot.addWidget(row)
    row.setFixedWidth(300)
    row.show()
    short_height = row.sizeHint().height()
    row.set_value("Graphene 5x5 | 300 K NVT AIMD | 2-hour pilot of a long comment")
    qtbot.wait(10)

    assert "…" not in row._value.text()
    assert row._value.text().endswith("long comment")
    assert row.sizeHint().height() > short_height, "it wrapped onto more lines"
    assert row.minimumSizeHint().width() <= 300, "and never widens the panel"


def test_equivalent_atoms_are_counted_not_elided(page, qtbot):
    page._on_atom_selected(0)
    qtbot.wait(20)
    text = page._info._rows["atom_equivalents"]._value.text()

    assert "…" not in text


# -- non-collinear spin components ------------------------------------------


def _canted_outcar(path) -> None:
    """A spin texture rotating in the xy plane with a uniform z canting --
    the case a non-collinear calculation exists to produce."""
    import numpy as _np

    angles = _np.radians([0, 90, 180, 270])
    components = {
        "x": _np.cos(angles) * 2.0,
        "y": _np.sin(angles) * 2.0,
        "z": _np.full(4, 0.4),
    }
    text = "header\n E-fermi : 1.0\n"
    for axis in ("x", "y", "z"):
        rows = "\n".join(
            f"    {i + 1}        0.0   0.0   {v:.3f}   {v:.3f}"
            for i, v in enumerate(components[axis])
        )
        text += (
            f"\n magnetization ({axis})\n"
            "\n# of ion       s       p       d       tot\n"
            "------------------------------------------\n" + rows +
            "\n------------------------------------------\ntot  0 0 0 0\n"
        )
    path.write_text(text)


def test_spin_components_are_read_per_axis(tmp_path):
    outcar = tmp_path / "OUTCAR"
    _canted_outcar(outcar)
    moments = parse_magnetic_moments(outcar).value

    assert not moments.collinear
    assert moments.component("x").tolist() == pytest.approx([2, 0, -2, 0], abs=1e-9)
    assert moments.component("y").tolist() == pytest.approx([0, 2, 0, -2], abs=1e-9)
    assert moments.component("z").tolist() == pytest.approx([0.4] * 4)


def test_net_moment_is_the_vector_sum_not_the_summed_magnitude(tmp_path):
    """An antiferromagnet has a large Σ|m| and a net of nearly zero. Reporting
    only one of the two describes it wrongly either way, so both are kept."""
    outcar = tmp_path / "OUTCAR"
    _canted_outcar(outcar)
    moments = parse_magnetic_moments(outcar).value

    # The in-plane components cancel; only the canting survives.
    assert moments.total_vector.tolist() == pytest.approx([0.0, 0.0, 1.6], abs=1e-9)
    assert moments.net_magnitude == pytest.approx(1.6)
    assert float(moments.magnitudes.sum()) > 8.0


def test_isolating_a_component_zeroes_the_others(tmp_path):
    outcar = tmp_path / "OUTCAR"
    _canted_outcar(outcar)
    moments = parse_magnetic_moments(outcar).value
    isolated = moments.component_overlay_vectors("y")

    assert isolated[:, 0].tolist() == pytest.approx([0.0] * 4)
    assert isolated[:, 2].tolist() == pytest.approx([0.0] * 4)
    assert isolated[:, 1].tolist() == pytest.approx([0, 2, 0, -2], abs=1e-9)


def test_dominant_axis_finds_where_the_weight_is(tmp_path):
    outcar = tmp_path / "OUTCAR"
    _canted_outcar(outcar)
    # |x| sums to 4.0, |y| to 4.0, |z| to 1.6 -- x wins the tie by argmax order.
    assert parse_magnetic_moments(outcar).value.dominant_axis() in ("x", "y")


def _open_page(qtbot, directory):
    """A Structure page showing ``directory``.

    The loader has to be pointed at the calculation as well as the page --
    the page asks it for the structure, so updating only one of the two
    leaves the request unanswered.
    """
    loader = DataLoader()
    widget = StructurePage(loader)
    qtbot.addWidget(widget)
    calculation = discover(directory)
    loader.set_calculation(calculation)
    widget.set_calculation(calculation)
    qtbot.waitUntil(lambda: widget._structure is not None, timeout=5000)
    return widget


@pytest.fixture()
def noncollinear_page(qtbot, tmp_path):
    fixtures.write_poscar(tmp_path, atoms=4)
    _canted_outcar(tmp_path / "OUTCAR")
    return _open_page(qtbot, tmp_path)


def test_the_component_selector_is_offered_only_for_non_collinear(
    noncollinear_page, qtbot, tmp_path
):
    """In a collinear run the moment *is* S_z by construction, so offering
    S_x and S_y would present a convention as a result."""
    page = noncollinear_page
    page._sidebar.show_moments.setChecked(True)
    qtbot.wait(20)
    assert page._sidebar.moment_component.isEnabled()

    collinear_dir = tmp_path / "collinear"
    collinear_dir.mkdir()
    fixtures.write_poscar(collinear_dir, atoms=2)
    _write_outcar(collinear_dir / "OUTCAR", [("x", 2.0, -2.0)])
    collinear = _open_page(qtbot, collinear_dir)
    collinear._sidebar.show_moments.setChecked(True)
    qtbot.wait(20)

    assert collinear._moments is not None and collinear._moments.collinear
    assert not collinear._sidebar.moment_component.isEnabled()


def test_each_component_draws_a_different_overlay(noncollinear_page, qtbot):
    page = noncollinear_page
    page._sidebar.show_moments.setChecked(True)
    seen = {}
    for key, label in (("x", "Sx only"), ("y", "Sy only"), ("z", "Sz only")):
        page._sidebar.moment_component.setCurrentText(label)
        qtbot.wait(20)
        overlay = page._view.vector_overlay("moments")
        seen[key] = overlay.vectors.copy()
        # The sign colour has to follow the axis being isolated, or every
        # arrow in an S_x view reads as "up" from the z test.
        assert overlay.signed_axis == "xyz".index(key)

    assert not np.allclose(seen["x"], seen["y"])
    assert not np.allclose(seen["y"], seen["z"])


def test_the_atom_card_reports_all_three_components(noncollinear_page, qtbot):
    page = noncollinear_page
    page._on_atom_selected(1)
    qtbot.wait(20)
    row = page._info._rows["atom_spin"]

    assert not row.isHidden()
    assert "+2.000" in row._value.text()   # S_y of atom 2


def test_the_magnetization_card_separates_net_from_summed(noncollinear_page, qtbot):
    page = noncollinear_page
    qtbot.wait(20)
    rows = page._info._rows

    assert "Non-collinear" in rows["mag_kind"]._value.text()
    assert not rows["mag_net_components"].isHidden()
    assert "+1.600" in rows["mag_net_components"]._value.text()
    assert rows["mag_net"]._value.text().startswith("1.600")
