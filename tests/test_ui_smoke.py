"""End-to-end checks that the window survives every partial calculation.

The central promise of the architecture is that missing or broken files remove
a capability instead of breaking the application. These tests open each
scenario in a real window and visit every page.
"""

from __future__ import annotations

import pytest

from vasp_analyzer.models.capabilities import Capability
from vasp_analyzer.ui.main_window import MainWindow
from vasp_analyzer.ui.plotting import apply_plot_style
from vasp_analyzer.ui.style import apply_theme

from . import fixtures

SCENARIOS = [
    ("complete", fixtures.case_a_complete),
    ("bands_only", fixtures.case_b_bands_only),
    ("no_eigenval", fixtures.case_c_no_eigenval),
    ("structure_only", fixtures.case_d_structure_only),
    ("partial_wannier", fixtures.case_e_partial_wannier),
    ("malformed_doscar", fixtures.case_f_malformed_doscar),
    ("empty", fixtures.case_g_empty),
]


@pytest.fixture(scope="module", autouse=True)
def _style(qapp):
    apply_theme(qapp)
    apply_plot_style()
    return qapp


@pytest.fixture
def window(qtbot, monkeypatch, tmp_path):
    # Keep the test run out of the user's real settings and recent list.
    #
    # Patching individual Settings methods isn't enough on its own: the base
    # class still opens the OS's real native preference store underneath
    # (QSettings(ORGANISATION, APPLICATION)), and services.appearance holds
    # its own separate Settings() built from that same real store — bound at
    # import time, so it is untouched by a monkeypatch on the Settings class
    # itself. Redirecting QSettings to a temp-file-backed store is the one
    # interception point that reliably isolates every path that opens one.
    from PySide6.QtCore import QSettings

    from vasp_analyzer.services import appearance
    from vasp_analyzer.services import settings as settings_module

    ini_path = str(tmp_path / "test_settings.ini")
    monkeypatch.setattr(
        settings_module,
        "QSettings",
        lambda *a, **k: QSettings(ini_path, QSettings.IniFormat),
    )
    monkeypatch.setattr(appearance, "_settings", None)
    monkeypatch.setattr(appearance, "_overrides", {})

    class _IsolatedSettings(settings_module.Settings):
        def __init__(self) -> None:
            super().__init__()
            self._store: dict = {}

        def last_directory(self):
            return None

        def set_last_directory(self, path):
            self._store["last"] = path

        def recent_directories(self):
            return []

        def window_geometry(self):
            return None

        def set_window_geometry(self, geometry):
            pass

        def sync(self):
            pass

    monkeypatch.setattr(settings_module, "Settings", _IsolatedSettings)
    monkeypatch.setattr(
        "vasp_analyzer.ui.main_window.Settings", _IsolatedSettings, raising=True
    )

    win = MainWindow()
    qtbot.addWidget(win)
    return win


@pytest.mark.parametrize("name,builder", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_every_scenario_opens_and_navigates(window, qtbot, tmp_path, name, builder):
    directory = tmp_path / name
    directory.mkdir()
    builder(directory)

    window.open_directory(directory)
    qtbot.wait(120)

    # Visit every page that navigation allows.
    for index in range(3):
        window._nav.setCurrentIndex(index)
        qtbot.wait(40)

    assert window.isEnabled()


def test_bands_page_shows_data_when_available(window, qtbot, tmp_path):
    directory = tmp_path / "complete"
    directory.mkdir()
    fixtures.case_a_complete(directory)

    window.open_directory(directory)
    qtbot.wait(200)

    page = window._bands
    assert page.calculation().can(Capability.BANDS)
    qtbot.waitUntil(lambda: page._bands is not None, timeout=3000)
    assert page._bands.n_kpoints > 0
    assert page._stack.currentIndex() == 0, "the plot, not the empty state"


def test_bands_page_shows_dos_only_without_eigenval(window, qtbot, tmp_path):
    """Case C (DOSCAR without EIGENVAL): DOS stands alone fine and needs
    nothing from EIGENVAL, so the page shows the plot area (DOS working,
    band plot empty) with an explanatory notice — not the full-page
    "unavailable" block, which used to fire for any missing EIGENVAL
    regardless of what else was available."""
    directory = tmp_path / "no_bands"
    directory.mkdir()
    fixtures.case_c_no_eigenval(directory)

    window.open_directory(directory)
    qtbot.wait(150)

    page = window._bands
    assert page._stack.currentIndex() == 0, "the plot area, not the empty state"
    qtbot.waitUntil(lambda: page._dos is not None, timeout=3000)
    assert page._bands is None
    # isVisible() needs the window to actually be shown on screen, which this
    # fixture deliberately doesn't do; the page's own tracked state is the
    # thing that actually matters here.
    assert page._dos_visible is True
    assert not page._dos_card.isHidden()


def test_bands_page_shows_empty_state_with_neither_bands_nor_dos(window, qtbot, tmp_path):
    directory = tmp_path / "truly_empty"
    directory.mkdir()
    fixtures.case_d_structure_only(directory)  # POSCAR + INCAR only

    window.open_directory(directory)
    qtbot.wait(150)

    assert window._bands._stack.currentIndex() == 1, "the empty state"


def test_dos_controls_disabled_without_doscar(window, qtbot, tmp_path):
    directory = tmp_path / "no_dos"
    directory.mkdir()
    fixtures.case_b_bands_only(directory)

    window.open_directory(directory)
    qtbot.wait(150)

    page = window._bands
    assert not page._sidebar.show_dos.isEnabled()
    assert "DOSCAR" in page._sidebar.show_dos.toolTip()
    assert not page._dos_card.isVisible()


def test_dos_export_and_copy_work_without_eigenval(window, qtbot, tmp_path, monkeypatch):
    """DOSCAR without EIGENVAL (Case C): export()/copy_to_clipboard() used to
    bail out unconditionally whenever bands were absent, even though DOS
    alone is perfectly exportable."""
    from PySide6.QtWidgets import QFileDialog

    directory = tmp_path / "dos_only"
    directory.mkdir()
    fixtures.case_c_no_eigenval(directory)

    window.open_directory(directory)
    qtbot.wait(150)

    page = window._bands
    qtbot.waitUntil(lambda: page._dos is not None, timeout=3000)
    assert page._bands is None

    assert page.copy_to_clipboard() is True

    out_path = tmp_path / "dos-export.png"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(out_path), "DOS image (*.png)")),
    )
    page.export()
    assert out_path.exists()


def test_export_is_a_no_op_with_neither_bands_nor_dos(window, qtbot, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    directory = tmp_path / "neither"
    directory.mkdir()
    fixtures.case_d_structure_only(directory)  # POSCAR + INCAR only

    window.open_directory(directory)
    qtbot.wait(100)

    page = window._bands
    assert page.copy_to_clipboard() is False

    called = []
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: called.append(1) or ("", "")),
    )
    page.export()
    assert not called, "the file dialog should never open with nothing to export"


def test_inputs_page_lists_and_opens_a_file(window, qtbot, tmp_path):
    directory = tmp_path / "inputs"
    directory.mkdir()
    fixtures.case_a_complete(directory)

    window.open_directory(directory)
    window._nav.setCurrentIndex(2)
    qtbot.wait(150)

    page = window._inputs
    assert page._current is not None
    assert page._editor.toPlainText()
    assert page._stack.currentIndex() == 0


def test_inputs_page_marks_outputs_read_only(window, qtbot, tmp_path):
    directory = tmp_path / "readonly"
    directory.mkdir()
    fixtures.case_a_complete(directory)

    window.open_directory(directory)
    window._nav.setCurrentIndex(2)
    qtbot.wait(120)

    page = window._inputs
    page.show_file("outcar")
    qtbot.wait(60)

    assert not page._edit_button.isEnabled(), "OUTCAR is output and must stay read-only"

    page.show_file("incar")
    qtbot.wait(60)
    assert page._edit_button.isEnabled()


def test_inputs_search_reports_matches(window, qtbot, tmp_path):
    directory = tmp_path / "search"
    directory.mkdir()
    fixtures.case_a_complete(directory)

    window.open_directory(directory)
    window._nav.setCurrentIndex(2)
    qtbot.wait(120)

    page = window._inputs
    page.show_file("incar")
    page._search.setText("ENCUT")
    qtbot.wait(60)

    assert "1 of" in page._match_label.text()


def test_element_projection_end_to_end(window, qtbot, tmp_path):
    directory = tmp_path / "projection"
    directory.mkdir()
    fixtures.write_eigenval(directory, n_kpoints=6, n_bands=4, n_electrons=8, n_ions=2)
    fixtures.write_poscar(directory, atoms=2)
    fixtures.write_procar(directory, n_kpoints=6, n_bands=4, n_ions=2)

    window.open_directory(directory)
    qtbot.wait(150)

    page = window._bands
    checkbox = page._sidebar.projections["element"]
    assert checkbox.isEnabled(), "PROCAR is present and matches EIGENVAL"

    checkbox.setChecked(True)
    qtbot.waitUntil(lambda: page._band_plot._projection is not None, timeout=3000)
    # write_poscar's default fixture is silicon.
    assert page._legend._entries.keys() & {"element:Si"}

    checkbox.setChecked(False)
    assert page._band_plot._projection is None
    assert not (page._legend._entries.keys() & {"element:Si"})


def test_orbital_selector_populates_and_filters(window, qtbot, tmp_path):
    directory = tmp_path / "orbital"
    directory.mkdir()
    fixtures.write_eigenval(directory, n_kpoints=6, n_bands=4, n_electrons=8, n_ions=2)
    fixtures.write_poscar(directory, atoms=2)
    fixtures.write_procar(directory, n_kpoints=6, n_bands=4, n_ions=2)

    window.open_directory(directory)
    qtbot.wait(150)

    page = window._bands
    sidebar = page._sidebar
    assert sidebar.orbital_box.count() == 1  # just "All Orbitals", nothing loaded yet
    assert not sidebar.orbital_box.isEnabled()

    sidebar.projections["element"].setChecked(True)
    qtbot.waitUntil(lambda: page._procar is not None, timeout=3000)

    assert sidebar.orbital_box.isEnabled()
    assert sidebar.orbital_box.findText("pz") >= 0

    sidebar.orbital_box.setCurrentText("pz")
    qtbot.wait(60)
    assert page._selected_orbital == "pz"
    assert page._band_plot._projection is not None

    # Unchecking the projection disables and resets the selector, without
    # forgetting the calculation's orbital list — re-checking should not
    # need to reload PROCAR to repopulate it.
    sidebar.projections["element"].setChecked(False)
    assert not sidebar.orbital_box.isEnabled()
    assert sidebar.orbital_box.currentText() == "All Orbitals"

    sidebar.projections["element"].setChecked(True)
    assert sidebar.orbital_box.isEnabled()
    assert sidebar.orbital_box.count() > 1


def test_band_gap_appears_in_the_info_bar(window, qtbot, tmp_path):
    directory = tmp_path / "gap"
    directory.mkdir()
    fixtures.case_a_complete(directory)  # EIGENVAL + DOSCAR (Fermi level) present

    window.open_directory(directory)
    qtbot.wait(150)

    qtbot.waitUntil(
        lambda: window._info_bar._fields["gap"]._value.text() != "—", timeout=3000
    )
    text = window._info_bar._fields["gap"]._value.text()
    assert "eV" in text
    assert "(direct)" in text or "(indirect)" in text or text == "Metallic"


def test_band_gap_resets_on_a_new_calculation_without_bands(window, qtbot, tmp_path):
    with_gap = tmp_path / "with_gap"
    with_gap.mkdir()
    fixtures.case_a_complete(with_gap)
    window.open_directory(with_gap)
    qtbot.waitUntil(
        lambda: window._info_bar._fields["gap"]._value.text() != "—", timeout=3000
    )

    without_bands = tmp_path / "without_bands"
    without_bands.mkdir()
    fixtures.case_d_structure_only(without_bands)
    window.open_directory(without_bands)
    qtbot.wait(100)

    assert window._info_bar._fields["gap"]._value.text() == "—"


def test_element_projection_unavailable_without_matching_procar(window, qtbot, tmp_path):
    directory = tmp_path / "no_procar"
    directory.mkdir()
    fixtures.case_b_bands_only(directory)  # EIGENVAL + POSCAR, no PROCAR

    window.open_directory(directory)
    qtbot.wait(150)

    checkbox = window._bands._sidebar.projections["element"]
    assert not checkbox.isEnabled()
    assert "PROCAR" in checkbox.toolTip()


def test_structure_page_renders_a_structure(window, qtbot, tmp_path):
    directory = tmp_path / "structure"
    directory.mkdir()
    fixtures.case_a_complete(directory)

    window.open_directory(directory)
    window._nav.setCurrentIndex(0)
    qtbot.wait(150)

    page = window._structure
    qtbot.waitUntil(lambda: page._structure is not None, timeout=5000)
    assert page._stack.currentIndex() == 0
    assert page._view._bonds is not None
    assert "Atoms:" in page._status.text()


def test_slab_vacuum_note_appears_and_tightens_the_camera(window, qtbot, tmp_path):
    directory = tmp_path / "slab"
    directory.mkdir()
    poscar = (
        "slab\n1.0\n"
        " 5.0 0.0 0.0\n 0.0 5.0 0.0\n 0.0 0.0 20.0\n"
        " Si\n 4\nDirect\n"
        " 0.1 0.1 0.40\n 0.6 0.1 0.42\n 0.1 0.6 0.48\n 0.6 0.6 0.50\n"
    )
    (directory / "POSCAR").write_text(poscar)

    window.open_directory(directory)
    window._nav.setCurrentIndex(0)
    qtbot.wait(150)

    page = window._structure
    qtbot.waitUntil(lambda: page._structure is not None, timeout=5000)

    vacuum = page._view.vacuum_info()
    assert vacuum is not None
    assert vacuum.axis_name == "c"
    assert "Vacuum:" in page._status.text()
    # The camera should be framed on the atoms, nowhere near the full 20 A
    # vacuum-inclusive cell.
    assert page._view._camera.extent < 10.0


def test_bulk_structure_shows_no_vacuum_note(window, qtbot, tmp_path):
    directory = tmp_path / "bulk"
    directory.mkdir()
    fixtures.case_a_complete(directory)

    window.open_directory(directory)
    window._nav.setCurrentIndex(0)
    qtbot.wait(150)

    page = window._structure
    qtbot.waitUntil(lambda: page._structure is not None, timeout=5000)
    assert page._view.vacuum_info() is None
    assert "Vacuum:" not in page._status.text()


def test_supercell_tiling_end_to_end(window, qtbot, tmp_path):
    directory = tmp_path / "supercell"
    directory.mkdir()
    fixtures.case_a_complete(directory)  # silicon, 2 atoms (write_poscar's default)

    window.open_directory(directory)
    window._nav.setCurrentIndex(0)
    qtbot.wait(150)

    page = window._structure
    qtbot.waitUntil(lambda: page._structure is not None, timeout=5000)
    base_atoms = page._base_structure.n_atoms
    space_group = page._base_structure.space_group_number
    assert base_atoms > 0 and space_group is not None

    sidebar = page._sidebar
    sidebar.supercell_boxes["a"].setValue(2)
    sidebar.supercell_boxes["b"].setValue(2)
    qtbot.wait(60)

    # The rendered structure is tiled...
    assert page._structure.n_atoms == base_atoms * 2 * 2 * 1
    assert page._view._structure.n_atoms == base_atoms * 2 * 2 * 1
    assert "atoms across 4 cells" in sidebar.supercell_hint.text()
    # ...but the info panel's lattice parameters still describe the
    # primitive cell, not an inflated supercell.
    assert "5.4667" in page._info._rows["a"]._value.text()
    # Symmetry is not silently dropped or re-derived on the bigger cell.
    assert str(space_group) in page._info._rows["space_group"]._value.text()

    page.reset()
    assert page._structure.n_atoms == base_atoms
    assert sidebar.supercell_boxes["a"].value() == 1


def test_measure_toolbar_button_switches_the_viewer_mode(window, qtbot, tmp_path):
    from vasp_analyzer.ui.pages.structure.viewer import StructureView

    directory = tmp_path / "measure_button"
    directory.mkdir()
    fixtures.case_a_complete(directory)

    window.open_directory(directory)
    window._nav.setCurrentIndex(0)
    qtbot.wait(150)

    page = window._structure
    qtbot.waitUntil(lambda: page._structure is not None, timeout=5000)
    assert page._view.mode() == StructureView.MODE_ROTATE  # the default tool

    measure_button = next(
        page._tool_group.button(i)
        for i in range(len(page._tool_group.buttons()))
        if page._tool_group.button(i).property("mode") == StructureView.MODE_MEASURE
    )
    measure_button.click()

    assert page._view.mode() == StructureView.MODE_MEASURE


def test_structure_atom_color_override_updates_the_legend_and_view(window, qtbot, tmp_path):
    from vasp_analyzer.services import appearance
    from vasp_analyzer.ui.pages.structure.page import _ElementSwatch

    directory = tmp_path / "colors"
    directory.mkdir()
    fixtures.case_a_complete(directory)

    window.open_directory(directory)
    window._nav.setCurrentIndex(0)
    qtbot.wait(150)

    page = window._structure
    qtbot.waitUntil(lambda: page._structure is not None, timeout=5000)

    swatches = page._legend.findChildren(_ElementSwatch)
    assert swatches, "the legend should list at least one element"
    swatch = swatches[0]

    # The modal QColorDialog can't be driven in a headless test; exercise the
    # same code path _pick_color would take after a colour is chosen.
    appearance.set_override(swatch._element, "#ff8800")
    swatch._apply_style()
    swatch.changed.emit()

    assert appearance.effective_color(swatch._element) == "#ff8800"
    assert "#ff8800" in swatch.styleSheet()


def test_structure_page_empty_without_a_structure_file(window, qtbot, tmp_path):
    directory = tmp_path / "no_structure"
    directory.mkdir()
    fixtures.write_eigenval(directory)
    fixtures.write_incar(directory)

    window.open_directory(directory)
    window._nav.setCurrentIndex(0)
    qtbot.wait(150)

    assert window._structure._stack.currentIndex() == 1


def test_structure_file_link_navigates_to_inputs(window, qtbot, tmp_path):
    directory = tmp_path / "linked"
    directory.mkdir()
    fixtures.case_a_complete(directory)

    window.open_directory(directory)
    qtbot.wait(150)

    window._open_input_file("incar")
    qtbot.wait(80)

    assert window._nav.currentIndex() == 2
    assert window._inputs._current is not None
    assert window._inputs._current.role.key == "incar"


def test_empty_directory_leaves_the_window_usable(window, qtbot, tmp_path):
    directory = tmp_path / "nothing"
    directory.mkdir()

    window.open_directory(directory)
    qtbot.wait(120)

    assert window._calculation.is_empty
    assert window._bands._stack.currentIndex() == 1
    window._nav.setCurrentIndex(2)
    qtbot.wait(60)
    assert window._inputs._stack.currentIndex() == 1
