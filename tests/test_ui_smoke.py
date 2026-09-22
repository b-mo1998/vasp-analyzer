"""End-to-end checks that the window survives every partial calculation.

The central promise of the architecture is that missing or broken files remove
a capability instead of breaking the application. These tests open each
scenario in a real window and visit every page.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from vasp_analyzer.models.capabilities import Capability
from vasp_analyzer.models.volumetric import FieldKind
from vasp_analyzer.ui.main_window import PAGES, MainWindow
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
    for index in range(len(PAGES)):
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


def test_paper_figure_export_writes_a_single_column_pdf(window, qtbot, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    directory = tmp_path / "gap"
    directory.mkdir()
    fixtures.case_a_complete(directory)  # EIGENVAL + DOSCAR

    window.open_directory(directory)
    qtbot.wait(150)

    page = window._bands
    qtbot.waitUntil(lambda: page._bands is not None, timeout=3000)

    out_path = tmp_path / "paper-export.pdf"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(
            lambda *a, **k: (
                str(out_path),
                "Paper figure — single column (*.pdf)",
            )
        ),
    )
    page.export()
    assert out_path.exists()
    assert out_path.stat().st_size > 0



def _auto_confirm_reload(monkeypatch):
    """Click the destructive Reload button the moment the dialog opens.

    _reload() now asks for confirmation with a modal QMessageBox before
    restarting; a headless test must answer it programmatically or exec()
    blocks forever waiting for a user who is not there.
    """
    from PySide6.QtWidgets import QMessageBox

    def fake_exec(self):
        for button in self.buttons():
            if self.buttonRole(button) == QMessageBox.DestructiveRole:
                button.click()
                return 0
        return 0

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)

def test_reload_calculation_restarts_the_process(window, qtbot, tmp_path, monkeypatch):
    """Reload Calculation is a real process restart (os.execv), not a plain
    data re-read -- the whole point being that it recovers from stuck Qt
    layout/widget state, which re-parsing the calculation's files can never
    touch. The current directory must be passed through so the new process
    reopens where this one left off."""
    directory = tmp_path / "reload_me"
    directory.mkdir()
    fixtures.case_a_complete(directory)
    window.open_directory(directory)
    qtbot.wait(100)

    _auto_confirm_reload(monkeypatch)
    captured = {}
    monkeypatch.setattr(
        "os.execv",
        lambda path, args: captured.update(path=path, args=args),
    )
    window._reload()

    assert captured, "os.execv must have been called"
    assert captured["path"] == sys.executable
    assert captured["args"][0] == sys.executable
    assert str(directory) in captured["args"]


def test_reload_calculation_restarts_even_with_nothing_open(window, qtbot, monkeypatch):
    _auto_confirm_reload(monkeypatch)
    captured = {}
    monkeypatch.setattr(
        "os.execv",
        lambda path, args: captured.update(path=path, args=args),
    )
    window._reload()

    assert captured, "os.execv must have been called even with no calculation open"
    assert captured["path"] == sys.executable


def test_restart_argv_goes_back_through_dash_m_when_not_frozen(monkeypatch):
    """`python -m vasp_analyzer` leaves sys.argv[0] pointing at __main__.py,
    and re-running that file directly dies on its own relative imports. The
    restart has to re-enter through -m."""
    monkeypatch.delattr(sys, "frozen", raising=False)
    executable, args = MainWindow.restart_argv(None)

    assert executable == sys.executable
    assert args == [sys.executable, "-m", "vasp_analyzer"]


def test_restart_argv_execs_the_bundle_binary_alone_when_frozen(monkeypatch, tmp_path):
    """In a PyInstaller bundle there is no -m to go through, and the old code
    passed sys.argv[0] as an extra argument -- which shoved the directory to
    argv[2], where nothing looked for it, so the restarted app came up
    blank."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    _executable, args = MainWindow.restart_argv(tmp_path)

    assert args == [sys.executable, str(tmp_path)]


def test_restart_argv_does_not_accumulate_directories(monkeypatch, tmp_path):
    """Built from scratch, not appended to the live argv: reloading twice
    used to leave the directory on the command line twice."""
    monkeypatch.setattr(sys, "argv", [sys.argv[0], str(tmp_path), str(tmp_path)])
    _executable, args = MainWindow.restart_argv(tmp_path)

    assert args.count(str(tmp_path)) == 1


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


def test_element_projection_shows_a_loading_indicator_while_procar_parses(
    window, qtbot, tmp_path
):
    """PROCAR is often a very large file on a real calculation — tens of
    seconds to parse, not a bug — and the checkbox used to give no feedback
    at all while that background load ran, reading as broken or frozen.
    A visible "Loading projection data…" label must appear the moment the
    checkbox is checked and disappear once PROCAR actually arrives."""
    directory = tmp_path / "projection_loading"
    directory.mkdir()
    fixtures.write_eigenval(directory, n_kpoints=6, n_bands=4, n_electrons=8, n_ions=2)
    fixtures.write_poscar(directory, atoms=2)
    fixtures.write_procar(directory, n_kpoints=6, n_bands=4, n_ions=2)

    window.open_directory(directory)
    qtbot.wait(150)

    # isHidden() rather than isVisible(): the test window is never shown, so
    # isVisible()'s effective-visibility chain reports False regardless —
    # isHidden() reflects only this widget's own explicit shown/hidden state.
    page = window._bands
    assert page._sidebar.projection_loading.isHidden()

    page._sidebar.projections["element"].setChecked(True)
    assert not page._sidebar.projection_loading.isHidden(), (
        "the indicator must appear immediately, before the background parse "
        "has had any chance to finish"
    )

    qtbot.waitUntil(lambda: page._procar is not None, timeout=3000)
    assert page._sidebar.projection_loading.isHidden()


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
    matrix = sidebar.orbital_container
    assert matrix.selection() == {}          # nothing loaded yet
    assert not matrix.isEnabled()

    sidebar.projections["element"].setChecked(True)
    qtbot.waitUntil(lambda: page._procar is not None, timeout=3000)

    assert matrix.isEnabled()
    # One row per element, every orbital selected to begin with — the plain
    # by-element view.
    assert matrix._rows, "the matrix should list the structure's elements"
    assert matrix.is_full_selection(page._procar.orbitals)
    assert any(k.startswith("element:") for k in page._legend._entries)
    assert not any(k.startswith("orbital:") for k in page._legend._entries)

    row = matrix._rows[0]
    assert "pz" in row._orbital_chips

    # Narrow every element to pz alone.
    for element_row in matrix._rows:
        for name, chip in element_row._orbital_chips.items():
            chip.setChecked(name == "pz")
    qtbot.wait(60)

    assert not matrix.is_full_selection(page._procar.orbitals)
    assert all(orbitals == ["pz"] for orbitals in matrix.selection().values())
    assert page._band_plot._projection is not None
    # This fixture is single-element, so a narrowed selection is far more
    # readable coloured by orbital than by a single flat element colour.
    assert "orbital:pz" in page._legend._entries

    # A shell chip completes its shell in one click, even from a partial
    # selection (only pz is on at this point).
    row._shell_chips["p"].click()
    qtbot.wait(60)
    assert {"px", "py", "pz"} <= set(row.selected())

    # Back to everything: element colouring returns.
    matrix._set_all(True)
    qtbot.wait(60)
    assert matrix.is_full_selection(page._procar.orbitals)
    assert any(k.startswith("element:") for k in page._legend._entries)
    assert not any(k.startswith("orbital:") for k in page._legend._entries)

    # Unchecking the projection disables the matrix without forgetting the
    # rows, so re-checking need not reload PROCAR.
    sidebar.projections["element"].setChecked(False)
    assert not matrix.isEnabled()

    sidebar.projections["element"].setChecked(True)
    assert matrix.isEnabled()
    assert matrix._rows


def test_atom_projection_end_to_end(window, qtbot, tmp_path):
    """The atom-ID counterpart to element projection: a specific dopant, not
    a whole species. Fixture puts atom 1 as C and atom 2 as O — write_procar
    puts all of ion 1's weight in orbital 0 and nothing on ion 2, so
    selecting "just atom 1" and "just atom 2" must light up disjoint bands."""
    directory = tmp_path / "atom_projection"
    directory.mkdir()
    fixtures.write_eigenval(directory, n_kpoints=6, n_bands=4, n_electrons=8, n_ions=2)
    fixtures.write_poscar(directory, atoms=2)  # ids 1=Si, 2=Si by default
    fixtures.write_procar(directory, n_kpoints=6, n_bands=4, n_ions=2)

    window.open_directory(directory)
    qtbot.wait(150)

    page = window._bands
    sidebar = page._sidebar
    checkbox = sidebar.projections["atom"]
    assert checkbox.isEnabled(), "PROCAR is present and matches EIGENVAL"
    assert sidebar.atom_selector_field.isEnabled() is False, "off until enabled"

    checkbox.setChecked(True)
    assert sidebar.atom_selector_field.isEnabled()
    assert sidebar.atom_orbitals.isEnabled()

    sidebar.atom_selector_field.setText("1")
    qtbot.waitUntil(lambda: page._band_plot._projection is not None, timeout=3000)
    assert sidebar.atom_status.text() == "1 atom selected"
    assert "atom:selection" in page._legend._entries

    # Nonsense text: falls back cleanly rather than showing a stale overlay.
    sidebar.atom_selector_field.setText("Xx")
    qtbot.wait(60)
    assert page._band_plot._projection is None
    assert "Xx" in sidebar.atom_status.text()

    # Back to a valid id.
    sidebar.atom_selector_field.setText("2")
    qtbot.wait(60)
    assert page._band_plot._projection is not None
    assert sidebar.atom_status.text() == "1 atom selected"

    checkbox.setChecked(False)
    assert page._band_plot._projection is None
    assert sidebar.atom_status.text() == ""
    assert not sidebar.atom_selector_field.isEnabled()


def test_atom_and_element_projection_share_one_overlay_slot(window, qtbot, tmp_path):
    directory = tmp_path / "atom_element_exclusive"
    directory.mkdir()
    fixtures.write_eigenval(directory, n_kpoints=4, n_bands=3, n_electrons=6, n_ions=2)
    fixtures.write_poscar(directory, atoms=2)
    fixtures.write_procar(directory, n_kpoints=4, n_bands=3, n_ions=2)

    window.open_directory(directory)
    qtbot.wait(150)

    page = window._bands
    sidebar = page._sidebar

    sidebar.projections["element"].setChecked(True)
    qtbot.wait(60)
    assert sidebar.projections["element"].isChecked()

    sidebar.atom_selector_field.setText("1")
    sidebar.projections["atom"].setChecked(True)
    qtbot.wait(60)
    assert sidebar.projections["atom"].isChecked()
    assert not sidebar.projections["element"].isChecked(), (
        "checking Atom must clear Element — they draw into the same slot"
    )

    sidebar.projections["element"].setChecked(True)
    qtbot.wait(60)
    assert sidebar.projections["element"].isChecked()
    assert not sidebar.projections["atom"].isChecked(), (
        "and the reverse must hold too"
    )


def test_wannier_bands_overlay_end_to_end(window, qtbot, tmp_path):
    directory = tmp_path / "wannier"
    directory.mkdir()
    fixtures.write_eigenval(directory, n_kpoints=6, n_bands=4, n_electrons=8, n_ions=2)
    fixtures.write_poscar(directory, atoms=2)
    fixtures.write_wannier_band_dat(directory, n_bands=4, n_points=10)

    window.open_directory(directory)
    qtbot.wait(150)

    page = window._bands
    checkbox = page._sidebar.projections["wannier"]
    assert checkbox.isEnabled(), "wannier90_band.dat is present"
    assert page._band_plot._wannier is None
    assert page._legend._entries["wannier"].isHidden()

    checkbox.setChecked(True)
    qtbot.waitUntil(lambda: page._band_plot._wannier is not None, timeout=3000)
    assert not page._legend._entries["wannier"].isHidden()

    checkbox.setChecked(False)
    assert page._band_plot._wannier is None
    assert page._legend._entries["wannier"].isHidden()


def test_spin_projection_end_to_end_and_mutually_exclusive_with_element(
    window, qtbot, tmp_path
):
    directory = tmp_path / "ncl"
    directory.mkdir()
    fixtures.write_eigenval(directory, n_kpoints=3, n_bands=4, n_electrons=8, n_ions=2)
    fixtures.write_poscar(directory, atoms=2)
    fixtures.write_incar(directory, LSORBIT="T")
    fixtures.write_procar_noncollinear(directory, n_kpoints=3, n_bands=4, n_ions=2)

    window.open_directory(directory)
    qtbot.wait(150)

    page = window._bands
    sidebar = page._sidebar
    spin_box = sidebar.projections["spin"]
    element_box = sidebar.projections["element"]
    assert spin_box.isEnabled(), "LSORBIT=T should enable spin projection"
    assert not sidebar.spin_axis_box.isEnabled()

    spin_box.setChecked(True)
    qtbot.waitUntil(lambda: page._band_plot._projection is not None, timeout=3000)
    assert sidebar.spin_axis_box.isEnabled()
    assert not page._spin_colorbar.isHidden()

    # Element and spin share the plot's one overlay slot: turning element on
    # must cleanly hand it over rather than have the two fight over it.
    element_box.setChecked(True)
    qtbot.wait(100)
    assert not spin_box.isChecked()
    assert page._spin_colorbar.isHidden()
    assert page._band_plot._projection is not None  # now the element overlay

    # And the reverse direction.
    spin_box.setChecked(True)
    qtbot.wait(100)
    assert not element_box.isChecked()
    assert not page._spin_colorbar.isHidden()

    spin_box.setChecked(False)
    assert page._band_plot._projection is None
    assert page._spin_colorbar.isHidden()


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


def test_band_gap_marker_checkbox_drives_the_plots_visibility_flag(window, qtbot, tmp_path):
    directory = tmp_path / "gap"
    directory.mkdir()
    fixtures.case_a_complete(directory)  # EIGENVAL + DOSCAR (Fermi level) present

    window.open_directory(directory)
    qtbot.wait(150)

    page = window._bands
    assert page._sidebar.show_band_gap.isChecked()
    assert page._band_plot._show_band_gap is True

    page._sidebar.show_band_gap.setChecked(False)
    assert page._band_plot._show_band_gap is False

    page._sidebar.show_band_gap.setChecked(True)
    assert page._band_plot._show_band_gap is True


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


def test_structure_page_neighbor_table_reflects_the_selected_atom(window, qtbot, tmp_path):
    """write_poscar's default fixture is silicon's diamond-cubic cell — each
    atom has exactly 4 nearest neighbours, all at the same tetrahedral bond
    length. Real physics, not just "some rows appeared"."""
    directory = tmp_path / "neighbors"
    directory.mkdir()
    fixtures.case_a_complete(directory)

    window.open_directory(directory)
    window._nav.setCurrentIndex(0)
    qtbot.wait(150)

    page = window._structure
    qtbot.waitUntil(lambda: page._structure is not None, timeout=5000)
    table = page._sidebar.neighbor_table

    # isHidden() rather than isVisible(): the test window is never shown, so
    # isVisible()'s effective-visibility chain reports False regardless —
    # isHidden() reflects only this widget's own explicit shown/hidden state.
    assert not table._placeholder.isHidden()
    assert not table._row_widgets

    page._on_atom_selected(0)
    qtbot.wait(30)
    assert table._placeholder.isHidden()
    assert len(table._row_widgets) >= 4

    distances = []
    for row in table._row_widgets[:4]:
        text = [l.text() for l in row.findChildren(type(table._placeholder))][-1]
        distances.append(float(text.replace(" Å", "")))
    assert all(d == pytest.approx(distances[0], abs=1e-2) for d in distances), (
        "silicon's 4 nearest neighbours are all the same tetrahedral bond length"
    )

    page._on_atom_selected(None)
    qtbot.wait(30)
    assert not table._placeholder.isHidden()
    assert not table._row_widgets


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

    # Through the shared toolbar's own API -- the page no longer builds a
    # private _tool_group of its own.
    page._toolbar.set_tool("measure")

    assert page._view.mode() == StructureView.MODE_MEASURE
    assert page._toolbar.current_tool() == "measure"


def test_structure_atom_color_override_updates_the_legend_and_view(window, qtbot, tmp_path):
    from vasp_analyzer.services import appearance
    from vasp_analyzer.ui.widgets.element_swatch import ElementSwatch

    directory = tmp_path / "colors"
    directory.mkdir()
    fixtures.case_a_complete(directory)

    window.open_directory(directory)
    window._nav.setCurrentIndex(0)
    qtbot.wait(150)

    page = window._structure
    qtbot.waitUntil(lambda: page._structure is not None, timeout=5000)

    swatches = page._legend.findChildren(ElementSwatch)
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

    # Derived from PAGES rather than hard-coded: adding a page shifts every
    # index after it, and that should not break an unrelated navigation test.
    assert window._nav.currentIndex() == PAGES.index("Inputs")
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


# --- Molecular Dynamics -------------------------------------------------------------


def test_md_page_plays_a_trajectory(window, qtbot, tmp_path):
    directory = tmp_path / "md"
    directory.mkdir()
    fixtures.case_md(directory, n_frames=12, n_atoms=4)

    window.open_directory(directory)
    qtbot.wait(150)

    page = window._md
    assert page.calculation().can(Capability.MD_TRAJECTORY)
    qtbot.waitUntil(lambda: page._trajectory is not None, timeout=3000)
    assert page._stack.currentIndex() == 0, "the viewer, not the empty state"
    assert page._trajectory.n_frames == 12
    assert page._playback._n_frames == 12

    # Stepping updates both the transport and the rendered frame.
    page._playback.go_last()
    assert page._frame == 11
    page._playback.go_first()
    assert page._frame == 0

    # Clicking a plot seeks: 1 fs steps means frame 5 sits at 0.005 ps.
    page._on_seek(0.005)
    assert page._frame == 5


def test_md_page_empty_state_without_xdatcar(window, qtbot, tmp_path):
    directory = tmp_path / "no_md"
    directory.mkdir()
    fixtures.case_a_complete(directory)

    window.open_directory(directory)
    qtbot.wait(120)

    page = window._md
    assert page._stack.currentIndex() == 1, "the empty state"
    assert "XDATCAR" in page._empty._message.text()


def test_md_page_survives_a_trajectory_without_oszicar(window, qtbot, tmp_path):
    """Trajectory playback must not depend on telemetry: the movie plays and
    the temperature/energy charts report themselves unavailable."""
    directory = tmp_path / "md_no_oszicar"
    directory.mkdir()
    fixtures.write_xdatcar(directory, n_frames=8, n_atoms=4)
    fixtures.write_incar_md(directory)

    window.open_directory(directory)
    qtbot.wait(150)

    page = window._md
    qtbot.waitUntil(lambda: page._trajectory is not None, timeout=3000)
    assert page._stack.currentIndex() == 0
    assert not page._trajectory.has("temperature")
    assert not page._temperature_plot._has_data
    assert page._msd is not None and "C" in page._msd


def test_md_bottom_bar_switches_layout(window, qtbot, tmp_path):
    directory = tmp_path / "md_info"
    directory.mkdir()
    fixtures.case_md(directory, n_frames=10, n_atoms=4)

    window.open_directory(directory)
    qtbot.wait(150)
    window._nav.setCurrentIndex(PAGES.index("MD"))
    qtbot.waitUntil(lambda: window._md._trajectory is not None, timeout=3000)
    qtbot.wait(80)

    bar = window._info_bar
    assert bar.current_layout() == "md"
    assert bar._fields["frames"]._value.text() == "10"
    assert bar._fields["timestep"]._value.text() == "1 fs"
    assert bar._fields["ensemble"]._value.text() == "NVT"

    # Leaving the page restores the electronic-structure fields.
    window._nav.setCurrentIndex(PAGES.index("Structure"))
    qtbot.wait(60)
    assert bar.current_layout() == "default"
    assert "system" in bar._fields


# --- NEB ------------------------------------------------------------------------------


def test_neb_page_loads_a_band(window, qtbot, tmp_path):
    directory = tmp_path / "neb"
    directory.mkdir()
    fixtures.case_neb(directory)

    window.open_directory(directory)
    qtbot.wait(150)

    page = window._neb
    assert page.calculation().can(Capability.NEB_PATH)
    qtbot.waitUntil(lambda: page._path is not None, timeout=3000)
    assert page._stack.currentIndex() == 0, "the viewer, not the empty state"
    assert page._path.n_images == 5
    assert page._strip._count == 5

    # The climbing image is selected on load, and the three views agree.
    assert page._selected == page._path.climbing_image
    assert page._strip.selected() == page._selected


def test_neb_selection_stays_synchronised(window, qtbot, tmp_path):
    directory = tmp_path / "neb_sync"
    directory.mkdir()
    fixtures.case_neb(directory)

    window.open_directory(directory)
    qtbot.wait(150)
    page = window._neb
    qtbot.waitUntil(lambda: page._path is not None, timeout=3000)

    # Selecting from the profile drives the strip and the image card.
    page._profile.imageSelected.emit(1)
    qtbot.wait(50)
    assert page._selected == 1
    assert page._strip.selected() == 1
    assert "01" in page._image_card._title.text()

    # And from the strip.
    page._strip.select(4)
    qtbot.wait(50)
    assert page._selected == 4
    assert "Final" in page._image_card._title.text()


def test_neb_page_reports_convergence_honestly(window, qtbot, tmp_path):
    directory = tmp_path / "neb_unconverged"
    directory.mkdir()
    fixtures.case_neb(directory, forces=(0.001, 0.9, 1.4, 0.7, 0.001))

    window.open_directory(directory)
    qtbot.wait(150)
    page = window._neb
    qtbot.waitUntil(lambda: page._path is not None, timeout=3000)

    assert page._path.converged is False
    assert page._model_card._values["convergence"].text() == "Not converged"


def test_neb_page_hides_the_barrier_when_images_are_unrun(window, qtbot, tmp_path):
    directory = tmp_path / "neb_partial"
    directory.mkdir()
    fixtures.write_neb_incar(directory)
    for index, energy in enumerate([-10.0, None, None, None, -9.5]):
        fixtures.write_neb_image(
            directory, index, shift=0.05 * index, energy=energy,
            max_force=0.001 if energy is not None else None,
        )

    window.open_directory(directory)
    qtbot.wait(150)
    page = window._neb
    qtbot.waitUntil(lambda: page._path is not None, timeout=3000)

    assert page._stack.currentIndex() == 0, "structures are still inspectable"
    assert page._summary._values["barrier"].text() == "—"
    assert page._model_card._values["convergence"].text() == "—"


def test_neb_page_empty_state_in_an_ordinary_directory(window, qtbot, tmp_path):
    directory = tmp_path / "not_neb"
    directory.mkdir()
    fixtures.case_a_complete(directory)

    window.open_directory(directory)
    qtbot.wait(120)
    assert window._neb._stack.currentIndex() == 1


def test_neb_offers_no_way_to_launch_a_calculation(window, qtbot, tmp_path):
    """This application analyses finished calculations; it must never offer to
    launch one. The Run NEB control was removed outright rather than shown
    disabled, so the check is that no such button exists at all."""
    from PySide6.QtWidgets import QPushButton

    directory = tmp_path / "neb_run"
    directory.mkdir()
    fixtures.case_neb(directory)

    window.open_directory(directory)
    qtbot.wait(120)

    labels = [
        button.text().lower()
        for button in window._neb._sidebar.findChildren(QPushButton)
    ]
    assert not any("run" in label for label in labels), labels


def test_neb_bottom_bar_layout(window, qtbot, tmp_path):
    directory = tmp_path / "neb_info"
    directory.mkdir()
    fixtures.case_neb(directory)

    window.open_directory(directory)
    qtbot.wait(150)
    window._nav.setCurrentIndex(PAGES.index("NEB"))
    qtbot.waitUntil(lambda: window._neb._path is not None, timeout=3000)
    qtbot.wait(80)

    bar = window._info_bar
    assert bar.current_layout() == "neb"
    assert bar._fields["images"]._value.text() == "5"
    assert "NEB" in bar._fields["method"]._value.text()


# --- Charge Density ---------------------------------------------------------------


def test_charge_page_loads_and_meshes(window, qtbot, tmp_path):
    directory = tmp_path / "charge"
    directory.mkdir()
    fixtures.case_charge(directory, shape=(12, 12, 12), electrons=8.0)

    window.open_directory(directory)
    qtbot.wait(150)

    page = window._charge
    assert page.calculation().can(Capability.CHARGE_DENSITY)
    qtbot.waitUntil(lambda: page._field is not None, timeout=4000)
    assert page._stack.currentIndex() == 0, "the viewer, not the empty state"

    # The integral must reproduce the electron count the fixture encoded.
    assert page._field.integral() == pytest.approx(8.0, abs=1e-6)
    qtbot.waitUntil(lambda: page._view.mesh() is not None, timeout=4000)


def test_charge_difference_against_a_second_file(window, qtbot, tmp_path):
    directory = tmp_path / "charge"
    directory.mkdir()
    fixtures.case_charge(directory, shape=(8, 8, 8), electrons=8.0)

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_path = fixtures.write_volumetric(
        other_dir, shape=(8, 8, 8), values=np.zeros((8, 8, 8))
    )

    window.open_directory(directory)
    qtbot.wait(150)
    page = window._charge
    qtbot.waitUntil(lambda: page._field is not None, timeout=4000)
    assert page._field.kind is FieldKind.CHARGE

    page._start_compare_load(other_path)
    qtbot.waitUntil(lambda: page._compare_field is not None, timeout=4000)

    assert page._sidebar.compare_enabled.isChecked()
    assert page._field.kind is FieldKind.DIFFERENCE
    assert page._field.kind.is_signed
    # Subtracting an all-zero field leaves the original values untouched.
    assert np.allclose(page._field.values, page._field_raw.values)

    page._clear_compare()
    assert page._field.kind is FieldKind.CHARGE
    assert page._compare_field is None


def test_charge_difference_rejects_a_mismatched_grid(window, qtbot, tmp_path):
    directory = tmp_path / "charge"
    directory.mkdir()
    fixtures.case_charge(directory, shape=(8, 8, 8), electrons=8.0)

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_path = fixtures.write_volumetric(other_dir, shape=(6, 6, 6))

    window.open_directory(directory)
    qtbot.wait(150)
    page = window._charge
    qtbot.waitUntil(lambda: page._field is not None, timeout=4000)

    page._start_compare_load(other_path)
    qtbot.wait(150)

    assert page._compare_field is None
    assert "mismatch" in page._sidebar.compare_status.text().lower()
    # The primary field is left exactly as it was.
    assert page._field.kind is FieldKind.CHARGE


def test_charge_page_empty_state_without_volumetric_data(window, qtbot, tmp_path):
    directory = tmp_path / "no_charge"
    directory.mkdir()
    fixtures.case_a_complete(directory)

    window.open_directory(directory)
    qtbot.wait(120)
    assert window._charge._stack.currentIndex() == 1
    assert "CHGCAR" in window._charge._empty._message.text()


def test_charge_page_offers_magnetization_for_a_spin_file(window, qtbot, tmp_path):
    directory = tmp_path / "spin"
    directory.mkdir()
    fixtures.write_volumetric(directory, shape=(8, 8, 8), n_blocks=2)

    window.open_directory(directory)
    qtbot.wait(150)
    page = window._charge
    qtbot.waitUntil(lambda: page._file is not None, timeout=4000)

    labels = [page._sidebar.component_box.itemText(i)
              for i in range(page._sidebar.component_box.count())]
    assert "Total Charge Density" in labels
    assert "Magnetization Density" in labels


def test_charge_page_adapts_units_and_integration_to_elf(window, qtbot, tmp_path):
    """ELF is dimensionless and has no electron count, so the density units and
    the Integration card must both change rather than be relabelled."""
    directory = tmp_path / "elf"
    directory.mkdir()
    fixtures.write_volumetric(directory, name="ELFCAR", shape=(8, 8, 8))

    window.open_directory(directory)
    qtbot.wait(150)
    page = window._charge
    qtbot.waitUntil(lambda: page._field is not None, timeout=4000)

    assert page._field.kind.units == ""
    assert not page._field.kind.integrates_to_electrons
    # The card is replaced, not filled with a meaningless "Total Charge".
    assert "integral" not in page._integration_card._values
    assert page._integration_card._title.text() == "Field Summary"


def test_charge_slice_and_plane_stay_synchronised(window, qtbot, tmp_path):
    directory = tmp_path / "slices"
    directory.mkdir()
    fixtures.case_charge(directory, shape=(10, 10, 10))

    window.open_directory(directory)
    qtbot.wait(150)
    page = window._charge
    qtbot.waitUntil(lambda: page._field is not None, timeout=4000)

    page._position_slider.setValue(0.75)
    qtbot.wait(60)
    assert page._slice_fraction == pytest.approx(0.75, abs=1e-6)
    # The 3D cutting plane follows the 2D slice.
    assert page._view._slice_quad is not None

    page._show_slice.setChecked(False)
    qtbot.wait(40)
    assert page._view._slice_quad is None


def test_charge_localization_masks_the_field_and_restores_on_toggle_off(
    window, qtbot, tmp_path
):
    """A CHGCAR carries no per-atom decomposition, so "project by element/
    atom" is implemented as a spatial mask around the chosen atoms rather
    than anything quantum-mechanical — real electron count in, less real
    electron count out, and turning it back off must reproduce the original
    field exactly, not an approximation of it."""
    directory = tmp_path / "localize"
    directory.mkdir()
    fixtures.case_charge(directory, shape=(16, 16, 16), electrons=8.0)

    window.open_directory(directory)
    qtbot.wait(150)
    page = window._charge
    qtbot.waitUntil(lambda: page._field is not None, timeout=4000)

    original_integral = page._field.integral()
    assert original_integral == pytest.approx(8.0, abs=1e-6)

    page._sidebar.localize_enabled.setChecked(True)
    page._sidebar.localize_field.setText("1")  # the fixture's first carbon
    qtbot.wait(60)

    assert page._sidebar.localize_status.text() == "1 atom within 2.0 Å"
    localized_integral = page._field.integral()
    assert 0.0 < localized_integral < original_integral, (
        "masking to one atom must genuinely reduce the represented charge, "
        "not just relabel the same field"
    )
    qtbot.waitUntil(lambda: page._view.mesh() is not None, timeout=4000)

    page._sidebar.localize_enabled.setChecked(False)
    qtbot.wait(60)
    assert page._field.integral() == pytest.approx(original_integral, abs=1e-6)
    assert page._sidebar.localize_status.text() == ""


def test_per_atom_integration_reports_one_row_per_selected_atom(window, qtbot, tmp_path):
    directory = tmp_path / "atom_integration"
    directory.mkdir()
    fixtures.case_charge(directory, shape=(16, 16, 16), electrons=8.0)

    window.open_directory(directory)
    qtbot.wait(150)
    page = window._charge
    qtbot.waitUntil(lambda: page._field is not None, timeout=4000)

    assert not page._sidebar.integrate_button.isEnabled()

    page._sidebar.localize_enabled.setChecked(True)
    page._sidebar.localize_field.setText("1,2")
    qtbot.wait(60)
    assert page._sidebar.integrate_button.isEnabled()

    table = page._atom_integration_table
    assert not table._placeholder.isHidden()

    page._sidebar.integrate_button.click()
    qtbot.wait(60)

    assert table._placeholder.isHidden()
    assert len(table._row_widgets) == 2

    # Changing the radius invalidates the shown breakdown until re-run.
    page._sidebar.localize_radius.setValue(3.0)
    qtbot.wait(60)
    assert not table._placeholder.isHidden()
    assert len(table._row_widgets) == 0


def test_charge_localization_reports_unrecognised_tokens(window, qtbot, tmp_path):
    directory = tmp_path / "localize_bad"
    directory.mkdir()
    fixtures.case_charge(directory, shape=(10, 10, 10))

    window.open_directory(directory)
    qtbot.wait(150)
    page = window._charge
    qtbot.waitUntil(lambda: page._field is not None, timeout=4000)

    page._sidebar.localize_enabled.setChecked(True)
    page._sidebar.localize_field.setText("Xx")
    qtbot.wait(60)

    assert "Xx" in page._sidebar.localize_status.text()
    # No atoms matched, so the field must fall back to the real one rather
    # than silently rendering an all-zero grid.
    assert page._field is page._field_raw


def test_charge_bottom_bar_layout(window, qtbot, tmp_path):
    directory = tmp_path / "charge_info"
    directory.mkdir()
    fixtures.case_charge(directory, shape=(8, 8, 8), electrons=8.0)

    window.open_directory(directory)
    qtbot.wait(150)
    window._nav.setCurrentIndex(PAGES.index("Charge"))
    qtbot.waitUntil(lambda: window._charge._field is not None, timeout=4000)
    qtbot.wait(80)

    bar = window._info_bar
    assert bar.current_layout() == "charge"
    assert bar._fields["grid"]._value.text() == "8 × 8 × 8"
    assert bar._fields["datatype"]._value.text() == "Total Charge Density"
    assert bar._fields["spin"]._value.text() == "Non-polarised"


def test_charge_page_survives_a_corrupt_file(window, qtbot, tmp_path):
    directory = tmp_path / "corrupt"
    directory.mkdir()
    (directory / "CHGCAR").write_text("this is not a CHGCAR\n" * 40)

    window.open_directory(directory)
    qtbot.wait(200)
    assert window.isEnabled(), "a bad file must not take the window down"


def test_inputs_search_has_previous_and_next_arrows(window, qtbot, tmp_path):
    """Enter already stepped forward through matches; there was no visible
    control for it and no way back. The arrows are that control, enabled only
    when there is more than one place to go."""
    directory = tmp_path / "search"
    directory.mkdir()
    fixtures.case_a_complete(directory)
    window.open_directory(directory)
    qtbot.wait(150)
    page = window._inputs
    page.show_file("incar")
    qtbot.wait(100)

    assert not page._search_next.isEnabled()
    assert not page._search_previous.isEnabled()

    page._search.setText("=")   # every INCAR tag line has one
    qtbot.wait(50)
    _current, total = page._editor.match_position()
    assert total > 1
    assert page._search_next.isEnabled() and page._search_previous.isEnabled()

    page._search_next.click()
    assert page._editor.match_position()[0] == 2
    page._search_previous.click()
    assert page._editor.match_position()[0] == 1
    # Wraps around backwards from the first match to the last.
    page._search_previous.click()
    assert page._editor.match_position()[0] == total
    assert page._match_label.text() == f"{total} of {total}"


# -- bottom info bar alignment ------------------------------------------------


def _right_edge(widget) -> int:
    """Exclusive right edge in global pixels (``QRect.right()`` is inclusive)."""
    return widget.mapToGlobal(widget.rect().topRight()).x() + 1


def _viewer_card_edge(page) -> int:
    from vasp_analyzer.ui.theme import SPACING

    column = page._right_column
    return column.mapToGlobal(column.rect().topLeft()).x() - SPACING.md


def _bottom_edge(widget) -> int:
    return widget.mapToGlobal(widget.rect().bottomLeft()).y() + 1


def test_info_bar_lines_up_with_the_viewer_card_on_3d_pages(window, qtbot, tmp_path):
    """On the pages with a side panel to the right of a 3D viewer, the bottom
    bar stops level with the viewer card instead of running on under the
    panel, and the panel runs on down to the bar's bottom edge rather than
    leaving a blank strip beside it. Every other page keeps the full width."""
    directory = tmp_path / "poscar_only"
    directory.mkdir()
    fixtures.write_poscar(directory, atoms=2)

    window.resize(1600, 900)
    window.show()
    window.open_directory(directory)
    qtbot.waitUntil(lambda: window._structure._structure is not None, timeout=5000)
    qtbot.wait(60)

    for name in ("Structure", "Editor"):
        window._nav.setCurrentIndex(PAGES.index(name))
        qtbot.wait(40)
        page = window._current_page()
        assert page._stack.currentIndex() == 0, f"{name} should be showing content"
        assert window._info_bar.isVisible(), name
        assert _right_edge(window._info_bar) == _viewer_card_edge(page), name
        assert _bottom_edge(page._right_column) == _bottom_edge(window._info_bar), name

    # No CHGCAR and no NEB images: those pages show their full-width empty
    # card, and the bar should match that card rather than a hidden panel.
    for name in ("Charge", "NEB", "Bands", "Inputs"):
        window._nav.setCurrentIndex(PAGES.index(name))
        qtbot.wait(40)
        assert window._info_bar.isVisible(), name
        assert _right_edge(window._info_bar) == _right_edge(window._page_stack), name


def test_info_bar_returns_to_full_width_when_the_page_empties(window, qtbot, tmp_path):
    """The inset must follow the page's content/empty state live, not just
    the page switch: opening a folder with no structure while Structure is
    in front flips it to the empty card, and the bar has to widen with it."""
    with_poscar = tmp_path / "with_poscar"
    with_poscar.mkdir()
    fixtures.write_poscar(with_poscar, atoms=2)
    without = tmp_path / "without"
    without.mkdir()

    window.resize(1600, 900)
    window.show()
    window.open_directory(with_poscar)
    qtbot.waitUntil(lambda: window._structure._structure is not None, timeout=5000)
    window._nav.setCurrentIndex(PAGES.index("Structure"))
    qtbot.wait(60)
    assert _right_edge(window._info_bar) < _right_edge(window._page_stack)

    window.open_directory(without)
    qtbot.waitUntil(lambda: window._structure._stack.currentIndex() == 1, timeout=5000)
    qtbot.wait(60)
    assert _right_edge(window._info_bar) == _right_edge(window._page_stack)
