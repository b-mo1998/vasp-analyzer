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


def test_bands_page_shows_empty_state_without_eigenval(window, qtbot, tmp_path):
    directory = tmp_path / "no_bands"
    directory.mkdir()
    fixtures.case_c_no_eigenval(directory)

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
