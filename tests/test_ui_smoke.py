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
from vasp_analyzer.ui.style import build_stylesheet

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
    qapp.setStyleSheet(build_stylesheet())
    apply_plot_style()
    return qapp


@pytest.fixture
def window(qtbot, monkeypatch, tmp_path):
    # Keep the test run out of the user's real settings and recent list.
    from vasp_analyzer.services import settings as settings_module

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
