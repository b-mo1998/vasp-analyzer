"""Opening one dropped file rather than a whole calculation directory.

The contract is deliberately narrow: whatever pages that one file can honestly
feed come up with real data, and every other page shows its ordinary "not
found" empty state. Nothing crashes, and nothing quietly borrows data from the
other files that happen to share the folder — a CHGCAR sitting beside an
unrelated EIGENVAL must not light up the Bands page with a different
calculation's eigenvalues.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QMimeData, QPoint, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent

from vasp_analyzer.models.capabilities import Capability
from vasp_analyzer.services.discovery import discover_file, scan_file
from vasp_analyzer.ui.main_window import PAGES, MainWindow

from . import fixtures


# -- discovery ------------------------------------------------------------


def test_a_lone_poscar_offers_structure_and_nothing_else(tmp_path):
    path = fixtures.write_poscar(tmp_path, atoms=4)
    calculation = discover_file(path)

    assert calculation.can(Capability.STRUCTURE)
    assert not calculation.can(Capability.BANDS)
    assert not calculation.can(Capability.CHARGE_DENSITY)
    assert not calculation.can(Capability.MD_TRAJECTORY)
    assert not calculation.can(Capability.NEB_PATH)


def test_a_lone_chgcar_offers_charge_density_and_its_embedded_structure(tmp_path):
    """A volumetric file opens with a complete POSCAR block, so it knows its
    own geometry -- the Structure and Editor pages have something real to
    show, not an empty state that isn't true."""
    path = fixtures.write_volumetric(tmp_path, species=("Si",), counts=(3,))
    calculation = discover_file(path)

    assert calculation.can(Capability.CHARGE_DENSITY)
    assert calculation.can(Capability.STRUCTURE)
    assert not calculation.can(Capability.BANDS)
    assert [source.path for source in calculation.volumetric_sources] == [path]


def test_a_dropped_file_ignores_its_neighbours(tmp_path):
    """The whole point of scanning one file: a complete calculation directory
    opened one file at a time must report only that file's capabilities."""
    fixtures.case_a_complete(tmp_path)
    calculation = discover_file(tmp_path / "POSCAR")

    assert calculation.can(Capability.STRUCTURE)
    assert not calculation.can(Capability.BANDS), "EIGENVAL is next door, not dropped"
    assert not calculation.can(Capability.DOS), "DOSCAR is next door, not dropped"


def test_a_dropped_chgcar_offers_only_itself_as_a_source(tmp_path):
    """Two densities in one folder, one of them dropped: the Charge page must
    list the dropped one alone rather than re-scanning the directory."""
    fixtures.write_volumetric(tmp_path, name="CHGCAR")
    elfcar = fixtures.write_volumetric(tmp_path, name="ELFCAR")
    calculation = discover_file(elfcar)

    assert [source.path for source in calculation.volumetric_sources] == [elfcar]


def test_an_unrecognised_file_says_so_instead_of_failing(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("this is not a VASP file\n")
    calculation = discover_file(path)

    assert not calculation.capabilities.available
    codes = [issue.code for issue in calculation.issues]
    assert "file_unrecognised" in codes


def test_a_missing_file_is_reported_rather_than_raising(tmp_path):
    calculation = discover_file(tmp_path / "gone" / "POSCAR")
    assert calculation.issues


def test_scan_file_claims_at_most_one_role(tmp_path):
    path = fixtures.write_poscar(tmp_path)
    index = scan_file(path)

    present = [entry.role.key for entry in index.entries.values() if entry.exists]
    assert present == ["poscar"]
    assert index.unrecognised == []


# -- the window -----------------------------------------------------------


def _page_states(window) -> dict[str, str]:
    """Which pages are showing data and which are showing an empty state."""
    states = {}
    for index, name in enumerate(PAGES):
        page = window._page_stack.widget(index)
        stack = getattr(page, "_stack", None)
        if stack is not None:
            states[name] = "data" if stack.currentIndex() == 0 else "empty"
    return states


@pytest.fixture()
def window(qtbot, monkeypatch):
    # Never reopen whatever the developer last had open: this suite is about
    # what one dropped file does, and a restored directory would mask it.
    monkeypatch.setattr(MainWindow, "_open_last_directory", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    return win


def test_dropping_a_poscar_leaves_structure_and_editor_usable(
    window, qtbot, tmp_path
):
    fixtures.write_poscar(tmp_path, atoms=4)
    window.open_file(tmp_path / "POSCAR")
    qtbot.wait(300)

    states = _page_states(window)
    assert states["Structure"] == "data"
    assert states["Editor"] == "data"
    assert states["Bands"] == "empty"
    assert states["Charge"] == "empty"
    assert states["MD"] == "empty"
    assert states["NEB"] == "empty"


def test_dropping_a_chgcar_adds_the_charge_page(window, qtbot, tmp_path):
    fixtures.write_volumetric(tmp_path, species=("Si",), counts=(3,))
    window.open_file(tmp_path / "CHGCAR")
    qtbot.wait(500)

    states = _page_states(window)
    assert states["Structure"] == "data"
    assert states["Editor"] == "data"
    assert states["Charge"] == "data"
    assert states["Bands"] == "empty"
    assert states["MD"] == "empty"


def test_the_editor_can_edit_a_chgcars_embedded_structure(window, qtbot, tmp_path):
    fixtures.write_volumetric(tmp_path, species=("Si",), counts=(3,))
    window.open_file(tmp_path / "CHGCAR")
    qtbot.wait(500)

    editor = window._editor
    assert editor._working is not None
    assert editor._working.n_atoms == 3


def test_the_title_bar_names_the_file_not_its_folder(window, qtbot, tmp_path):
    fixtures.write_poscar(tmp_path)
    window.open_file(tmp_path / "POSCAR")
    qtbot.wait(200)

    assert "POSCAR" in window.windowTitle()


def test_an_unrecognised_drop_explains_itself(window, qtbot, tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(MainWindow, "_show_issue_details", lambda self, i: shown.append(i))

    path = tmp_path / "slurm-1234.out"
    path.write_text("job finished\n")
    window.open_file(path)
    qtbot.wait(100)

    assert shown, "the user must be told why nothing opened"
    assert shown[0].code == "file_unrecognised"


def _drop_event(kind, path: Path):
    """Build a drag event plus the QMimeData behind it.

    The mime data must be returned alongside the event and kept alive by the
    caller: the event does not own it, so letting it fall out of scope frees
    it under Qt and segfaults the moment the handler reads its urls.
    """
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    event = kind(QPoint(10, 10), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    return event, mime


def test_a_dragged_file_is_accepted_not_just_a_folder(window, tmp_path):
    """Files used to be refused outright at dragEnter, which gave no hint the
    app had any opinion about them."""
    path = fixtures.write_poscar(tmp_path)
    event, _mime = _drop_event(QDragEnterEvent, path)
    window.dragEnterEvent(event)

    assert event.isAccepted()


def test_dropping_a_file_opens_it(window, qtbot, tmp_path):
    path = fixtures.write_poscar(tmp_path, atoms=4)
    event, _mime = _drop_event(QDropEvent, path)
    window.dropEvent(event)
    qtbot.wait(300)

    assert window._calculation is not None
    assert window._calculation.can(Capability.STRUCTURE)
    assert _page_states(window)["Bands"] == "empty"


def test_a_directory_opened_before_the_deferred_restore_is_not_wiped(
    qtbot, tmp_path, monkeypatch
):
    """MainWindow defers "reopen the last session" to the first event-loop
    turn so the window can paint first. Anything that opened a calculation
    before that turn -- a path on the command line, a Finder "Open With", a
    drop onto a freshly shown window -- used to be silently discarded a
    moment later when the deferred restore found no saved directory and
    cleared every page.
    """
    from vasp_analyzer.services import settings as settings_module

    monkeypatch.setattr(
        settings_module.Settings, "last_directory", lambda self: None
    )
    win = MainWindow()
    qtbot.addWidget(win)

    fixtures.write_poscar(tmp_path, atoms=4)
    win.open_file(tmp_path / "POSCAR")
    # Let the deferred _open_last_directory fire.
    qtbot.wait(200)

    assert win._calculation is not None, "the deferred restore wiped the open file"
    assert win._calculation.can(Capability.STRUCTURE)
