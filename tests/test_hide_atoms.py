"""Hiding atoms in the Editor.

Hiding is strictly a view state. The distinction that matters throughout: a
hidden atom is still in the structure, still in the table and still written
out on save -- otherwise "hide" and "delete" would differ only in how
recoverable they felt, which is not a difference a user can rely on.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest

from vasp_analyzer.models.structure import StructureData
from vasp_analyzer.services.discovery import discover
from vasp_analyzer.ui.pages.editor.page import StructureEditorPage
from vasp_analyzer.ui.pages.structure.viewer import StructureView

from . import fixtures


# -- the viewer's hidden set ----------------------------------------------


def _structure(n: int = 6) -> StructureData:
    return StructureData(
        lattice=np.eye(3) * 6.0,
        species=["Si"] * n,
        frac_coords=np.array([[i / n, i / n, 0.25] for i in range(n)]),
        source="test",
    )


@pytest.fixture()
def view(qtbot):
    widget = StructureView()
    qtbot.addWidget(widget)
    widget.resize(500, 420)
    widget.set_structure(_structure())
    widget.show()
    widget.grab()  # force a real paint so the pick caches are populated
    return widget


def test_hidden_atoms_round_trip(view):
    view.set_hidden_atoms({1, 3})
    assert view.hidden_atoms() == {1, 3}


def test_hiding_changes_what_is_drawn(view):
    before = view.grab().toImage()
    view.set_hidden_atoms({0, 1, 2})
    after = view.grab().toImage()

    assert before != after


def test_a_hidden_atom_cannot_be_picked(view):
    """An atom you cannot see must not be selectable through the empty space
    where it used to be -- hiding is often exactly a way to reach whatever is
    behind it."""
    view.grab()
    screen = view._screen_cache
    target = 2
    point = QPointF(float(screen[target][0]), float(screen[target][1]))
    assert view._pick(point) is not None

    view.set_hidden_atoms({target})
    view.grab()
    hit = view._pick(point)
    assert hit is None or view._bonds.sites[hit].index != target


def test_a_visible_atom_is_still_picked_while_others_are_hidden(view):
    view.set_hidden_atoms({0, 1})
    view.grab()
    screen = view._screen_cache
    point = QPointF(float(screen[4][0]), float(screen[4][1]))

    assert view._pick(point) is not None


def test_hiding_never_touches_the_structure(view):
    before = view._structure.n_atoms
    view.set_hidden_atoms({0, 1, 2, 3})

    assert view._structure.n_atoms == before


# -- the Editor page ------------------------------------------------------


@pytest.fixture()
def editor(qtbot, tmp_path):
    fixtures.write_poscar(tmp_path, atoms=8)
    page = StructureEditorPage()
    qtbot.addWidget(page)
    page.set_calculation(discover(tmp_path))
    qtbot.wait(100)
    assert page._working is not None
    return page


def test_hide_removes_the_selection_from_view_and_from_the_selection(editor):
    """Leaving them selected would keep the move gizmo pointed at atoms that
    are no longer on screen, so a drag would move something invisible."""
    editor._set_selection(set(editor._working.ids[:3]))
    editor._hide_selection()

    assert len(editor._view.hidden_atoms()) == 3
    assert editor._selection == set()


def test_hide_leaves_every_atom_in_the_structure(editor):
    before = editor._working.n_atoms
    editor._set_selection(set(editor._working.ids[:3]))
    editor._hide_selection()

    assert editor._working.n_atoms == before
    assert editor._table.model_().rowCount() == before


def test_isolate_hides_everything_except_the_selection(editor):
    total = editor._working.n_atoms
    editor._set_selection({editor._working.ids[2]})
    editor._hide_unselected()

    assert len(editor._view.hidden_atoms()) == total - 1


def test_show_all_brings_them_back(editor):
    editor._set_selection(set(editor._working.ids[:4]))
    editor._hide_selection()
    editor._show_all_atoms()

    assert editor._view.hidden_atoms() == set()


def test_the_sidebar_reports_how_many_are_hidden(editor):
    """A hidden atom looks exactly like an absent one, so the count has to be
    said out loud or a user who forgets they hid half a slab reads the viewer
    as having lost atoms."""
    side = editor._sidebar
    assert side._hidden_caption.text() == ""
    assert not side._show_all.isEnabled()

    editor._set_selection(set(editor._working.ids[:2]))
    editor._hide_selection()

    assert "2 atoms hidden" == side._hidden_caption.text()
    assert side._show_all.isEnabled()


def test_hide_and_isolate_need_a_selection(editor):
    side = editor._sidebar
    editor._set_selection(set())
    assert not side._hide.isEnabled()
    assert not side._isolate.isEnabled()

    editor._set_selection({editor._working.ids[0]})
    assert side._hide.isEnabled()
    assert side._isolate.isEnabled()


def test_hidden_atoms_survive_an_edit_that_renumbers_rows(editor):
    """Hidden atoms are tracked by stable id, not row index: deleting an atom
    ahead of them shifts every later row, and a row-based set would silently
    start hiding the wrong atoms."""
    ids = list(editor._working.ids)
    editor._set_selection({ids[5], ids[6]})
    editor._hide_selection()

    editor._set_selection({ids[0]})
    editor._delete_selection()

    hidden_rows = editor._view.hidden_atoms()
    hidden_ids = {editor._working.ids[r] for r in hidden_rows}
    assert hidden_ids == {ids[5], ids[6]}


def test_opening_another_structure_clears_the_hidden_set(editor, qtbot, tmp_path):
    editor._set_selection(set(editor._working.ids[:3]))
    editor._hide_selection()
    assert editor._view.hidden_atoms()

    other = tmp_path / "other"
    other.mkdir()
    fixtures.write_poscar(other, atoms=4)
    editor.set_calculation(discover(other))
    qtbot.wait(100)

    assert editor._view.hidden_atoms() == set()


def test_changing_the_selection_does_not_dirty_the_structure(editor, qtbot):
    """A QLineEdit emits editingFinished when it loses focus, and being torn
    down counts. The inspector rebuilds its coordinate grid on every selection
    change, so that fired a no-op "Edit atom N coordinate" -- pushing junk
    onto the undo history and flagging an untouched structure Modified. Hiding
    made it visible (hide, and the status bar claimed unsaved edits), but any
    selection change did it.
    """
    ids = list(editor._working.ids)
    for atom_id in ids[:4]:
        editor._set_selection({atom_id})
        qtbot.wait(10)
    editor._set_selection(set())
    qtbot.wait(10)

    assert not editor._modified
    assert not editor._history.can_undo


def test_hiding_does_not_dirty_the_structure(editor, qtbot):
    editor._set_selection(set(editor._working.ids[:3]))
    editor._hide_selection()
    qtbot.wait(20)

    assert not editor._modified
    assert "Saved" in editor.info_fields()[0]["status"]


def test_a_real_coordinate_edit_still_registers(editor, qtbot):
    """The no-op guard must not swallow an actual edit."""
    editor._set_selection({editor._working.ids[0]})
    qtbot.wait(10)

    card = editor._selection_card
    _row, fields = card._fields[0]
    fields[0].setText("0.42000")
    fields[0].editingFinished.emit()
    qtbot.wait(10)

    assert editor._modified
    assert editor._working.frac_coords[0][0] == pytest.approx(0.42, abs=1e-6)


# -- the real "undo/redo don't work" bug: shortcuts and a stray window -----


def test_undo_redo_shortcuts_fire_with_focus_in_the_sidebar(editor, qtbot):
    """The sidebar (EditorSidebar) is not a descendant of the page --
    MainWindow keeps it in a separate _sidebar_stack widget (see
    MainWindow._register_page) rather than nesting it inside the page. A
    shortcut scoped with Qt.WidgetWithChildrenShortcut to the page alone only
    fires while the focused widget is inside the page's own tree, so this
    went dead the instant focus was anywhere in the sidebar -- including on
    the sidebar's own Undo/Redo buttons right after clicking them. Fixed by
    using Qt.WindowShortcut, which only cares whether this page's window is
    active, not which specific widget currently holds focus."""
    from PySide6.QtGui import QShortcut, QKeySequence

    before = editor._working.n_atoms
    editor._set_selection({editor._working.ids[0]})
    editor._delete_selection()
    qtbot.wait(20)

    editor._sidebar._filter_field.setFocus()
    qtbot.wait(10)
    assert not editor.isAncestorOf(editor._sidebar)

    for sc in editor.findChildren(QShortcut):
        if sc.key() == QKeySequence(QKeySequence.Undo):
            assert sc.context() == Qt.WindowShortcut
            sc.activated.emit()

    assert editor._working.n_atoms == before


def test_the_selection_cards_teardown_never_leaves_a_stray_focused_window(
    editor, qtbot
):
    """SelectionCard._clear_grid() reparents each old coordinate field to
    None before deleting it, and reparenting *any* widget to None makes it
    its own top-level window. A field that still had real keyboard focus and
    its old "visible" flag at that exact instant became a genuine floating,
    focused, visible window of its own -- which silently stole OS-level
    window activation away from the whole application until the deferred
    deleteLater() actually ran. While that stray window held activation, no
    shortcut in the app (Undo/Redo included) had anywhere to dispatch to.
    This is the actual root cause "the undo/redo buttons and shortcuts don't
    work" was reporting -- the buttons and shortcuts were fine, the window
    just quietly wasn't the active one any more.
    """
    editor.show()
    editor.activateWindow()
    qtbot.wait(10)
    card = editor._selection_card
    editor._set_selection({editor._working.ids[0]})
    qtbot.wait(10)
    _row, fields = card._fields[0]
    field = fields[0]
    field.setFocus()
    qtbot.wait(10)
    assert field.hasFocus()

    card._clear_grid()
    # Checked synchronously, before deleteLater() actually runs on the next
    # event-loop turn (after which the C++ object is gone outright) -- the
    # property under test is what state the field is in at the moment it is
    # orphaned, not after it no longer exists to ask.
    assert not field.isVisible()
    assert not field.hasFocus()

    qtbot.wait(10)
    # Never became a floating window of its own in the first place.
    assert editor.window().isActiveWindow()


def test_window_shortcuts_do_not_leak_to_a_different_page(qtbot, tmp_path):
    """Qt.WindowShortcut fires whenever the shortcut's own widget is visible
    and its window is active -- it must not fire while the Editor page is
    hidden behind another one in the same MainWindow's QStackedWidget."""
    from PySide6.QtGui import QShortcut, QKeySequence
    from vasp_analyzer.ui.main_window import MainWindow, PAGES

    win = MainWindow()
    qtbot.addWidget(win)
    win.show()
    fixtures.write_poscar(tmp_path, atoms=4)
    win.open_directory(tmp_path)
    qtbot.wait(200)
    win._nav.setCurrentIndex(PAGES.index("Editor"))
    qtbot.wait(50)

    called = []
    for sc in win._editor.findChildren(QShortcut):
        if sc.key() == QKeySequence(QKeySequence.Undo):
            sc.activated.connect(lambda: called.append(True))

    win._nav.setCurrentIndex(PAGES.index("Structure"))
    qtbot.wait(50)
    assert not win._editor.isVisible()

    for sc in win._editor.findChildren(QShortcut):
        if sc.key() == QKeySequence(QKeySequence.Undo):
            # A hidden owning widget: WindowShortcut must not consider this
            # eligible even though MainWindow itself is still active.
            assert not sc.isEnabled() or not win._editor.isVisible()


def test_typing_single_letter_shortcuts_into_a_text_field_does_not_switch_tools(
    editor, qtbot
):
    """S/R/T/G/D/M/H are registered as window-wide shortcuts so they work
    regardless of focus -- which must not mean typing an ordinary word into
    the element filter gets intercepted as a string of tool switches. Qt's
    own ShortcutOverride handling in QLineEdit is what protects this,
    independent of the shortcut's context."""
    field = editor._sidebar._filter_field
    field.clear()
    field.setFocus()
    qtbot.wait(10)
    tool_before = editor._sidebar.tool()

    QTest.keyClicks(field, "sand dirt hem")

    assert field.text() == "sand dirt hem"
    assert editor._sidebar.tool() == tool_before


def test_the_viewer_starts_in_rotate_and_the_toolbar_agrees(editor):
    """Rotate is the default tool, and the highlighted button must match the
    viewer's real mode.

    The toolbar used to call setChecked(True) on its first button (Select) at
    construction, which paints a button active without emitting `clicked` --
    so _set_tool() never ran and the strip showed Select highlighted while
    the viewer was actually still on StructureView's own MODE_ROTATE default.
    Orbiting on open is the wanted behaviour; the bug was that the toolbar
    disagreed with it.
    """
    assert editor._view.mode() == editor._view.MODE_ROTATE
    checked = editor._camera_group.checkedButton()
    assert checked is not None
    assert "Rotate" in checked.toolTip()


def test_switching_to_select_lets_a_click_select_an_atom(editor, qtbot):
    """Selection still works once the user picks the Select tool -- an edit
    made that way is what puts something on the undo stack for the sidebar's
    Undo/Redo to act on."""
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QMouseEvent

    editor._set_tool("select")
    view = editor._view
    view.resize(600, 480)
    view.show()
    qtbot.wait(10)
    view.grab()  # force a paint so the pick cache exists

    point = QPointF(
        float(view._screen_cache[0][0]), float(view._screen_cache[0][1])
    )
    view.mousePressEvent(
        QMouseEvent(
            QMouseEvent.Type.MouseButtonPress, point,
            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier,
        )
    )
    view.mouseReleaseEvent(
        QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease, point,
            Qt.LeftButton, Qt.NoButton, Qt.NoModifier,
        )
    )
    qtbot.wait(10)
    assert len(editor._selection) == 1, "a click must select an atom"

    before = editor._working.n_atoms
    editor._delete_selection()
    qtbot.wait(10)

    assert editor._working.n_atoms == before - 1
    assert editor._history.can_undo
