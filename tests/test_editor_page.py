"""The Editor page driven through a real window.

The behaviours checked here are the ones a user would notice immediately if
they broke: the three views agreeing about what is selected, a coordinate-mode
switch not moving anything, and undo returning the structure exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from vasp_analyzer.ui.main_window import PAGES, MainWindow
from vasp_analyzer.ui.pages.editor.page import StructureEditorPage
from vasp_analyzer.ui.pages.editor.viewer import NEIGHBOR_DISTANCE_RATIO
from vasp_analyzer.ui.plotting import apply_plot_style
from vasp_analyzer.ui.style import apply_theme

from . import fixtures

EDITOR_INDEX = PAGES.index("Editor")


@pytest.fixture(scope="module", autouse=True)
def _style(qapp):
    apply_theme(qapp)
    apply_plot_style()
    return qapp


@pytest.fixture
def window(qtbot, monkeypatch, tmp_path):
    from vasp_analyzer.services import settings as settings_module

    monkeypatch.setattr(
        settings_module.Settings, "set_last_directory", lambda self, path: None
    )
    monkeypatch.setattr(
        settings_module.Settings, "recent_directories", lambda self: []
    )
    window = MainWindow()
    qtbot.addWidget(window)
    return window


@pytest.fixture
def editor(window, qtbot, tmp_path):
    directory = tmp_path / "edit"
    directory.mkdir()
    fixtures.case_a_complete(directory)
    window.open_directory(directory)
    window._nav.setCurrentIndex(EDITOR_INDEX)
    qtbot.wait(150)

    page = window._editor
    assert isinstance(page, StructureEditorPage)
    qtbot.waitUntil(lambda: page._working is not None, timeout=5000)
    return page


def test_editor_is_registered_after_structure():
    assert PAGES[0] == "Structure"
    assert PAGES[1] == "Editor"


def test_opening_a_calculation_loads_a_working_structure(editor):
    assert editor._working is not None
    assert editor._working.n_atoms > 0
    # A freshly opened structure is not yet modified.
    assert editor._modified is False
    assert editor._stack.currentIndex() == 0


def test_original_saved_and_working_start_as_separate_objects(editor):
    assert editor._original is not None and editor._saved is not None
    assert editor._original is not editor._working
    assert editor._saved is not editor._working


def test_selection_propagates_to_viewer_table_and_inspector(editor):
    atom_id = editor._working.ids[0]
    editor._set_selection({atom_id})

    assert editor._view.selection() == {0}
    assert {
        editor._table.model_().structure_row(index.row())
        for index in editor._table.selectionModel().selectedRows()
    } == {0}
    assert "1 atom" in editor._selection_card._count.text()


def test_table_selection_reaches_the_viewer(editor, qtbot):
    editor._on_table_selection({1})
    assert editor._view.selection() == {1}
    assert editor._selection == {editor._working.ids[1]}


def test_switching_coordinate_mode_never_moves_an_atom(editor):
    before = editor._working.frac_coords.copy()

    editor._set_coordinate_mode(False)
    editor._set_coordinate_mode(True)

    np.testing.assert_allclose(editor._working.frac_coords, before)
    assert editor._modified is False


def test_cartesian_display_matches_the_lattice(editor):
    editor._set_coordinate_mode(False)
    model = editor._table.model_()
    shown = float(model.data(model.index(0, 2)))
    expected = editor._working.cart_coords[0][0]
    assert shown == pytest.approx(expected, abs=1e-4)


def test_editing_a_coordinate_marks_modified_and_is_undoable(editor):
    before = editor._working.frac_coords[0].copy()

    editor._edit_coordinate(0, 2, 0.4252)
    assert editor._working.frac_coords[0][2] == pytest.approx(0.4252)
    assert editor._modified is True
    assert editor._history.can_undo

    editor._undo()
    np.testing.assert_allclose(editor._working.frac_coords[0], before)


def test_add_delete_and_duplicate_keep_the_atom_count_honest(editor):
    start = editor._working.n_atoms

    editor._insert_atom("O", np.array([0.9, 0.9, 0.9]), True)
    assert editor._working.n_atoms == start + 1
    assert editor._working.species[-1] == "O"
    # The new atom becomes the selection, ready to be moved.
    assert len(editor._selection) == 1

    editor._duplicate_selection()
    assert editor._working.n_atoms == start + 2

    editor._delete_selection()
    assert editor._working.n_atoms == start + 1
    assert editor._selection == set()


def test_undo_restores_a_deletion_exactly(editor):
    before = editor._working
    editor._set_selection({editor._working.ids[0]})
    editor._delete_selection()
    assert editor._working.n_atoms == before.n_atoms - 1

    editor._undo()
    assert editor._working.n_atoms == before.n_atoms
    assert editor._working.species == before.species
    np.testing.assert_allclose(editor._working.frac_coords, before.frac_coords)


def test_a_drag_records_one_history_entry(editor):
    editor._set_selection({editor._working.ids[0]})
    depth = len(editor._history._done)
    origin = editor._working

    editor._drag_origin = origin
    # Several intermediate positions, as a real drag would produce.
    for step in (0.1, 0.2, 0.3):
        editor._on_drag_moved(np.array([step, 0.0, 0.0]))
    editor._on_drag_finished(np.array([0.3, 0.0, 0.0]))

    assert len(editor._history._done) == depth + 1
    editor._undo()
    np.testing.assert_allclose(editor._working.frac_coords, origin.frac_coords)


def test_supercell_builds_a_real_structure_and_undoes_cleanly(editor):
    before = editor._working
    n_before = before.n_atoms
    a_before = before.lattice_parameters[0]

    editor._build_supercell(2, 2, 1)
    assert editor._working.n_atoms == n_before * 4
    assert editor._working.lattice_parameters[0] == pytest.approx(a_before * 2)

    editor._undo()
    assert editor._working.n_atoms == n_before
    np.testing.assert_allclose(editor._working.lattice, before.lattice)


def test_fixing_atoms_records_selective_dynamics(editor):
    editor._set_selection({editor._working.ids[0]})
    editor._toggle_fixed(None)

    assert editor._working.has_selective_dynamics
    assert not editor._working.selective[0].any()

    editor._toggle_fixed(None)
    assert editor._working.selective[0].all()


def test_clear_selection_does_not_delete_anything(editor):
    n_before = editor._working.n_atoms
    editor._set_selection(set(editor._working.ids[:2]))
    editor._sidebar.clearSelectionRequested.emit()

    assert editor._selection == set()
    assert editor._working.n_atoms == n_before


def test_wrap_brings_a_strayed_atom_back_into_the_cell(editor):
    editor._edit_coordinate(0, 0, 1.7)
    editor._wrap_into_cell()

    coords = editor._working.frac_coords
    assert ((coords >= 0) & (coords < 1)).all()


def test_saving_a_poscar_reopens_with_the_same_structure(editor, tmp_path, monkeypatch):
    editor._insert_atom("O", np.array([0.31, 0.42, 0.53]), True)
    editor._set_selection({editor._working.ids[0]})
    editor._toggle_fixed(None)

    target = tmp_path / "POSCAR_out"
    monkeypatch.setattr(
        "vasp_analyzer.ui.pages.editor.page.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(target), ""),
    )
    editor._save_poscar()

    assert target.exists()
    assert editor._modified is False

    from vasp_analyzer.models.editable import EditableStructure
    from vasp_analyzer.parsers.poscar import parse_poscar

    result = parse_poscar(target)
    assert result.ok
    reopened = EditableStructure.from_structure(result.value)

    assert reopened.n_atoms == editor._working.n_atoms
    assert reopened.composition == editor._working.composition
    # The constraint applied above survived the write.
    assert int((~reopened.selective).all(axis=1).sum()) == 1


def test_save_never_targets_the_source_file_without_confirmation(editor, monkeypatch):
    """The calculation's own POSCAR must not be replaced by an accidental
    Enter in the save dialog."""
    source = editor._sources[editor._sidebar._source.currentText()]
    monkeypatch.setattr(
        "vasp_analyzer.ui.pages.editor.page.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(source), ""),
    )
    asked = {"count": 0}

    def _decline(*args, **kwargs):
        from PySide6.QtWidgets import QMessageBox

        asked["count"] += 1
        return QMessageBox.Cancel

    monkeypatch.setattr(
        "vasp_analyzer.ui.pages.editor.page.QMessageBox.warning", _decline
    )
    before = source.read_bytes()
    editor._save_poscar()

    assert asked["count"] == 1
    assert source.read_bytes() == before


def test_validation_rejects_a_structure_with_no_atoms(editor):
    editor._set_selection(set(editor._working.ids))
    editor._delete_selection()

    assert editor._validation_errors()
    assert editor._guard_export() is False


def test_editor_survives_a_calculation_without_a_structure(window, qtbot, tmp_path):
    directory = tmp_path / "no_structure"
    directory.mkdir()
    # EIGENVAL but deliberately no POSCAR/CONTCAR: there is nothing to edit.
    fixtures.write_eigenval(directory)
    fixtures.write_incar(directory)
    window.open_directory(directory)
    window._nav.setCurrentIndex(EDITOR_INDEX)
    qtbot.wait(150)

    page = window._editor
    # The empty state, not a crash.
    assert page._stack.currentIndex() == 1


def test_editor_handles_a_closed_calculation(window, qtbot):
    window._editor.set_calculation(None)
    qtbot.wait(50)
    assert window._editor._working is None
    assert window._editor._stack.currentIndex() == 1


# --- move gizmo: neighbour handles, pair-move, centroid ---------------------------


def test_single_atom_selection_offers_neighbor_handles(editor):
    """Real data: graphene's carbon has three physical bonds, all to the same
    other basis atom via different periodic images."""
    editor._set_selection({editor._working.ids[0]})
    handles = editor._neighbor_handles_for(set(editor._rows_for_selection()))
    assert len(handles) >= 1
    for _row, direction, _distance in handles:
        assert pytest.approx(1.0) == float((direction**2).sum() ** 0.5)


def test_neighbor_handles_are_empty_for_multi_atom_selection(editor):
    editor._set_selection(set(editor._working.ids[:2]))
    handles = editor._neighbor_handles_for(set(editor._rows_for_selection()))
    assert handles == []


def test_sparse_neighbors_beyond_the_distance_ratio_are_excluded(editor):
    """An atom with one genuinely close neighbour and others much farther
    away should not sprout handles pointing at the far ones."""
    working = editor._working
    # Add an isolated atom far from everything else.
    grown, atom_id = working.with_atom_added("H", [10.0, 10.0, 10.0])
    editor._working = grown
    editor._set_selection({atom_id})
    handles = editor._neighbor_handles_for(set(editor._rows_for_selection()))
    # Whatever it finds must all be within NEIGHBOR_DISTANCE_RATIO of the
    # nearest, not an arbitrary k=4 regardless of how far away they are.
    if handles:
        nearest = min(h[2] for h in handles)
        assert all(h[2] <= nearest * NEIGHBOR_DISTANCE_RATIO for h in handles)


def test_two_atom_selection_offers_a_pair_handle_toward_the_second(editor, qtbot):
    ids = list(editor._working.ids)
    editor._set_selection({ids[0]})
    editor._set_selection({ids[0], ids[1]})
    # Order isn't guaranteed here (programmatic selection falls back to
    # sorted row order), but *some* pair handle must appear either way.
    editor._apply_pair_handle(set(editor._rows_for_selection()))
    assert editor._view._pair_handle is not None


def test_pair_handle_targets_whichever_atom_was_selected_second(editor):
    """The direction must point toward the *second*-selected atom, in the
    order the view actually recorded — not just "the other one" by row
    number, since sorted-fallback and click order can disagree."""
    working = editor._working
    rows = list(range(working.n_atoms))
    assert len(rows) >= 2
    # Force a specific, deliberately reversed click order.
    editor._view._selection = {rows[1], rows[0]}
    editor._view._selection_order = [rows[1], rows[0]]
    editor._selection = {working.ids[rows[1]], working.ids[rows[0]]}
    editor._apply_pair_handle({rows[1], rows[0]})

    direction, distance, target_row = editor._view._pair_handle
    expected_direction, expected_distance = working.direction_to(rows[1], rows[0])
    np.testing.assert_allclose(direction, expected_direction)
    assert distance == pytest.approx(expected_distance)
    assert target_row == rows[0]


def test_pair_drag_moves_only_the_first_selected_atom(editor):
    """The end-to-end path: press the pair handle, drag it, release — only
    the anchor (first-selected) atom's coordinates may change."""
    working = editor._working
    editor._view._selection = {0, 1}
    editor._view._selection_order = [0, 1]
    editor._selection = {working.ids[0], working.ids[1]}
    editor._apply_pair_handle({0, 1})
    assert editor._view._pair_handle is not None

    before = working.frac_coords.copy()
    editor._view._drag_target_rows = {0}
    editor._drag_origin = working
    editor._on_drag_moved(np.array([0.05, 0.0, 0.0]))
    editor._on_drag_finished(np.array([0.05, 0.0, 0.0]))

    np.testing.assert_allclose(editor._working.frac_coords[1], before[1])
    assert not np.allclose(editor._working.frac_coords[0], before[0])
    assert "toward its neighbour" in editor._history.undo_label


def test_pair_handle_is_cleared_when_selection_stops_being_a_pair(editor):
    ids = list(editor._working.ids)
    editor._set_selection({ids[0]})
    editor._apply_pair_handle(set(editor._rows_for_selection()))
    assert editor._view._pair_handle is None  # only one atom selected

    editor._set_selection({ids[0], ids[1]})
    editor._apply_pair_handle(set(editor._rows_for_selection()))
    assert editor._view._pair_handle is not None

    editor._set_selection({ids[0], ids[1], ids[2] if len(ids) > 2 else ids[0]})


def test_centroid_shown_for_any_nonempty_selection(editor):
    """A single selected atom's "centre" is a well-defined answer too — its
    own position — so the readout is not withheld until a second atom joins;
    only an empty selection hides it."""
    card = editor._selection_card
    ids = list(editor._working.ids)

    editor._set_selection(set())
    assert card._centroid_host.isHidden()

    editor._set_selection({ids[0]})
    assert not card._centroid_host.isHidden()
    assert card._centroid_value.text() != "—"

    editor._set_selection({ids[0], ids[1]})
    assert not card._centroid_host.isHidden()
    assert card._centroid_value.text() != "—"


def test_centroid_of_a_single_atom_is_its_own_position(editor):
    card = editor._selection_card
    working = editor._working
    editor._set_selection({working.ids[0]})

    expected = working.frac_coords[0]
    shown = card._centroid_value.text()
    parts = [float(p.strip()) for p in shown.split(",")]
    np.testing.assert_allclose(parts, expected, atol=1e-4)


def test_centroid_value_matches_the_mean_of_selected_coordinates(editor):
    card = editor._selection_card
    working = editor._working
    rows = [0, 1] if working.n_atoms > 1 else [0]
    if len(rows) < 2:
        pytest.skip("fixture has fewer than two atoms")
    ids = [working.ids[r] for r in rows]
    editor._set_selection(set(ids))

    expected = working.frac_coords[rows].mean(axis=0)
    shown = card._centroid_value.text().split(" Å")[0] if " Å" in card._centroid_value.text() else card._centroid_value.text()
    parts = [float(p.strip()) for p in shown.split(",")]
    np.testing.assert_allclose(parts, expected, atol=1e-4)


def test_centroid_copy_button_writes_to_the_clipboard(editor, qtbot):
    from PySide6.QtGui import QGuiApplication

    ids = list(editor._working.ids)
    editor._set_selection({ids[0], ids[1]})
    card = editor._selection_card
    card._copy_centroid()
    assert card._centroid_value.text() in QGuiApplication.clipboard().text()


# --- structure operations: Relax Selected removed, Sort renamed -------------


def test_relax_selected_has_been_removed(editor):
    """It was a placeholder button wired to a backend that never existed —
    disabled, with a tooltip saying so. Rather than leave dead UI around,
    it's gone outright; nothing should still reference it."""
    card = editor._operations_card
    assert not hasattr(card, "relaxRequested")

    grid = card.body().itemAt(card.body().count() - 1).layout()
    labels = {
        grid.itemAt(i).widget().text()
        for i in range(grid.count())
        if grid.itemAt(i).widget() is not None
    }
    assert "Relax Selected" not in labels


def test_sort_button_is_labelled_sort_element_list(editor):
    card = editor._operations_card
    grid = card.body().itemAt(card.body().count() - 1).layout()
    labels = {
        grid.itemAt(i).widget().text()
        for i in range(grid.count())
        if grid.itemAt(i).widget() is not None
    }
    assert "Sort Element List" in labels
    assert "Sort by Element" not in labels


def test_sort_element_list_still_groups_atoms_by_species(editor):
    """The rename didn't change what the button does — still a valid grouping
    for a POSCAR counts line to describe."""
    editor._insert_atom("O", np.array([0.9, 0.9, 0.9]), True)
    editor._set_tool("select")

    editor._sort_by_element()
    species = editor._working.species

    # Each element's rows must be contiguous — the property a POSCAR counts
    # line actually needs, regardless of which order the elements come in.
    order: list[str] = []
    for symbol in species:
        if symbol not in order:
            order.append(symbol)
    rebuilt = [s for element in order for s in species if s == element]
    assert species == rebuilt


# --- Fixed column: a read-only lock icon, not a checkbox --------------------


def test_fixed_column_has_no_checkbox_and_shows_a_lock_icon_when_fixed(editor):
    from PySide6.QtCore import Qt

    working = editor._working
    working.selective[0] = [False, False, False]  # fully fixed
    working.selective[1] = [True, True, True] if working.n_atoms > 1 else [False, False, False]

    model = editor._table.model_()
    model.set_structure(working)

    last = model.columnCount() - 1
    # No text label on the Fixed column — it's a single icon, and a header
    # word above a column that is otherwise blank for most rows reads
    # stranger than no label at all.
    assert model.headerData(last, Qt.Horizontal) == ""
    assert "Occupancy" not in [
        model.headerData(c, Qt.Horizontal) for c in range(model.columnCount())
    ]
    assert "Fixed" not in [
        model.headerData(c, Qt.Horizontal) for c in range(model.columnCount())
    ]

    fixed_index = model.index(0, last)
    assert not bool(model.flags(fixed_index) & Qt.ItemIsUserCheckable)
    icon = model.data(fixed_index, Qt.DecorationRole)
    assert icon is not None and not icon.isNull()

    if working.n_atoms > 1:
        free_index = model.index(1, last)
        assert model.data(free_index, Qt.DecorationRole) is None


def test_fixed_column_is_not_editable_via_setData(editor):
    """The checkbox is gone deliberately — Fix/Free now only happen through
    the Fix button (whole selection) or the per-atom Fix action, both of
    which go through _commit and are undoable. setData on this column must
    no longer be able to bypass that."""
    from PySide6.QtCore import Qt

    model = editor._table.model_()
    last = model.columnCount() - 1
    index = model.index(0, last)
    before = editor._working.selective[0].copy()

    ok = model.setData(index, Qt.Checked, Qt.CheckStateRole)

    assert ok is False
    np.testing.assert_array_equal(editor._working.selective[0], before)


# --- Grab and Move: the renamed tool -----------------------------------------


def test_move_tool_is_labelled_grab_and_move(editor):
    button = editor._sidebar._tool_buttons["move"]
    assert button.text() == "Grab and Move"

    from vasp_analyzer.ui.pages.editor.sidebar import TOOLS

    tips = {key: tip for key, _caption, tip in TOOLS}
    assert "(G)" in tips["move"]

    assert editor._selection_card._move.text() == "Grab and Move"


# --- keyboard shortcuts: S / R / G ---------------------------------------------


def _shown(editor, qtbot):
    """A QShortcut only fires through Qt's real shortcut map, which requires
    the window to actually be shown and focused — the ``editor`` fixture
    never calls ``show()`` because most tests here drive the page directly
    and don't need a real window. These do."""
    window = editor.window()
    window.show()
    qtbot.waitExposed(window)
    window.activateWindow()
    editor.setFocus()
    qtbot.waitUntil(lambda: editor.window().isActiveWindow(), timeout=2000)
    return window


def test_s_r_g_switch_tools(editor, qtbot):
    from PySide6.QtCore import Qt

    from vasp_analyzer.ui.pages.editor.viewer import EditorView

    _shown(editor, qtbot)

    qtbot.keyClick(editor, Qt.Key_G)
    assert editor._view.mode() == EditorView.MODE_MOVE

    qtbot.keyClick(editor, Qt.Key_R)
    assert editor._view.mode() == EditorView.MODE_ROTATE

    qtbot.keyClick(editor, Qt.Key_S)
    assert editor._view.mode() == EditorView.MODE_SELECT


def test_m_now_switches_to_measure_since_move_took_g(editor, qtbot):
    """R used to mean Measure; it now means Rotate, so Measure moved to M —
    the letter Move gave up. Both keys must land on their new tool, not the
    old one."""
    from PySide6.QtCore import Qt

    from vasp_analyzer.ui.pages.editor.viewer import EditorView

    _shown(editor, qtbot)

    qtbot.keyClick(editor, Qt.Key_M)
    assert editor._view.mode() == EditorView.MODE_MEASURE


def test_t_switches_to_pan(editor, qtbot):
    from PySide6.QtCore import Qt

    from vasp_analyzer.ui.pages.editor.viewer import EditorView

    _shown(editor, qtbot)

    qtbot.keyClick(editor, Qt.Key_T)
    assert editor._view.mode() == EditorView.MODE_PAN


def test_add_atom_tool_is_gone_from_the_edit_tools_list(editor):
    """It only ever switched to Select mode under a different label — real
    insertion happens through the AddAtomCard form, unaffected by this."""
    from vasp_analyzer.ui.pages.editor.sidebar import TOOLS

    keys = [key for key, _caption, _tip in TOOLS]
    assert "add" not in keys
    assert "add" not in editor._sidebar._tool_buttons

    # The form itself must still work.
    start = editor._working.n_atoms
    editor._insert_atom("O", np.array([0.9, 0.9, 0.9]), True)
    assert editor._working.n_atoms == start + 1


# --- click-to-select: Ctrl/Cmd toggles, Shift only extends -------------------


def _press_atom(view, site_index, modifiers=None):
    """Simulate a left-click that resolves to a given site, without needing a
    real on-screen position: ``_pick`` is monkeypatched to the target index,
    isolating the modifier-handling logic in ``_select_click`` from actual
    screen-space picking (which needs a shown, painted widget)."""
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    view._pick = lambda _pos: site_index
    modifiers = modifiers if modifiers is not None else Qt.NoModifier
    event = QMouseEvent(
        QEvent.MouseButtonPress, QPointF(0, 0), QPointF(0, 0),
        Qt.LeftButton, Qt.LeftButton, modifiers,
    )
    view.mousePressEvent(event)


def test_shift_click_extends_the_selection(editor):
    from PySide6.QtCore import Qt

    if editor._working.n_atoms < 2:
        pytest.skip("fixture has fewer than two atoms")

    view = editor._view
    editor._set_tool("select")

    _press_atom(view, 0)
    assert view.selection() == {0}

    _press_atom(view, 1, Qt.ShiftModifier)
    assert view.selection() == {0, 1}, "shift-click must add, not replace"


def test_shift_click_on_an_already_selected_atom_deselects_just_that_atom(editor):
    """Shift toggles, the same as Ctrl/Cmd — clicking an atom you already
    shift-clicked is how you drop it back out of a multi-atom selection
    without losing the rest."""
    from PySide6.QtCore import Qt

    if editor._working.n_atoms < 2:
        pytest.skip("fixture has fewer than two atoms")

    view = editor._view
    editor._set_tool("select")

    _press_atom(view, 0)
    _press_atom(view, 1, Qt.ShiftModifier)
    assert view.selection() == {0, 1}

    _press_atom(view, 1, Qt.ShiftModifier)
    assert view.selection() == {0}, "shift-click on a selected atom removes only it"


def test_ctrl_click_still_toggles_membership(editor):
    from PySide6.QtCore import Qt

    view = editor._view
    editor._set_tool("select")

    _press_atom(view, 0)
    assert view.selection() == {0}

    _press_atom(view, 0, Qt.ControlModifier)
    assert view.selection() == set(), "ctrl-click on a selected atom removes it"


def test_move_tool_click_with_no_prior_selection_selects_and_raises_the_gizmo(editor):
    """The reported gap: hitting G with nothing selected yet, then clicking an
    atom, used to do nothing at all — the gizmo has nowhere to anchor until
    something is selected, and Move mode didn't fall back to picking. It now
    does, so G-then-click is a complete one-step "grab this atom" gesture."""
    view = editor._view
    editor._set_tool("move")
    assert view.selection() == set()

    _press_atom(view, 0)

    assert view.selection() == {0}
    assert view._gizmo_handles() is not None, "the gizmo must appear immediately"


# --- pop-out: the viewer's toolbar, axis bar, and shortcuts travel with it --


def test_popped_out_viewer_carries_its_toolbar_and_axis_bar(editor):
    """Regression: the pop-out originally moved only the bare 3D view,
    leaving the floating select/rotate/pan toolbar and the axis-orientation
    bar behind on the now-empty card — the popped-out window had no way to
    switch tools, zoom, fit, or reorient at all. Popping out self._holder
    (the toolbar and axis bar are its floating children) instead of
    self._view fixes that; both must still be visible once popped out."""
    editor._popout._toggle()
    try:
        assert editor._toolbar.isVisible()
        assert editor._axis_bar.isVisible()
        assert editor._view.window() is editor._popout._host
    finally:
        editor._popout._host.close()

    assert editor._view.window() is not editor._popout._host
    assert editor._holder.parentWidget() is not None


def test_editor_shortcuts_still_work_while_the_popout_is_the_active_window(
    editor, qtbot
):
    """QShortcut with WidgetWithChildrenShortcut context only fires while
    focus sits inside the *page's* own widget tree — which the popped-out
    window is no longer part of, since its content now lives in a separate
    top-level window. PopoutButton's shortcuts= parameter re-installs the
    same (key, slot) pairs on that window; this confirms one of them —
    G for Grab and Move — actually reaches the same handler there."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from vasp_analyzer.ui.pages.editor.viewer import EditorView

    window = editor.window()
    window.show()
    qtbot.waitExposed(window)

    editor._popout._toggle()
    host = editor._popout._host
    try:
        host.show()
        qtbot.waitExposed(host)
        host.activateWindow()
        qtbot.waitUntil(lambda: host.isActiveWindow(), timeout=2000)

        assert editor._view.mode() != EditorView.MODE_MOVE
        QTest.keyClick(host, Qt.Key_G)
        assert editor._view.mode() == EditorView.MODE_MOVE
    finally:
        host.close()
