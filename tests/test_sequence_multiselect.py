"""Multi-selection in the sequence viewer (MD and NEB pages).

Plain click keeps its historical meaning -- select exactly this atom -- and
Ctrl/Shift+click toggles membership instead, matching the Editor's
conventions. ``atomSelected`` keeps firing with the last-clicked atom so the
tracked-atom machinery and the sidebar chip need no special cases for single
clicks.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent

from vasp_analyzer.models.structure import StructureData
from vasp_analyzer.ui.pages.sequence_view import SequenceView


def _structure(n: int = 8) -> StructureData:
    return StructureData(
        lattice=np.eye(3) * 8.0,
        species=["C"] * n,
        frac_coords=np.array([[i / n, (i * 2 % n) / n, 0.3] for i in range(n)]),
        source="test",
    )


@pytest.fixture()
def view(qtbot):
    widget = SequenceView()
    qtbot.addWidget(widget)
    widget.resize(520, 440)
    widget.set_structure(_structure())
    widget.set_mode(widget.MODE_SELECT)
    widget.show()
    widget.grab()  # populate the pick caches
    return widget


def _click(view, atom, mods=Qt.NoModifier):
    view.grab()  # fresh screen cache -- tracking can move the camera
    point = QPointF(
        float(view._screen_cache[atom][0]), float(view._screen_cache[atom][1])
    )
    view.mousePressEvent(
        QMouseEvent(QMouseEvent.Type.MouseButtonPress, point,
                    Qt.LeftButton, Qt.LeftButton, mods)
    )
    view.mouseReleaseEvent(
        QMouseEvent(QMouseEvent.Type.MouseButtonRelease, point,
                    Qt.LeftButton, Qt.NoButton, mods)
    )


def test_plain_click_selects_exactly_one(view):
    _click(view, 0)
    assert view.selection() == [0]
    assert view.selected_index() == 0


def test_shift_click_accumulates_and_ctrl_click_too(view):
    _click(view, 0)
    _click(view, 3, Qt.ShiftModifier)
    _click(view, 5, Qt.ControlModifier)
    assert view.selection() == [0, 3, 5]
    # Tracking follows the atom clicked last.
    assert view.selected_index() == 5


def test_modified_click_on_a_selected_atom_toggles_it_off(view):
    _click(view, 0)
    _click(view, 3, Qt.ShiftModifier)
    _click(view, 0, Qt.ShiftModifier)
    assert view.selection() == [3]


def test_plain_click_replaces_a_multi_selection(view):
    _click(view, 0)
    _click(view, 3, Qt.ShiftModifier)
    _click(view, 6)
    assert view.selection() == [6]


def test_clicking_empty_space_clears(view):
    _click(view, 0)
    view.mousePressEvent(
        QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(1.0, 1.0),
                    Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    )
    assert view.selection() == []
    assert view.selected_index() is None


def test_selection_changed_signal_reports_the_full_set(view, qtbot):
    seen = []
    view.selectionChanged.connect(seen.append)
    _click(view, 1)
    _click(view, 4, Qt.ShiftModifier)
    assert seen == [[1], [1, 4]]


def test_every_selected_atom_gets_a_ring(view):
    """The ring condition goes through _selection_set, so all members are
    highlighted -- not just the primary."""
    _click(view, 0)
    _click(view, 3, Qt.ShiftModifier)
    assert view._selection_set() == {0, 3}


def test_a_new_structure_drops_the_old_selection(view):
    _click(view, 0)
    _click(view, 3, Qt.ShiftModifier)
    view.set_structure(_structure(4))
    assert view.selection() == []


def test_selection_survives_frame_playback(view):
    """show_frame with the same atom count moves sites without rebuilding,
    so a selection made mid-playback must not vanish on the next frame."""
    _click(view, 2)
    frame = _structure()
    view.show_frame(frame, 5)
    assert view.selection() == [2]
