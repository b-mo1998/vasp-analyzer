"""Coverage for the wheel guard, the uniform colouring mode and the Editor's
frozen-camera drag."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QAbstractScrollArea, QComboBox

from vasp_analyzer.models.structure import StructureData
from vasp_analyzer.ui.pages.structure.viewer import UNIFORM_COLOR, StructureView, ViewOptions


def _cell() -> StructureData:
    return StructureData(
        lattice=np.eye(3) * 5.43,
        species=["Si", "C"],
        frac_coords=np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]),
    )


def test_uniform_colouring_paints_every_species_the_same(qtbot):
    view = StructureView()
    qtbot.addWidget(view)
    view.set_structure(_cell())

    by_element = {s: view.element_color(s).name() for s in ("Si", "C")}
    assert by_element["Si"] != by_element["C"], "the default keeps species apart"

    view.set_options(ViewOptions(coloring="Uniform"))
    uniform = {s: view.element_color(s).name() for s in ("Si", "C")}
    assert uniform["Si"] == uniform["C"] == UNIFORM_COLOR


def test_wheel_over_a_combo_box_scrolls_the_panel_instead(qtbot):
    """The complaint this answers: scrolling down a sidebar silently changed
    whichever dropdown the pointer crossed, and the panel stayed put."""
    from PySide6.QtWidgets import QApplication, QScrollArea, QVBoxLayout, QWidget

    from vasp_analyzer.ui.widgets.scroll_guard import ScrollGuard

    guard = ScrollGuard()
    QApplication.instance().installEventFilter(guard)
    try:
        area = QScrollArea()
        qtbot.addWidget(area)
        area.setWidgetResizable(True)
        content = QWidget()
        column = QVBoxLayout(content)
        box = QComboBox()
        box.addItems([f"item {i}" for i in range(10)])
        column.addWidget(box)
        for index in range(40):
            column.addWidget(QComboBox())
        area.setWidget(content)
        area.resize(220, 160)
        area.show()
        qtbot.waitExposed(area)

        assert area.verticalScrollBar().maximum() > 0
        before_index = box.currentIndex()
        before_scroll = area.verticalScrollBar().value()

        position = QPointF(box.width() / 2, box.height() / 2)
        event = QWheelEvent(
            position,
            box.mapToGlobal(position.toPoint()),
            QPoint(0, 0),
            QPoint(0, -120),
            Qt.NoButton,
            Qt.NoModifier,
            Qt.NoScrollPhase,
            False,
        )
        QApplication.sendEvent(box, event)

        assert box.currentIndex() == before_index, "the dropdown must not change"
        assert area.verticalScrollBar().value() > before_scroll, "the panel must scroll"
    finally:
        QApplication.instance().removeEventFilter(guard)


def test_scroll_bars_keep_their_own_wheel_behaviour(qtbot):
    """A scroll bar is a QAbstractSlider too. Guarding it would hand its own
    event back to the area that produced it, which is an infinite loop, not a
    fix."""
    from PySide6.QtWidgets import QScrollBar

    from vasp_analyzer.ui.widgets.scroll_guard import EXEMPT, GUARDED

    assert issubclass(QScrollBar, GUARDED)
    assert issubclass(QScrollBar, EXEMPT)


def test_editor_drag_does_not_move_the_camera(qtbot, tmp_path):
    """The reported symptom was that moving an atom snapped the view back to
    the default orientation on every mouse-move. The cause was the refresh
    path reframing the camera; a drag has to happen against a frozen view."""
    from vasp_analyzer.models.editable import EditableStructure
    from vasp_analyzer.ui.pages.editor.viewer import EditorView

    view = EditorView()
    qtbot.addWidget(view)
    view.resize(600, 480)
    view.show()
    qtbot.waitExposed(view)

    working = EditableStructure.from_structure(_cell())
    view.set_structure(working.to_structure())
    view.set_mode(EditorView.MODE_MOVE)
    view.set_selection({0})
    view.camera().orbit(45.0, 30.0)
    view.camera().scale(1.7)
    qtbot.wait(10)

    rotation = view.camera().rotation.copy()
    zoom = view.camera().zoom
    centre = view.camera().centre.copy()

    deltas: list[np.ndarray] = []
    view.dragMoved.connect(deltas.append)

    geometry = view._gizmo_handles()
    assert geometry is not None
    origin, handles = geometry
    grab = QPointF((origin.x() + handles[0].tip.x()) / 2, (origin.y() + handles[0].tip.y()) / 2)

    def send(kind, point, buttons=Qt.LeftButton):
        event = QMouseEvent(
            kind, point, view.mapToGlobal(point.toPoint()),
            Qt.LeftButton, buttons, Qt.NoModifier,
        )
        view.event(event)

    send(QEvent.MouseButtonPress, grab)
    for step in range(1, 5):
        # Each move feeds the *page* a new structure in the real application;
        # here the view is driven directly, which is what isolates the camera.
        send(QEvent.MouseMove, QPointF(grab.x() + 12 * step, grab.y()))
    send(QEvent.MouseButtonRelease, QPointF(grab.x() + 48, grab.y()), Qt.NoButton)

    assert np.allclose(view.camera().rotation, rotation), "orientation moved"
    assert view.camera().zoom == pytest.approx(zoom), "zoom moved"
    assert np.allclose(view.camera().centre, centre), "framing moved"

    # Equal pointer steps must give equal displacement steps: the original bug
    # measured motion against a reference frame that was itself moving, so the
    # atom accelerated away across the cell.
    lengths = [float(np.linalg.norm(d)) for d in deltas]
    assert len(lengths) == 4
    increments = np.diff(lengths)
    assert np.allclose(increments, increments[0], rtol=1e-6)


# -- move gizmo: drag sensitivity stays bounded under any camera --------------


def _handle_sensitivities(view):
    """Sensitivity of every handle currently on the gizmo, plus the
    well-behaved baseline (1 / pixels-per-Angstrom at the gizmo's origin)
    those sensitivities are measured against."""
    geometry = view._gizmo_handles()
    assert geometry is not None
    origin, handles = geometry
    _screen, _depth, origin_scale = view.camera().project(
        view._gizmo_origin[None, :], view.width(), view.height()
    )
    baseline = 1.0 / max(float(origin_scale[0]), 1e-9)
    return handles, baseline


def test_gizmo_sensitivity_stays_bounded_across_many_camera_orientations(qtbot):
    """The bug report: the gizmo "broke" under some perspectives and over
    large distances. Root cause was that a handle's drag sensitivity (Å moved
    per pixel dragged) could be inferred from a one-shot linear extrapolation
    that assumed screen displacement scales with 3D arm length — an
    assumption that fails badly for a handle pointing toward the camera,
    where screen position barely changes however far the arm extends. That
    let sensitivity run away to thousands of times its normal value, so a
    small mouse move produced a huge, unpredictable jump.

    Swept here across a dense grid of orbit angles and zoom levels on a real
    (if small) structure: sensitivity must stay finite, non-negative, and
    within a bounded multiple of the well-behaved baseline, for every handle,
    everywhere on the sweep — not just the couple of orientations a
    hand-picked test would think to check.
    """
    from vasp_analyzer.models.structure import StructureData
    from vasp_analyzer.ui.pages.editor.viewer import EditorView, MAX_SENSITIVITY_MULTIPLE

    structure = StructureData(
        lattice=np.array([[2.47, 0.0, 0.0], [-1.235, 2.139, 0.0], [0.0, 0.0, 14.0]]),
        species=["C", "C"],
        frac_coords=np.array([[0.0, 0.0, 0.5], [1 / 3, 2 / 3, 0.5]]),
    )
    view = EditorView()
    qtbot.addWidget(view)
    view.resize(700, 560)
    view.set_structure(structure)
    view.set_mode(EditorView.MODE_MOVE)
    view.set_selection({0})
    cam = view.camera()

    checked = 0
    for zoom in (0.2, 1.0, 6.0):
        for yaw in range(0, 360, 30):
            for pitch in (-70, -20, 20, 70):
                cam.reset()
                cam.orbit(yaw, pitch)
                cam.zoom = zoom
                view.refresh_gizmo()
                handles, baseline = _handle_sensitivities(view)
                for handle in handles:
                    checked += 1
                    assert np.isfinite(handle.sensitivity)
                    assert handle.sensitivity >= 0.0
                    assert handle.sensitivity <= MAX_SENSITIVITY_MULTIPLE * baseline + 1e-9

    assert checked > 100, "the sweep should have exercised a substantial number of handles"


def test_gizmo_handle_pointing_at_the_camera_is_capped_not_wild(qtbot):
    """A handle aligned with the view axis is the specific degenerate case
    that broke before: screen position is nearly independent of arm length
    there, so extrapolating arm length from a target screen size blew up
    without bound. Oriented straight down an axis, the axis handle pointing
    at the camera must respond to a drag with a bounded, sane displacement —
    not the multi-hundred-Angstrom jump the bug produced."""
    from vasp_analyzer.models.structure import StructureData
    from vasp_analyzer.ui.pages.editor.viewer import EditorView

    structure = StructureData(
        lattice=np.eye(3) * 5.43,
        species=["Si", "Si"],
        frac_coords=np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]),
    )
    view = EditorView()
    qtbot.addWidget(view)
    view.resize(700, 560)
    view.set_structure(structure)
    view.set_mode(EditorView.MODE_MOVE)
    view.set_selection({0})

    cam = view.camera()
    cam.orient("c")  # the z axis handle now points straight at the camera
    cam.zoom = 1.0
    view.refresh_gizmo()
    origin, handles = view._gizmo_handles()

    view_dir = cam.rotation[2]
    end_on = max(
        (h for h in handles if h.kind == "axis"),
        key=lambda h: abs(float(np.dot(h.direction, view_dir))),
    )
    assert abs(float(np.dot(end_on.direction, view_dir))) == pytest.approx(1.0, abs=1e-6)

    idx = handles.index(end_on)
    view._drag_target_rows = None
    view._drag_axis = idx
    view._drag_start = QPointF(origin.x(), origin.y())
    view._drag_delta = np.zeros(3)
    view._drag_geometry = (origin, handles)
    view._update_drag(QPointF(origin.x() + 50, origin.y()))

    delta = float(np.linalg.norm(view._drag_delta))
    assert np.isfinite(delta)
    # Silicon's lattice parameter is ~5.4 Å; a 50px drag landing within a
    # generous multiple of it is "responsive," not "flew off across the
    # cell" — the original bug produced displacements of hundreds of
    # Angstroms from exactly this configuration.
    assert delta < 30.0
