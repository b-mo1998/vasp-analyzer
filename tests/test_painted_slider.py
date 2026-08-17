"""PaintedSlider: value/range behaviour, plus the QScrollArea regression it
exists to avoid.

QSlider's ``::groove``/``::add-page`` stylesheet subcontrols silently revert
to native Fusion painting once nested inside a QScrollArea — and every
sidebar in this app is one. That was invisible in isolated widget tests (a
bare QSlider outside a scroll area styles correctly) and only showed up
embedded in a real sidebar, which is what made it easy to miss. The rendering
test below reproduces the exact nesting rather than trusting the fix on paper.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from vasp_analyzer.ui.pages.bands.sidebar import BandsSidebar
from vasp_analyzer.ui.style import apply_theme
from vasp_analyzer.ui.theme import PALETTE
from vasp_analyzer.ui.widgets.painted_slider import PaintedSlider
from vasp_analyzer.ui.widgets.slider_row import SliderRow


def _luminance(color) -> float:
    return 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()


# --- value/range behaviour ---------------------------------------------------


def test_value_clamped_to_range(qapp):
    slider = PaintedSlider(0.0, 10.0, 5.0)
    slider.setValue(50.0)
    assert slider.value() == 10.0
    slider.setValue(-50.0)
    assert slider.value() == 0.0


def test_set_value_emits_by_default_but_can_be_silent(qtbot):
    slider = PaintedSlider(0.0, 10.0, 0.0)
    with qtbot.waitSignal(slider.valueChanged, timeout=1000):
        slider.setValue(5.0)

    received = []
    slider.valueChanged.connect(received.append)
    slider.setValue(7.0, notify=False)
    assert received == []
    assert slider.value() == 7.0


def test_set_range_reclamps_the_current_value(qapp):
    slider = PaintedSlider(0.0, 10.0, 9.0)
    slider.setRange(0.0, 5.0)
    assert slider.value() == 5.0


def test_click_at_x_sets_a_proportional_value(qtbot):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtCore import QPointF

    slider = PaintedSlider(0.0, 100.0, 0.0)
    slider.resize(200, 22)
    qtbot.addWidget(slider)

    # Click at the horizontal centre of the track.
    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(100, 11),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    slider.mousePressEvent(event)
    assert 40.0 < slider.value() < 60.0  # near the middle, not exact due to margins


def test_disabled_slider_ignores_clicks(qtbot):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    slider = PaintedSlider(0.0, 100.0, 0.0)
    slider.resize(200, 22)
    slider.setEnabled(False)
    qtbot.addWidget(slider)

    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(180, 11),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    slider.mousePressEvent(event)
    assert slider.value() == 0.0


# --- SliderRow integration ----------------------------------------------------


def test_slider_row_keeps_field_and_slider_in_sync(qapp):
    row = SliderRow("Fermi Level", -5.0, 5.0, 0.0, 0.01, "eV", decimals=3)
    row.setValue(2.5)
    assert row.value() == 2.5
    assert row.value() == row._slider.value()
    assert "2.500" in row._field.text()


def test_slider_row_set_range_rescales_slider_bounds(qapp):
    row = SliderRow("Y-Axis Min", -20.0, 0.0, -10.0, 0.1, "eV")
    row.setRange(-30.0, -10.0)
    assert row._slider.minimum() == -30.0
    assert row._slider.maximum() == -10.0


# --- the QScrollArea regression ------------------------------------------------


def test_slider_in_a_scroll_area_still_shows_no_track(qapp, qtbot):
    """A bare-QSlider version of this exact test used to pass while the real
    app, which always nests sliders inside a QScrollArea, showed a solid gray
    track. Reproduce the real nesting, not an easier case."""
    apply_theme(qapp)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    content = QWidget()
    layout = QVBoxLayout(content)
    layout.setContentsMargins(20, 20, 20, 20)
    row = SliderRow("Y-Axis Min", -20.0, 0.0, -10.0, 0.1, "eV")
    layout.addWidget(row)
    scroll.setWidget(content)
    scroll.resize(300, 150)
    qtbot.addWidget(scroll)
    scroll.show()
    qtbot.waitExposed(scroll)

    image = scroll.grab().toImage()
    # Sample past the knob, where the old QSlider left a solid grey bar.
    sampled = image.pixelColor(250, 75)
    assert _luminance(sampled) > 240, (
        f"slider track is visible past the knob ({sampled.name()}) inside a "
        "QScrollArea — the native-QSlider-in-a-scroll-area bug has regressed"
    )


def test_real_bands_sidebar_slider_track_is_invisible(qapp, qtbot):
    apply_theme(qapp)
    sidebar = BandsSidebar()
    sidebar.resize(280, 500)
    qtbot.addWidget(sidebar)
    sidebar.show()
    qtbot.waitExposed(sidebar)

    slider = sidebar.y_min._slider
    global_point = slider.mapTo(sidebar, slider.rect().bottomRight())
    sampled = sidebar.grab().toImage().pixelColor(
        global_point.x() - 5, global_point.y() - slider.height() // 2
    )
    assert _luminance(sampled) > 240, (
        f"Y-Axis Min slider shows a track past the knob ({sampled.name()})"
    )
