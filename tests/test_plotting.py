"""PlotCanvas's preferred size must stay fixed regardless of how large the
canvas is actually drawn -- matplotlib's FigureCanvasQTAgg otherwise reports
sizeHint() from the figure's *current* pixel size, which only ever grows once
the canvas has been stretched by a pop-out window, so the card it lives in
keeps demanding more room on every trip through the pop-out."""

from __future__ import annotations

from vasp_analyzer.ui.plotting import PlotCanvas


def test_size_hint_is_fixed_regardless_of_current_widget_size(qtbot):
    canvas = PlotCanvas()
    qtbot.addWidget(canvas)
    original = canvas.sizeHint()

    canvas.resize(900, 700)
    canvas.figure.set_size_inches(9.0, 7.0, forward=True)

    assert canvas.sizeHint() == original


def test_size_hint_survives_a_simulated_popout_resize(qtbot):
    """Reproduces what widgets/popout.py's pop-out window does: resize the
    canvas much larger, as FigureCanvasQTAgg.resizeEvent naturally would."""
    canvas = PlotCanvas()
    qtbot.addWidget(canvas)
    original = canvas.sizeHint()

    canvas.resize(760, 580)
    qtbot.wait(20)

    assert canvas.sizeHint() == original


def test_minimum_size_hint_matches_the_explicit_minimum(qtbot):
    canvas = PlotCanvas()
    qtbot.addWidget(canvas)
    assert canvas.minimumSizeHint() == canvas.minimumSize()
