"""The pop-out button: a plot or 3D viewer must move into its own window and
back into exactly where it started, unchanged and still the same live widget."""

from __future__ import annotations

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from vasp_analyzer.ui.plotting import PlotCanvas
from vasp_analyzer.ui.widgets.popout import PopoutButton, _PlotPopoutToolbar


def test_pop_out_moves_the_target_into_its_own_top_level_window(qtbot):
    host = QWidget()
    layout = QVBoxLayout(host)
    before = QLabel("before")
    target = QLabel("target")
    after = QLabel("after")
    layout.addWidget(before)
    layout.addWidget(target)
    layout.addWidget(after)
    qtbot.addWidget(host)

    button = PopoutButton(target, "Test Plot")
    qtbot.addWidget(button)

    assert target.parentWidget() is host
    button._toggle()

    assert button.is_popped_out()
    assert target.parentWidget() is not host
    assert target.window() is not host
    assert target.text() == "target", "the same widget, not a copy"


def test_pop_in_restores_the_original_position_in_the_layout(qtbot):
    host = QWidget()
    layout = QVBoxLayout(host)
    before = QLabel("before")
    target = QLabel("target")
    after = QLabel("after")
    layout.addWidget(before)
    layout.addWidget(target)
    layout.addWidget(after)
    qtbot.addWidget(host)

    button = PopoutButton(target, "Test Plot")
    qtbot.addWidget(button)

    button._toggle()  # out
    button._toggle()  # back in

    assert not button.is_popped_out()
    assert target.parentWidget() is host
    assert layout.indexOf(target) == 1, "back at its original position, not appended"


def test_closing_the_popout_window_also_restores_the_target(qtbot):
    """The button is one way back in; the window's own close control is the
    other, and both must do the same thing."""
    host = QWidget()
    layout = QVBoxLayout(host)
    target = QLabel("target")
    layout.addWidget(target)
    qtbot.addWidget(host)

    button = PopoutButton(target, "Test Plot")
    qtbot.addWidget(button)

    button._toggle()
    assert button.is_popped_out()

    button._host.close()

    assert not button.is_popped_out()
    assert target.parentWidget() is host


def test_pop_in_finds_a_widget_added_via_a_nested_layout(qtbot):
    """Regression: Charge's Sectional Slices card builds a QHBoxLayout on its
    own (``row = QHBoxLayout()``) and installs it into the card's own layout
    via ``addLayout`` rather than giving it its own widget — so the plot's
    ``parentWidget()`` is the *card*, several levels above the layout that
    actually manages it. The naive ``parentWidget().layout()`` lookup landed
    on the card's outer layout instead, and popping back in appended the
    plot there — leaving a gap in the row it used to share with the other
    controls, with the plot itself full width beneath everything else. This
    is the same shape, deliberately reproduced without needing the Charge
    page itself."""
    card = QWidget()
    card_layout = QVBoxLayout(card)
    label = QLabel("Sectional Slices")
    card_layout.addWidget(label)

    row = QHBoxLayout()  # deliberately not owned by its own widget
    controls = QLabel("controls")
    target = QLabel("plot")
    row.addWidget(controls, 0)
    row.addWidget(target, 1)
    card_layout.addLayout(row)

    qtbot.addWidget(card)
    button = PopoutButton(target, "Test Plot")
    qtbot.addWidget(button)

    assert target.parentWidget() is card, "confirms the nested-layout shape"

    button._toggle()  # out
    assert row.count() == 1, "the plot must have left the row it shared"

    button._toggle()  # back in
    assert row.count() == 2, "and returned to that same row, not the card"
    assert row.indexOf(target) == 1, "at its original position within it"
    assert card_layout.indexOf(target) == -1, (
        "not appended directly onto the outer card layout"
    )
    assert row.stretch(1) == 1, "its stretch factor must survive the round trip"


def test_repeated_pop_cycles_leave_the_widget_tree_consistent(qtbot):
    host = QWidget()
    layout = QVBoxLayout(host)
    target = QLabel("target")
    layout.addWidget(target)
    qtbot.addWidget(host)

    button = PopoutButton(target, "Test Plot")
    qtbot.addWidget(button)

    for _ in range(4):
        button._toggle()
        button._toggle()

    assert not button.is_popped_out()
    assert target.parentWidget() is host
    assert layout.count() == 1


# --- Copy / Export on every pop-out ---------------------------------------------


def test_popout_window_carries_a_copy_and_export_toolbar(qtbot):
    host = QWidget()
    layout = QVBoxLayout(host)
    target = QLabel("target")
    layout.addWidget(target)
    qtbot.addWidget(host)

    button = PopoutButton(target, "Test Plot")
    qtbot.addWidget(button)
    button._toggle()

    toolbar = button._host.findChild(_PlotPopoutToolbar)
    assert toolbar is not None
    button._toggle()  # tidy up


def test_copy_puts_a_real_image_on_the_clipboard_for_a_plain_widget(qtbot):
    host = QWidget()
    layout = QVBoxLayout(host)
    target = QLabel("target")
    target.resize(50, 30)
    layout.addWidget(target)
    qtbot.addWidget(host)

    button = PopoutButton(target, "Test Plot")
    qtbot.addWidget(button)
    button._toggle()

    toolbar = button._host.findChild(_PlotPopoutToolbar)
    toolbar._copy()
    image = QGuiApplication.clipboard().image()
    assert not image.isNull()
    button._toggle()


def test_export_writes_a_png_for_a_plain_widget(qtbot, tmp_path, monkeypatch):
    host = QWidget()
    layout = QVBoxLayout(host)
    target = QLabel("target")
    target.resize(50, 30)
    layout.addWidget(target)
    qtbot.addWidget(host)

    button = PopoutButton(target, "Test Plot")
    qtbot.addWidget(button)
    button._toggle()

    toolbar = button._host.findChild(_PlotPopoutToolbar)
    out = tmp_path / "shot.png"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (str(out), "PNG image (*.png)")),
    )
    toolbar._export()

    assert out.exists()
    assert out.stat().st_size > 0
    button._toggle()


def test_export_of_a_plot_canvas_offers_vector_formats(qtbot, tmp_path, monkeypatch):
    """A matplotlib canvas should be exported through its own figure (real
    vector output), not a raster screen-grab, so PDF/SVG must be offered."""
    host = QWidget()
    layout = QVBoxLayout(host)
    target = PlotCanvas()
    layout.addWidget(target)
    qtbot.addWidget(host)

    button = PopoutButton(target, "Test Plot")
    qtbot.addWidget(button)
    button._toggle()

    toolbar = button._host.findChild(_PlotPopoutToolbar)
    seen_filters = []

    def fake_dialog(*args, **kwargs):
        seen_filters.append(args[3] if len(args) > 3 else kwargs.get("filter", ""))
        return str(tmp_path / "figure.pdf"), "PDF document (*.pdf)"

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(fake_dialog))
    toolbar._export()

    assert "SVG" in seen_filters[0] and "PDF" in seen_filters[0]
    assert (tmp_path / "figure.pdf").exists()
    button._toggle()


def test_the_popout_toolbar_sits_directly_above_the_plot(qtbot):
    """The Copy/Export strip must stay a button-height tall.

    A matplotlib canvas reports Preferred/Preferred rather than Expanding, so
    a plain QWidget toolbar above it was just as entitled to the window's
    spare height -- and took most of it. A 26px button row rendered 240px tall
    in a 720x540 pop-out, parking the icons a visible inch above the plot they
    act on, and leaving the plot itself squashed against the bottom edge.
    """
    host = QWidget()
    layout = QVBoxLayout(host)
    target = PlotCanvas()
    layout.addWidget(target)
    qtbot.addWidget(host)
    host.resize(500, 400)
    host.show()

    button = PopoutButton(target, "Test Plot")
    qtbot.addWidget(button)
    button._toggle()
    window = button._host
    window.resize(720, 540)
    window.layout().activate()

    toolbar = window.findChild(_PlotPopoutToolbar)
    assert toolbar.height() <= 48, "the toolbar is chrome, not a panel"
    # The plot starts where the toolbar ends, and fills the rest of the window.
    assert target.y() == toolbar.height()
    assert target.height() >= window.height() - toolbar.height() - 4

    button._toggle()


def test_the_popout_toolbar_does_not_grow_with_the_window(qtbot):
    host = QWidget()
    layout = QVBoxLayout(host)
    target = PlotCanvas()
    layout.addWidget(target)
    qtbot.addWidget(host)
    host.show()

    button = PopoutButton(target, "Test Plot")
    qtbot.addWidget(button)
    button._toggle()
    window = button._host
    toolbar = window.findChild(_PlotPopoutToolbar)

    window.resize(700, 500)
    window.layout().activate()
    small = toolbar.height()

    window.resize(1400, 1000)
    window.layout().activate()
    assert toolbar.height() == small

    button._toggle()
