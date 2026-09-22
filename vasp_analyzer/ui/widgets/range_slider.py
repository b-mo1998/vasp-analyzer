"""A two-handled slider for choosing an axis window.

Setting a plot's limits needs a low *and* a high value, and giving each its own
single-handle slider costs two rows per axis and still lets the user drag the
minimum above the maximum. One control with two knobs says "this is a range",
fits on one row, and cannot express an inverted window because the handles
clamp against each other.

Painted directly rather than composed from QSlider, for the same reason as
:mod:`vasp_analyzer.ui.widgets.painted_slider`: stylesheet subcontrols fall
back to native painting inside a QScrollArea, and every sidebar here is one.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QAbstractButton, QHBoxLayout, QLabel, QWidget

from ..theme import PALETTE, RADIUS, SPACING, TYPE

TRACK_HEIGHT = 4.0
KNOB_DIAMETER = 14.0
WIDGET_HEIGHT = 22
# A range narrower than this fraction of the span is treated as degenerate:
# the plot would have nothing meaningful to show inside it.
MIN_SPAN_FRACTION = 0.01


class RangeSlider(QWidget):
    """A horizontal slider with independent low and high handles."""

    rangeChanged = Signal(float, float)

    def __init__(
        self,
        minimum: float = 0.0,
        maximum: float = 1.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._min = float(minimum)
        self._max = float(maximum)
        self._low = self._min
        self._high = self._max
        self._active: str | None = None       # "low" | "high"
        self._hovered: str | None = None

        self.setFixedHeight(WIDGET_HEIGHT)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)

    # -- values ------------------------------------------------------------

    def values(self) -> tuple[float, float]:
        return self._low, self._high

    def bounds(self) -> tuple[float, float]:
        return self._min, self._max

    def setBounds(  # noqa: N802 - Qt naming
        self, minimum: float, maximum: float, reset: bool = True
    ) -> None:
        self._min, self._max = float(minimum), float(maximum)
        if reset or self._low < self._min or self._high > self._max:
            self._low, self._high = self._min, self._max
        self.update()

    def setValues(self, low: float, high: float, notify: bool = True) -> None:  # noqa: N802
        low, high = min(low, high), max(low, high)
        self._low = max(self._min, min(self._max, float(low)))
        self._high = max(self._min, min(self._max, float(high)))
        self.update()
        if notify:
            self.rangeChanged.emit(self._low, self._high)

    def reset(self, notify: bool = True) -> None:
        self.setValues(self._min, self._max, notify=notify)

    def is_full_range(self) -> bool:
        span = self._max - self._min
        if span <= 0:
            return True
        return (self._low - self._min) / span < 1e-9 and (self._max - self._high) / span < 1e-9

    # -- geometry ----------------------------------------------------------

    def _track_rect(self) -> QRectF:
        margin = KNOB_DIAMETER / 2
        y = self.height() / 2 - TRACK_HEIGHT / 2
        return QRectF(margin, y, max(0.0, self.width() - 2 * margin), TRACK_HEIGHT)

    def _x_for(self, value: float) -> float:
        track = self._track_rect()
        span = self._max - self._min
        fraction = 0.0 if span <= 0 else (value - self._min) / span
        return track.left() + track.width() * fraction

    def _value_at(self, x: float) -> float:
        track = self._track_rect()
        if track.width() <= 0:
            return self._min
        fraction = max(0.0, min(1.0, (x - track.left()) / track.width()))
        return self._min + fraction * (self._max - self._min)

    def _handle_at(self, position: QPointF) -> str | None:
        for name, value in (("low", self._low), ("high", self._high)):
            centre = QPointF(self._x_for(value), self.height() / 2)
            if (position - centre).manhattanLength() <= KNOB_DIAMETER:
                return name
        return None

    # -- painting ----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        enabled = self.isEnabled()
        track = self._track_rect()

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(PALETTE.border if enabled else PALETTE.surface_sunken))
        painter.drawRoundedRect(track, track.height() / 2, track.height() / 2)

        low_x, high_x = self._x_for(self._low), self._x_for(self._high)
        if high_x > low_x:
            painter.setBrush(
                QColor(PALETTE.accent if enabled else PALETTE.border_strong)
            )
            painter.drawRoundedRect(
                QRectF(low_x, track.top(), high_x - low_x, track.height()),
                track.height() / 2,
                track.height() / 2,
            )

        for name, value in (("low", self._low), ("high", self._high)):
            centre = QPointF(self._x_for(value), self.height() / 2)
            active = enabled and self._active == name
            hovered = enabled and self._hovered == name
            border = PALETTE.accent if (active or hovered) else PALETTE.border_strong
            painter.setPen(QColor(border if enabled else PALETTE.border))
            painter.setBrush(
                QColor(PALETTE.surface if enabled else PALETTE.surface_sunken)
            )
            painter.drawEllipse(centre, KNOB_DIAMETER / 2, KNOB_DIAMETER / 2)
        painter.end()

    # -- interaction -------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if not self.isEnabled() or event.button() != Qt.LeftButton:
            return
        handle = self._handle_at(event.position())
        if handle is None:
            # Clicking the track moves whichever handle is nearer, so the bar
            # responds to a click anywhere rather than only on a 14px knob.
            value = self._value_at(event.position().x())
            handle = "low" if abs(value - self._low) <= abs(value - self._high) else "high"
        self._active = handle
        self._drag_to(event.position().x())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._active is not None:
            self._drag_to(event.position().x())
            return
        hovered = self._handle_at(event.position())
        if hovered != self._hovered:
            self._hovered = hovered
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._active = None
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._hovered = None
        self.update()

    def _drag_to(self, x: float) -> None:
        value = self._value_at(x)
        span = self._max - self._min
        gap = span * MIN_SPAN_FRACTION
        if self._active == "low":
            self._low = min(value, self._high - gap)
        else:
            self._high = max(value, self._low + gap)
        self.update()
        self.rangeChanged.emit(self._low, self._high)


class ResetButton(QAbstractButton):
    """A small circular-arrow icon, drawn rather than set from a glyph.

    The obvious choice — a QPushButton with a unicode "↺"-style character as
    its text — depends on the system font actually shipping that glyph. Where
    it doesn't, Qt draws its "notdef" box instead: a blank rounded rectangle
    that looks like a UI element with nothing in it, which is exactly the
    "weird empty box" this replaced. Painting the arrow ourselves has no such
    dependency — it renders identically on every machine.
    """

    def __init__(self, tooltip: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(20, 20)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(tooltip)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        enabled = self.isEnabled()
        hovered = enabled and self.underMouse()
        painter.setPen(QPen(QColor(PALETTE.border_strong if not hovered else PALETTE.accent), 1.0))
        painter.setBrush(QColor(PALETTE.surface_hover if hovered else PALETTE.surface))
        painter.drawRoundedRect(rect, RADIUS.sm, RADIUS.sm)

        colour = QColor(
            PALETTE.disabled_text if not enabled else (PALETTE.accent if hovered else PALETTE.text_secondary)
        )
        centre = QPointF(self.width() / 2, self.height() / 2)
        radius = 4.6
        # A ~270-degree arc with an arrowhead at the open end — "circle back
        # to the start" is a more legible reset symbol than a bare refresh
        # icon and reads correctly even at 20px.
        span_start, span_degrees = -40, 270
        path = QPainterPath()
        path.arcMoveTo(
            centre.x() - radius, centre.y() - radius, radius * 2, radius * 2, span_start
        )
        path.arcTo(
            centre.x() - radius, centre.y() - radius, radius * 2, radius * 2,
            span_start, span_degrees,
        )
        pen = QPen(colour, 1.4)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)

        # Arrowhead at the end of the arc, tangent to it.
        import math

        end_angle = math.radians(span_start + span_degrees)
        tip = centre + QPointF(math.cos(end_angle), -math.sin(end_angle)) * radius
        tangent = end_angle + math.pi / 2
        back = math.pi * 0.78
        left = tip + QPointF(math.cos(tangent + back), -math.sin(tangent + back)) * 3.4
        right = tip + QPointF(math.cos(tangent - back), -math.sin(tangent - back)) * 3.4
        painter.setPen(Qt.NoPen)
        painter.setBrush(colour)
        painter.drawPolygon([tip, left, right])
        painter.end()

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.update()
        super().leaveEvent(event)


class RangeRow(QWidget):
    """A :class:`RangeSlider` with a caption, a live readout and a reset.

    Axis-limit controls appear on more than one page, and a plot window the
    user has narrowed needs to say what it is showing — otherwise a zoomed
    axis is indistinguishable from data that simply stops there.
    """

    rangeChanged = Signal(float, float)

    def __init__(
        self, caption: str, unit: str = "", parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._unit = unit

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING.sm)

        label = QLabel(caption)
        label.setFixedWidth(52)
        label.setStyleSheet(
            f"font-size: {TYPE.small}px; color: {PALETTE.text_secondary};"
        )
        row.addWidget(label)

        self._slider = RangeSlider()
        self._slider.rangeChanged.connect(self._on_changed)
        row.addWidget(self._slider, 1)

        self._readout = QLabel("—")
        self._readout.setFixedWidth(118)
        self._readout.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._readout.setStyleSheet(
            f"font-size: {TYPE.small}px; color: {PALETTE.text_tertiary};"
        )
        row.addWidget(self._readout)

        reset = ResetButton("Reset to the full range")
        # Not a direct connection: QAbstractButton.clicked emits a `bool
        # checked` argument, which would land on RangeSlider.reset's own
        # `notify` parameter and silently pass notify=False — the slider
        # would then visually snap back but never emit rangeChanged, so nothing
        # downstream (the plot's view limits) would hear about the change.
        reset.clicked.connect(lambda: self._slider.reset())
        row.addWidget(reset)

    def _on_changed(self, low: float, high: float) -> None:
        suffix = f" {self._unit}" if self._unit else ""
        self._readout.setText(f"{low:.3g} → {high:.3g}{suffix}")
        self.rangeChanged.emit(low, high)

    def set_bounds(self, minimum: float, maximum: float) -> None:
        """Adopt a new data extent and show the whole of it."""
        self._slider.setBounds(minimum, maximum, reset=True)
        self._on_changed(*self._slider.values())

    def values(self) -> tuple[float, float]:
        return self._slider.values()

    def window(self) -> tuple[float, float] | None:
        """The chosen window, or None when it still spans the whole range —
        which callers pass straight through as "no limit"."""
        return None if self._slider.is_full_range() else self._slider.values()
