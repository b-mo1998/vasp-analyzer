"""A minimal, custom-painted slider.

QSlider's stylesheet subcontrols (``::groove``, ``::add-page``, ``::sub-page``)
turn out to silently fall back to native Fusion painting once the slider sits
inside a QScrollArea — and every sidebar in this app is one. Rather than fight
that Qt/style quirk, the slider is painted directly: a grey track for the full
range, an accent-coloured fill from the left edge to the knob, and the knob
itself — identically wherever it's placed.

The full-width track matters for legibility, not decoration: without it a
slider sitting at or near its minimum is just a dot floating in whitespace,
with nothing to show how far it can travel or that it is a slider at all.

Values are kept as floats throughout. QSlider only understands integers, so
the previous implementation converted a float range to slider steps and back;
that conversion is gone along with QSlider, which also removes a source of
rounding drift between the displayed value and the slider position.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter
from PySide6.QtWidgets import QWidget

from ..theme import PALETTE

TRACK_HEIGHT = 4.0
KNOB_DIAMETER = 15.0
WIDGET_HEIGHT = 22
HOVER_RADIUS = KNOB_DIAMETER  # generous hit target around the knob


class PaintedSlider(QWidget):
    """A flat horizontal slider: an accent fill and a knob, nothing behind it."""

    valueChanged = Signal(float)

    def __init__(
        self,
        minimum: float = 0.0,
        maximum: float = 1.0,
        value: float = 0.0,
        single_step: float | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._min = float(minimum)
        self._max = float(maximum)
        self._value = self._clamp(value)
        # Dragging is continuous (pixel-limited, not step-quantised) — a
        # genuine improvement over the QSlider it replaces. single_step only
        # sets the increment for arrow keys and the scroll wheel.
        self._single_step = float(single_step) if single_step else (maximum - minimum) / 100.0
        self._dragging = False
        self._hovered = False

        self.setFixedHeight(WIDGET_HEIGHT)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)

    # -- public API (mirrors the QSlider surface SliderRow relies on) -------

    def value(self) -> float:
        return self._value

    def setValue(self, value: float, notify: bool = True) -> None:
        clamped = self._clamp(value)
        if abs(clamped - self._value) < 1e-12:
            return
        self._value = clamped
        self.update()
        if notify:
            self.valueChanged.emit(self._value)

    def setRange(self, minimum: float, maximum: float) -> None:
        self._min, self._max = float(minimum), float(maximum)
        self.setValue(self._value, notify=False)
        self.update()

    def minimum(self) -> float:
        return self._min

    def maximum(self) -> float:
        return self._max

    def setSingleStep(self, step: float) -> None:  # noqa: N802 - Qt naming
        self._single_step = float(step)

    def _clamp(self, value: float) -> float:
        return max(self._min, min(self._max, float(value)))

    # -- geometry -------------------------------------------------------------

    def _track_rect(self) -> QRectF:
        margin = KNOB_DIAMETER / 2
        y = self.height() / 2 - TRACK_HEIGHT / 2
        width = max(0.0, self.width() - 2 * margin)
        return QRectF(margin, y, width, TRACK_HEIGHT)

    def _fraction(self) -> float:
        span = self._max - self._min
        return 0.0 if span <= 0 else (self._value - self._min) / span

    def _knob_center(self) -> QPointF:
        track = self._track_rect()
        return QPointF(track.left() + track.width() * self._fraction(), self.height() / 2)

    def _value_at_x(self, x: float) -> float:
        track = self._track_rect()
        if track.width() <= 0:
            return self._min
        fraction = max(0.0, min(1.0, (x - track.left()) / track.width()))
        return self._min + fraction * (self._max - self._min)

    # -- painting ---------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        enabled = self.isEnabled()
        track = self._track_rect()
        knob = self._knob_center()

        # The unfilled track first, so the accent fill paints over it.
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(PALETTE.border if enabled else PALETTE.surface_sunken))
        painter.drawRoundedRect(track, track.height() / 2, track.height() / 2)

        fill_width = max(0.0, knob.x() - track.left())
        if fill_width > 0:
            fill_color = QColor(PALETTE.accent if enabled else PALETTE.border_strong)
            fill_rect = QRectF(track.left(), track.top(), fill_width, track.height())
            painter.setPen(Qt.NoPen)
            painter.setBrush(fill_color)
            painter.drawRoundedRect(fill_rect, track.height() / 2, track.height() / 2)

        active = enabled and (self._hovered or self._dragging)
        knob_border = PALETTE.accent if active else PALETTE.border_strong
        painter.setPen(QColor(knob_border if enabled else PALETTE.border))
        painter.setBrush(QColor(PALETTE.surface if enabled else PALETTE.surface_sunken))
        painter.drawEllipse(knob, KNOB_DIAMETER / 2, KNOB_DIAMETER / 2)
        painter.end()

    # -- interaction --------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if not self.isEnabled() or event.button() != Qt.LeftButton:
            return
        self.setFocus(Qt.MouseFocusReason)
        self._dragging = True
        self.setValue(self._value_at_x(event.position().x()))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._dragging:
            self.setValue(self._value_at_x(event.position().x()))
            return
        hovered = (event.position() - self._knob_center()).manhattanLength() <= HOVER_RADIUS
        if hovered != self._hovered:
            self._hovered = hovered
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._dragging = False
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._hovered = False
        self.update()

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if not self.isEnabled():
            return
        direction = 1 if event.angleDelta().y() > 0 else -1
        self.setValue(self._value + direction * self._single_step)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if not self.isEnabled():
            super().keyPressEvent(event)
            return
        step = self._single_step
        key = event.key()
        if key in (Qt.Key_Left, Qt.Key_Down):
            self.setValue(self._value - step)
        elif key in (Qt.Key_Right, Qt.Key_Up):
            self.setValue(self._value + step)
        elif key == Qt.Key_Home:
            self.setValue(self._min)
        elif key == Qt.Key_End:
            self.setValue(self._max)
        else:
            super().keyPressEvent(event)

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.type() == QEvent.EnabledChange:
            self.update()
        super().changeEvent(event)
