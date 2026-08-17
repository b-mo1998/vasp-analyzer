"""An animated switch, painted rather than styled.

Qt has no native switch widget and QCheckBox cannot be coerced into one that
looks right, so this is drawn directly. It follows the macOS switch: a rounded
track that fills with the accent colour and a white knob that slides.
"""

from __future__ import annotations

from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QPropertyAnimation,
    QRectF,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor, QPainter, QPainterPath
from PySide6.QtWidgets import QAbstractButton, QWidget

from ..theme import PALETTE

TRACK_WIDTH = 40
TRACK_HEIGHT = 23
KNOB_MARGIN = 2


class ToggleSwitch(QAbstractButton):
    """A checkable switch. Emits :attr:`toggled` like any Qt button."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(TRACK_WIDTH, TRACK_HEIGHT)
        self._offset = 0.0
        self._animation = QPropertyAnimation(self, b"offset", self)
        self._animation.setDuration(140)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        self.toggled.connect(self._animate)

    # -- animated property -------------------------------------------------

    def _get_offset(self) -> float:
        return self._offset

    def _set_offset(self, value: float) -> None:
        self._offset = float(value)
        self.update()

    offset = Property(float, _get_offset, _set_offset)

    def _animate(self, checked: bool) -> None:
        self._animation.stop()
        self._animation.setStartValue(self._offset)
        self._animation.setEndValue(1.0 if checked else 0.0)
        self._animation.start()

    # -- painting ----------------------------------------------------------

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        return QSize(TRACK_WIDTH, TRACK_HEIGHT)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        enabled = self.isEnabled()
        off_color = QColor(PALETTE.border if enabled else PALETTE.surface_sunken)
        on_color = QColor(PALETTE.accent if enabled else PALETTE.border_strong)
        track = QColor(
            int(off_color.red() + (on_color.red() - off_color.red()) * self._offset),
            int(off_color.green() + (on_color.green() - off_color.green()) * self._offset),
            int(off_color.blue() + (on_color.blue() - off_color.blue()) * self._offset),
        )

        radius = TRACK_HEIGHT / 2
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, TRACK_WIDTH, TRACK_HEIGHT), radius, radius)
        painter.fillPath(path, track)

        if not enabled:
            painter.setPen(QColor(PALETTE.border))
            painter.drawPath(path)

        knob_size = TRACK_HEIGHT - 2 * KNOB_MARGIN
        travel = TRACK_WIDTH - knob_size - 2 * KNOB_MARGIN
        x = KNOB_MARGIN + travel * self._offset
        knob = QRectF(x, KNOB_MARGIN, knob_size, knob_size)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 26))
        painter.drawEllipse(knob.translated(0, 0.6))
        painter.setBrush(QColor("#ffffff" if enabled else PALETTE.surface))
        painter.drawEllipse(knob)
        painter.end()


class LabelledToggle(QWidget):
    """A caption plus a switch, e.g. ``DOS [switch]``."""

    toggled = Signal(bool)

    def __init__(
        self, text: str, checked: bool = False, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        from PySide6.QtWidgets import QHBoxLayout, QLabel

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self._label = QLabel(text)
        layout.addWidget(self._label)
        self._switch = ToggleSwitch()
        self._switch.setChecked(checked)
        self._switch.toggled.connect(self.toggled)
        layout.addWidget(self._switch)

    def isChecked(self) -> bool:  # noqa: N802 - Qt naming
        return self._switch.isChecked()

    def setChecked(self, checked: bool) -> None:  # noqa: N802 - Qt naming
        self._switch.setChecked(checked)

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802 - Qt naming
        super().setEnabled(enabled)
        self._switch.setEnabled(enabled)
        self._label.setStyleSheet(
            "" if enabled else f"color: {PALETTE.disabled_text};"
        )

    def setToolTip(self, text: str) -> None:  # noqa: N802 - Qt naming
        super().setToolTip(text)
        self._switch.setToolTip(text)
        self._label.setToolTip(text)
