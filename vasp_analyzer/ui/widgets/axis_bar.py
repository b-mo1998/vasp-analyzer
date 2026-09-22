"""The viewer's orientation bar: a, b, c, a*, b*, c* and a default view.

Modelled on the orientation strip crystallography tools put above the render
window, because that is the vocabulary the people using this application
already have: "look down c" is a question about the crystal, and answering it
by dragging until the cell happens to line up is both slow and imprecise.

The buttons are plain text apart from the last one, whose cube is painted
rather than set as a glyph — a polyhedron character is not present in every
system font, and a missing glyph renders as a tofu box that reads as a bug.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from ..theme import METRICS, PALETTE, RADIUS, TYPE

# (key passed to Camera.orient, button caption, tooltip)
ORIENTATIONS = [
    ("a", "a", "View down the a axis"),
    ("b", "b", "View down the b axis"),
    ("c", "c", "View down the c axis"),
    ("a*", "a*", "View down a* — the normal to the b-c plane"),
    ("b*", "b*", "View down b* — the normal to the c-a plane"),
    ("c*", "c*", "View down c* — the normal to the a-b plane"),
]

BUTTON_WIDTH = 30
# Pill padding above and below the segments.
BAR_PADDING = 3
# Segments fill the pill so that the pill is exactly as tall as the tool
# buttons and pop-out button it shares the top of the viewer with.
BUTTON_HEIGHT = METRICS.viewer_control_height - 2 * BAR_PADDING


class _CubeButton(QPushButton):
    """The default-view button, drawn as a wireframe cell."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(BUTTON_WIDTH, BUTTON_HEIGHT)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Restore the default view")

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # A cell drawn in the same one-point-perspective the default view
        # produces, so the button previews the orientation it restores.
        centre = QPointF(self.width() / 2.0, self.height() / 2.0)
        front, back, offset = 5.4, 3.2, 2.6
        front_corners = [
            centre + QPointF(-front, -front + 0.8),
            centre + QPointF(front, -front + 0.8),
            centre + QPointF(front, front + 0.8),
            centre + QPointF(-front, front + 0.8),
        ]
        back_corners = [
            centre + QPointF(-back + offset, -back - offset + 0.8),
            centre + QPointF(back + offset, -back - offset + 0.8),
            centre + QPointF(back + offset, back - offset + 0.8),
            centre + QPointF(-back + offset, back - offset + 0.8),
        ]

        colour = QColor(PALETTE.text if self.isEnabled() else PALETTE.disabled_text)
        painter.setPen(QPen(colour, 1.0))
        for corners in (back_corners, front_corners):
            for index in range(4):
                painter.drawLine(corners[index], corners[(index + 1) % 4])
        painter.setPen(QPen(colour, 0.8))
        for near, far in zip(front_corners, back_corners):
            painter.drawLine(near, far)
        painter.end()


class AxisBar(QWidget):
    """A row of orientation presets. Emits the key ``Camera.orient`` wants."""

    orientationRequested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("AxisBar")
        # Without this a plain QWidget skips its stylesheet background, and
        # the bar has no surface at all where it floats over a dark viewer.
        self.setAttribute(Qt.WA_StyledBackground, True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(BAR_PADDING, BAR_PADDING, BAR_PADDING, BAR_PADDING)
        layout.setSpacing(1)

        self._buttons: list[QPushButton] = []
        for key, caption, tip in ORIENTATIONS:
            button = QPushButton(caption)
            button.setObjectName("AxisButton")
            button.setFixedSize(BUTTON_WIDTH, BUTTON_HEIGHT)
            button.setCursor(Qt.PointingHandCursor)
            button.setToolTip(tip)
            button.clicked.connect(lambda _c=False, k=key: self.orientationRequested.emit(k))
            layout.addWidget(button)
            self._buttons.append(button)

        self._default = _CubeButton()
        self._default.setObjectName("AxisButton")
        self._default.clicked.connect(lambda: self.orientationRequested.emit("default"))
        layout.addWidget(self._default)
        self._buttons.append(self._default)

        self.setStyleSheet(
            f"#AxisBar {{ background: {PALETTE.surface};"
            f" border: 1px solid {PALETTE.border};"
            f" border-radius: {RADIUS.md}px; }}"
            f"#AxisButton {{ background: transparent; border: none;"
            f" border-radius: {RADIUS.sm}px; color: {PALETTE.text};"
            f" font-size: {TYPE.body}px; font-weight: 600; }}"
            f"#AxisButton:hover:enabled {{ background: {PALETTE.surface_hover}; }}"
            f"#AxisButton:pressed:enabled {{ background: {PALETTE.border}; }}"
            f"#AxisButton:disabled {{ color: {PALETTE.disabled_text}; }}"
        )

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802 - Qt naming
        super().setEnabled(enabled)
        for button in self._buttons:
            button.setEnabled(enabled)


__all__ = ["AxisBar"]
