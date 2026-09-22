"""A compact diverging colour scale for spin-projected bands.

Painted directly rather than borrowed from matplotlib: the band plot's own
margins are pixel-managed by hand (see ``plotting.PixelMargins``) so the two
panels line up exactly, and inserting a real matplotlib colorbar axes would
fight that alignment system. This sits beside the plot instead, the same way
``PlotLegend`` sits in its header.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..theme import PALETTE, TYPE

# Matches projection.SPIN_CMAP's endpoints (matplotlib "bwr_r": low -> red,
# high -> blue) so the bar reads as the same scale the scatter points use.
SPIN_DOWN_COLOR = "#ff0000"
SPIN_UP_COLOR = "#0000ff"

BAR_WIDTH = 14
TITLE_HEIGHT = 16
LABEL_HEIGHT = 14


class DivergingColorbar(QWidget):
    """A narrow vertical red-white-blue gradient, fixed to a +1/2 (top) to
    -1/2 (bottom) scale — the physical range of a single electron's spin
    expectation value — with a title above it (e.g. "Spin (z)")."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label = "Spin"
        self.setMinimumWidth(46)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

    def set_label(self, label: str) -> None:
        self._label = label
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        top = TITLE_HEIGHT + LABEL_HEIGHT
        bar_left = (self.width() - BAR_WIDTH) / 2
        bar_rect = QRectF(
            bar_left, top, BAR_WIDTH, max(self.height() - top - LABEL_HEIGHT, 1)
        )

        gradient = QLinearGradient(0, bar_rect.top(), 0, bar_rect.bottom())
        gradient.setColorAt(0.0, QColor(SPIN_UP_COLOR))  # top: +1/2, spin up
        gradient.setColorAt(0.5, QColor("#ffffff"))
        gradient.setColorAt(1.0, QColor(SPIN_DOWN_COLOR))  # bottom: -1/2, spin down
        painter.fillRect(bar_rect, gradient)
        painter.setPen(QPen(QColor(PALETTE.text_secondary), 1))
        painter.drawRect(bar_rect)

        font = painter.font()
        font.setPointSizeF(max(TYPE.small - 2, 7))
        painter.setFont(font)
        painter.setPen(QColor(PALETTE.text_secondary))

        painter.drawText(
            QRectF(0, 0, self.width(), TITLE_HEIGHT),
            Qt.AlignHCenter | Qt.AlignVCenter,
            self._label,
        )
        painter.drawText(
            QRectF(0, TITLE_HEIGHT, self.width(), LABEL_HEIGHT),
            Qt.AlignHCenter | Qt.AlignVCenter,
            "+1/2",
        )
        painter.drawText(
            QRectF(0, bar_rect.bottom(), self.width(), LABEL_HEIGHT),
            Qt.AlignHCenter | Qt.AlignVCenter,
            "-1/2",
        )
        painter.end()
