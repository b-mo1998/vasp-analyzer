"""Plot legend drawn as a widget rather than inside the figure.

Keeping it out of the axes means it never steals plot area, it stays crisp at
any DPI, and it can sit in the card header exactly where the design puts it.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from ..theme import PALETTE, SPACING, TYPE


class LineSample(QWidget):
    """A short line drawn in a given colour and dash pattern."""

    def __init__(
        self, color: str, dashed: bool = False, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._color = color
        self._dashed = dashed
        self.setFixedSize(22, 12)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        return QSize(22, 12)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(self._color))
        pen.setWidthF(1.6)
        if self._dashed:
            pen.setDashPattern([3.0, 2.0])
        painter.setPen(pen)
        middle = self.height() / 2
        painter.drawLine(1, int(middle), self.width() - 1, int(middle))
        painter.end()


class DotSample(QWidget):
    """A small filled circle — used for element-colour legend entries, where
    a dot reads as "this colour" more directly than a line does."""

    def __init__(self, color: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._color = color
        self.setFixedSize(10, 10)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        return QSize(10, 10)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(0, 0, 0, 38), 1))
        painter.setBrush(QColor(self._color))
        painter.drawEllipse(1, 1, self.width() - 2, self.height() - 2)
        painter.end()


class PlotLegend(QWidget):
    """A row of line or dot samples with captions."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(SPACING.lg)
        self._entries: dict[str, QWidget] = {}

    def add_entry(self, key: str, text: str, color: str, dashed: bool = False) -> None:
        self._add(key, text, LineSample(color, dashed))

    def add_dot_entry(self, key: str, text: str, color: str) -> None:
        self._add(key, text, DotSample(color))

    def _add(self, key: str, text: str, sample: QWidget) -> None:
        self.remove_entry(key)
        label = QLabel(text)
        label.setStyleSheet(f"font-size: {TYPE.small}px; color: {PALETTE.text};")

        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING.sm)
        row.addWidget(sample, 0, Qt.AlignVCenter)
        row.addWidget(label, 0, Qt.AlignVCenter)

        self._layout.addWidget(holder)
        self._entries[key] = holder
        holder.setProperty("legend_key", key)

    def set_entry_visible(self, key: str, visible: bool) -> None:
        holder = self._entries.get(key)
        if holder is not None:
            holder.setVisible(visible)

    def remove_entry(self, key: str) -> None:
        holder = self._entries.pop(key, None)
        if holder is not None:
            self._layout.removeWidget(holder)
            holder.setParent(None)
            holder.deleteLater()

    def remove_entries_with_prefix(self, prefix: str) -> None:
        for key in [k for k in self._entries if k.startswith(prefix)]:
            self.remove_entry(key)
