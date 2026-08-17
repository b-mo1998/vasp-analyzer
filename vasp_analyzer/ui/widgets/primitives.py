"""Small shared building blocks: cards, labels, dividers."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ..theme import PALETTE, SPACING


class Card(QFrame):
    """A white rounded panel with a hairline border and a very soft shadow."""

    def __init__(
        self,
        parent: QWidget | None = None,
        padding: int = SPACING.lg,
        shadow: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(padding, padding, padding, padding)
        layout.setSpacing(SPACING.md)
        self._layout = layout
        if shadow:
            effect = QGraphicsDropShadowEffect(self)
            effect.setBlurRadius(18)
            effect.setOffset(0, 2)
            effect.setColor(QColor(0, 0, 0, 14))
            self.setGraphicsEffect(effect)

    def body(self) -> QVBoxLayout:
        return self._layout


class Label(QLabel):
    """A themed label; ``role`` selects one of the styles in the stylesheet."""

    def __init__(
        self, text: str = "", role: str = "", parent: QWidget | None = None
    ) -> None:
        super().__init__(text, parent)
        if role:
            self.setObjectName(role)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)


class SectionHeader(QLabel):
    """An upper-case section title, as used down the left sidebars."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("SectionLabel")


class Divider(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Divider")
        self.setFixedHeight(1)


class Row(QWidget):
    """A horizontal strip with no margins — used constantly, so it earns a name."""

    def __init__(self, spacing: int = SPACING.sm, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(spacing)
        self._layout = layout

    def body(self) -> QHBoxLayout:
        return self._layout

    def add(self, widget: QWidget, stretch: int = 0) -> QWidget:
        self._layout.addWidget(widget, stretch)
        return widget

    def add_stretch(self, stretch: int = 1) -> None:
        self._layout.addStretch(stretch)


def status_dot(color: str, size: int = 8) -> QLabel:
    """A small coloured disc for inline status."""
    dot = QLabel()
    dot.setFixedSize(size, size)
    dot.setStyleSheet(
        f"background: {color}; border-radius: {size // 2}px;"
    )
    return dot


__all__ = ["Card", "Label", "SectionHeader", "Divider", "Row", "status_dot", "PALETTE"]
