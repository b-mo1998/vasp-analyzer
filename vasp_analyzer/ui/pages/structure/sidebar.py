"""Display controls for the Structure page."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...theme import PALETTE, SPACING, TYPE
from ...widgets.primitives import Divider, Label, SectionHeader

STYLES = ["Ball-and-Stick", "Space Filling", "Wireframe"]
COLORINGS = ["By Element"]


class StructureSidebar(QWidget):
    styleChanged = Signal(str)
    coloringChanged = Signal(str)
    showCellChanged = Signal(bool)
    showAxesChanged = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        outer.addWidget(scroll)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(SPACING.xl, SPACING.xl, SPACING.xl, SPACING.xl)
        layout.setSpacing(SPACING.lg)
        scroll.setWidget(content)

        layout.addWidget(Label("Structure Display", "TitleLabel"))
        layout.addSpacing(SPACING.xs)
        layout.addWidget(Divider())

        layout.addWidget(self._caption("Visualization Style"))
        self.style_box = QComboBox()
        self.style_box.addItems(STYLES)
        self.style_box.currentTextChanged.connect(self.styleChanged)
        layout.addWidget(self.style_box)

        layout.addWidget(self._caption("Coloring"))
        self.coloring_box = QComboBox()
        self.coloring_box.addItems(COLORINGS)
        self.coloring_box.currentTextChanged.connect(self.coloringChanged)
        layout.addWidget(self.coloring_box)

        layout.addSpacing(SPACING.xs)

        self.show_cell = QCheckBox("Show Unit Cell")
        self.show_cell.setChecked(True)
        self.show_cell.toggled.connect(self.showCellChanged)
        layout.addWidget(self.show_cell)

        self.show_axes = QCheckBox("Show Axes")
        self.show_axes.setChecked(True)
        self.show_axes.toggled.connect(self.showAxesChanged)
        layout.addWidget(self.show_axes)

        layout.addStretch(1)

    @staticmethod
    def _caption(text: str) -> Label:
        label = Label(text)
        label.setStyleSheet(
            f"color: {PALETTE.text_secondary}; font-size: {TYPE.body}px;"
        )
        return label

    def set_enabled(self, enabled: bool) -> None:
        for widget in (self.style_box, self.coloring_box, self.show_cell, self.show_axes):
            widget.setEnabled(enabled)

    def reset(self) -> None:
        self.style_box.setCurrentIndex(0)
        self.coloring_box.setCurrentIndex(0)
        self.show_cell.setChecked(True)
        self.show_axes.setChecked(True)


__all__ = ["StructureSidebar", "SectionHeader"]
