"""Display controls for the Structure page."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...theme import PALETTE, SPACING, TYPE
from ...widgets.primitives import Divider, Label, SectionHeader

STYLES = ["Ball-and-Stick", "Space Filling", "Wireframe"]
COLORINGS = ["By Element"]
MAX_SUPERCELL_REPEAT = 6


class StructureSidebar(QWidget):
    styleChanged = Signal(str)
    coloringChanged = Signal(str)
    showCellChanged = Signal(bool)
    showAxesChanged = Signal(bool)
    supercellChanged = Signal(int, int, int)

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

        layout.addSpacing(SPACING.xs)
        layout.addWidget(Divider())

        layout.addWidget(SectionHeader("Supercell"))
        cell_row = QHBoxLayout()
        cell_row.setSpacing(SPACING.sm)
        self.supercell_boxes: dict[str, QSpinBox] = {}
        for axis in ("a", "b", "c"):
            axis_layout = QVBoxLayout()
            axis_layout.setSpacing(2)
            axis_label = self._caption(axis)
            axis_layout.addWidget(axis_label)
            box = QSpinBox()
            box.setRange(1, MAX_SUPERCELL_REPEAT)
            box.setValue(1)
            box.valueChanged.connect(self._on_supercell_value_changed)
            axis_layout.addWidget(box)
            self.supercell_boxes[axis] = box
            cell_row.addLayout(axis_layout)
        layout.addLayout(cell_row)

        self.supercell_hint = Label("", "TertiaryLabel")
        self.supercell_hint.setWordWrap(True)
        layout.addWidget(self.supercell_hint)

        layout.addStretch(1)

    @staticmethod
    def _caption(text: str) -> Label:
        label = Label(text)
        label.setStyleSheet(
            f"color: {PALETTE.text_secondary}; font-size: {TYPE.body}px;"
        )
        return label

    def _on_supercell_value_changed(self, _value: int) -> None:
        self.supercellChanged.emit(
            self.supercell_boxes["a"].value(),
            self.supercell_boxes["b"].value(),
            self.supercell_boxes["c"].value(),
        )

    def set_supercell_hint(self, text: str) -> None:
        self.supercell_hint.setText(text)

    def set_enabled(self, enabled: bool) -> None:
        widgets = [self.style_box, self.coloring_box, self.show_cell, self.show_axes]
        widgets.extend(self.supercell_boxes.values())
        for widget in widgets:
            widget.setEnabled(enabled)

    def reset(self) -> None:
        self.style_box.setCurrentIndex(0)
        self.coloring_box.setCurrentIndex(0)
        self.show_cell.setChecked(True)
        self.show_axes.setChecked(True)
        for box in self.supercell_boxes.values():
            box.blockSignals(True)
            box.setValue(1)
            box.blockSignals(False)
        self.supercellChanged.emit(1, 1, 1)


__all__ = ["StructureSidebar", "SectionHeader"]
