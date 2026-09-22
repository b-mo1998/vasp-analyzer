"""Display controls for the Structure page."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...theme import PALETTE, RADIUS, SPACING, TYPE
from ...widgets.element_swatch import ElementSwatch
from ...widgets.slider_row import SliderRow
from ...widgets.primitives import Divider, Label, SectionHeader
from ...widgets.stepper import IntegerStepper

# Beyond this a table is noise, not information — the point is "what is this
# atom actually bonded to," not an exhaustive distance-sorted census.
MAX_NEIGHBOR_ROWS = 6


class NeighborTable(QWidget):
    """Nearest-neighbour distances for whichever atom is currently selected.

    A plain list of rows rather than a QTableWidget: six rows of "element,
    distance" is not dense enough to need a real table's header/sorting/
    selection machinery, and a handful of styled QLabels is far cheaper to
    build and rebuild on every click.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._rows_layout = QVBoxLayout()
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)
        layout.addLayout(self._rows_layout)

        self._placeholder = Label(
            "Select an atom to see its nearest neighbours.", "TertiaryLabel"
        )
        self._placeholder.setWordWrap(True)
        layout.addWidget(self._placeholder)

        self._row_widgets: list[QWidget] = []

    def clear(self) -> None:
        for widget in self._row_widgets:
            self._rows_layout.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()
        self._row_widgets.clear()
        self._placeholder.show()

    def set_neighbors(
        self, neighbors: list[tuple[str, int, float]]
    ) -> None:
        """``neighbors`` is (symbol, 1-based atom id, distance in Å),
        nearest first — already resolved by the page, which is the one that
        knows both the geometry and the atom numbering."""
        self.clear()
        if not neighbors:
            return
        self._placeholder.hide()
        for symbol, atom_id, distance in neighbors[:MAX_NEIGHBOR_ROWS]:
            row = self._build_row(symbol, atom_id, distance)
            self._rows_layout.addWidget(row)
            self._row_widgets.append(row)

    def _build_row(self, symbol: str, atom_id: int, distance: float) -> QWidget:
        row = QWidget()
        row.setObjectName("NeighborRow")
        row.setStyleSheet(
            f"#NeighborRow {{ background: {PALETTE.surface_sunken};"
            f" border-radius: {RADIUS.sm}px; }}"
        )
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(SPACING.sm, SPACING.xs, SPACING.sm, SPACING.xs)
        row_layout.setSpacing(SPACING.xs)

        swatch = ElementSwatch(symbol, size=10)
        row_layout.addWidget(swatch)

        label = Label(f"{symbol} #{atom_id}")
        label.setStyleSheet(f"font-size: {TYPE.body}px; font-weight: 500;")
        row_layout.addWidget(label)
        row_layout.addStretch(1)

        value = Label(f"{distance:.3f} Å")
        value.setStyleSheet(
            f"font-size: {TYPE.body}px; color: {PALETTE.text_secondary};"
        )
        row_layout.addWidget(value)
        return row

STYLES = ["Ball-and-Stick", "Space Filling", "Wireframe"]
# "Uniform" paints every atom one neutral colour. It is what to reach for when
# the shape of a framework matters more than which species sits where — the
# element palette is high-contrast by design and fights a dense cell.
COLORINGS = ["By Element", "Uniform"]
MAX_SUPERCELL_REPEAT = 6


class StructureSidebar(QWidget):
    styleChanged = Signal(str)
    coloringChanged = Signal(str)
    showCellChanged = Signal(bool)
    showAxesChanged = Signal(bool)
    supercellChanged = Signal(int, int, int)
    momentsToggled = Signal(bool)
    momentComponentChanged = Signal(str)
    displacementsToggled = Signal(bool)
    displacementScaleChanged = Signal(float)
    equivalentToggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        # A sidebar is a fixed-width column; anything too wide for it is a
        # layout bug to fix at the source, not something to scroll sideways.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        content = QWidget()
        layout = QVBoxLayout(content)
        # Side padding is lg rather than xl so the widest row here — the three
        # a/b/c supercell steppers side by side — fits the fixed sidebar width
        # instead of being clipped by it.
        layout.setContentsMargins(SPACING.lg, SPACING.xl, SPACING.lg, SPACING.xl)
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
        # xs rather than sm: the three steppers side by side were landing 2px
        # past the fixed sidebar width at the wider spacing.
        cell_row.setSpacing(SPACING.xs)
        self.supercell_boxes: dict[str, IntegerStepper] = {}
        for axis in ("a", "b", "c"):
            axis_layout = QVBoxLayout()
            axis_layout.setSpacing(SPACING.xs)
            axis_label = self._caption(axis)
            axis_label.setStyleSheet(
                axis_label.styleSheet() + f" font-weight: 600; font-size: {TYPE.body}px;"
            )
            axis_layout.addWidget(axis_label)
            box = IntegerStepper(minimum=1, maximum=MAX_SUPERCELL_REPEAT, value=1)
            box.valueChanged.connect(self._on_supercell_value_changed)
            axis_layout.addWidget(box)
            self.supercell_boxes[axis] = box
            cell_row.addLayout(axis_layout)
        layout.addLayout(cell_row)

        self.supercell_hint = Label("", "TertiaryLabel")
        self.supercell_hint.setWordWrap(True)
        layout.addWidget(self.supercell_hint)

        layout.addSpacing(SPACING.xs)
        layout.addWidget(Divider())

        layout.addWidget(SectionHeader("Overlays"))

        self.show_moments = QCheckBox("Magnetic Moments")
        self.show_moments.setToolTip(
            "Per-atom moments from OUTCAR's magnetization table, drawn as "
            "arrows. Needs a spin-polarised run with LORBIT set."
        )
        self.show_moments.setEnabled(False)
        self.show_moments.toggled.connect(self.momentsToggled)
        self.show_moments.toggled.connect(lambda _v: self._sync_component_enabled())
        layout.addWidget(self.show_moments)

        # Which part of the moment the arrows represent. Only ever enabled
        # for a non-collinear run: in a collinear one the moment *is* S_z by
        # construction, so offering S_x and S_y would present a convention
        # as if it were a result.
        self.moment_component = QComboBox()
        self.moment_component.addItems(
            ["Total vector", "Sx only", "Sy only", "Sz only"]
        )
        self.moment_component.setToolTip(
            "Non-collinear runs carry a spin vector per atom. View the whole "
            "vector, or isolate one component to read how it is arranged."
        )
        self.moment_component.setEnabled(False)
        self.moment_component.hide()
        self.moment_component.currentTextChanged.connect(
            lambda text: self.momentComponentChanged.emit(
                {"Total vector": "vector", "Sx only": "x",
                 "Sy only": "y", "Sz only": "z"}[text]
            )
        )
        layout.addWidget(self.moment_component)

        self.show_displacements = QCheckBox("Relaxation Displacements")
        self.show_displacements.setToolTip(
            "Arrows from each atom's POSCAR position to where CONTCAR left it "
            "-- what the relaxation actually did."
        )
        self.show_displacements.setEnabled(False)
        self.show_displacements.toggled.connect(self.displacementsToggled)
        self.show_displacements.toggled.connect(
            lambda _on: self._sync_displacement_scale()
        )
        layout.addWidget(self.show_displacements)

        # Relaxation moves are usually hundredths of an ångström, invisible at
        # 1:1 next to a 1 Å ball. The exaggeration is display-only; the
        # numbers in the info panel are always the true values.
        self.displacement_scale = SliderRow(
            "Exaggeration", 1.0, 30.0, 10.0, 1.0, "×", decimals=0
        )
        self.displacement_scale.setEnabled(False)
        self.displacement_scale.hide()
        self.displacement_scale.valueChanged.connect(self.displacementScaleChanged)
        layout.addWidget(self.displacement_scale)

        self.highlight_equivalent = QCheckBox("Highlight Equivalent Atoms")
        self.highlight_equivalent.setToolTip(
            "When an atom is selected, ring every atom the space group maps "
            "it onto, and report its Wyckoff position."
        )
        self.highlight_equivalent.setChecked(True)
        self.highlight_equivalent.toggled.connect(self.equivalentToggled)
        layout.addWidget(self.highlight_equivalent)

        layout.addSpacing(SPACING.xs)
        layout.addWidget(Divider())

        layout.addWidget(SectionHeader("Nearest Neighbours"))
        self.neighbor_table = NeighborTable()
        layout.addWidget(self.neighbor_table)

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

    def set_moments_available(
        self, available: bool, reason: str = "", non_collinear: bool = False
    ) -> None:
        """Enable the moments toggle only when OUTCAR actually has a table.

        A disabled checkbox with the reason in its tooltip says more than a
        live one that silently draws nothing.
        """
        self.show_moments.setEnabled(available)
        if not available:
            self.show_moments.blockSignals(True)
            self.show_moments.setChecked(False)
            self.show_moments.blockSignals(False)
        if reason:
            self.show_moments.setToolTip(reason)
        self._moments_non_collinear = non_collinear
        self._sync_component_enabled()

    def _sync_component_enabled(self) -> None:
        # Shown only while it can do something: a greyed selector under an
        # unchecked box is clutter that also hints at a feature that is
        # not there for a collinear run.
        relevant = (
            getattr(self, "_moments_non_collinear", False)
            and self.show_moments.isChecked()
        )
        self.moment_component.setEnabled(relevant)
        self.moment_component.setVisible(relevant)

    def _sync_displacement_scale(self) -> None:
        relevant = self.show_displacements.isEnabled() and self.show_displacements.isChecked()
        self.displacement_scale.setEnabled(relevant)
        self.displacement_scale.setVisible(relevant)

    def moment_component_key(self) -> str:
        return {"Total vector": "vector", "Sx only": "x",
                "Sy only": "y", "Sz only": "z"}[self.moment_component.currentText()]

    def set_moment_component(self, key: str) -> None:
        label = {"vector": "Total vector", "x": "Sx only",
                 "y": "Sy only", "z": "Sz only"}[key]
        if self.moment_component.currentText() != label:
            self.moment_component.setCurrentText(label)

    def set_displacements_available(self, available: bool, reason: str = "") -> None:
        self.show_displacements.setEnabled(available)
        if not available:
            self.show_displacements.blockSignals(True)
            self.show_displacements.setChecked(False)
            self.show_displacements.blockSignals(False)
        self._sync_displacement_scale()
        if reason:
            self.show_displacements.setToolTip(reason)

    def set_neighbor_table(self, neighbors: list[tuple[str, int, float]]) -> None:
        self.neighbor_table.set_neighbors(neighbors)

    def set_enabled(self, enabled: bool) -> None:
        widgets = [
            self.style_box, self.coloring_box, self.show_cell, self.show_axes,
            self.highlight_equivalent,
        ]
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
        self.neighbor_table.clear()
        for box in (self.show_moments, self.show_displacements):
            if box.isChecked():
                box.setChecked(False)
        self.displacement_scale.setValue(10.0, notify=False)
        self.moment_component.setCurrentIndex(0)
        if not self.highlight_equivalent.isChecked():
            self.highlight_equivalent.setChecked(True)


__all__ = ["StructureSidebar", "SectionHeader"]
