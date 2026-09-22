"""Display controls for the Molecular Dynamics page."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...theme import PALETTE, RADIUS, SPACING, TYPE
from ...widgets.primitives import Divider, Label, SectionHeader
from ...widgets.slider_row import SliderRow
from ..structure.sidebar import COLORINGS, STYLES

DEFAULT_TRAIL_LENGTH = 10
MAX_TRAIL_LENGTH = 100

WRAPPED = "wrapped"
UNWRAPPED = "unwrapped"


class AtomChip(QWidget):
    """The selected atom, its element colour, and a button to clear it."""

    cleared = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(SPACING.sm, SPACING.xs, SPACING.xs, SPACING.xs)
        row.setSpacing(SPACING.sm)

        self._label = QLabel("No atom selected")
        self._label.setStyleSheet(f"font-size: {TYPE.body}px;")
        row.addWidget(self._label, 1)

        self._dot = QLabel()
        self._dot.setFixedSize(12, 12)
        row.addWidget(self._dot)

        self._clear = QPushButton("✕")
        self._clear.setFixedSize(18, 18)
        self._clear.setObjectName("ChipClear")
        self._clear.setCursor(self.cursor())
        self._clear.clicked.connect(self.cleared)
        row.addWidget(self._clear)

        self.setObjectName("AtomChip")
        self.setStyleSheet(
            f"#AtomChip {{ background: {PALETTE.surface_sunken};"
            f" border: 1px solid {PALETTE.border};"
            f" border-radius: {RADIUS.sm}px; }}"
            f"#ChipClear {{ border: none; background: transparent;"
            f" color: {PALETTE.text_secondary}; font-size: 11px; }}"
            f"#ChipClear:hover {{ color: {PALETTE.text}; }}"
        )
        self.set_atom(None, "")

    def set_atom(self, text: str | None, color: str) -> None:
        self._label.setText(text or "No atom selected")
        self._label.setStyleSheet(
            f"font-size: {TYPE.body}px;"
            f" color: {PALETTE.text if text else PALETTE.text_tertiary};"
        )
        self._dot.setVisible(bool(text))
        self._clear.setVisible(bool(text))
        if text:
            self._dot.setStyleSheet(
                f"background: {color}; border-radius: 6px;"
                f" border: 1px solid rgba(0,0,0,0.15);"
            )


class MDSidebar(QWidget):
    """Visualisation style, trail settings, boundary handling and selection."""

    styleChanged = Signal(str)
    coloringChanged = Signal(str)
    showCellChanged = Signal(bool)
    showAxesChanged = Signal(bool)
    trailsChanged = Signal()
    wrapChanged = Signal(bool)          # True when atoms are wrapped into the cell
    centreChanged = Signal(bool)
    boundaryModeChanged = Signal(str)   # WRAPPED / UNWRAPPED
    trackChanged = Signal(bool)
    showTrailChanged = Signal(bool)
    selectionCleared = Signal()

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
        layout.setContentsMargins(SPACING.xl, SPACING.xl, SPACING.xl, SPACING.xl)
        layout.setSpacing(SPACING.lg)
        scroll.setWidget(content)

        layout.addWidget(Label("Display Controls", "TitleLabel"))
        layout.addSpacing(SPACING.xs)
        layout.addWidget(Divider())

        # --- appearance -------------------------------------------------------
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

        layout.addWidget(Divider())

        # --- MD display -------------------------------------------------------
        layout.addWidget(SectionHeader("MD Display"))

        self.atom_trails = QCheckBox("Atom Trails")
        self.atom_trails.setChecked(True)
        self.atom_trails.toggled.connect(self._on_trails_toggled)
        layout.addWidget(self.atom_trails)

        self.wrap_atoms = QCheckBox("Wrap Atoms Into Cell")
        # Unchecked to match the Unwrapped default of the radio group below,
        # which expresses the same physical choice.
        self.wrap_atoms.setChecked(False)
        self.wrap_atoms.toggled.connect(self._on_wrap_toggled)
        layout.addWidget(self.wrap_atoms)

        self.centre_structure = QCheckBox("Center Structure")
        self.centre_structure.setChecked(True)
        self.centre_structure.toggled.connect(self.centreChanged)
        layout.addWidget(self.centre_structure)

        self.trail_length = SliderRow(
            "Trail Length", 2, MAX_TRAIL_LENGTH, DEFAULT_TRAIL_LENGTH, 1, "frames",
            decimals=0,
        )
        self.trail_length.valueChanged.connect(lambda _v: self.trailsChanged.emit())
        layout.addWidget(self.trail_length)

        layout.addWidget(Divider())

        # --- periodic boundaries ------------------------------------------------
        layout.addWidget(SectionHeader("Periodic Boundaries"))
        self._boundary_group = QButtonGroup(self)
        self._boundary_group.setExclusive(True)

        self.boundary_wrapped = QRadioButton("Wrapped (in cell)")
        self.boundary_unwrapped = QRadioButton("Unwrapped")
        # Unwrapped by default: it shows continuous motion, which is what a
        # trajectory is being watched for. Wrapping teleports an atom to the
        # opposite face the moment it crosses a boundary, which reads as the
        # structure glitching rather than as diffusion, and it is the same
        # continuous coordinate set the MSD is measured from -- so the
        # default now matches what the plots on this page describe.
        self.boundary_unwrapped.setChecked(True)
        self.boundary_unwrapped.setToolTip(
            "Reconstruct continuous trajectories through periodic boundaries — "
            "the coordinates diffusion is measured from."
        )
        for index, button in enumerate(
            (self.boundary_wrapped, self.boundary_unwrapped)
        ):
            self._boundary_group.addButton(button, index)
            layout.addWidget(button)
        self._boundary_group.idToggled.connect(self._on_boundary_changed)

        layout.addWidget(Divider())

        # --- atom selection -------------------------------------------------------
        layout.addWidget(SectionHeader("Atom Selection"))
        self.atom_chip = AtomChip()
        self.atom_chip.cleared.connect(self.selectionCleared)
        layout.addWidget(self.atom_chip)

        self.track_atom = QCheckBox("Track Atom")
        self.track_atom.setChecked(True)
        self.track_atom.toggled.connect(self.trackChanged)
        layout.addWidget(self.track_atom)

        self.show_trail = QCheckBox("Show Trail")
        self.show_trail.setChecked(True)
        self.show_trail.toggled.connect(self.showTrailChanged)
        layout.addWidget(self.show_trail)

        self.clear_button = QPushButton("Clear Selection")
        self.clear_button.clicked.connect(self.selectionCleared)
        layout.addWidget(self.clear_button)

        layout.addStretch(1)
        self._sync_trail_enabled()
        self.set_selected_atom(None, "")

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _caption(text: str) -> Label:
        label = Label(text)
        label.setStyleSheet(
            f"color: {PALETTE.text_secondary}; font-size: {TYPE.body}px;"
        )
        return label

    def _on_trails_toggled(self, _checked: bool) -> None:
        self._sync_trail_enabled()
        self.trailsChanged.emit()

    def _sync_trail_enabled(self) -> None:
        # Shown only with trails on: a length control for trails that are not
        # drawn is something to drag that does nothing.
        on = self.atom_trails.isChecked()
        self.trail_length.setEnabled(on)
        self.trail_length.setVisible(on)

    def _on_wrap_toggled(self, checked: bool) -> None:
        # The checkbox and the radio group express the same physical choice, so
        # they are kept in step rather than allowed to contradict each other.
        self._set_boundary(WRAPPED if checked else UNWRAPPED, from_checkbox=True)
        self.wrapChanged.emit(checked)

    def _on_boundary_changed(self, index: int, checked: bool) -> None:
        if not checked:
            return
        mode = WRAPPED if index == 0 else UNWRAPPED
        self.wrap_atoms.blockSignals(True)
        self.wrap_atoms.setChecked(mode == WRAPPED)
        self.wrap_atoms.blockSignals(False)
        self.boundaryModeChanged.emit(mode)

    def _set_boundary(self, mode: str, from_checkbox: bool = False) -> None:
        button = self.boundary_wrapped if mode == WRAPPED else self.boundary_unwrapped
        if button.isChecked():
            return
        button.blockSignals(True)
        button.setChecked(True)
        button.blockSignals(False)
        if from_checkbox:
            self.boundaryModeChanged.emit(mode)

    # -- state ---------------------------------------------------------------

    def boundary_mode(self) -> str:
        return WRAPPED if self.boundary_wrapped.isChecked() else UNWRAPPED

    def trail_frames(self) -> int:
        return int(self.trail_length.value())

    def set_selected_atom(self, text: str | None, color: str) -> None:
        self.atom_chip.set_atom(text, color)
        for widget in (self.track_atom, self.show_trail, self.clear_button):
            widget.setEnabled(text is not None)

    def set_enabled(self, enabled: bool) -> None:
        for widget in (
            self.style_box,
            self.coloring_box,
            self.show_cell,
            self.show_axes,
            self.atom_trails,
            self.wrap_atoms,
            self.centre_structure,
            self.trail_length,
            self.boundary_wrapped,
            self.boundary_unwrapped,
        ):
            widget.setEnabled(enabled)
        if enabled:
            self._sync_trail_enabled()

    def reset(self) -> None:
        self.style_box.setCurrentIndex(0)
        self.coloring_box.setCurrentIndex(0)
        self.show_cell.setChecked(True)
        self.show_axes.setChecked(True)
        self.atom_trails.setChecked(True)
        self.wrap_atoms.setChecked(False)
        self.centre_structure.setChecked(True)
        self.trail_length.setValue(DEFAULT_TRAIL_LENGTH, notify=False)
        self.boundary_unwrapped.setChecked(True)
        self.track_atom.setChecked(True)
        self.show_trail.setChecked(True)
        self.set_selected_atom(None, "")
        self._sync_trail_enabled()
