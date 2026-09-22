"""Controls for the NEB page.

Several controls here describe the calculation rather than drive it: the path
method, the convergence criterion and the climbing-image switch all report what
the INCAR asked for. They are shown read-only, because this application
analyses finished calculations and silently rewriting an INCAR from a display
control would be a genuinely dangerous thing for it to do.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...theme import PALETTE, SPACING, TYPE
from ...widgets.primitives import Divider, Label, SectionHeader
from ..md.sidebar import MAX_TRAIL_LENGTH
from ..structure.sidebar import COLORINGS, STYLES

class NEBSidebar(QWidget):
    """Path description and visualisation controls."""

    styleChanged = Signal(str)
    coloringChanged = Signal(str)
    showCellChanged = Signal(bool)
    showAxesChanged = Signal(bool)
    trailsChanged = Signal()
    wrapChanged = Signal(bool)
    centreChanged = Signal(bool)
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

        layout.addWidget(Label("NEB Controls", "TitleLabel"))
        layout.addWidget(Label("Explore the reaction pathway profile", "SubtitleLabel"))
        layout.addSpacing(SPACING.xs)
        layout.addWidget(Divider())

        # --- what the calculation is ------------------------------------------
        #
        # Read-only facts about the finished run, so they are plain text.
        # They were disabled dropdowns, which drew a control affordance —
        # a bordered box implying something to open — around values that can
        # never be changed from here.
        layout.addWidget(self._caption("Path Calculation"))
        self.method_value = self._readout("—")
        self.method_value.setToolTip(
            "Read from the calculation's INCAR (LCLIMB / IMAGES)."
        )
        layout.addWidget(self.method_value)

        layout.addWidget(self._caption("Method"))
        self.optimiser_value = self._readout("—")
        self.optimiser_value.setToolTip(
            "Read from the calculation's INCAR (IBRION / IOPT)."
        )
        layout.addWidget(self.optimiser_value)

        layout.addWidget(Divider())

        # --- appearance ----------------------------------------------------------
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

        # --- image display ---------------------------------------------------------
        layout.addWidget(SectionHeader("Image Display"))

        self.atom_trails = QCheckBox("Atom Trails")
        self.atom_trails.setChecked(True)
        self.atom_trails.setToolTip(
            "Show how each atom moves along the reaction path, not through time."
        )
        self.atom_trails.toggled.connect(self._on_trails_toggled)
        layout.addWidget(self.atom_trails)

        self.wrap_atoms = QCheckBox("Wrap Atoms Into Cell")
        self.wrap_atoms.setChecked(True)
        self.wrap_atoms.toggled.connect(self.wrapChanged)
        layout.addWidget(self.wrap_atoms)

        self.centre_structure = QCheckBox("Center Structure")
        self.centre_structure.setChecked(True)
        self.centre_structure.toggled.connect(self.centreChanged)
        layout.addWidget(self.centre_structure)


        layout.addWidget(Divider())

        self.clear_button = QPushButton("Clear Selection")
        self.clear_button.clicked.connect(self.selectionCleared)
        layout.addWidget(self.clear_button)

        layout.addStretch(1)

    # -- helpers -------------------------------------------------------------------

    @staticmethod
    def _caption(text: str) -> Label:
        label = Label(text)
        label.setStyleSheet(
            f"color: {PALETTE.text_secondary}; font-size: {TYPE.body}px;"
        )
        return label

    @staticmethod
    def _readout(text: str) -> Label:
        """A read-only value: emphasised text, no control chrome."""
        label = Label(text)
        label.setWordWrap(True)
        label.setStyleSheet(
            f"color: {PALETTE.text}; font-size: {TYPE.body}px; font-weight: 500;"
        )
        return label

    def _on_trails_toggled(self, _checked: bool) -> None:
        self.trailsChanged.emit()


    # -- population -----------------------------------------------------------------

    def describe_path(
        self,
        method: str,
        optimiser: str,
        threshold: float | None,
        climbing: bool,
    ) -> None:
        """Show what the calculation actually asked for.

        ``threshold`` and ``climbing`` are accepted but no longer shown here:
        the convergence readout that displayed them was removed from this
        sidebar, and both facts are still reported — the force threshold in
        the bottom bar, and the climbing image in the method name and the
        Model Information card.
        """
        _ = (threshold, climbing)
        self.method_value.setText(method or "—")
        self.optimiser_value.setText(optimiser or "—")

    def trail_images(self) -> int:
        """Always the whole band.

        A reaction path is a handful of images that are meant to be read as
        one continuous path, so trimming the trail to a subset of them only
        ever hid part of the thing the page exists to show. There is no
        length control any more; MAX_TRAIL_LENGTH stands in for "all of it"
        and is clamped against the real image count by the caller.
        """
        return MAX_TRAIL_LENGTH

    def set_enabled(self, enabled: bool) -> None:
        for widget in (
            self.style_box,
            self.coloring_box,
            self.show_cell,
            self.show_axes,
            self.atom_trails,
            self.wrap_atoms,
            self.centre_structure,
            self.clear_button,
        ):
            widget.setEnabled(enabled)

    def reset(self) -> None:
        self.style_box.setCurrentIndex(0)
        self.coloring_box.setCurrentIndex(0)
        self.show_cell.setChecked(True)
        self.show_axes.setChecked(True)
        self.atom_trails.setChecked(True)
        self.wrap_atoms.setChecked(True)
        self.centre_structure.setChecked(True)


__all__ = ["NEBSidebar", "QLabel"]
