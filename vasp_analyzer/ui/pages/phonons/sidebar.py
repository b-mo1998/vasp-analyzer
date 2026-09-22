"""Controls for the Phonons page."""

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
from ...widgets.slider_row import SliderRow

BRANCH_ALL = "All Branches"
BRANCH_ACOUSTIC = "Acoustic (1–3)"
BRANCH_OPTICAL = "Optical (4+)"

DOS_TOTAL = "Total DOS"
DOS_ELEMENT = "Element Projected"

SPEEDS = ["0.25×", "0.5×", "1×", "2×", "4×"]
SPEED_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0]


class PhononSidebar(QWidget):
    """Dispersion, DOS, animation and display controls."""

    branchFilterChanged = Signal(str)
    imaginaryToggled = Signal(bool)
    qpathChanged = Signal(str)
    coloringChanged = Signal(str)
    unitsChanged = Signal(str)
    dosTypeChanged = Signal(str)
    projectedToggled = Signal(bool)
    smearingChanged = Signal(float)
    amplitudeChanged = Signal(float)
    speedChanged = Signal(float)
    repeatToggled = Signal(bool)
    displayChanged = Signal()

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
        # Tighter than the other sidebars on purpose: this page carries
        # roughly twice as many controls, and at the shared spacing it was
        # the only one that overflowed its column vertically.
        layout.setContentsMargins(SPACING.sm, SPACING.lg, SPACING.sm, SPACING.lg)
        layout.setSpacing(SPACING.sm)
        scroll.setWidget(content)

        header = QVBoxLayout()
        header.setSpacing(SPACING.xs)
        header.addWidget(Label("Phonon Controls", "TitleLabel"))
        subtitle = Label("Dispersion, DOS and mode animation", "SubtitleLabel")
        subtitle.setWordWrap(True)
        header.addWidget(subtitle)
        layout.addLayout(header)
        layout.addWidget(Divider())

        # -- dispersion -----------------------------------------------------
        layout.addWidget(SectionHeader("Phonon Dispersion"))
        layout.addWidget(self._caption("Branch"))
        self.branch_box = QComboBox()
        self.branch_box.addItems([BRANCH_ALL, BRANCH_ACOUSTIC, BRANCH_OPTICAL])
        self.branch_box.setFixedHeight(27)
        self.branch_box.currentTextChanged.connect(self.branchFilterChanged)
        layout.addWidget(self.branch_box)

        self.show_imaginary = QCheckBox("Show Imaginary Modes")
        self.show_imaginary.setChecked(True)
        self.show_imaginary.setToolTip(
            "Imaginary (unstable) modes are drawn as negative frequencies in red."
        )
        self.show_imaginary.toggled.connect(self.imaginaryToggled)
        layout.addWidget(self.show_imaginary)

        self.acoustic_sum = QCheckBox("Acoustic Sum Rule")
        self.acoustic_sum.setChecked(True)
        self.acoustic_sum.setEnabled(False)
        self.acoustic_sum.setToolTip(
            "Reports whether the loaded data already satisfies the acoustic sum "
            "rule. Applying it here would mean recomputing the force constants, "
            "which this page does not do."
        )
        layout.addWidget(self.acoustic_sum)

        # The q-path is a fact about the loaded band.yaml, not a choice — a
        # dropdown around a single unchangeable value is control chrome
        # implying something to open. Shown inline as "label: value".
        qpath_row = QHBoxLayout()
        qpath_row.setContentsMargins(0, 0, 0, 0)
        qpath_row.setSpacing(SPACING.sm)
        qpath_row.addWidget(self._caption("Q-Path"))
        self.qpath_value = QLabel("—")
        self.qpath_value.setStyleSheet(
            f"font-size: {TYPE.small}px; font-weight: 500; color: {PALETTE.text};"
        )
        qpath_row.addWidget(self.qpath_value, 1)
        layout.addLayout(qpath_row)

        layout.addWidget(self._caption("Frequency Units"))
        self.units_box = QComboBox()
        self.units_box.addItems(["THz", "cm⁻¹", "meV"])
        self.units_box.setFixedHeight(27)
        self.units_box.setToolTip(
            "Display only. Frequencies are held internally in THz and converted "
            "for display, so switching units never rescales the data."
        )
        self.units_box.currentTextChanged.connect(self.unitsChanged)
        layout.addWidget(self.units_box)

        layout.addWidget(self._caption("Coloring"))
        self.coloring_box = QComboBox()
        self.coloring_box.addItems(["By Branch", "By Stability"])
        self.coloring_box.setFixedHeight(27)
        self.coloring_box.currentTextChanged.connect(self.coloringChanged)
        layout.addWidget(self.coloring_box)

        # -- DOS -------------------------------------------------------------
        layout.addWidget(Divider())
        layout.addWidget(SectionHeader("Phonon DOS"))
        layout.addWidget(self._caption("Type"))
        self.dos_box = QComboBox()
        self.dos_box.addItems([DOS_TOTAL, DOS_ELEMENT])
        self.dos_box.setFixedHeight(27)
        self.dos_box.currentTextChanged.connect(self.dosTypeChanged)
        layout.addWidget(self.dos_box)

        self.show_projected = QCheckBox("Show Projected DOS")
        self.show_projected.setChecked(True)
        self.show_projected.toggled.connect(self.projectedToggled)
        layout.addWidget(self.show_projected)

        # Why the two controls above are greyed, stated inline rather than
        # only in a tooltip. A disabled control with no visible explanation
        # reads as broken — which is exactly how the single-element case
        # (where a per-element projection *is* the total curve) was reported.
        self.dos_note = Label("", "TertiaryLabel")
        self.dos_note.setWordWrap(True)
        self.dos_note.hide()
        layout.addWidget(self.dos_note)

        self.smearing = SliderRow(
            "Gaussian Smearing", 0.01, 1.0, 0.10, 0.01, "THz", decimals=2
        )
        self.smearing.valueChanged.connect(self.smearingChanged)
        layout.addWidget(self.smearing)

        # -- animation --------------------------------------------------------
        layout.addWidget(Divider())
        layout.addWidget(SectionHeader("Mode Animation"))
        self.amplitude = SliderRow(
            "Displacement Scale", 0.1, 3.0, 1.0, 0.1, "Å", decimals=1
        )
        self.amplitude.setToolTip(
            "A visualisation multiplier. Phonon amplitude is set by temperature "
            "and is not 1 Å; this only controls how far the atoms are drawn to "
            "move."
        )
        self.amplitude.valueChanged.connect(self.amplitudeChanged)
        layout.addWidget(self.amplitude)

        layout.addWidget(self._caption("Animation Speed"))
        speed_row = QHBoxLayout()
        speed_row.setSpacing(SPACING.xs)
        self.speed_buttons: list[QPushButton] = []
        for index, text in enumerate(SPEEDS):
            button = QPushButton(text)
            button.setCheckable(True)
            button.setObjectName("ToolButton")
            button.setFixedHeight(24)
            button.setCursor(Qt.PointingHandCursor)
            # Override the shared ToolButton horizontal padding: it is sized
            # for word labels, and five of these in a row is the widest thing
            # in this column by a margin.
            button.setStyleSheet("#ToolButton { padding: 0 4px; }")
            button.clicked.connect(lambda _c=False, i=index: self._select_speed(i))
            speed_row.addWidget(button)
            self.speed_buttons.append(button)
        self.speed_buttons[2].setChecked(True)
        layout.addLayout(speed_row)

        self.repeat = QCheckBox("Repeat")
        self.repeat.setChecked(True)
        self.repeat.toggled.connect(self.repeatToggled)
        layout.addWidget(self.repeat)

        # -- display ----------------------------------------------------------
        layout.addWidget(Divider())
        layout.addWidget(SectionHeader("Display Options"))
        self.show_cell = QCheckBox("Show Unit Cell")
        self.show_axes = QCheckBox("Show Axes")
        self.show_atoms = QCheckBox("Show Atoms")
        self.atom_labels = QCheckBox("Atom Labels")
        for box, checked in (
            (self.show_cell, True),
            (self.show_axes, True),
            (self.show_atoms, True),
            (self.atom_labels, False),
        ):
            box.setChecked(checked)
            box.toggled.connect(lambda _v: self.displayChanged.emit())
            layout.addWidget(box)

        layout.addStretch(1)

    @staticmethod
    def _caption(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(
            f"color: {PALETTE.text_secondary}; font-size: {TYPE.small}px;"
        )
        return label

    def _select_speed(self, index: int) -> None:
        for position, button in enumerate(self.speed_buttons):
            button.setChecked(position == index)
        self.speedChanged.emit(SPEED_VALUES[index])

    # -- state -------------------------------------------------------------

    def set_qpaths(self, names: list[str]) -> None:
        self.qpath_value.setText(names[0] if names else "—")

    def set_projection_available(self, available: bool, reason: str = "") -> None:
        self.show_projected.setEnabled(available)
        self.dos_box.setEnabled(available)
        self.show_projected.setToolTip("" if available else reason)
        self.dos_box.setToolTip("" if available else reason)
        self.dos_note.setText("" if available else reason)
        self.dos_note.setVisible(not available and bool(reason))
        if not available:
            self.show_projected.setChecked(False)
            # Fall back to Total, so the mode shown matches the mode that is
            # actually reachable.
            self.dos_box.setCurrentIndex(0)

    def set_smearing_active(self, active: bool, reason: str = "") -> None:
        """Grey the smearing slider when the DOS on screen came from a file
        this page did not broaden, so the control cannot claim an effect it
        does not have."""
        self.smearing.setEnabled(active)
        self.smearing.setToolTip(reason)

    def set_animation_available(self, available: bool, reason: str = "") -> None:
        for widget in (self.amplitude, self.repeat, *self.speed_buttons):
            widget.setEnabled(available)
            widget.setToolTip("" if available else reason)

    def set_acoustic_sum_state(self, satisfied: bool | None) -> None:
        """Report, rather than apply, the acoustic sum rule."""
        if satisfied is None:
            self.acoustic_sum.setChecked(False)
            self.acoustic_sum.setText("Acoustic Sum Rule")
            return
        self.acoustic_sum.setChecked(satisfied)
        self.acoustic_sum.setText(
            "Acoustic Sum Rule  ✓" if satisfied else "Acoustic Sum Rule  ⚠"
        )
        self.acoustic_sum.setToolTip(
            "The three acoustic branches go to zero at Γ, so the loaded force "
            "constants satisfy the acoustic sum rule."
            if satisfied
            else "The acoustic branches do not reach zero at Γ. The force "
            "constants may need an acoustic-sum-rule correction; this page "
            "reports that rather than silently changing the data."
        )

    def speed(self) -> float:
        for index, button in enumerate(self.speed_buttons):
            if button.isChecked():
                return SPEED_VALUES[index]
        return 1.0

    def reset(self) -> None:
        self.branch_box.setCurrentIndex(0)
        self.show_imaginary.setChecked(True)
        self.coloring_box.setCurrentIndex(0)
        self.units_box.setCurrentIndex(0)
        self.dos_box.setCurrentIndex(0)
        self.show_projected.setChecked(True)
        self.smearing.setValue(0.10)
        self.amplitude.setValue(1.0)
        self._select_speed(2)
        self.repeat.setChecked(True)
        for box, checked in (
            (self.show_cell, True),
            (self.show_axes, True),
            (self.show_atoms, True),
            (self.atom_labels, False),
        ):
            box.setChecked(checked)
