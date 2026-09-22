"""Controls for the Bands page."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...theme import PALETTE, SPACING, TYPE
from ...widgets.primitives import Divider, Label, SectionHeader
from .orbital_matrix import OrbitalMatrix, OrbitalShellPicker
from ...widgets.slider_row import SliderRow

DEFAULT_Y_MIN = -10.0
DEFAULT_Y_MAX = 10.0
FERMI_SPAN = 5.0


class BandsSidebar(QWidget):
    """Energy range, Fermi reference, projections and display options."""

    yMinChanged = Signal(float)
    yMaxChanged = Signal(float)
    fermiChanged = Signal(float)
    dosToggled = Signal(bool)
    projectionToggled = Signal(str, bool)
    orbitalsChanged = Signal(dict)  # {element: [orbital, ...]} currently chosen
    atomProjectionChanged = Signal()  # atom selector text or orbital chips changed
    bandGapMarkersToggled = Signal(bool)
    hoverInfoToggled = Signal(bool)
    spinAxisChanged = Signal(int)  # 0=x, 1=y, 2=z

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._orbitals: list[str] = []

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

        # -- header ---------------------------------------------------------
        layout.addWidget(Label("Band Structure", "TitleLabel"))
        layout.addWidget(Label("Electronic structure analysis", "SubtitleLabel"))
        layout.addSpacing(SPACING.xs)
        layout.addWidget(Divider())

        # -- energy range ---------------------------------------------------
        layout.addWidget(SectionHeader("Bands Range (Energy in eV)"))

        self.y_min = SliderRow("Y-Axis Min", -20.0, 0.0, DEFAULT_Y_MIN, 0.1, "eV")
        self.y_min.valueChanged.connect(self.yMinChanged)
        layout.addWidget(self.y_min)

        self.y_max = SliderRow("Y-Axis Max", 0.0, 20.0, DEFAULT_Y_MAX, 0.1, "eV")
        self.y_max.valueChanged.connect(self.yMaxChanged)
        layout.addWidget(self.y_max)

        self.fermi = SliderRow(
            "Fermi Level", -FERMI_SPAN, FERMI_SPAN, 0.0, 0.01, "eV", decimals=3
        )
        self.fermi.valueChanged.connect(self.fermiChanged)
        layout.addWidget(self.fermi)

        layout.addWidget(Divider())

        # -- projections ------------------------------------------------------
        layout.addWidget(SectionHeader("Projections"))

        # Element/spin projection both come from PROCAR, which on a real
        # calculation can be a very large file (hundreds of MB to a couple
        # of GB) — parsing it is genuinely slow, seconds to tens of seconds,
        # not a bug. Without this, checking the box did nothing visible
        # until the plot suddenly updated later, which reads as broken or
        # frozen rather than "working on it."
        self.projection_loading = Label("Loading projection data…", "TertiaryLabel")
        self.projection_loading.hide()
        layout.addWidget(self.projection_loading)

        self.projections: dict[str, QCheckBox] = {}
        for key, text in (
            ("spin", "Spin Projection"),
            ("element", "Element Projection"),
            ("wannier", "Wannier Bands"),
        ):
            box = QCheckBox(text)
            layout.addWidget(box)
            self.projections[key] = box

            if key == "spin":
                # Indented under its checkbox: which spin component drives
                # the colour, for a non-collinear (SOC) calculation's mx/my/
                # mz. Ignored for a plain ISPIN=2 collinear calculation,
                # which only has one "spin" axis (up minus down) to begin
                # with.
                axis_row = QHBoxLayout()
                axis_row.setContentsMargins(SPACING.xl, 0, 0, 0)
                axis_label = QLabel("Axis")
                axis_label.setStyleSheet(
                    f"font-size: {TYPE.small}px; color: {PALETTE.text_secondary};"
                )
                axis_label.hide()
                self.spin_axis_label = axis_label
                axis_row.addWidget(axis_label)
                self.spin_axis_box = QComboBox()
                self.spin_axis_box.addItems(["X", "Y", "Z"])
                self.spin_axis_box.setCurrentIndex(2)
                self.spin_axis_box.setEnabled(False)
                self.spin_axis_box.hide()
                self.spin_axis_box.currentIndexChanged.connect(self.spinAxisChanged)
                axis_row.addWidget(self.spin_axis_box, 1)
                layout.addLayout(axis_row)

            if key == "element":
                # Indented under its checkbox: which orbitals to project,
                # chosen per element. A shared orbital filter can only ask
                # one question of the whole cell; the matrix lets carbon p
                # be compared against tellurium d, which is the question a
                # doped system actually poses.
                self.orbital_container = OrbitalMatrix()
                self.orbital_container.changed.connect(self._on_orbital_selection_changed)
                self.orbital_container.setEnabled(False)
                self.orbital_container.hide()
                layout.addWidget(self.orbital_container)

        # Element and spin projection share the same single overlay slot on
        # the plot, so they're mutually exclusive: checking one unchecks the
        # other. Wired *before* the projectionToggled emission below so that
        # turning the old mode off (and page.py clearing its overlay) fully
        # completes before the newly-checked mode's projectionToggled(True)
        # arrives and builds its own overlay — otherwise the new overlay
        # would be built first and then immediately stomped by the old
        # mode's teardown.
        self.projections["element"].toggled.connect(self._enforce_projection_exclusivity)
        self.projections["spin"].toggled.connect(self._enforce_projection_exclusivity)

        for key, box in self.projections.items():
            box.toggled.connect(
                lambda checked, name=key: self.projectionToggled.emit(name, checked)
            )

        self.projections["element"].toggled.connect(self._on_element_toggled)
        self.projections["spin"].toggled.connect(self._on_spin_toggled)

        layout.addWidget(Divider())

        # -- atom projection ---------------------------------------------------
        #
        # Element projection can't isolate one Te dopant from 112 carbons —
        # a species filter is either all of an element or none of it. This
        # asks the more specific question VESTA's atom-picker asks: project
        # onto exactly these atoms (by the same ID the Editor's atom table
        # already shows), restricted to whichever orbitals matter. It shares
        # the plot's one overlay slot with Element/Spin above, so it joins
        # the same self.projections dict and gets the same exclusivity and
        # projectionToggled wiring for free — this is a fourth *kind* of
        # projection, not a separate feature.
        layout.addWidget(SectionHeader("Atom Projection"))
        atom_box = QCheckBox("Enable")
        atom_box.setToolTip(
            "Project onto specific atoms rather than a whole element — enter "
            "atom IDs (the same numbering the Editor's atom table uses), "
            "ranges, or element symbols below."
        )
        layout.addWidget(atom_box)
        self.projections["atom"] = atom_box

        self.atom_selector_field = QLineEdit()
        self.atom_selector_field.setObjectName("ValueField")
        self.atom_selector_field.setPlaceholderText("Atom IDs or elements, e.g. Te, 12, 45-50")
        self.atom_selector_field.setEnabled(False)
        self.atom_selector_field.hide()
        self.atom_selector_field.textChanged.connect(self._on_atom_selector_changed)
        layout.addWidget(self.atom_selector_field)

        self.atom_orbitals = OrbitalShellPicker()
        self.atom_orbitals.setEnabled(False)
        self.atom_orbitals.hide()
        self.atom_orbitals.changed.connect(self._on_atom_selector_changed)
        layout.addWidget(self.atom_orbitals)

        self.atom_status = Label("", "TertiaryLabel")
        self.atom_status.setWordWrap(True)
        self.atom_status.hide()
        layout.addWidget(self.atom_status)

        atom_box.toggled.connect(self._enforce_projection_exclusivity)
        atom_box.toggled.connect(self._on_atom_toggled)
        atom_box.toggled.connect(
            lambda checked: self.projectionToggled.emit("atom", checked)
        )

        layout.addWidget(Divider())

        # -- display ------------------------------------------------------------
        layout.addWidget(SectionHeader("Display"))
        self.show_dos = QCheckBox("Show Density of States")
        self.show_dos.setChecked(True)
        self.show_dos.toggled.connect(self.dosToggled)
        layout.addWidget(self.show_dos)

        self.show_band_gap = QCheckBox("Show VBM/CBM Markers")
        self.show_band_gap.setChecked(True)
        self.show_band_gap.toggled.connect(self.bandGapMarkersToggled)
        layout.addWidget(self.show_band_gap)

        self.show_hover_info = QCheckBox("Show Hover Info")
        self.show_hover_info.setToolTip(
            "The tooltip that reports k-point, band, energy and orbital "
            "character under the cursor."
        )
        self.show_hover_info.setChecked(True)
        self.show_hover_info.toggled.connect(self.hoverInfoToggled)
        layout.addWidget(self.show_hover_info)

        layout.addStretch(1)

    # -- state ---------------------------------------------------------------

    def set_fermi_range(self, centre: float | None) -> None:
        """Centre the Fermi control on the value parsed from the calculation.

        With no parsed value the control still works — it just starts at zero,
        which is a display reference rather than a claim about the calculation.
        """
        base = 0.0 if centre is None else float(centre)
        self.fermi.setRange(base - FERMI_SPAN, base + FERMI_SPAN)
        self.fermi.setValue(base, notify=False)

    def set_projection_loading(self, loading: bool) -> None:
        self.projection_loading.setVisible(loading)

    def set_projection_enabled(self, key: str, enabled: bool, reason: str = "") -> None:
        box = self.projections.get(key)
        if box is None:
            return
        box.setEnabled(enabled)
        box.setToolTip("" if enabled else reason)
        if not enabled and box.isChecked():
            box.setChecked(False)

    def set_orbitals(self, orbitals: list[str], elements: list[str]) -> None:
        """Build the matrix once PROCAR and the structure are both known —
        neither the orbital columns nor the element list exist before then.
        Everything starts selected, which is the plain by-element view."""
        self._orbitals = list(orbitals)
        self.orbital_container.populate(elements, orbitals)
        self._on_element_toggled(self.projections["element"].isChecked())

    def clear_orbitals(self) -> None:
        self._orbitals = []
        self.orbital_container.clear()
        self.orbital_container.setEnabled(False)
        self.orbital_container.hide()

    def orbital_selection(self) -> dict[str, list[str]]:
        """{element: [orbital, ...]}, restricted to what is actually chosen."""
        return self.orbital_container.selection()

    def orbital_selection_is_full(self) -> bool:
        return self.orbital_container.is_full_selection(self._orbitals)

    def _on_orbital_selection_changed(self) -> None:
        self.orbitalsChanged.emit(self.orbital_container.selection())

    # Dependent controls are shown only while their parent box is on. A
    # greyed matrix or field under an unchecked box adds length to the
    # sidebar without offering anything to do.

    def _on_element_toggled(self, checked: bool) -> None:
        relevant = checked and self.orbital_container.has_any()
        self.orbital_container.setEnabled(relevant)
        self.orbital_container.setVisible(relevant)

    def _on_spin_toggled(self, checked: bool) -> None:
        self.spin_axis_box.setEnabled(checked)
        self.spin_axis_box.setVisible(checked)
        self.spin_axis_label.setVisible(checked)

    def _on_atom_toggled(self, checked: bool) -> None:
        for widget in (self.atom_selector_field, self.atom_orbitals):
            widget.setEnabled(checked)
            widget.setVisible(checked)
        if not checked:
            self.set_atom_status("")

    def _on_atom_selector_changed(self, *_args) -> None:
        self.atomProjectionChanged.emit()

    def atom_selector_text(self) -> str:
        return self.atom_selector_field.text()

    def atom_orbital_selection(self) -> list[str]:
        return self.atom_orbitals.selected()

    def set_atom_orbitals(self, orbitals: list[str]) -> None:
        self.atom_orbitals.populate(orbitals)

    def set_atom_status(self, text: str) -> None:
        self.atom_status.setText(text)
        self.atom_status.setVisible(bool(text))

    def spin_axis(self) -> int:
        return self.spin_axis_box.currentIndex()

    def _enforce_projection_exclusivity(self, checked: bool) -> None:
        """Element, Spin and Atom projection all draw into the plot's one
        overlay slot, so checking any of them clears the other two. Wannier
        is not included — it draws a separate comparison line rather than
        colouring the same points, so it can stay on alongside whichever of
        the other three is active."""
        if not checked:
            return
        sender = self.sender()
        for key in ("element", "spin", "atom"):
            box = self.projections[key]
            if box is not sender and box.isChecked():
                box.setChecked(False)

    def set_dos_enabled(self, enabled: bool, reason: str = "") -> None:
        self.show_dos.setEnabled(enabled)
        self.show_dos.setToolTip("" if enabled else reason)

    def set_controls_enabled(self, enabled: bool) -> None:
        for widget in (self.y_min, self.y_max, self.fermi):
            widget.setEnabled(enabled)

    def reset(self, fermi_centre: float | None) -> None:
        self.y_min.setValue(DEFAULT_Y_MIN, notify=False)
        self.y_max.setValue(DEFAULT_Y_MAX, notify=False)
        self.set_fermi_range(fermi_centre)
        for box in self.projections.values():
            if box.isChecked():
                box.setChecked(False)
