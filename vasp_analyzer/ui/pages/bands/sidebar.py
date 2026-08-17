"""Controls for the Bands page."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...theme import SPACING
from ...widgets.primitives import Divider, Label, SectionHeader
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
    orbitalChanged = Signal(object)  # str, or None for "All Orbitals"

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
        self.projections: dict[str, QCheckBox] = {}
        for key, text in (
            ("spin", "Spin Projection"),
            ("element", "Element Projection"),
            ("wannier", "Wannier Bands"),
        ):
            box = QCheckBox(text)
            box.toggled.connect(
                lambda checked, name=key: self.projectionToggled.emit(name, checked)
            )
            layout.addWidget(box)
            self.projections[key] = box

            if key == "element":
                # Indented under its checkbox: which orbital to colour and
                # size by, once PROCAR is loaded and the projection is on.
                orbital_row = QHBoxLayout()
                orbital_row.setContentsMargins(SPACING.xl, 0, 0, 0)
                self.orbital_box = QComboBox()
                self.orbital_box.addItem("All Orbitals")
                self.orbital_box.setEnabled(False)
                self.orbital_box.currentTextChanged.connect(self._on_orbital_changed)
                orbital_row.addWidget(self.orbital_box, 1)
                layout.addLayout(orbital_row)

        self.projections["element"].toggled.connect(self._on_element_toggled)

        layout.addWidget(Divider())

        # -- display ------------------------------------------------------------
        layout.addWidget(SectionHeader("Display"))
        self.show_dos = QCheckBox("Show Density of States")
        self.show_dos.setChecked(True)
        self.show_dos.toggled.connect(self.dosToggled)
        layout.addWidget(self.show_dos)

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

    def set_projection_enabled(self, key: str, enabled: bool, reason: str = "") -> None:
        box = self.projections.get(key)
        if box is None:
            return
        box.setEnabled(enabled)
        box.setToolTip("" if enabled else reason)
        if not enabled and box.isChecked():
            box.setChecked(False)

    def set_orbitals(self, orbitals: list[str]) -> None:
        """Populate the orbital selector once PROCAR has actually been read
        — the orbital list isn't known until then."""
        self.orbital_box.blockSignals(True)
        self.orbital_box.clear()
        self.orbital_box.addItem("All Orbitals")
        self.orbital_box.addItems(orbitals)
        self.orbital_box.blockSignals(False)
        self.orbital_box.setEnabled(self.projections["element"].isChecked())

    def clear_orbitals(self) -> None:
        self.orbital_box.blockSignals(True)
        self.orbital_box.clear()
        self.orbital_box.addItem("All Orbitals")
        self.orbital_box.blockSignals(False)
        self.orbital_box.setEnabled(False)

    def selected_orbital(self) -> str | None:
        text = self.orbital_box.currentText()
        return None if text == "All Orbitals" else text

    def _on_element_toggled(self, checked: bool) -> None:
        self.orbital_box.setEnabled(checked and self.orbital_box.count() > 1)
        if not checked:
            self.orbital_box.blockSignals(True)
            self.orbital_box.setCurrentIndex(0)
            self.orbital_box.blockSignals(False)

    def _on_orbital_changed(self, text: str) -> None:
        self.orbitalChanged.emit(None if text == "All Orbitals" else text)

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
