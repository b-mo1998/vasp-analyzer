"""The bottom information strip.

Shows the facts that describe the calculation. Anything the app could not
determine shows an em dash — never a plausible-looking default.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ...models.calculation import Calculation
from ...models.capabilities import Capability
from ...models.metadata import CalculationMetadata
from ..theme import METRICS, PALETTE, SPACING, TYPE
from .primitives import Card


class InfoField(QWidget):
    """One caption/value pair in the strip."""

    def __init__(self, caption: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        self._caption = QLabel(caption)
        self._caption.setObjectName("TertiaryLabel")
        layout.addWidget(self._caption)

        self._value = QLabel("—")
        self._value.setStyleSheet(f"font-size: {TYPE.body}px; font-weight: 500;")
        layout.addWidget(self._value)

    def set_value(self, text: str, tooltip: str = "", muted: bool = False) -> None:
        self._value.setText(text)
        color = PALETTE.text_tertiary if muted else PALETTE.text
        self._value.setStyleSheet(
            f"font-size: {TYPE.body}px; font-weight: 500; color: {color};"
        )
        self.setToolTip(tooltip)


class InfoBar(Card):
    """The horizontal metadata strip at the bottom of the window."""

    detailsRequested = Signal()

    FIELDS = [
        ("system", "System"),
        ("method", "Method"),
        ("fermi", "Fermi Level"),
        ("kpoints", "K-Points"),
        ("encut", "ENCUT"),
        ("wannier", "Wannier90"),
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, padding=SPACING.md)
        self.setFixedHeight(METRICS.info_bar_height)

        row = QHBoxLayout()
        row.setContentsMargins(SPACING.sm, 0, 0, 0)
        row.setSpacing(SPACING.xxl)

        self._fields: dict[str, InfoField] = {}
        for key, caption in self.FIELDS:
            field = InfoField(caption)
            self._fields[key] = field
            row.addWidget(field)

        row.addStretch(1)

        self._details = QPushButton("ⓘ")
        self._details.setObjectName("ToolButton")
        self._details.setFixedSize(30, 30)
        self._details.setCursor(Qt.PointingHandCursor)
        self._details.setToolTip("Calculation details")
        self._details.clicked.connect(self.detailsRequested)
        row.addWidget(self._details)

        self.body().addLayout(row)

    # -- population --------------------------------------------------------

    def clear(self) -> None:
        for field in self._fields.values():
            field.set_value("—", muted=True)

    def update_from(self, calculation: Calculation | None) -> None:
        if calculation is None:
            self.clear()
            return

        meta = calculation.metadata
        display = CalculationMetadata.display

        self._set("system", display(meta.system), meta.system)
        self._set("method", self._method_text(meta), meta.functional)

        if meta.e_fermi is not None:
            self._fields["fermi"].set_value(
                f"{meta.e_fermi.value:.4f} eV",
                tooltip=f"Read from {meta.e_fermi.source}",
            )
        else:
            self._fields["fermi"].set_value(
                "—",
                tooltip=(
                    "No Fermi level available: EIGENVAL does not store one and "
                    "no usable DOSCAR, OUTCAR or vasprun.xml was found."
                ),
                muted=True,
            )

        self._set("kpoints", display(meta.kmesh), meta.kmesh)
        self._set(
            "encut",
            "—" if meta.encut is None else f"{meta.encut.value:g} eV",
            meta.encut,
        )

        if calculation.can(Capability.WANNIER_BANDS):
            self._fields["wannier"].set_value("Loaded ✓")
        else:
            reason = calculation.why_not(Capability.WANNIER_BANDS)
            has_any = any(
                entry.exists
                for key, entry in calculation.index.entries.items()
                if key.startswith("w90_")
            )
            self._fields["wannier"].set_value(
                "Partial" if has_any else "—", tooltip=reason, muted=not has_any
            )

    def _set(self, key: str, text: str, sourced) -> None:
        tooltip = f"Read from {sourced.source}" if sourced is not None else ""
        self._fields[key].set_value(text, tooltip=tooltip, muted=text == "—")

    @staticmethod
    def _method_text(meta: CalculationMetadata) -> str:
        if meta.functional is not None:
            return str(meta.functional.value)
        if meta.spin_mode is not None:
            return str(meta.spin_mode.value)
        return "—"
