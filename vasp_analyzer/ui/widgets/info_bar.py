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

    def set_value(
        self, text: str, tooltip: str = "", muted: bool = False, color: str | None = None
    ) -> None:
        """``color`` overrides the text colour -- for a value that *is* a
        state, like the Editor's green "Saved" / amber "Modified"."""
        self._value.setText(text)
        color = color or (PALETTE.text_tertiary if muted else PALETTE.text)
        self._value.setStyleSheet(
            f"font-size: {TYPE.body}px; font-weight: 500; color: {color};"
        )
        self.setToolTip(tooltip)


class InfoBar(Card):
    """The horizontal metadata strip at the bottom of the window.

    The strip describes whatever the current page is about, so its fields are
    swappable: an electronic-structure page wants the Fermi level and the gap,
    while a trajectory page wants frames, timestep and ensemble. Pages that
    need the alternative set call :meth:`set_layout` and then :meth:`set_fields`.
    """

    detailsRequested = Signal()

    FIELDS = [
        ("system", "System"),
        ("method", "Method"),
        ("fermi", "Fermi Level"),
        ("gap", "Gap"),
        ("kpoints", "K-Points"),
        ("encut", "ENCUT"),
        ("wannier", "Wannier90"),
    ]

    MD_FIELDS = [
        ("system", "System"),
        ("frames", "Frames"),
        ("timestep", "Timestep"),
        ("duration", "Duration"),
        ("ensemble", "Ensemble"),
        ("temperature", "Temperature"),
    ]

    NEB_FIELDS = [
        ("system", "System"),
        ("images", "Images"),
        ("method", "Method"),
        ("barrier", "Barrier"),
        ("force", "Max Force"),
        ("convergence", "Convergence"),
    ]

    CHARGE_FIELDS = [
        ("system", "System"),
        ("source", "Source"),
        ("datatype", "Data Type"),
        ("grid", "Grid"),
        ("volume", "Cell Volume"),
        ("integrated", "Integrated e⁻"),
        ("spin", "Spin"),
    ]

    # The Editor's old status strip, folded in here so the page has one
    # bottom bar rather than two stacked ones both starting with "System".
    EDITOR_FIELDS = [
        ("system", "System"),
        ("atoms", "Atoms"),
        ("selected", "Selected"),
        ("mode", "Mode"),
        ("gap", "Gap"),
        ("kpoints", "K-Points"),
        ("encut", "ENCUT"),
        ("status", "Status"),
    ]

    PHONON_FIELDS = [
        ("system", "System"),
        ("calculation", "Calculation"),
        ("supercell", "Supercell"),
        ("atoms", "Atoms"),
        ("qgrid", "Q-point Grid"),
        ("branches", "Branches"),
        ("imaginary", "Imaginary Modes"),
        ("maxfreq", "Max Frequency"),
    ]

    LAYOUTS = {
        "default": FIELDS,
        "md": MD_FIELDS,
        "neb": NEB_FIELDS,
        "charge": CHARGE_FIELDS,
        "phonon": PHONON_FIELDS,
        "editor": EDITOR_FIELDS,
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, padding=SPACING.md)
        self.setFixedHeight(METRICS.info_bar_height)

        row = QHBoxLayout()
        row.setContentsMargins(SPACING.sm, 0, 0, 0)
        row.setSpacing(SPACING.xxl)
        self._row = row

        self._fields: dict[str, InfoField] = {}
        self._layout_name = "default"
        self._build_fields(self.FIELDS)

        row.addStretch(1)

        self._details = QPushButton("ⓘ")
        self._details.setObjectName("ToolButton")
        self._details.setFixedSize(30, 30)
        self._details.setCursor(Qt.PointingHandCursor)
        self._details.setToolTip("Calculation details")
        self._details.clicked.connect(self.detailsRequested)
        row.addWidget(self._details)

        self.body().addLayout(row)

    def _build_fields(self, spec: list[tuple[str, str]]) -> None:
        # Called only during construction, before the stretch and the details
        # button exist, so plain appends are in the right place.
        for key, caption in spec:
            field = InfoField(caption)
            self._fields[key] = field
            self._row.addWidget(field)

    # -- layout ------------------------------------------------------------

    def set_layout(self, name: str) -> None:
        """Switch to a named field set, rebuilding the strip if it changed."""
        if name == self._layout_name:
            return
        spec = self.LAYOUTS.get(name)
        if spec is None:
            return
        self._layout_name = name

        for field in self._fields.values():
            self._row.removeWidget(field)
            field.setParent(None)
            field.deleteLater()
        self._fields.clear()

        for index, (key, caption) in enumerate(spec):
            field = InfoField(caption)
            self._fields[key] = field
            self._row.insertWidget(index, field)
        self.clear()

    def current_layout(self) -> str:
        return self._layout_name

    def set_fields(
        self, values: dict[str, str], colors: dict[str, str] | None = None
    ) -> None:
        """Populate the current layout from a page-supplied mapping."""
        colors = colors or {}
        for key, field in self._fields.items():
            text = values.get(key, "—")
            field.set_value(
                text or "—", muted=text in ("", "—"), color=colors.get(key)
            )

    # -- population --------------------------------------------------------

    def clear(self) -> None:
        for field in self._fields.values():
            field.set_value("—", muted=True)

    def update_from(self, calculation: Calculation | None) -> None:
        if self._layout_name != "default":
            return
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

        # The gap needs the full EIGENVAL, which the Bands page reads lazily
        # in the background — reset to unknown here and let set_gap() below
        # fill it in once (and if) that data arrives, rather than showing a
        # previous calculation's stale value in the meantime.
        self._fields["gap"].set_value("—", muted=True)

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

    def set_gap(self, text: str, tooltip: str = "") -> None:
        """Called by the Bands page once EIGENVAL (and a Fermi level) are
        actually available — the gap can't be known at discovery time.

        "Metallic" is a determined fact, not a missing value, so only a
        literal "—" is shown muted.
        """
        field = self._fields.get("gap")
        if field is not None:
            field.set_value(text, tooltip=tooltip, muted=text == "—")

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
