"""Structure information and the calculation's files, on the right-hand side."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ....models.calculation import Calculation
from ....models.files import FileGroup, FileStatus
from ....models.structure import StructureData
from ....services.symmetry import space_group_label
from ...theme import PALETTE, RADIUS, SPACING, TYPE
from ...widgets.primitives import Card, Label as ThemedLabel

STATUS_GLYPH = {
    FileStatus.FOUND: ("✓", PALETTE.success),
    FileStatus.MISSING: ("○", PALETTE.text_tertiary),
    FileStatus.MALFORMED: ("⚠", PALETTE.warning),
    FileStatus.UNREADABLE: ("⚠", PALETTE.warning),
    FileStatus.INCONSISTENT: ("⚠", PALETTE.warning),
}


class InfoRow(QWidget):
    """A caption on the left, a value on the right, with a hairline beneath.

    A value too long for the space beside its caption wraps onto further
    lines rather than being cut off. The panel has a fixed width by design
    (so the viewer gets every spare pixel), which means "make the window
    bigger" can never reveal a truncated value -- so nothing may be
    truncated. Wrapping keeps every character readable at any width; the
    row simply grows taller.
    """

    def __init__(
        self, caption: str, value: str = "—", parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        row = QHBoxLayout()
        row.setContentsMargins(0, 6, 0, 6)
        row.setSpacing(SPACING.md)
        self._caption = QLabel(caption)
        self._caption.setStyleSheet(f"color: {PALETTE.text_secondary};")
        self._caption.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        row.addWidget(self._caption, 0)

        self._value = QLabel(value)
        self._value.setStyleSheet(f"color: {PALETTE.text}; font-weight: 500;")
        self._value.setAlignment(Qt.AlignRight | Qt.AlignTop)
        self._value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._value.setWordWrap(True)
        # Ignored horizontally so the *text* never dictates the row's width
        # -- a QLabel's minimum size is otherwise its full unwrapped text,
        # which is exactly what pushed the whole panel past the window edge
        # on a long POSCAR comment. Given only the space left over, the label
        # wraps to fit it.
        self._value.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._value.setMinimumWidth(0)
        row.addWidget(self._value, 1)
        layout.addLayout(row)

        rule = QWidget()
        rule.setFixedHeight(1)
        rule.setStyleSheet(f"background: {PALETTE.border};")
        layout.addWidget(rule)

    def set_value(self, text: str, muted: bool = False) -> None:
        colour = PALETTE.text_tertiary if muted else PALETTE.text
        self._value.setStyleSheet(f"color: {colour}; font-weight: 500;")
        self._value.setText(text)


class FileLink(QWidget):
    """A file name plus its status; clicking opens it on the Inputs page."""

    clicked = Signal(str)

    def __init__(self, role_key: str, filename: str, status: FileStatus,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._role_key = role_key
        self._enabled = status is not FileStatus.MISSING
        self.setFixedHeight(30)
        if self._enabled:
            self.setCursor(Qt.PointingHandCursor)

        row = QHBoxLayout(self)
        row.setContentsMargins(SPACING.sm, 0, SPACING.sm, 0)
        row.setSpacing(SPACING.sm)

        icon = QLabel("🗎")
        icon.setStyleSheet(f"color: {PALETTE.text_tertiary}; font-size: 11px;")
        row.addWidget(icon)

        name = QLabel(filename)
        colour = PALETTE.text if self._enabled else PALETTE.disabled_text
        name.setStyleSheet(f"color: {colour}; font-size: {TYPE.body}px;")
        row.addWidget(name, 1)

        glyph, glyph_colour = STATUS_GLYPH.get(status, ("○", PALETTE.text_tertiary))
        badge = QLabel(glyph)
        badge.setStyleSheet(f"color: {glyph_colour}; font-size: 12px;")
        row.addWidget(badge)

        chevron = QLabel("›")
        chevron.setStyleSheet(f"color: {PALETTE.text_tertiary};")
        row.addWidget(chevron)

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._enabled:
            self.setStyleSheet(
                f"FileLink {{ background: {PALETTE.surface_hover};"
                f" border-radius: {RADIUS.sm}px; }}"
            )
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.setStyleSheet("")
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._enabled:
            self.clicked.emit(self._role_key)
        super().mousePressEvent(event)


class StructureInfoPanel(QWidget):
    """Structure Information, Lattice Parameters, Lattice Angles and files."""

    fileRequested = Signal(str)

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
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING.md)
        scroll.setWidget(content)

        # --- structure information -----------------------------------------
        info_card = Card()
        info_body = info_card.body()
        info_body.addWidget(ThemedLabel("Structure Information", "CardTitle"))

        self._rows: dict[str, InfoRow] = {}
        for key, caption in (
            ("system", "System"),
            ("space_group", "Space Group"),
            ("lattice", "Lattice"),
        ):
            row = InfoRow(caption)
            self._rows[key] = row
            info_body.addWidget(row)

        info_body.addSpacing(SPACING.sm)
        info_body.addWidget(ThemedLabel("Lattice Parameters", "CardTitle"))
        for key, caption in (("a", "a"), ("b", "b"), ("c", "c")):
            row = InfoRow(caption)
            self._rows[key] = row
            info_body.addWidget(row)

        info_body.addSpacing(SPACING.sm)
        info_body.addWidget(ThemedLabel("Lattice Angles", "CardTitle"))
        for key, caption in (("ab", "a-b"), ("bc", "b-c"), ("ca", "c-a")):
            row = InfoRow(caption)
            self._rows[key] = row
            info_body.addWidget(row)

        layout.addWidget(info_card)

        # --- selected atom ---------------------------------------------------
        self._atom_card = Card()
        atom_body = self._atom_card.body()
        atom_body.addWidget(ThemedLabel("Selected Atom", "CardTitle"))
        self._atom_title = QLabel("—")
        self._atom_title.setStyleSheet("font-weight: 600;")
        atom_body.addWidget(self._atom_title)
        for key, caption in (
            ("element", "Element"),
            ("frac", "Fractional"),
            ("cart", "Cartesian (Å)"),
            ("wyckoff", "Wyckoff"),
            ("equivalents", "Equivalent atoms"),
            ("moment", "Moment (μB)"),
            ("spin", "S(x, y, z) μB"),
            ("displacement", "Relaxed by"),
        ):
            row = InfoRow(caption)
            self._rows[f"atom_{key}"] = row
            atom_body.addWidget(row)
        # The last four only mean something when the data behind them
        # exists; hidden otherwise rather than shown as a column of dashes.
        for key in ("wyckoff", "equivalents", "moment", "spin", "displacement"):
            self._rows[f"atom_{key}"].hide()
        self._atom_card.hide()
        layout.addWidget(self._atom_card)

        # --- relaxation --------------------------------------------------------
        self._relax_card = Card()
        relax_body = self._relax_card.body()
        relax_body.addWidget(ThemedLabel("Relaxation (POSCAR → CONTCAR)", "CardTitle"))
        for key, caption in (
            ("relax_mean", "Mean"),
            ("relax_rms", "RMS"),
        ):
            row = InfoRow(caption)
            self._rows[key] = row
            relax_body.addWidget(row)
        relax_body.addSpacing(SPACING.xs)
        movers_title = QLabel("Most displaced")
        movers_title.setStyleSheet(
            f"color: {PALETTE.text_secondary}; font-size: {TYPE.small}px;"
        )
        relax_body.addWidget(movers_title)
        self._movers_layout = QVBoxLayout()
        self._movers_layout.setContentsMargins(0, 0, 0, 0)
        self._movers_layout.setSpacing(2)
        relax_body.addLayout(self._movers_layout)
        self._mover_labels: list[QLabel] = []
        self._relax_card.hide()
        layout.addWidget(self._relax_card)

        # --- magnetism -----------------------------------------------------------
        self._mag_card = Card()
        mag_body = self._mag_card.body()
        mag_body.addWidget(ThemedLabel("Magnetization", "CardTitle"))
        for key, caption in (
            ("mag_kind", "Type"),
            ("mag_net", "Net moment"),
            ("mag_net_components", "Net S(x, y, z)"),
            ("mag_total", "Σ |m| per atom"),
            ("mag_largest", "Largest |m|"),
        ):
            row = InfoRow(caption)
            self._rows[key] = row
            mag_body.addWidget(row)
        self._mag_card.hide()
        layout.addWidget(self._mag_card)

        # --- measurement -------------------------------------------------------
        self._measure_card = Card()
        measure_body = self._measure_card.body()
        measure_body.addWidget(ThemedLabel("Measurement", "CardTitle"))
        self._measure_value = QLabel("—")
        self._measure_value.setStyleSheet("font-weight: 600; font-size: 20px;")
        measure_body.addWidget(self._measure_value)
        self._measure_detail = QLabel("")
        self._measure_detail.setStyleSheet(f"color: {PALETTE.text_secondary};")
        measure_body.addWidget(self._measure_detail)
        hint = QLabel("Right-click or Esc to clear")
        hint.setStyleSheet(f"color: {PALETTE.text_tertiary}; font-size: {TYPE.small}px;")
        measure_body.addWidget(hint)
        self._measure_card.hide()
        layout.addWidget(self._measure_card)

        # --- input files -------------------------------------------------------
        self._files_card = Card()
        self._files_body = self._files_card.body()
        self._files_body.addWidget(ThemedLabel("Input Files", "CardTitle"))
        self._file_links: list[FileLink] = []
        layout.addWidget(self._files_card)

        layout.addStretch(1)

    # -- population -----------------------------------------------------------------

    def clear(self) -> None:
        for row in self._rows.values():
            row.set_value("—", muted=True)
        self._atom_card.hide()
        self._measure_card.hide()
        self._relax_card.hide()
        self._mag_card.hide()
        self._clear_files()

    def update_structure(
        self, structure: StructureData | None, calculation: Calculation | None
    ) -> None:
        if structure is None:
            self.clear()
            return

        system = None
        if calculation is not None and calculation.metadata.system is not None:
            system = str(calculation.metadata.system.value)
        self._rows["system"].set_value(system or structure.formula or "—")

        self._rows["space_group"].set_value(
            space_group_label(structure),
            muted=structure.space_group_number is None,
        )
        self._rows["lattice"].set_value(
            structure.crystal_system or "—", muted=structure.crystal_system is None
        )

        a, b, c = structure.lattice_parameters
        for key, value in (("a", a), ("b", b), ("c", c)):
            self._rows[key].set_value(f"{value:.4f} Å")

        alpha, beta, gamma = structure.lattice_angles
        for key, value in (("ab", gamma), ("bc", alpha), ("ca", beta)):
            self._rows[key].set_value(f"{value:.2f}°")

    def show_atom(
        self,
        structure: StructureData | None,
        index: int | None,
        *,
        wyckoff: str | None = None,
        equivalents: list[int] | None = None,
        moment: float | None = None,
        spin: tuple[float, float, float] | None = None,
        displacement: float | None = None,
    ) -> None:
        """The selected atom, plus whichever extras the page has for it.

        Each keyword is optional and its row is shown only when given, so a
        calculation without magnetization or without a CONTCAR simply has a
        shorter card rather than rows of dashes.
        """
        if structure is None or index is None or index >= structure.n_atoms:
            self._atom_card.hide()
            return

        self._atom_title.setText(f"Atom {index + 1}")
        self._rows["atom_element"].set_value(structure.species[index])
        frac = structure.frac_coords[index]
        cart = structure.cart_coords[index]
        self._rows["atom_frac"].set_value(
            f"{frac[0]:.4f}, {frac[1]:.4f}, {frac[2]:.4f}"
        )
        self._rows["atom_cart"].set_value(
            f"{cart[0]:.3f}, {cart[1]:.3f}, {cart[2]:.3f}"
        )

        row = self._rows["atom_wyckoff"]
        row.setVisible(wyckoff is not None)
        if wyckoff is not None:
            row.set_value(wyckoff)

        row = self._rows["atom_equivalents"]
        row.setVisible(equivalents is not None)
        if equivalents is not None:
            others = [i for i in equivalents if i != index]
            if not others:
                row.set_value("none (unique site)", muted=True)
            else:
                # The row wraps, so a dozen indices fit; beyond that, say how
                # many are left rather than trailing off.
                shown = ", ".join(str(i + 1) for i in others[:12])
                if len(others) > 12:
                    shown += f" and {len(others) - 12} more"
                row.set_value(f"{len(others)} atoms: {shown}")

        row = self._rows["atom_moment"]
        row.setVisible(moment is not None)
        if moment is not None:
            row.set_value(f"{moment:+.3f}")

        row = self._rows["atom_spin"]
        row.setVisible(spin is not None)
        if spin is not None:
            row.set_value(f"{spin[0]:+.3f}, {spin[1]:+.3f}, {spin[2]:+.3f}")

        row = self._rows["atom_displacement"]
        row.setVisible(displacement is not None)
        if displacement is not None:
            row.set_value(f"{displacement:.4f} Å")

        self._atom_card.show()

    def show_relaxation(self, result) -> None:
        """``result`` is a services.relaxation.RelaxationResult, or None."""
        for label in self._mover_labels:
            self._movers_layout.removeWidget(label)
            label.hide()
            label.setParent(None)
            label.deleteLater()
        self._mover_labels = []
        if result is None or result.n_atoms == 0:
            self._relax_card.hide()
            return

        self._rows["relax_mean"].set_value(f"{result.mean_displacement:.4f} Å")
        self._rows["relax_rms"].set_value(f"{result.rms_displacement:.4f} Å")
        magnitudes = result.magnitudes
        for index in result.ranked(5):
            label = QLabel(
                f"{result.final.species[index]} #{index + 1}"
                f"    {magnitudes[index]:.4f} Å"
            )
            label.setStyleSheet(f"font-size: {TYPE.small}px; color: {PALETTE.text};")
            self._movers_layout.addWidget(label)
            self._mover_labels.append(label)
        self._relax_card.show()

    def show_magnetization(self, moments) -> None:
        """``moments`` is a parsers.outcar.MagneticMoments, or None."""
        if moments is None or moments.n_atoms == 0:
            self._mag_card.hide()
            return
        largest = int(moments.magnitudes.argmax())
        self._rows["mag_kind"].set_value(
            "Collinear (ISPIN=2)" if moments.collinear
            else "Non-collinear (spin-orbit)"
        )
        # The net moment is the vector sum; Σ|m| is the sum of magnitudes.
        # They differ exactly where it matters: an antiferromagnet has a
        # large Σ|m| and a net of nearly zero, and reporting only one of the
        # two would describe it wrongly either way.
        self._rows["mag_net"].set_value(f"{moments.net_magnitude:.3f} μB")
        net_row = self._rows["mag_net_components"]
        net_row.setVisible(not moments.collinear)
        if not moments.collinear:
            net = moments.total_vector
            net_row.set_value(
                f"{net[0]:+.3f}, {net[1]:+.3f}, {net[2]:+.3f}"
            )
        self._rows["mag_total"].set_value(
            f"{float(moments.magnitudes.sum()):.3f} μB"
        )
        self._rows["mag_largest"].set_value(
            f"{moments.magnitudes[largest]:.3f} μB  (atom {largest + 1})"
        )
        self._mag_card.show()

    def show_measurement(self, measurement) -> None:
        if measurement is None:
            self._measure_card.hide()
            return

        if measurement.kind == "distance":
            a, b = measurement.symbols
            self._measure_value.setText(f"{measurement.value:.4f} Å")
            self._measure_detail.setText(f"{a} – {b}")
        else:
            a, b, c = measurement.symbols
            self._measure_value.setText(f"{measurement.value:.2f}°")
            self._measure_detail.setText(f"{a} – {b} – {c}")
        self._measure_card.show()

    def update_files(self, calculation: Calculation | None) -> None:
        self._clear_files()
        if calculation is None:
            self._files_card.hide()
            return

        entries = [
            entry
            for entry in calculation.index.in_group(FileGroup.VASP)
            if entry.role.key in ("poscar", "contcar", "incar", "kpoints", "outcar")
        ]
        if not entries:
            self._files_card.hide()
            return

        for entry in entries:
            link = FileLink(entry.role.key, entry.role.filename, entry.status)
            link.clicked.connect(self.fileRequested)
            self._files_body.addWidget(link)
            self._file_links.append(link)
        self._files_card.show()

    def _clear_files(self) -> None:
        for link in self._file_links:
            link.setParent(None)
            link.deleteLater()
        self._file_links.clear()
