"""Structure information and the calculation's files, on the right-hand side."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
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
    """A caption on the left, a value on the right, with a hairline beneath."""

    def __init__(
        self, caption: str, value: str = "—", parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        row = QHBoxLayout()
        row.setContentsMargins(0, 6, 0, 6)
        self._caption = QLabel(caption)
        self._caption.setStyleSheet(f"color: {PALETTE.text_secondary};")
        row.addWidget(self._caption)
        row.addStretch(1)

        self._value = QLabel(value)
        self._value.setStyleSheet(f"color: {PALETTE.text}; font-weight: 500;")
        self._value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row.addWidget(self._value)
        layout.addLayout(row)

        rule = QWidget()
        rule.setFixedHeight(1)
        rule.setStyleSheet(f"background: {PALETTE.border};")
        layout.addWidget(rule)

    def set_value(self, text: str, muted: bool = False) -> None:
        self._value.setText(text)
        colour = PALETTE.text_tertiary if muted else PALETTE.text
        self._value.setStyleSheet(f"color: {colour}; font-weight: 500;")


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
        ):
            row = InfoRow(caption)
            self._rows[f"atom_{key}"] = row
            atom_body.addWidget(row)
        self._atom_card.hide()
        layout.addWidget(self._atom_card)

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

    def show_atom(self, structure: StructureData | None, index: int | None) -> None:
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
        self._atom_card.show()

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
