"""Per-element orbital selection for the band projection.

A single flat list of orbital checkboxes can only ask one question of the
whole cell: "show me pz". The question a multi-element system actually poses
is per element — in a Te-doped nanotube the interesting comparison is carbon
*p* against tellurium *d*, and a shared filter cannot express it.

So the control is a small matrix: one row per element, chips for the orbital
shells that element's PROCAR actually carries. Chips rather than checkboxes
because a row of three or four short labels reads as a set of options at a
glance, where a grid of checkboxes reads as a form.

Shells (s/p/d/f) rather than individual orbitals are the default rung: they
are how the question is usually asked, and they keep a row to four chips
instead of nine. The individual orbitals stay reachable through the row's
disclosure arrow for the cases that need pz specifically.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ....services import appearance
from ...theme import PALETTE, RADIUS, SPACING, TYPE

# Orbital chips per row when a row is expanded. Four keeps even a full
# nine-orbital PROCAR to a few tidy rows inside the sidebar's fixed width,
# rather than one long line that overflows it.
DETAIL_COLUMNS = 4

# Which orbital names belong to which shell, in VASP's PROCAR column order.
SHELLS: list[tuple[str, tuple[str, ...]]] = [
    ("s", ("s",)),
    ("p", ("py", "pz", "px")),
    ("d", ("dxy", "dyz", "dz2", "dxz", "x2-y2", "dx2")),
    ("f", ("f-3", "f-2", "f-1", "f0", "f1", "f2", "f3")),
]


def shells_present(orbitals: list[str]) -> list[tuple[str, list[str]]]:
    """The shells this PROCAR actually carries, each with its own columns."""
    available = set(orbitals)
    found: list[tuple[str, list[str]]] = []
    for name, members in SHELLS:
        columns = [o for o in members if o in available]
        if columns:
            found.append((name, columns))
    return found


class Chip(QPushButton):
    """A small toggleable pill."""

    def __init__(
        self, text: str, color: str | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(text, parent)
        self._color = color or PALETTE.accent
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(22)
        self.setMinimumWidth(28)
        self._apply_style()
        self.toggled.connect(lambda _c: self._apply_style())

    def _apply_style(self) -> None:
        if self.isChecked():
            self.setStyleSheet(
                f"QPushButton {{ background: {self._color}; color: #ffffff;"
                f" border: 1px solid {self._color}; border-radius: {RADIUS.sm}px;"
                f" padding: 0 8px; font-size: {TYPE.small}px; font-weight: 600; }}"
            )
        else:
            self.setStyleSheet(
                f"QPushButton {{ background: transparent;"
                f" color: {PALETTE.text_secondary};"
                f" border: 1px solid {PALETTE.border_strong};"
                f" border-radius: {RADIUS.sm}px; padding: 0 8px;"
                f" font-size: {TYPE.small}px; }}"
            )


class ElementRow(QWidget):
    """One element's shell chips, with an optional per-orbital expansion."""

    changed = Signal()

    def __init__(
        self,
        element: str,
        orbitals: list[str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._element = element
        self._orbitals = list(orbitals)
        self._expanded = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(SPACING.xs)

        self._disclosure = QPushButton("›")
        self._disclosure.setFlat(True)
        self._disclosure.setFixedSize(16, 22)
        self._disclosure.setCursor(Qt.PointingHandCursor)
        self._disclosure.setToolTip(f"Individual {element} orbitals")
        self._disclosure.clicked.connect(self._toggle_expanded)
        self._disclosure.setStyleSheet(
            f"QPushButton {{ border: none; color: {PALETTE.text_tertiary};"
            f" font-size: {TYPE.body}px; }}"
        )
        top.addWidget(self._disclosure)

        swatch = QLabel()
        swatch.setFixedSize(8, 8)
        swatch.setStyleSheet(
            f"background: {appearance.effective_color(element)};"
            f" border-radius: 4px;"
        )
        top.addWidget(swatch)

        name = QLabel(element)
        name.setFixedWidth(30)
        name.setStyleSheet(f"font-size: {TYPE.small}px; font-weight: 600;")
        top.addWidget(name)

        self._shell_chips: dict[str, Chip] = {}
        self._shell_members: dict[str, list[str]] = {}
        for shell, columns in shells_present(orbitals):
            chip = Chip(shell)
            chip.setChecked(True)
            # clicked, not toggled: a shell that is only *partly* selected
            # still shows as checked, so reacting to the toggle state would
            # make a click on it clear the shell when the obvious intent is
            # to complete it.
            chip.clicked.connect(self._on_shell_clicked)
            top.addWidget(chip)
            self._shell_chips[shell] = chip
            self._shell_members[shell] = columns
        top.addStretch(1)
        outer.addLayout(top)

        # Individual orbitals, hidden until the row is expanded. Laid out as a
        # grid rather than one long row: a d-shell alone is nine chips, which
        # in a single line overflows the sidebar's fixed width and forces a
        # horizontal scrollbar across the whole panel.
        self._detail = QWidget()
        detail_layout = QGridLayout(self._detail)
        detail_layout.setContentsMargins(22, 0, 0, 0)
        detail_layout.setHorizontalSpacing(SPACING.xs)
        detail_layout.setVerticalSpacing(2)
        self._orbital_chips: dict[str, Chip] = {}
        for index, orbital in enumerate(orbitals):
            chip = Chip(orbital)
            chip.setChecked(True)
            chip.toggled.connect(self._on_orbital_toggled)
            detail_layout.addWidget(
                chip, index // DETAIL_COLUMNS, index % DETAIL_COLUMNS
            )
            self._orbital_chips[orbital] = chip
        detail_layout.setColumnStretch(DETAIL_COLUMNS, 1)
        self._detail.hide()
        outer.addWidget(self._detail)

        self._suppress = False

    @property
    def element(self) -> str:
        return self._element

    def _toggle_expanded(self) -> None:
        self._expanded = not self._expanded
        self._detail.setVisible(self._expanded)
        self._disclosure.setText("⌄" if self._expanded else "›")

    def _on_shell_clicked(self) -> None:
        """A shell chip fills its orbitals, or clears them if already full."""
        if self._suppress:
            return
        chip = self.sender()
        shell = next((s for s, c in self._shell_chips.items() if c is chip), None)
        if shell is None:
            return
        members = self._shell_members[shell]
        was_full = all(self._orbital_chips[o].isChecked() for o in members)
        target = not was_full

        self._suppress = True
        try:
            for orbital in members:
                self._orbital_chips[orbital].setChecked(target)
            chip.setChecked(target)
        finally:
            self._suppress = False
        self.changed.emit()

    def _on_orbital_toggled(self, _checked: bool) -> None:
        """Individual orbitals drive their shell chip back the other way, so
        the collapsed row still tells the truth about what is selected."""
        if self._suppress:
            return
        self._suppress = True
        try:
            for shell, columns in self._shell_members.items():
                states = [self._orbital_chips[o].isChecked() for o in columns]
                self._shell_chips[shell].setChecked(any(states))
        finally:
            self._suppress = False
        self.changed.emit()

    def selected(self) -> list[str]:
        """Selected orbital columns, in PROCAR order."""
        return [o for o in self._orbitals if self._orbital_chips[o].isChecked()]

    def set_all(self, checked: bool) -> None:
        self._suppress = True
        try:
            for chip in self._orbital_chips.values():
                chip.setChecked(checked)
            for chip in self._shell_chips.values():
                chip.setChecked(checked)
        finally:
            self._suppress = False
        self.changed.emit()


class OrbitalShellPicker(QWidget):
    """A single, standalone row of shell chips (with the same per-orbital
    disclosure as :class:`ElementRow`), for a selection that isn't keyed by
    element — the Atom Projection tool asks "which orbitals, on this
    specific handful of atoms", one answer for the whole selection rather
    than one per element, so it needs the shell-chip control without the
    element swatch/label ElementRow always draws alongside it.
    """

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._orbitals: list[str] = []
        self._expanded = False
        self._suppress = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(SPACING.xs)

        self._disclosure = QPushButton("›")
        self._disclosure.setFlat(True)
        self._disclosure.setFixedSize(16, 22)
        self._disclosure.setCursor(Qt.PointingHandCursor)
        self._disclosure.setToolTip("Individual orbitals")
        self._disclosure.clicked.connect(self._toggle_expanded)
        self._disclosure.setStyleSheet(
            f"QPushButton {{ border: none; color: {PALETTE.text_tertiary};"
            f" font-size: {TYPE.body}px; }}"
        )
        top.addWidget(self._disclosure)

        self._shell_row = QHBoxLayout()
        self._shell_row.setContentsMargins(0, 0, 0, 0)
        self._shell_row.setSpacing(SPACING.xs)
        top.addLayout(self._shell_row)
        top.addStretch(1)
        outer.addLayout(top)

        self._detail = QWidget()
        self._detail_layout = QGridLayout(self._detail)
        self._detail_layout.setContentsMargins(22, 0, 0, 0)
        self._detail_layout.setHorizontalSpacing(SPACING.xs)
        self._detail_layout.setVerticalSpacing(2)
        self._detail.hide()
        outer.addWidget(self._detail)

        self._shell_chips: dict[str, Chip] = {}
        self._shell_members: dict[str, list[str]] = {}
        self._orbital_chips: dict[str, Chip] = {}

    def populate(self, orbitals: list[str]) -> None:
        """Rebuild the chips for a (possibly new) orbital set — called
        whenever PROCAR's own orbital list becomes known or changes."""
        for chip in list(self._shell_chips.values()):
            self._shell_row.removeWidget(chip)
            chip.setParent(None)
            chip.deleteLater()
        for chip in list(self._orbital_chips.values()):
            self._detail_layout.removeWidget(chip)
            chip.setParent(None)
            chip.deleteLater()
        self._shell_chips.clear()
        self._shell_members.clear()
        self._orbital_chips.clear()
        self._orbitals = list(orbitals)

        for shell, columns in shells_present(orbitals):
            chip = Chip(shell)
            chip.setChecked(True)
            chip.clicked.connect(self._on_shell_clicked)
            self._shell_row.addWidget(chip)
            self._shell_chips[shell] = chip
            self._shell_members[shell] = columns

        for index, orbital in enumerate(orbitals):
            chip = Chip(orbital)
            chip.setChecked(True)
            chip.toggled.connect(self._on_orbital_toggled)
            self._detail_layout.addWidget(
                chip, index // DETAIL_COLUMNS, index % DETAIL_COLUMNS
            )
            self._orbital_chips[orbital] = chip
        self._detail_layout.setColumnStretch(DETAIL_COLUMNS, 1)

    def _toggle_expanded(self) -> None:
        self._expanded = not self._expanded
        self._detail.setVisible(self._expanded)
        self._disclosure.setText("⌄" if self._expanded else "›")

    def _on_shell_clicked(self) -> None:
        if self._suppress:
            return
        chip = self.sender()
        shell = next((s for s, c in self._shell_chips.items() if c is chip), None)
        if shell is None:
            return
        members = self._shell_members[shell]
        was_full = all(self._orbital_chips[o].isChecked() for o in members)
        target = not was_full

        self._suppress = True
        try:
            for orbital in members:
                self._orbital_chips[orbital].setChecked(target)
            chip.setChecked(target)
        finally:
            self._suppress = False
        self.changed.emit()

    def _on_orbital_toggled(self, _checked: bool) -> None:
        if self._suppress:
            return
        self._suppress = True
        try:
            for shell, columns in self._shell_members.items():
                states = [self._orbital_chips[o].isChecked() for o in columns]
                self._shell_chips[shell].setChecked(any(states))
        finally:
            self._suppress = False
        self.changed.emit()

    def selected(self) -> list[str]:
        """Selected orbital columns, in PROCAR order."""
        return [o for o in self._orbitals if self._orbital_chips[o].isChecked()]

    def is_full_selection(self) -> bool:
        return bool(self._orbitals) and set(self.selected()) == set(self._orbitals)

    def has_orbitals(self) -> bool:
        """Whether :meth:`populate` has been given PROCAR's orbital list yet."""
        return bool(self._orbitals)


class OrbitalMatrix(QWidget):
    """One :class:`ElementRow` per element in the structure."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING.lg, SPACING.xs, 0, 0)
        layout.setSpacing(SPACING.xs)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(SPACING.xs)
        self._all = QPushButton("All")
        self._none = QPushButton("None")
        for button in (self._all, self._none):
            button.setFlat(True)
            button.setFixedHeight(20)
            button.setCursor(Qt.PointingHandCursor)
            button.setStyleSheet(
                f"QPushButton {{ border: none; color: {PALETTE.accent};"
                f" font-size: {TYPE.small}px; }}"
            )
            controls.addWidget(button)
        controls.addStretch(1)
        self._all.clicked.connect(lambda: self._set_all(True))
        self._none.clicked.connect(lambda: self._set_all(False))
        layout.addLayout(controls)

        self._rows_layout = QVBoxLayout()
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)
        layout.addLayout(self._rows_layout)

        self._rows: list[ElementRow] = []

    def populate(self, elements: list[str], orbitals: list[str]) -> None:
        self.clear()
        for element in elements:
            row = ElementRow(element, orbitals)
            row.changed.connect(self.changed)
            self._rows_layout.addWidget(row)
            self._rows.append(row)

    def clear(self) -> None:
        for row in self._rows:
            self._rows_layout.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
        self._rows.clear()

    def _set_all(self, checked: bool) -> None:
        for row in self._rows:
            row.set_all(checked)

    def selection(self) -> dict[str, list[str]]:
        """{element: [orbital, ...]} for every element with anything chosen."""
        return {row.element: row.selected() for row in self._rows if row.selected()}

    def is_full_selection(self, orbitals: list[str]) -> bool:
        """True when every element has every orbital chosen — the plain
        by-element view, with no filtering to describe."""
        if not self._rows:
            return True
        expected = set(orbitals)
        return all(set(row.selected()) == expected for row in self._rows)

    def has_any(self) -> bool:
        return any(row.selected() for row in self._rows)
