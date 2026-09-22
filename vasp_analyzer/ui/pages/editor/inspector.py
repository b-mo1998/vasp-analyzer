"""The Editor's right-hand column: selection, add-atom, supercell, operations.

These cards read from the working structure and emit intent; they never mutate
anything themselves. The page owns every change so that all of them go through
one place and therefore land in the undo history — a panel that quietly edited
the structure would be a change the user could not take back.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDoubleValidator, QGuiApplication, QIcon
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ....models.editable import EditableStructure
from ...style import icon_url
from ...theme import PALETTE, RADIUS, SPACING, TYPE
from ...widgets.primitives import Card
from ...widgets.segmented import SegmentedControl
from ...widgets.stepper import IntegerStepper

# Higher than the Structure page's read-only supercell preview (capped at 6):
# building a real editable supercell here is gated separately, by the actual
# resulting atom count (see StructureEditorPage._build_supercell), so the
# stepper itself only needs to keep the field from accepting nonsense.
MAX_SUPERCELL_REPEAT = 20

# Elements offered in the Add Atom picker beyond those already present.
COMMON_ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Hf", "Ta", "W",
    "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
]


def _caption(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(
        f"color: {PALETTE.text_secondary}; font-size: {TYPE.small}px;"
    )
    return label


def _coord_field(placeholder: str = "0.00000") -> QLineEdit:
    field = QLineEdit()
    field.setObjectName("ValueField")
    field.setPlaceholderText(placeholder)
    field.setValidator(QDoubleValidator())
    field.setFixedHeight(28)
    return field


class SelectionCard(Card):
    """What is selected, its exact coordinates, and what can be done to it."""

    coordinateEdited = Signal(int, int, float)   # structure row, axis, value
    moveRequested = Signal()
    deleteRequested = Signal()
    fixRequested = Signal(object)                # (3,) bool free-flags, or None to cycle
    duplicateRequested = Signal()
    coordinateModeChanged = Signal(bool)         # True = fractional

    # Beyond this many rows the per-atom coordinate list stops being a useful
    # inspector and becomes a second, worse atom table.
    MAX_ROWS = 12

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self._structure: EditableStructure | None = None
        self._selection: set[int] = set()
        self._fractional = True
        self._fields: list[tuple[int, list[QLineEdit]]] = []

        body = self.body()
        body.setSpacing(SPACING.sm)

        header = QHBoxLayout()
        title = QLabel("Selection")
        title.setObjectName("CardTitle")
        header.addWidget(title)
        header.addStretch(1)
        self._count = _caption("No selection")
        header.addWidget(self._count)
        body.addLayout(header)

        summary = QGridLayout()
        summary.setHorizontalSpacing(SPACING.lg)
        summary.setVerticalSpacing(2)
        summary.addWidget(_caption("Atom ID"), 0, 0)
        summary.addWidget(_caption("Element"), 0, 1)
        self._ids = QLabel("—")
        self._elements = QLabel("—")
        for widget in (self._ids, self._elements):
            widget.setStyleSheet(f"font-size: {TYPE.body}px; font-weight: 500;")
            widget.setWordWrap(True)
        summary.addWidget(self._ids, 1, 0)
        summary.addWidget(self._elements, 1, 1)
        body.addLayout(summary)

        # Only meaningful for two or more atoms — a single atom's "centre" is
        # just its own position, already shown below. Handy for measuring a
        # gap, placing a new atom at a midpoint, or just knowing where a
        # cluster of selected atoms sits.
        centroid_row = QHBoxLayout()
        centroid_row.setSpacing(SPACING.sm)
        self._centroid_caption = _caption("Center")
        centroid_row.addWidget(self._centroid_caption)
        self._centroid_value = QLabel("—")
        self._centroid_value.setStyleSheet(f"font-size: {TYPE.body}px; font-weight: 500;")
        # Wrapped so its full "x, y, z Å" width is not a hard floor on the
        # card, and through it on the whole sidebar — that one row is what
        # pushed the panel a couple of pixels past the fixed column width.
        # It has a whole row to itself in practice and still fits on one line.
        self._centroid_value.setWordWrap(True)
        centroid_row.addWidget(self._centroid_value, 1)
        self._centroid_copy = QPushButton("Copy")
        self._centroid_copy.setObjectName("ToolButton")
        self._centroid_copy.setCursor(Qt.PointingHandCursor)
        self._centroid_copy.setFixedHeight(24)
        self._centroid_copy.clicked.connect(self._copy_centroid)
        centroid_row.addWidget(self._centroid_copy)
        self._centroid_host = QWidget()
        self._centroid_host.setLayout(centroid_row)
        self._centroid_host.hide()
        body.addWidget(self._centroid_host)

        mode_row = QHBoxLayout()
        self._coord_title = _caption("Coordinates")
        mode_row.addWidget(self._coord_title)
        mode_row.addStretch(1)
        self._mode = SegmentedControl(["Frac", "Cart"])
        self._mode.currentChanged.connect(
            lambda index: self.coordinateModeChanged.emit(index == 0)
        )
        mode_row.addWidget(self._mode)
        body.addLayout(mode_row)

        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(SPACING.sm)
        self._grid.setVerticalSpacing(4)
        body.addWidget(self._grid_host)

        # A 2x2 grid rather than one row: "Grab and Move" is long enough that
        # four buttons abreast would either overflow the fixed sidebar width
        # or need font-shrinking tricks. Two columns give every button the
        # same headroom "Duplicate" already needed.
        actions = QGridLayout()
        actions.setSpacing(SPACING.sm)
        self._move = QPushButton("Grab and Move")
        self._delete = QPushButton("Delete")
        self._fix = QPushButton("Fix")
        self._duplicate = QPushButton("Duplicate")
        for index, (button, slot) in enumerate((
            (self._move, self.moveRequested),
            (self._delete, self.deleteRequested),
            (self._fix, None),
            (self._duplicate, self.duplicateRequested),
        )):
            button.setObjectName("ToolButton")
            button.setCursor(Qt.PointingHandCursor)
            button.setFixedHeight(30)
            actions.addWidget(button, index // 2, index % 2)
            if slot is not None:
                button.clicked.connect(slot.emit)
        self._fix.clicked.connect(lambda: self.fixRequested.emit(None))
        # Same lock glyph the atom table uses for a fixed row — one visual
        # vocabulary for "held in place" rather than two.
        self._fix.setIcon(QIcon(icon_url("lock.svg")))
        # Restrained rather than a filled red button: destructive, but it is
        # undoable and used often, so it should not shout.
        self._delete.setStyleSheet(f"QPushButton {{ color: {PALETTE.danger}; }}")
        body.addLayout(actions)

    def set_coordinate_mode(self, fractional: bool) -> None:
        self._fractional = fractional
        self._mode.setCurrentIndex(0 if fractional else 1)
        self.refresh()

    def set_state(self, structure: EditableStructure | None, selection: set[int]) -> None:
        self._structure = structure
        self._selection = set(selection)
        self.refresh()

    def refresh(self) -> None:
        structure = self._structure
        rows = sorted(r for r in self._selection if structure and r < structure.n_atoms)
        enabled = bool(rows)
        for button in (self._move, self._delete, self._fix, self._duplicate):
            button.setEnabled(enabled)

        self._clear_grid()
        if structure is None or not rows:
            self._count.setText("No selection")
            self._ids.setText("—")
            self._elements.setText("—")
            self._coord_title.setText("Coordinates")
            self._centroid_host.hide()
            return

        self._count.setText(
            f"{len(rows)} atom{'s' if len(rows) != 1 else ''} selected"
        )
        self._ids.setText(", ".join(str(r + 1) for r in rows[:8]) + ("…" if len(rows) > 8 else ""))

        # A mixed selection is reported as a breakdown rather than as whichever
        # element happened to come first, which would be simply wrong.
        counts: dict[str, int] = {}
        for row in rows:
            symbol = structure.species[row]
            counts[symbol] = counts.get(symbol, 0) + 1
        if len(counts) == 1:
            self._elements.setText(next(iter(counts)))
        else:
            self._elements.setText(
                ", ".join(f"{symbol} ×{n}" for symbol, n in counts.items())
            )

        self._coord_title.setText(
            f"Coordinates ({'Fractional' if self._fractional else 'Cartesian'})"
        )
        self._build_grid(structure, rows)

        if rows:
            # Shown for a single atom too, not just 2+: "the centre of the
            # selected atoms" is a well-defined answer at N=1 as well (that
            # atom's own position), and the alternative — the readout
            # appearing only once a second atom joins the selection — reads
            # as the tool being unavailable rather than simply agreeing with
            # the one coordinate already visible below.
            coords = structure.frac_coords if self._fractional else structure.cart_coords
            centroid = coords[rows].mean(axis=0)
            unit = "" if self._fractional else " Å"
            # Abbreviated where the Coordinates row below spells the same
            # mode out in full: three five-decimal numbers and a Copy button
            # need the width more than a repeated word does.
            self._centroid_caption.setText(
                f"Center ({'frac' if self._fractional else 'cart'})"
            )
            self._centroid_value.setText(
                f"{centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}{unit}"
            )
            self._centroid_host.show()
        else:
            self._centroid_host.hide()

    def _copy_centroid(self) -> None:
        text = self._centroid_value.text()
        if text and text != "—":
            QGuiApplication.clipboard().setText(text)

    def _clear_grid(self) -> None:
        self._fields.clear()
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # Muted before it is torn down. A QLineEdit emits
                # editingFinished when it loses focus, and being reparented
                # away counts — so simply rebuilding this grid (which happens
                # on every selection change) fired _emit_edit for whichever
                # coordinate field the user had last clicked into. That
                # committed a no-op "Edit atom N coordinate" to the undo
                # history and flagged the structure Modified without anyone
                # having typed anything.
                widget.blockSignals(True)
                # Hidden, and focus explicitly released, before the
                # reparent — not an optional pair of extra steps.
                # setParent(None) makes any widget a top-level window of its
                # own; a widget that still has both real focus and its old
                # "visible" flag at that instant becomes a genuine floating,
                # focused, visible top-level window in its own right, and
                # doing that to the exact field the user was just typing into
                # stole OS-level window activation away from the whole
                # application until deleteLater() actually ran on the next
                # event-loop turn. While that window silently held
                # activation, no shortcut anywhere in the app — Undo/Redo
                # included — had anywhere to dispatch to, since Qt shortcut
                # contexts other than ApplicationShortcut all require their
                # own window to be the active one first.
                if widget.hasFocus():
                    widget.clearFocus()
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()

    def _build_grid(self, structure: EditableStructure, rows: list[int]) -> None:
        for column, axis in enumerate("xyz"):
            self._grid.addWidget(_caption(axis), 0, column + 1, alignment=Qt.AlignHCenter)

        shown = rows[: self.MAX_ROWS]
        coords = structure.frac_coords if self._fractional else structure.cart_coords
        for line, row in enumerate(shown, start=1):
            self._grid.addWidget(_caption(str(row + 1)), line, 0)
            fields: list[QLineEdit] = []
            for axis in range(3):
                field = _coord_field()
                shown_text = f"{float(coords[row][axis]):.5f}"
                field.setText(shown_text)
                # What the field was populated with, so _emit_edit can tell a
                # real edit from a field the user merely clicked through.
                field.setProperty("originalText", shown_text)
                field.editingFinished.connect(
                    lambda r=row, a=axis, f=field: self._emit_edit(r, a, f)
                )
                self._grid.addWidget(field, line, axis + 1)
                fields.append(field)
            self._fields.append((row, fields))

        if len(rows) > self.MAX_ROWS:
            more = _caption(f"… and {len(rows) - self.MAX_ROWS} more")
            self._grid.addWidget(more, len(shown) + 1, 0, 1, 4)

    def _emit_edit(self, row: int, axis: int, field: QLineEdit) -> None:
        text = field.text()
        # Nothing was typed, so nothing happened. editingFinished fires on
        # every focus change, and treating a click-through as an edit filled
        # the undo history with no-ops and marked an untouched structure
        # Modified.
        if text == field.property("originalText"):
            return
        try:
            value = float(text)
        except ValueError:
            self.refresh()
            return
        self.coordinateEdited.emit(row, axis, value)


class AddAtomCard(Card):
    """Insert one atom at an exact coordinate."""

    insertRequested = Signal(str, object, bool)   # element, (3,) coords, fractional

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        body = self.body()
        body.setSpacing(SPACING.sm)

        title = QLabel("Add Atom")
        title.setObjectName("CardTitle")
        body.addWidget(title)

        row = QHBoxLayout()
        row.addWidget(_caption("Element"))
        self._element = QComboBox()
        self._element.setEditable(True)
        self._element.setFixedHeight(28)
        row.addWidget(self._element, 1)
        body.addLayout(row)

        mode_row = QHBoxLayout()
        self._coord_caption = _caption("Coordinates (Fractional)")
        mode_row.addWidget(self._coord_caption)
        mode_row.addStretch(1)
        self._mode = SegmentedControl(["Frac", "Cart"])
        self._mode.currentChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self._mode)
        body.addLayout(mode_row)

        fields = QHBoxLayout()
        fields.setSpacing(SPACING.sm)
        self._coords = [_coord_field() for _ in range(3)]
        for field in self._coords:
            field.setText("0.00000")
            fields.addWidget(field)
        body.addLayout(fields)

        self._insert = QPushButton("＋  Insert Atom")
        self._insert.setObjectName("PrimaryButton")
        self._insert.setCursor(Qt.PointingHandCursor)
        self._insert.setFixedHeight(32)
        self._insert.clicked.connect(self._emit_insert)
        body.addWidget(self._insert)

    def _on_mode_changed(self, index: int) -> None:
        self._coord_caption.setText(
            f"Coordinates ({'Fractional' if index == 0 else 'Cartesian'})"
        )

    def set_elements(self, present: list[str]) -> None:
        """Offer the structure's own elements first, then the periodic table —
        the next atom added is usually one already in the cell."""
        current = self._element.currentText()
        self._element.clear()
        ordered = list(present) + [e for e in COMMON_ELEMENTS if e not in present]
        self._element.addItems(ordered)
        if current in ordered:
            self._element.setCurrentText(current)

    def set_coordinate_mode(self, fractional: bool) -> None:
        self._mode.setCurrentIndex(0 if fractional else 1)
        self._on_mode_changed(0 if fractional else 1)

    def _emit_insert(self) -> None:
        element = self._element.currentText().strip()
        if not element:
            return
        try:
            coords = np.array([float(f.text()) for f in self._coords])
        except ValueError:
            return
        self.insertRequested.emit(element, coords, self._mode.currentIndex() == 0)


class SupercellCard(Card):
    """Replication factors and the resulting lattice."""

    buildRequested = Signal(int, int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        body = self.body()
        body.setSpacing(SPACING.sm)

        title = QLabel("Supercell & Lattice")
        title.setObjectName("CardTitle")
        body.addWidget(title)

        body.addWidget(_caption("Replicate (a × b × c)"))
        row = QHBoxLayout()
        row.setSpacing(SPACING.md)
        # One labelled a/b/c stepper per axis, matching the Structure page's
        # supercell control exactly — same IntegerStepper widget, same
        # label-above-stepper layout — rather than the plain QSpinBox row
        # this used before.
        self._spins: list[IntegerStepper] = []
        for axis in ("a", "b", "c"):
            axis_layout = QVBoxLayout()
            axis_layout.setSpacing(SPACING.xs)
            axis_label = _caption(axis)
            axis_label.setStyleSheet(
                axis_label.styleSheet() + f" font-weight: 600; font-size: {TYPE.body}px;"
            )
            axis_layout.addWidget(axis_label)
            stepper = IntegerStepper(minimum=1, maximum=MAX_SUPERCELL_REPEAT, value=1)
            stepper.valueChanged.connect(self._update_estimate)
            axis_layout.addWidget(stepper)
            self._spins.append(stepper)
            row.addLayout(axis_layout)
        body.addLayout(row)

        self._estimate = _caption("")
        body.addWidget(self._estimate)

        self._build = QPushButton("Build Supercell")
        self._build.setObjectName("ToolButton")
        self._build.setCursor(Qt.PointingHandCursor)
        self._build.setFixedHeight(30)
        self._build.clicked.connect(
            lambda: self.buildRequested.emit(*[s.value() for s in self._spins])
        )
        body.addWidget(self._build)

        self._lattice = QLabel("—")
        self._lattice.setStyleSheet(
            f"color: {PALETTE.text_secondary}; font-size: {TYPE.small}px;"
        )
        body.addWidget(self._lattice)

        self._n_atoms = 0

    def set_structure(self, structure: EditableStructure | None) -> None:
        if structure is None:
            self._lattice.setText("—")
            self._n_atoms = 0
            self._update_estimate()
            return
        a, b, c = structure.lattice_parameters
        alpha, beta, gamma = structure.lattice_angles
        self._lattice.setText(
            f"a = {a:.3f}   b = {b:.3f}   c = {c:.3f} Å\n"
            f"α = {alpha:.2f}°   β = {beta:.2f}°   γ = {gamma:.2f}°"
        )
        self._n_atoms = structure.n_atoms
        self._update_estimate()

    def _update_estimate(self) -> None:
        factor = 1
        for spin in self._spins:
            factor *= spin.value()
        total = self._n_atoms * factor
        if factor == 1 or not self._n_atoms:
            self._estimate.setText("")
            self._estimate.setStyleSheet(
                f"color: {PALETTE.text_secondary}; font-size: {TYPE.small}px;"
            )
            return
        # Warn before the user commits to a structure large enough to make the
        # page sluggish, rather than after.
        heavy = total > 4000
        self._estimate.setText(
            f"Result: {total:,} atoms" + ("  — this may be slow" if heavy else "")
        )
        self._estimate.setStyleSheet(
            f"color: {PALETTE.warning if heavy else PALETTE.text_secondary};"
            f" font-size: {TYPE.small}px;"
        )


class OperationsCard(Card):
    """Whole-structure operations."""

    wrapRequested = Signal()
    centreRequested = Signal()
    sortRequested = Signal()
    overlapsRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        body = self.body()
        body.setSpacing(SPACING.sm)

        title = QLabel("Structure Operations")
        title.setObjectName("CardTitle")
        body.addWidget(title)

        grid = QGridLayout()
        grid.setSpacing(SPACING.sm)
        entries = [
            ("Wrap Into Cell", self.wrapRequested, True, ""),
            ("Center Structure", self.centreRequested, True, ""),
            ("Sort Element List", self.sortRequested, True, ""),
            ("Remove Overlaps", self.overlapsRequested, True, ""),
        ]
        for index, (text, signal, enabled, tip) in enumerate(entries):
            button = QPushButton(text)
            button.setObjectName("ToolButton")
            button.setCursor(Qt.PointingHandCursor)
            button.setFixedHeight(30)
            button.setEnabled(enabled)
            if tip:
                button.setToolTip(tip)
            if enabled:
                button.clicked.connect(signal.emit)
            grid.addWidget(button, index // 2, index % 2)
        body.addLayout(grid)


class SaveCard(Card):
    """Explicit file writes."""

    savePoscarRequested = Signal()
    exportCifRequested = Signal()
    exportXyzRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        body = self.body()
        body.setSpacing(SPACING.sm)

        title = QLabel("Save / Export")
        title.setObjectName("CardTitle")
        body.addWidget(title)

        row = QHBoxLayout()
        row.setSpacing(SPACING.sm)
        for text, signal in (
            ("Save POSCAR", self.savePoscarRequested),
            ("Export CIF", self.exportCifRequested),
            ("Export XYZ", self.exportXyzRequested),
        ):
            button = QPushButton(text)
            button.setObjectName("ToolButton")
            button.setCursor(Qt.PointingHandCursor)
            button.setFixedHeight(30)
            button.clicked.connect(signal.emit)
            row.addWidget(button)
        body.addLayout(row)
