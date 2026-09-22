"""The atom table.

A real ``QAbstractTableModel`` rather than a grid of widgets: a 4×4×4 supercell
of a 100-atom cell is 6400 rows, and building six widgets per row would cost
tens of thousands of QWidgets and make the page unusable at exactly the moment
the user most needs it. The model hands Qt only the cells it actually paints.

Coordinates are shown in whichever mode the page is in. Switching modes is a
change of *display*, never of data — the underlying fractional coordinates are
untouched, and the Cartesian values are derived through the full lattice
matrix so oblique cells convert correctly.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import (
    QAbstractTableModel,
    QItemSelectionModel,
    QModelIndex,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor, QIcon
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableView, QWidget

from ....models.editable import EditableStructure
from ....services import appearance
from ...style import icon_url
from ...theme import PALETTE, TYPE

COLUMNS = ["Index", "Element", "x", "y", "z", "Fixed"]
COL_INDEX, COL_ELEMENT, COL_X, COL_Y, COL_Z, COL_FIXED = range(6)
COORD_COLUMNS = (COL_X, COL_Y, COL_Z)


class AtomTableModel(QAbstractTableModel):
    """Rows are atoms of the working structure, in structure order."""

    coordinateEdited = Signal(int, int, float)   # row, axis, new value (display units)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._structure: EditableStructure | None = None
        self._fractional = True
        self._rows: list[int] = []   # visible structure indices after filtering

    # -- data plumbing -----------------------------------------------------

    def set_structure(self, structure: EditableStructure | None) -> None:
        self.beginResetModel()
        self._structure = structure
        self._rows = list(range(structure.n_atoms)) if structure else []
        self.endResetModel()

    def set_fractional(self, fractional: bool) -> None:
        if fractional == self._fractional:
            return
        self.beginResetModel()
        self._fractional = fractional
        self.endResetModel()

    def set_visible_rows(self, rows: list[int]) -> None:
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()

    def structure_row(self, view_row: int) -> int:
        return self._rows[view_row]

    def view_row(self, structure_row: int) -> int | None:
        try:
            return self._rows.index(structure_row)
        except ValueError:
            return None

    @property
    def fractional(self) -> bool:
        return self._fractional

    def coords_for_display(self, row: int) -> np.ndarray:
        assert self._structure is not None
        if self._fractional:
            return self._structure.frac_coords[row]
        return self._structure.cart_coords[row]

    # -- Qt model interface ------------------------------------------------

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):  # noqa: N802
        if role != Qt.DisplayRole or orientation != Qt.Horizontal:
            return None
        if section in COORD_COLUMNS:
            axis = COLUMNS[section]
            return f"{axis} (Frac)" if self._fractional else f"{axis} (Å)"
        if section == COL_FIXED:
            # No text label — the column is a single icon, self-explanatory
            # once you've seen it once, and the tooltip on the cell itself
            # carries the actual per-axis detail a label couldn't show anyway.
            return ""
        return COLUMNS[section]

    def flags(self, index):
        base = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if index.column() in COORD_COLUMNS:
            return base | Qt.ItemIsEditable
        # Fixed is a status readout, not a control — toggling it lives on the
        # Fix button (single atom) and the sidebar (the whole selection),
        # both of which act on a chosen group rather than one checkbox at a
        # time and both record a proper undo step. A checkbox here duplicated
        # that with none of it.
        return base

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or self._structure is None:
            return None
        row = self._rows[index.row()]
        column = index.column()

        if role in (Qt.DisplayRole, Qt.EditRole):
            if column == COL_INDEX:
                return row + 1
            if column == COL_ELEMENT:
                return self._structure.species[row]
            if column in COORD_COLUMNS:
                value = float(self.coords_for_display(row)[column - COL_X])
                # Enough precision to be scientifically useful, and the same
                # in both modes so a value never appears to change when the
                # display mode does.
                return f"{value:.5f}"
            return None

        if role == Qt.DecorationRole and column == COL_FIXED:
            # A lock icon rather than a checkmark: "fixed" means "held in
            # place," and a lock says that at a glance, where a checkmark
            # reads as "yes/complete" — the wrong association for a
            # constraint. A partially-fixed atom (free on some axes, not
            # others) is real — selective dynamics is per-axis — so it is
            # left blank rather than guessed at; the tooltip below spells
            # out exactly which axes are which.
            free = self._structure.selective[row]
            return QIcon(icon_url("lock.svg")) if not free.any() else None

        if role == Qt.ForegroundRole and column == COL_ELEMENT:
            return QColor(appearance.effective_color(self._structure.species[row]))

        if role == Qt.TextAlignmentRole:
            # Every column centered under its header — right-aligning the
            # numbers while the header text sits left/center made the values
            # drift out from under their own column title.
            return int(Qt.AlignCenter)

        if role == Qt.ToolTipRole and column == COL_FIXED:
            free = self._structure.selective[row]
            flags = " ".join("T" if f else "F" for f in free)
            return f"Selective dynamics: {flags}"

        return None

    def setData(self, index, value, role=Qt.EditRole):  # noqa: N802
        if self._structure is None or not index.isValid():
            return False
        row = self._rows[index.row()]
        column = index.column()

        if role == Qt.EditRole and column in COORD_COLUMNS:
            try:
                number = float(str(value).strip())
            except (TypeError, ValueError):
                return False
            self.coordinateEdited.emit(row, column - COL_X, number)
            return True

        return False


class AtomTable(QTableView):
    """The table view, wired for multi-row selection."""

    selectionSet = Signal(object)   # set[int] of structure rows

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = AtomTableModel(self)
        self.setModel(self._model)
        self._suppress = False

        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setAlternatingRowColors(False)
        self.setShowGrid(False)
        self.setCornerButtonEnabled(False)
        self.verticalHeader().setVisible(False)
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )

        header = self.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setHighlightSections(False)
        # Matches the cells' own AlignCenter — a centered header over
        # right-aligned (or left-default) values was what made the numbers
        # look adrift from their own column title.
        header.setDefaultAlignment(Qt.AlignCenter)
        self.setColumnWidth(COL_INDEX, 62)
        self.setColumnWidth(COL_ELEMENT, 78)
        for column in COORD_COLUMNS:
            self.setColumnWidth(column, 104)
        self.setColumnWidth(COL_FIXED, 60)
        self.verticalHeader().setDefaultSectionSize(26)

        self.setStyleSheet(
            f"QTableView {{ border: none; font-size: {TYPE.body}px;"
            f" selection-background-color: {PALETTE.accent_soft};"
            f" selection-color: {PALETTE.text}; }}"
            f"QHeaderView::section {{ background: transparent; border: none;"
            f" border-bottom: 1px solid {PALETTE.border};"
            f" padding: 6px 4px; font-size: {TYPE.small}px;"
            f" color: {PALETTE.text_secondary}; }}"
        )
        self.selectionModel().selectionChanged.connect(self._on_selection_changed)

    def model_(self) -> AtomTableModel:
        return self._model

    def _on_selection_changed(self, *_args) -> None:
        if self._suppress:
            return
        rows = {
            self._model.structure_row(index.row())
            for index in self.selectionModel().selectedRows()
        }
        self.selectionSet.emit(rows)

    def apply_selection(self, structure_rows: set[int]) -> None:
        """Reflect a selection that came from elsewhere, without echoing it
        back out as a fresh change."""
        self._suppress = True
        try:
            selection_model = self.selectionModel()
            selection_model.clearSelection()
            first: int | None = None
            for structure_row in sorted(structure_rows):
                view_row = self._model.view_row(structure_row)
                if view_row is None:
                    continue
                if first is None:
                    first = view_row
                selection_model.select(
                    self._model.index(view_row, 0),
                    QItemSelectionModel.Select | QItemSelectionModel.Rows,
                )
            if first is not None:
                self.scrollTo(self._model.index(first, 0), QAbstractItemView.EnsureVisible)
        finally:
            self._suppress = False
