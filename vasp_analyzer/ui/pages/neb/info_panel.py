"""The NEB right-hand column: barriers, the selected image, and provenance.

Every value here is either read from the calculation or derived from values
that were; anything undetermined shows an em dash. That matters more on this
page than most: a barrier is the number an NEB exists to produce, and a
plausible-looking one from an unfinished band would be worse than none.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ....models.neb import NEBPath
from ...theme import PALETTE, RADIUS, SPACING, TYPE


def _caption(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(
        f"font-size: {TYPE.small}px; color: {PALETTE.text_secondary};"
    )
    return label


def _value(text: str = "—", strong: bool = False, color: str | None = None) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(
        f"font-size: {TYPE.body}px; font-weight: {600 if strong else 500};"
        f" color: {color or PALETTE.text};"
    )
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return label


class SummaryStrip(QWidget):
    """Initial energy, forward barrier and final energy, side by side."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, SPACING.xs, 0, 0)
        row.setSpacing(SPACING.md)

        self._values: dict[str, QLabel] = {}
        for key, caption, colour in (
            ("initial", "Initial Energy", None),
            ("barrier", "Barrier (Ea)", PALETTE.danger),
            ("final", "Final Energy", None),
        ):
            cell = QWidget()
            column = QVBoxLayout(cell)
            column.setContentsMargins(0, 0, 0, 0)
            column.setSpacing(2)
            column.addWidget(_caption(caption))
            value = _value(color=colour)
            self._values[key] = value
            column.addWidget(value)
            row.addWidget(cell, 1)

    def update_from(self, path: NEBPath | None) -> None:
        if path is None:
            for label in self._values.values():
                label.setText("—")
            return

        initial = path.initial_energy
        final = path.final_energy
        barrier = path.forward_barrier

        self._values["initial"].setText(
            "—" if initial is None else f"{initial:.2f} eV"
        )
        self._values["final"].setText("—" if final is None else f"{final:.2f} eV")
        if barrier is None:
            self._values["barrier"].setText("—")
            self._values["barrier"].setToolTip(
                "A barrier needs an energy for every image; this band has gaps."
            )
        else:
            self._values["barrier"].setText(f"{barrier:.2f} eV")
            reverse = path.reverse_barrier
            self._values["barrier"].setToolTip(
                f"Forward barrier E_max - E_initial."
                + (f" Reverse: {reverse:.2f} eV." if reverse is not None else "")
            )


class KeyValueCard(QWidget):
    """A titled block of caption/value pairs in two columns."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("InfoBlock")
        self.setAttribute(Qt.WA_StyledBackground, True)
        column = QVBoxLayout(self)
        column.setContentsMargins(SPACING.md, SPACING.sm, SPACING.md, SPACING.md)
        column.setSpacing(SPACING.sm)

        self._title = QLabel(title)
        self._title.setStyleSheet(
            f"font-size: {TYPE.body}px; font-weight: 600; color: {PALETTE.text};"
        )
        column.addWidget(self._title)

        self._grid = QGridLayout()
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(SPACING.md)
        self._grid.setVerticalSpacing(SPACING.sm)
        column.addLayout(self._grid)

        self._values: dict[str, QLabel] = {}
        self.setStyleSheet(
            f"#InfoBlock {{ background: {PALETTE.surface_sunken};"
            f" border: 1px solid {PALETTE.border};"
            f" border-radius: {RADIUS.md}px; }}"
        )

    def set_title(self, text: str) -> None:
        self._title.setText(text)

    def build(self, entries: list[tuple[str, str]], columns: int = 2) -> None:
        """Lay out ``(key, caption)`` pairs; values are filled in later."""
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._values.clear()

        for position, (key, caption) in enumerate(entries):
            row, column = divmod(position, columns)
            cell = QWidget()
            box = QVBoxLayout(cell)
            box.setContentsMargins(0, 0, 0, 0)
            box.setSpacing(2)
            box.addWidget(_caption(caption))
            value = _value()
            self._values[key] = value
            box.addWidget(value)
            self._grid.addWidget(cell, row, column)

    def set_values(self, values: dict[str, str], tooltips: dict[str, str] | None = None) -> None:
        tooltips = tooltips or {}
        for key, label in self._values.items():
            text = values.get(key, "—")
            label.setText(text or "—")
            label.setStyleSheet(
                f"font-size: {TYPE.body}px; font-weight: 500;"
                f" color: {PALETTE.text_tertiary if text in ('', '—') else PALETTE.text};"
            )
            label.setToolTip(tooltips.get(key, ""))

    def clear(self) -> None:
        self.set_values({key: "—" for key in self._values})
