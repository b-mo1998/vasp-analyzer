"""The pill navigation used at the top of every page."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QPushButton, QWidget


class SegmentedControl(QWidget):
    """An exclusive row of pill buttons.

    Individual segments can be disabled with an explanatory tooltip — that is
    how a page whose capability is missing presents itself, rather than
    vanishing and leaving the user wondering where it went.
    """

    currentChanged = Signal(int)

    def __init__(self, items: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("SegmentedControl")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(3)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: list[QPushButton] = []

        for position, text in enumerate(items):
            button = QPushButton(text)
            button.setCheckable(True)
            button.setCursor(button.cursor())
            layout.addWidget(button)
            self._group.addButton(button, position)
            self._buttons.append(button)

        if self._buttons:
            self._buttons[0].setChecked(True)
        self._group.idClicked.connect(self._on_clicked)

    def _on_clicked(self, index: int) -> None:
        self.currentChanged.emit(index)

    def currentIndex(self) -> int:  # noqa: N802 - Qt naming
        return self._group.checkedId()

    def setCurrentIndex(self, index: int) -> None:  # noqa: N802 - Qt naming
        if 0 <= index < len(self._buttons):
            self._buttons[index].setChecked(True)
            self.currentChanged.emit(index)

    def setItemEnabled(self, index: int, enabled: bool, reason: str = "") -> None:  # noqa: N802
        if 0 <= index < len(self._buttons):
            button = self._buttons[index]
            button.setEnabled(enabled)
            button.setToolTip(reason if not enabled else "")
