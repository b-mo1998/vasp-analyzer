"""The top-right tool cluster: Reset, Copy, Export and an overflow menu."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QMenu, QPushButton, QWidget

from ..theme import SPACING


class PageToolbar(QWidget):
    """Actions that operate on whatever the current page is showing."""

    resetRequested = Signal()
    copyRequested = Signal()
    exportRequested = Signal()
    openRequested = Signal()
    revealRequested = Signal()
    logRequested = Signal()
    reloadRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING.sm)

        self._reset = self._button("Reset", self.resetRequested)
        self._copy = self._button("Copy", self.copyRequested)
        self._export = self._button("Export", self.exportRequested)
        layout.addWidget(self._reset)
        layout.addWidget(self._copy)
        layout.addWidget(self._export)

        self._more = QPushButton("•••")
        self._more.setObjectName("ToolButton")
        self._more.setCursor(Qt.PointingHandCursor)
        self._more.setFixedWidth(44)
        self._more.clicked.connect(self._show_menu)
        layout.addWidget(self._more)

    def _button(self, text: str, signal) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName("ToolButton")
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(signal)
        return button

    def _show_menu(self) -> None:
        menu = QMenu(self)
        menu.addAction("Open Calculation…", self.openRequested.emit)
        menu.addAction("Reload Calculation", self.reloadRequested.emit)
        menu.addSeparator()
        menu.addAction("Reveal in Finder", self.revealRequested.emit)
        menu.addAction("Show Log File", self.logRequested.emit)
        menu.exec(self._more.mapToGlobal(self._more.rect().bottomLeft()))

    def set_actions_enabled(self, enabled: bool) -> None:
        for button in (self._reset, self._copy, self._export):
            button.setEnabled(enabled)
