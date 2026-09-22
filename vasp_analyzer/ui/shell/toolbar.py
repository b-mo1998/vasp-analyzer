"""The top-right tool cluster: a page action, Copy, Export and an overflow menu."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QMenu, QPushButton, QWidget

from ..theme import SPACING


class PageToolbar(QWidget):
    """Actions that operate on whatever the current page is showing."""

    DEFAULT_ACTION = ("Reset", "Put this page's controls back to their defaults")

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

        # The first slot is relabelled by whichever page is in front. Most
        # pages use it to put their own controls back to defaults, but the
        # Editor holds unsaved edits, where "Reset" would be an alarmingly
        # vague word for discarding them — it says "Revert Changes" there,
        # matching the button beside Save Structure.
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

    def set_page_action(self, label: str, tooltip: str = "") -> None:
        self._reset.setText(label)
        self._reset.setToolTip(tooltip)
        # A relabelled button keeps the width it was laid out with until it is
        # asked to resize, which clips the longer caption.
        self._reset.adjustSize()

    def set_actions_enabled(self, enabled: bool) -> None:
        for button in (self._reset, self._copy, self._export):
            button.setEnabled(enabled)
