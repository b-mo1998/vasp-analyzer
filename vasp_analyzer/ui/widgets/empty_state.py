"""Empty and unavailable states.

Used wherever a page cannot show its content: no directory open, a capability
missing, or a file that failed to parse. These are quiet, centred explanations
— never dialogs.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from ..theme import PALETTE, SPACING, TYPE


class EmptyState(QWidget):
    """A centred glyph, title, explanation and optional action button."""

    actionTriggered = Signal()

    def __init__(
        self,
        title: str,
        message: str = "",
        glyph: str = "",
        action: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING.xl, SPACING.xl, SPACING.xl, SPACING.xl)
        layout.setSpacing(SPACING.sm)
        layout.addStretch(1)

        if glyph:
            glyph_label = QLabel(glyph)
            glyph_label.setAlignment(Qt.AlignCenter)
            glyph_label.setStyleSheet(
                f"font-size: 34px; color: {PALETTE.text_tertiary};"
            )
            layout.addWidget(glyph_label)
            layout.addSpacing(SPACING.xs)

        self._title = QLabel(title)
        self._title.setAlignment(Qt.AlignCenter)
        self._title.setStyleSheet(
            f"font-size: {TYPE.body + 2}px; font-weight: 600; color: {PALETTE.text};"
        )
        layout.addWidget(self._title)

        self._message = QLabel(message)
        self._message.setAlignment(Qt.AlignCenter)
        self._message.setWordWrap(True)
        self._message.setMaximumWidth(420)
        self._message.setStyleSheet(
            f"font-size: {TYPE.body}px; color: {PALETTE.text_secondary};"
        )
        layout.addWidget(self._message, 0, Qt.AlignCenter)

        self._button: QPushButton | None = None
        if action:
            layout.addSpacing(SPACING.md)
            self._button = QPushButton(action)
            self._button.setObjectName("PrimaryButton")
            self._button.setCursor(Qt.PointingHandCursor)
            self._button.setFixedWidth(200)
            self._button.clicked.connect(self.actionTriggered)
            layout.addWidget(self._button, 0, Qt.AlignCenter)

        layout.addStretch(1)

    def set_text(self, title: str, message: str = "") -> None:
        self._title.setText(title)
        self._message.setText(message)
