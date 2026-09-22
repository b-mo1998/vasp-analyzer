"""Empty and unavailable states.

Used wherever a page cannot show its content: no directory open, a capability
missing, or a file that failed to parse. These are quiet, centred explanations
— never dialogs.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from ..theme import PALETTE, SPACING, TYPE

MESSAGE_WIDTH = 360


def _wrap_to_width(metrics: QFontMetrics, text: str, max_width: int) -> str:
    """Break ``text`` into lines no wider than ``max_width`` pixels.

    Used instead of ``QLabel.setWordWrap(True)`` for centred multi-line
    messages: a word-wrapped QLabel only computes its height correctly when
    added to a layout *without* an alignment flag, and only centres correctly
    *with* one — there is no combination that gets both right. Wrapping the
    text by hand up front turns it into an ordinary fixed-line-count label,
    which has neither problem.
    """
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if not current or metrics.horizontalAdvance(candidate) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


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

        self._message = QLabel()
        self._message.setAlignment(Qt.AlignCenter)
        self._message.setStyleSheet(
            f"font-size: {TYPE.body}px; color: {PALETTE.text_secondary};"
        )
        layout.addWidget(self._message, 0, Qt.AlignCenter)
        self._set_message(message)

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

    def _set_message(self, message: str) -> None:
        metrics = QFontMetrics(self._message.font())
        self._message.setText(_wrap_to_width(metrics, message, MESSAGE_WIDTH))

    def set_text(self, title: str, message: str = "") -> None:
        self._title.setText(title)
        self._set_message(message)
