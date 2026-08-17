"""Inline, non-blocking notices.

The error philosophy in one widget: a thin coloured strip with one short line
of text and an optional "View details" link. No modal dialogs, and the rest of
the page keeps working.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from ...core.errors import Issue, Severity
from ..theme import PALETTE, RADIUS, SPACING, TYPE

_STYLES = {
    Severity.INFO: (PALETTE.accent, "#eff4ff", "#dbe6fd"),
    Severity.WARNING: (PALETTE.warning, "#fff8ed", "#f8e3c4"),
    Severity.ERROR: (PALETTE.danger, "#fef2f2", "#f8d7d7"),
}


class Notice(QWidget):
    """A single inline message strip."""

    detailsRequested = Signal(object)  # Issue

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._issue: Issue | None = None
        # An id selector is essential: a bare rule on this widget would cascade
        # into the child label and box the text as well.
        self.setObjectName("NoticeStrip")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACING.md, SPACING.sm, SPACING.md, SPACING.sm)
        layout.setSpacing(SPACING.sm)

        self._icon = QLabel()
        self._icon.setFixedSize(7, 7)
        layout.addWidget(self._icon, 0, Qt.AlignVCenter)

        self._text = QLabel()
        self._text.setWordWrap(True)
        self._text.setStyleSheet(f"font-size: {TYPE.small}px;")
        layout.addWidget(self._text, 1)

        self._details = QPushButton("View details")
        self._details.setCursor(Qt.PointingHandCursor)
        self._details.setFlat(True)
        self._details.setStyleSheet(
            f"border: none; background: transparent; color: {PALETTE.accent};"
            f" font-size: {TYPE.small}px; text-decoration: underline;"
        )
        self._details.clicked.connect(self._emit_details)
        layout.addWidget(self._details, 0, Qt.AlignVCenter)

        self.hide()

    def show_issue(self, issue: Issue) -> None:
        self._issue = issue
        accent, background, border = _STYLES.get(
            issue.severity, _STYLES[Severity.INFO]
        )
        self.setStyleSheet(
            f"#NoticeStrip {{ background: {background};"
            f" border: 1px solid {border};"
            f" border-radius: {RADIUS.md}px; }}"
        )
        self._icon.setStyleSheet(f"background: {accent}; border-radius: 3px;")
        self._text.setText(issue.message)
        self._text.setStyleSheet(f"font-size: {TYPE.small}px; color: {PALETTE.text};")
        self._details.setVisible(bool(issue.detail))
        self.show()

    def clear(self) -> None:
        self._issue = None
        self.hide()

    def _emit_details(self) -> None:
        if self._issue is not None:
            self.detailsRequested.emit(self._issue)
