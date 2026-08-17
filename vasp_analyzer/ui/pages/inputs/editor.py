"""A read-only-by-default code view with line numbers and search."""

from __future__ import annotations

from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QPainter,
    QTextCursor,
    QTextFormat,
)
from PySide6.QtWidgets import QPlainTextEdit, QTextEdit, QWidget

from ...theme import PALETTE, TYPE

GUTTER_BACKGROUND = "#fafafa"
GUTTER_TEXT = "#b0b0b6"
CURRENT_LINE = "#f5f8ff"
MATCH_BACKGROUND = "#fde68a"
CURRENT_MATCH_BACKGROUND = "#fbbf24"


class _LineNumberArea(QWidget):
    def __init__(self, editor: "CodeEditor") -> None:
        super().__init__(editor)
        self._editor = editor

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        return QSize(self._editor.gutter_width(), 0)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._editor.paint_gutter(event)


class CodeEditor(QPlainTextEdit):
    """Monospaced viewer/editor with a line-number gutter and match highlighting."""

    modifiedChanged = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._gutter = _LineNumberArea(self)
        self._matches: list[int] = []
        self._current_match = -1
        self._search_term = ""

        # Named families only — an unresolvable family makes Qt substitute a
        # font that renders the whole document oblique.
        font = QFont()
        font.setFamilies(["Menlo", "Monaco", "Andale Mono", "Courier New"])
        font.setStyleHint(QFont.Monospace)
        font.setFixedPitch(True)
        font.setPointSizeF(11.5)
        self.setFont(font)
        self.setTabStopDistance(QFontMetricsF(font).horizontalAdvance(" ") * 4)

        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.setReadOnly(True)
        self.setFrameShape(QPlainTextEdit.NoFrame)
        self.setStyleSheet(
            f"QPlainTextEdit {{ background: {PALETTE.surface};"
            f" color: {PALETTE.text}; border: none;"
            f" selection-background-color: #cfe0ff;"
            f" selection-color: {PALETTE.text}; }}"
        )

        self.blockCountChanged.connect(self._update_gutter_width)
        self.updateRequest.connect(self._update_gutter)
        self.cursorPositionChanged.connect(self._highlight_extras)
        self.document().modificationChanged.connect(self.modifiedChanged)
        self._update_gutter_width(0)

    # -- gutter ------------------------------------------------------------

    def gutter_width(self) -> int:
        digits = max(3, len(str(max(1, self.blockCount()))))
        return 18 + self.fontMetrics().horizontalAdvance("9") * digits

    def _update_gutter_width(self, _count: int) -> None:
        self.setViewportMargins(self.gutter_width(), 0, 0, 0)

    def _update_gutter(self, rect: QRect, dy: int) -> None:
        if dy:
            self._gutter.scroll(0, dy)
        else:
            self._gutter.update(0, rect.y(), self._gutter.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_gutter_width(0)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        area = self.contentsRect()
        self._gutter.setGeometry(
            QRect(area.left(), area.top(), self.gutter_width(), area.height())
        )

    def paint_gutter(self, event) -> None:
        painter = QPainter(self._gutter)
        painter.fillRect(event.rect(), QColor(GUTTER_BACKGROUND))
        painter.setPen(QColor(PALETTE.border))
        painter.drawLine(
            self._gutter.width() - 1,
            event.rect().top(),
            self._gutter.width() - 1,
            event.rect().bottom(),
        )

        block = self.firstVisibleBlock()
        number = block.blockNumber()
        top = self.blockBoundingGeometry(block).translated(self.contentOffset()).top()
        bottom = top + self.blockBoundingRect(block).height()
        painter.setFont(self.font())

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                painter.setPen(QColor(GUTTER_TEXT))
                painter.drawText(
                    0,
                    int(top),
                    self._gutter.width() - 10,
                    int(self.fontMetrics().height()),
                    Qt.AlignRight | Qt.AlignVCenter,
                    str(number + 1),
                )
            block = block.next()
            top = bottom
            bottom = top + self.blockBoundingRect(block).height()
            number += 1
        painter.end()

    # -- highlighting ---------------------------------------------------------

    def _highlight_extras(self) -> None:
        selections: list[QTextEdit.ExtraSelection] = []

        if not self.isReadOnly():
            selection = QTextEdit.ExtraSelection()
            selection.format.setBackground(QColor(CURRENT_LINE))
            selection.format.setProperty(QTextFormat.FullWidthSelection, True)
            selection.cursor = self.textCursor()
            selection.cursor.clearSelection()
            selections.append(selection)

        selections.extend(self._match_selections())
        self.setExtraSelections(selections)

    def _match_selections(self) -> list[QTextEdit.ExtraSelection]:
        selections: list[QTextEdit.ExtraSelection] = []
        if not self._search_term:
            return selections
        for order, position in enumerate(self._matches):
            selection = QTextEdit.ExtraSelection()
            is_current = order == self._current_match
            selection.format.setBackground(
                QColor(CURRENT_MATCH_BACKGROUND if is_current else MATCH_BACKGROUND)
            )
            cursor = self.textCursor()
            cursor.setPosition(position)
            cursor.setPosition(
                position + len(self._search_term), QTextCursor.KeepAnchor
            )
            selection.cursor = cursor
            selections.append(selection)
        return selections

    # -- search ----------------------------------------------------------------

    def search(self, term: str) -> int:
        """Highlight every occurrence; returns the number of matches."""
        self._search_term = term
        self._matches = []
        self._current_match = -1

        if term:
            text = self.toPlainText().lower()
            needle = term.lower()
            start = text.find(needle)
            while start != -1:
                self._matches.append(start)
                start = text.find(needle, start + 1)
            if self._matches:
                self._current_match = 0
                self._scroll_to_match()

        self._highlight_extras()
        return len(self._matches)

    def next_match(self) -> None:
        if not self._matches:
            return
        self._current_match = (self._current_match + 1) % len(self._matches)
        self._scroll_to_match()
        self._highlight_extras()

    def previous_match(self) -> None:
        if not self._matches:
            return
        self._current_match = (self._current_match - 1) % len(self._matches)
        self._scroll_to_match()
        self._highlight_extras()

    def match_position(self) -> tuple[int, int]:
        return (self._current_match + 1, len(self._matches))

    def _scroll_to_match(self) -> None:
        cursor = self.textCursor()
        cursor.setPosition(self._matches[self._current_match])
        self.setTextCursor(cursor)
        self.centerCursor()
