"""Syntax highlighting for VASP and Wannier90 input files.

Deliberately restrained: tags, numbers, strings and comments only. Scientific
input files are read for their values, and a rainbow of colour makes them
harder, not easier, to scan.
"""

from __future__ import annotations

import re

from PySide6.QtCore import QRegularExpression
from PySide6.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat

TAG_COLOR = "#1e40af"        # dark blue
VALUE_COLOR = "#7c2d12"      # dark orange/brown
NUMBER_COLOR = "#6d28d9"     # muted purple
COMMENT_COLOR = "#4d7c58"    # subdued green
LOGICAL_COLOR = "#0f766e"    # teal
TEXT_COLOR = "#3a3a3c"


def _format(color: str, bold: bool = False, italic: bool = False) -> QTextCharFormat:
    fmt = QTextCharFormat()
    fmt.setForeground(QColor(color))
    if bold:
        fmt.setFontWeight(QFont.DemiBold)
    if italic:
        fmt.setFontItalic(True)
    return fmt


class BaseHighlighter(QSyntaxHighlighter):
    """Applies an ordered list of (pattern, format) rules per line."""

    language = "Plain text"

    def __init__(self, document) -> None:
        super().__init__(document)
        self._rules: list[tuple[QRegularExpression, QTextCharFormat, int]] = []
        self._build_rules()

    def _build_rules(self) -> None:
        """Subclasses populate ``self._rules``."""

    def add_rule(self, pattern: str, fmt: QTextCharFormat, group: int = 0) -> None:
        self._rules.append((QRegularExpression(pattern), fmt, group))

    def highlightBlock(self, text: str) -> None:  # noqa: N802 - Qt naming
        for expression, fmt, group in self._rules:
            iterator = expression.globalMatch(text)
            while iterator.hasNext():
                match = iterator.next()
                start = match.capturedStart(group)
                length = match.capturedLength(group)
                if start >= 0 and length > 0:
                    self.setFormat(start, length, fmt)


class IncarHighlighter(BaseHighlighter):
    """``TAG = VALUE  # comment``."""

    language = "VASP INCAR"

    def _build_rules(self) -> None:
        self.add_rule(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", _format(TAG_COLOR, bold=True), 1)
        self.add_rule(r";\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", _format(TAG_COLOR, bold=True), 1)
        self.add_rule(r"\.(TRUE|FALSE|T|F)\.", _format(LOGICAL_COLOR))
        self.add_rule(r"\b-?\d+\.?\d*([EeDd][-+]?\d+)?\b", _format(NUMBER_COLOR))
        self.add_rule(r"[#!].*$", _format(COMMENT_COLOR, italic=True))


class KpointsHighlighter(BaseHighlighter):
    language = "VASP KPOINTS"

    def _build_rules(self) -> None:
        self.add_rule(
            r"^\s*(Auto\w*|Gamma\w*|Monkhorst\w*|Line[- ]?mode|Reciprocal|Cartesian)",
            _format(TAG_COLOR, bold=True),
            1,
        )
        self.add_rule(r"\b-?\d+\.?\d*([EeDd][-+]?\d+)?\b", _format(NUMBER_COLOR))
        self.add_rule(r"[#!].*$", _format(COMMENT_COLOR, italic=True))


class PoscarHighlighter(BaseHighlighter):
    language = "VASP POSCAR"

    def _build_rules(self) -> None:
        self.add_rule(
            r"^\s*(Direct|Cartesian|direct|cartesian|Selective\s+dynamics|S)\s*$",
            _format(TAG_COLOR, bold=True),
            1,
        )
        self.add_rule(r"\b-?\d+\.\d+([EeDd][-+]?\d+)?\b", _format(NUMBER_COLOR))
        self.add_rule(r"\b[TF]\b", _format(LOGICAL_COLOR))
        self.add_rule(r"[#!].*$", _format(COMMENT_COLOR, italic=True))


class WannierHighlighter(BaseHighlighter):
    language = "Wannier90 input"

    def _build_rules(self) -> None:
        self.add_rule(
            r"^\s*(begin|end)\s+(\w+)", _format(TAG_COLOR, bold=True), 0
        )
        self.add_rule(r"^\s*([a-zA-Z_][\w]*)\s*[=:]", _format(TAG_COLOR, bold=True), 1)
        self.add_rule(r"\b-?\d+\.?\d*([EeDd][-+]?\d+)?\b", _format(NUMBER_COLOR))
        self.add_rule(r"\b(true|false|\.true\.|\.false\.|T|F)\b", _format(LOGICAL_COLOR))
        self.add_rule(r"[#!].*$", _format(COMMENT_COLOR, italic=True))


class OutcarHighlighter(BaseHighlighter):
    """OUTCAR is output, not input: only key = value lines get emphasis."""

    language = "VASP OUTCAR"

    def _build_rules(self) -> None:
        self.add_rule(r"^\s*([A-Z][A-Z0-9_-]{2,})\s*[=:]", _format(TAG_COLOR, bold=True), 1)
        self.add_rule(r"\b-?\d+\.\d+([EeDd][-+]?\d+)?\b", _format(NUMBER_COLOR))


_BY_ROLE = {
    "incar": IncarHighlighter,
    "kpoints": KpointsHighlighter,
    "poscar": PoscarHighlighter,
    "contcar": PoscarHighlighter,
    "outcar": OutcarHighlighter,
    "w90_win": WannierHighlighter,
    "w90_wout": WannierHighlighter,
}


def highlighter_for(role_key: str):
    """Return the highlighter class for a file role, or None for plain text."""
    return _BY_ROLE.get(role_key)


def language_name(role_key: str) -> str:
    cls = _BY_ROLE.get(role_key)
    return cls.language if cls is not None else "Plain text"


def looks_binary(sample: bytes) -> bool:
    """Heuristic used before showing a file as text."""
    if b"\x00" in sample:
        return True
    printable = sum(
        1 for byte in sample if 32 <= byte <= 126 or byte in (9, 10, 13)
    )
    return bool(sample) and printable / len(sample) < 0.85


_NUMBER = re.compile(r"^-?\d+\.?\d*([EeDd][-+]?\d+)?$")
