"""The Inputs page: inspect — and carefully edit — calculation files."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ....core.errors import Issue, Severity
from ....core.logging_setup import get_logger
from ....models.calculation import Calculation
from ....models.files import FileEntry, FileStatus
from ....parsers.incar import parse_incar
from ....parsers.kpoints import parse_kpoints
from ....parsers.poscar import parse_poscar
from ....parsers.wannier import parse_win
from ...theme import PALETTE, RADIUS, SPACING, TYPE
from ...widgets.empty_state import EmptyState
from ...widgets.notice import Notice
from ...widgets.primitives import Card
from ..base import Page
from .editor import CodeEditor
from .highlighters import highlighter_for, language_name, looks_binary
from .sidebar import InputsSidebar

log = get_logger("ui.inputs")

# Files the user may edit. Everything else is calculation output: showing it is
# useful, letting the app rewrite it is not.
EDITABLE_ROLES = {"incar", "kpoints", "poscar", "contcar", "w90_win"}

# Above this size the file is opened as a preview of its first lines. VASP
# outputs reach hundreds of megabytes and would otherwise freeze the window.
LARGE_FILE_BYTES = 2_000_000
PREVIEW_LINES = 5_000

VALIDATORS = {
    "incar": parse_incar,
    "kpoints": parse_kpoints,
    "poscar": parse_poscar,
    "contcar": parse_poscar,
    "w90_win": parse_win,
}


class InputsPage(Page):
    """File browser plus a large viewer/editor surface."""

    title = "Inputs"
    issueRaised = Signal(object)
    navigationRequested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sidebar = InputsSidebar()
        self._sidebar.fileSelected.connect(self.show_file)
        self._current: FileEntry | None = None
        self._truncated = False
        self._backed_up: set[Path] = set()

        self._build()

    # -- construction ---------------------------------------------------------

    def _build(self) -> None:
        layout = self.body()
        layout.setSpacing(SPACING.md)

        self._notice = Notice()
        self._notice.detailsRequested.connect(self.issueRaised)
        layout.addWidget(self._notice)

        self._stack = QStackedWidget()
        layout.addWidget(self._stack, 1)

        self._stack.addWidget(self._build_viewer_card())

        empty_card = Card()
        self._empty = EmptyState(
            title="No file selected",
            message="Choose a file from the list to inspect it.",
            glyph="🗎",
        )
        empty_card.body().addWidget(self._empty)
        self._stack.addWidget(empty_card)
        self._stack.setCurrentIndex(1)

    def _build_viewer_card(self) -> QWidget:
        card = Card(padding=0)
        body = card.body()
        body.setSpacing(0)

        body.addWidget(self._build_header())
        body.addWidget(self._build_editor(), 1)
        body.addWidget(self._build_footer())
        return card

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setFixedHeight(56)
        row = QHBoxLayout(header)
        row.setContentsMargins(SPACING.lg, SPACING.sm, SPACING.lg, SPACING.sm)
        row.setSpacing(SPACING.md)

        self._filename = QLabel("—")
        self._filename.setStyleSheet(
            f"font-size: {TYPE.body + 3}px; font-weight: 600;"
        )
        row.addWidget(self._filename)

        self._badge = QLabel("")
        self._badge.setAlignment(Qt.AlignCenter)
        row.addWidget(self._badge)
        row.addStretch(1)

        self._search = QLineEdit()
        self._search.setObjectName("ValueField")
        self._search.setPlaceholderText("Search in file…")
        self._search.setFixedWidth(210)
        self._search.textChanged.connect(self._on_search)
        self._search.returnPressed.connect(self._on_search_next)
        row.addWidget(self._search)

        self._match_label = QLabel("")
        self._match_label.setObjectName("TertiaryLabel")
        self._match_label.setFixedWidth(64)
        row.addWidget(self._match_label)

        self._copy_button = QPushButton("Copy")
        self._copy_button.setObjectName("ToolButton")
        self._copy_button.setCursor(Qt.PointingHandCursor)
        self._copy_button.clicked.connect(self.copy_to_clipboard)
        row.addWidget(self._copy_button)

        self._edit_button = QPushButton("Edit")
        self._edit_button.setObjectName("ToolButton")
        self._edit_button.setCursor(Qt.PointingHandCursor)
        self._edit_button.setCheckable(True)
        self._edit_button.toggled.connect(self._set_editing)
        row.addWidget(self._edit_button)

        self._more = QPushButton("•••")
        self._more.setObjectName("ToolButton")
        self._more.setFixedWidth(44)
        self._more.setCursor(Qt.PointingHandCursor)
        self._more.clicked.connect(self._show_menu)
        row.addWidget(self._more)

        return header

    def _build_editor(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        top_rule = QWidget()
        top_rule.setFixedHeight(1)
        top_rule.setStyleSheet(f"background: {PALETTE.border};")
        layout.addWidget(top_rule)

        self._editor = CodeEditor()
        self._editor.modifiedChanged.connect(self._on_modified)
        layout.addWidget(self._editor, 1)

        save = QShortcut(QKeySequence.Save, self._editor)
        save.activated.connect(self._save)
        return container

    def _build_footer(self) -> QWidget:
        footer = QWidget()
        footer.setFixedHeight(34)
        row = QHBoxLayout(footer)
        row.setContentsMargins(SPACING.lg, 0, SPACING.lg, 0)
        row.setSpacing(SPACING.xl)

        rule = QWidget()
        rule.setFixedHeight(1)

        self._stats: dict[str, QLabel] = {}
        for key, caption in (
            ("lines", "Lines"),
            ("characters", "Characters"),
            ("encoding", "Encoding"),
            ("language", "Language"),
        ):
            label = QLabel(f"{caption}: —")
            label.setObjectName("TertiaryLabel")
            row.addWidget(label)
            self._stats[key] = label
            label.setProperty("caption", caption)

        row.addStretch(1)
        self._modified_label = QLabel("")
        self._modified_label.setObjectName("TertiaryLabel")
        row.addWidget(self._modified_label)
        return footer

    # -- Page interface -----------------------------------------------------------

    def sidebar(self) -> QWidget:
        return self._sidebar

    def set_calculation(self, calculation: Calculation | None) -> None:
        super().set_calculation(calculation)
        self._current = None
        self._backed_up.clear()
        self._notice.clear()
        self._sidebar.populate(calculation)

        if calculation is None or calculation.is_empty:
            self._empty.set_text(
                "No calculation open",
                "Open a calculation directory to inspect its input files.",
            )
            self._stack.setCurrentIndex(1)
            return

        first = self._sidebar.first_available()
        if first is None:
            self._empty.set_text(
                "No readable files",
                "This folder contains no recognised VASP or Wannier90 files.",
            )
            self._stack.setCurrentIndex(1)
            return

        self._sidebar.select(first)
        self.show_file(first)

    # -- file display ---------------------------------------------------------------

    def show_file(self, role_key: str) -> None:
        calculation = self.calculation()
        if calculation is None:
            return
        entry = calculation.index.entries.get(role_key)
        if entry is None or entry.path is None or not entry.exists:
            return

        if self._confirm_discard() is False:
            # Put the selection back where it was.
            if self._current is not None:
                self._sidebar.select(self._current.role.key)
            return

        self._current = entry
        self._notice.clear()
        self._edit_button.setChecked(False)
        self._search.clear()

        text, issue = self._read_file(entry)
        if issue is not None:
            self._notice.show_issue(issue)

        highlighter = highlighter_for(role_key)
        # Highlighting a very large document is slower than reading it.
        self._editor.setPlainText(text)
        if highlighter is not None and len(text) < LARGE_FILE_BYTES:
            highlighter(self._editor.document())

        self._editor.document().setModified(False)
        self._filename.setText(entry.name)
        self._apply_badge(entry, text)
        self._update_stats(text, role_key)

        editable = role_key in EDITABLE_ROLES and not self._truncated
        self._edit_button.setEnabled(editable)
        self._edit_button.setToolTip(
            ""
            if editable
            else (
                "This file is calculation output and is shown read-only."
                if not self._truncated
                else "A partially loaded file cannot be edited."
            )
        )
        self._stack.setCurrentIndex(0)
        self._sidebar.select(role_key)

    def _read_file(self, entry: FileEntry) -> tuple[str, Issue | None]:
        self._truncated = False
        path = entry.path
        assert path is not None
        try:
            with path.open("rb") as handle:
                sample = handle.read(4096)
            if looks_binary(sample):
                return (
                    "",
                    Issue(
                        code="binary_file",
                        message=f"{entry.name} is a binary file and cannot be shown.",
                        severity=Severity.INFO,
                        source=entry.name,
                    ),
                )

            if entry.size > LARGE_FILE_BYTES:
                lines: list[str] = []
                with path.open("r", errors="replace") as handle:
                    for _ in range(PREVIEW_LINES):
                        line = handle.readline()
                        if not line:
                            break
                        lines.append(line)
                self._truncated = True
                return (
                    "".join(lines),
                    Issue(
                        code="large_file_preview",
                        message=(
                            f"{entry.name} is {entry.size_label}; showing the first "
                            f"{len(lines):,} lines."
                        ),
                        severity=Severity.INFO,
                        source=entry.name,
                    ),
                )

            return path.read_text(errors="replace"), None
        except OSError as exc:
            log.exception("Could not read %s", path)
            return (
                "",
                Issue(
                    code="file_unreadable",
                    message=f"{entry.name} could not be read.",
                    detail=str(exc),
                    severity=Severity.ERROR,
                    source=entry.name,
                ),
            )

    def _apply_badge(self, entry: FileEntry, text: str) -> None:
        """Validate the file by actually parsing it, where a parser exists."""
        if entry.status is FileStatus.INCONSISTENT:
            label, color, background = "Different run", PALETTE.warning, "#fff8ed"
        elif self._truncated:
            label, color, background = "Preview", PALETTE.text_secondary, "#f2f2f4"
        else:
            validator = VALIDATORS.get(entry.role.key)
            if validator is None or not text:
                label, color, background = "", "", ""
            else:
                result = validator(entry.path)
                if result.ok and not result.errors:
                    label, color, background = "Valid", PALETTE.success, "#eefaf1"
                else:
                    label, color, background = "Parse issue", PALETTE.warning, "#fff8ed"
                    if result.issues:
                        self._notice.show_issue(result.issues[0])

        self._badge.setText(label)
        self._badge.setVisible(bool(label))
        if label:
            self._badge.setStyleSheet(
                f"background: {background}; color: {color};"
                f" border-radius: {RADIUS.sm}px; padding: 3px 9px;"
                f" font-size: {TYPE.small}px; font-weight: 500;"
            )

    def _update_stats(self, text: str, role_key: str) -> None:
        lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
        values = {
            "lines": f"{lines:,}" + (" (preview)" if self._truncated else ""),
            "characters": f"{len(text):,}",
            "encoding": "UTF-8",
            "language": language_name(role_key),
        }
        for key, label in self._stats.items():
            caption = label.property("caption")
            label.setText(f"{caption}: {values[key]}")

    # -- search -------------------------------------------------------------------------

    def _on_search(self, term: str) -> None:
        count = self._editor.search(term)
        if not term:
            self._match_label.setText("")
        elif count == 0:
            self._match_label.setText("No results")
        else:
            current, total = self._editor.match_position()
            self._match_label.setText(f"{current} of {total}")

    def _on_search_next(self) -> None:
        self._editor.next_match()
        current, total = self._editor.match_position()
        if total:
            self._match_label.setText(f"{current} of {total}")

    # -- editing -------------------------------------------------------------------------

    def _set_editing(self, editing: bool) -> None:
        self._editor.setReadOnly(not editing)
        self._edit_button.setText("Editing" if editing else "Edit")
        if editing:
            self._notice.show_issue(
                Issue(
                    code="editing_enabled",
                    message=(
                        "Editing enabled. Saving keeps a timestamped backup of the "
                        "original alongside it."
                    ),
                    severity=Severity.INFO,
                )
            )
        else:
            self._notice.clear()

    def _on_modified(self, modified: bool) -> None:
        self._modified_label.setText("Unsaved changes" if modified else "")
        self._modified_label.setStyleSheet(
            f"color: {PALETTE.warning};" if modified else ""
        )

    def _save(self) -> None:
        entry = self._current
        if entry is None or entry.path is None or self._editor.isReadOnly():
            return
        if not self._editor.document().isModified():
            return

        path = entry.path
        try:
            if path not in self._backed_up:
                stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
                backup = path.with_name(f"{path.name}.backup-{stamp}")
                backup.write_bytes(path.read_bytes())
                self._backed_up.add(path)
                log.info("Backed up %s to %s", path, backup.name)

            path.write_text(self._editor.toPlainText())
            self._editor.document().setModified(False)
            log.info("Saved %s", path)
            self._notice.show_issue(
                Issue(
                    code="file_saved",
                    message=f"Saved {entry.name}. The original was kept as a backup.",
                    severity=Severity.INFO,
                )
            )
            self._apply_badge(entry, self._editor.toPlainText())
        except OSError as exc:
            log.exception("Could not save %s", path)
            self._notice.show_issue(
                Issue(
                    code="save_failed",
                    message=f"{entry.name} could not be saved.",
                    detail=str(exc),
                    severity=Severity.ERROR,
                )
            )

    def _confirm_discard(self) -> bool | None:
        """Ask before losing edits. Returns False when the user cancels."""
        if not self._editor.document().isModified():
            return None
        box = QMessageBox(self)
        box.setWindowTitle("Unsaved changes")
        box.setIcon(QMessageBox.Warning)
        box.setText(
            f"{self._current.name if self._current else 'This file'} has unsaved changes."
        )
        box.setStandardButtons(
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
        )
        choice = box.exec()
        if choice == QMessageBox.Save:
            self._save()
            return None
        if choice == QMessageBox.Discard:
            self._editor.document().setModified(False)
            return None
        return False

    # -- misc -------------------------------------------------------------------------------

    def _show_menu(self) -> None:
        menu = QMenu(self)
        save = menu.addAction("Save", self._save)
        save.setEnabled(
            not self._editor.isReadOnly() and self._editor.document().isModified()
        )
        menu.addAction("Reload from disk", self._reload)
        menu.addSeparator()
        reveal = menu.addAction("Reveal in Finder", self._reveal)
        reveal.setEnabled(self._current is not None)
        menu.exec(self._more.mapToGlobal(self._more.rect().bottomLeft()))

    def _reload(self) -> None:
        if self._current is not None:
            self._editor.document().setModified(False)
            self.show_file(self._current.role.key)

    def _reveal(self) -> None:
        import subprocess
        import sys

        if self._current is None or self._current.path is None:
            return
        if sys.platform == "darwin":
            subprocess.run(["open", "-R", str(self._current.path)], check=False)

    def copy_to_clipboard(self) -> bool:
        text = self._editor.toPlainText()
        if not text:
            return False
        QGuiApplication.clipboard().setText(text)
        return True

    def reset(self) -> None:
        self._search.clear()
        self._edit_button.setChecked(False)
        self._reload()
