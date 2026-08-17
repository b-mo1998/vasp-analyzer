"""File browser for the Inputs page.

Files are grouped by origin and always shown with their state, so a missing
file is visible information rather than an absence the user has to notice.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ....models.calculation import Calculation
from ....models.files import FileEntry, FileGroup, FileStatus
from ...theme import PALETTE, RADIUS, SPACING, TYPE
from ...widgets.primitives import Divider, Label

STATUS_GLYPH = {
    FileStatus.FOUND: ("✓", PALETTE.success),
    FileStatus.MISSING: ("○", PALETTE.text_tertiary),
    FileStatus.MALFORMED: ("⚠", PALETTE.warning),
    FileStatus.UNREADABLE: ("⚠", PALETTE.warning),
    FileStatus.INCONSISTENT: ("⚠", PALETTE.warning),
}

STATUS_TOOLTIP = {
    FileStatus.FOUND: "File found",
    FileStatus.MISSING: "Not present in this directory",
    FileStatus.MALFORMED: "Found, but could not be parsed",
    FileStatus.UNREADABLE: "Found, but could not be read",
    FileStatus.INCONSISTENT: "Found, but describes a different calculation",
}


class FileRow(QWidget):
    """One selectable file entry."""

    clicked = Signal(str)  # role key

    def __init__(self, entry: FileEntry, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._entry = entry
        self._selected = False
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(30)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACING.sm, 0, SPACING.sm, 0)
        layout.setSpacing(SPACING.sm)

        icon = QLabel("🗎")
        icon.setStyleSheet(f"color: {PALETTE.text_tertiary}; font-size: 12px;")
        layout.addWidget(icon)

        self._name = QLabel(entry.role.filename)
        layout.addWidget(self._name, 1)

        glyph, color = STATUS_GLYPH.get(
            entry.status, ("○", PALETTE.text_tertiary)
        )
        self._status = QLabel(glyph)
        self._status.setStyleSheet(f"color: {color}; font-size: 12px;")
        tooltip = STATUS_TOOLTIP.get(entry.status, "")
        if entry.note:
            tooltip = f"{tooltip} — {entry.note}"
        self._status.setToolTip(tooltip)
        layout.addWidget(self._status)

        self._apply_style()

    @property
    def entry(self) -> FileEntry:
        return self._entry

    @property
    def role_key(self) -> str:
        return self._entry.role.key

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._apply_style()

    def _apply_style(self) -> None:
        available = self._entry.exists
        if self._selected:
            background = PALETTE.accent_soft
            weight = "500"
        else:
            background = "transparent"
            weight = "400"
        color = PALETTE.text if available else PALETTE.disabled_text
        self.setStyleSheet(
            f"FileRow {{ background: {background}; border-radius: {RADIUS.sm}px; }}"
        )
        self._name.setStyleSheet(
            f"color: {color}; font-weight: {weight}; font-size: {TYPE.body}px;"
        )

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if not self._selected:
            self.setStyleSheet(
                f"FileRow {{ background: {PALETTE.surface_hover};"
                f" border-radius: {RADIUS.sm}px; }}"
            )
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._apply_style()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._entry.exists:
            self.clicked.emit(self.role_key)
        super().mousePressEvent(event)


class FileSection(QWidget):
    """A collapsible group of file rows."""

    rowClicked = Signal(str)

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._header = QLabel(f"⌄  {title}")
        self._header.setCursor(Qt.PointingHandCursor)
        self._header.setStyleSheet(
            f"color: {PALETTE.text}; font-weight: 600;"
            f" font-size: {TYPE.body}px; padding: 4px;"
        )
        self._header.mousePressEvent = self._toggle  # type: ignore[method-assign]
        layout.addWidget(self._header)

        self._container = QWidget()
        self._rows_layout = QVBoxLayout(self._container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(1)
        layout.addWidget(self._container)

        self._title = title
        self._expanded = True
        self._rows: list[FileRow] = []

    def _toggle(self, _event) -> None:
        self._expanded = not self._expanded
        self._container.setVisible(self._expanded)
        self._header.setText(f"{'⌄' if self._expanded else '›'}  {self._title}")

    def clear(self) -> None:
        for row in self._rows:
            row.setParent(None)
            row.deleteLater()
        self._rows.clear()

    def add_entry(self, entry: FileEntry) -> FileRow:
        row = FileRow(entry)
        row.clicked.connect(self.rowClicked)
        self._rows_layout.addWidget(row)
        self._rows.append(row)
        return row

    def rows(self) -> list[FileRow]:
        return list(self._rows)

    def set_visible_if_populated(self) -> None:
        self.setVisible(bool(self._rows))


class InputsSidebar(QWidget):
    """Header, then one collapsible section per file group."""

    fileSelected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        outer.addWidget(scroll)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(SPACING.lg, SPACING.xl, SPACING.lg, SPACING.xl)
        layout.setSpacing(SPACING.md)
        scroll.setWidget(content)

        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(SPACING.sm, 0, SPACING.sm, 0)
        header_layout.setSpacing(SPACING.xs)
        header_layout.addWidget(Label("Inputs", "TitleLabel"))
        header_layout.addWidget(Label("Manage and inspect input files", "SubtitleLabel"))
        layout.addWidget(header)

        layout.addWidget(Divider())

        files_label = Label("FILES", "SectionLabel")
        files_label.setContentsMargins(SPACING.sm, 0, 0, 0)
        layout.addWidget(files_label)

        self._sections: dict[FileGroup, FileSection] = {}
        for group in (FileGroup.VASP, FileGroup.WANNIER90):
            section = FileSection(group.value)
            section.rowClicked.connect(self._on_row_clicked)
            layout.addWidget(section)
            self._sections[group] = section

        self._empty = Label(
            "No recognised files in this folder.", "TertiaryLabel"
        )
        self._empty.setContentsMargins(SPACING.sm, 0, 0, 0)
        self._empty.setWordWrap(True)
        self._empty.hide()
        layout.addWidget(self._empty)

        layout.addStretch(1)
        self._selected: str | None = None

    # -- population -------------------------------------------------------------

    def populate(self, calculation: Calculation | None) -> None:
        for section in self._sections.values():
            section.clear()
        self._selected = None

        if calculation is None:
            for section in self._sections.values():
                section.hide()
            self._empty.setText("No calculation is open.")
            self._empty.show()
            return

        any_rows = False
        for group, section in self._sections.items():
            entries = calculation.index.in_group(group)
            # Wannier90 files are only worth listing when the calculation shows
            # some sign of a Wannier run; otherwise the section is noise.
            if group is FileGroup.WANNIER90 and not any(e.exists for e in entries):
                section.hide()
                continue
            for entry in entries:
                section.add_entry(entry)
            section.show()
            any_rows = any_rows or bool(entries)

        self._empty.setVisible(not any_rows)
        if not any_rows:
            self._empty.setText("No recognised files in this folder.")

    def select(self, role_key: str) -> None:
        self._selected = role_key
        for section in self._sections.values():
            for row in section.rows():
                row.set_selected(row.role_key == role_key)

    def first_available(self) -> str | None:
        """The file a freshly opened calculation should show."""
        preferred = ["incar", "poscar", "kpoints", "contcar", "outcar"]
        available = {
            row.role_key
            for section in self._sections.values()
            for row in section.rows()
            if row.entry.exists
        }
        for key in preferred:
            if key in available:
                return key
        return next(iter(sorted(available)), None)

    def _on_row_clicked(self, role_key: str) -> None:
        self.select(role_key)
        self.fileSelected.emit(role_key)
