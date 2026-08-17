"""The application shell: sidebar, navigation, page stack and information bar.

The shell owns navigation and the current calculation; pages own their content.
Opening a directory runs discovery synchronously (it is a stat-level scan) and
then hands the result to every page, which decides for itself what it can show.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.errors import Issue
from ..core.logging_setup import get_logger, log_file
from ..models.calculation import Calculation
from ..services.discovery import discover
from ..services.loader import DataLoader
from ..services.settings import Settings
from .pages.bands.page import BandsPage
from .shell.toolbar import PageToolbar
from .theme import METRICS, SPACING
from .widgets.info_bar import InfoBar
from .widgets.segmented import SegmentedControl

log = get_logger("ui.main_window")

PAGES = ["Bands", "Structure", "Inputs"]


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VASP Analyzer")
        self.setMinimumSize(1180, 720)
        self.setAcceptDrops(True)

        self._settings = Settings()
        self._loader = DataLoader(self)
        self._calculation: Calculation | None = None

        self._build_ui()
        self._build_menus()
        self._restore_geometry()

        QTimer.singleShot(0, self._open_last_directory)

    # -- construction --------------------------------------------------------

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("Root")
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- sidebar ---------------------------------------------------------
        self._sidebar_stack = QStackedWidget()
        self._sidebar_stack.setObjectName("Sidebar")
        self._sidebar_stack.setFixedWidth(METRICS.sidebar_width)
        layout.addWidget(self._sidebar_stack)

        # --- content ---------------------------------------------------------
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(
            SPACING.lg, SPACING.lg, SPACING.lg, SPACING.lg
        )
        content_layout.setSpacing(SPACING.md)
        layout.addWidget(content, 1)

        # Navigation and toolbar share one grid cell so the pill stays centred
        # regardless of how wide the toolbar becomes.
        header = QWidget()
        header_grid = QGridLayout(header)
        header_grid.setContentsMargins(0, 0, 0, 0)

        self._nav = SegmentedControl(PAGES)
        self._nav.currentChanged.connect(self._on_page_changed)
        header_grid.addWidget(self._nav, 0, 0, Qt.AlignCenter)

        self._toolbar = PageToolbar()
        header_grid.addWidget(self._toolbar, 0, 0, Qt.AlignRight | Qt.AlignVCenter)
        content_layout.addWidget(header)

        self._page_stack = QStackedWidget()
        content_layout.addWidget(self._page_stack, 1)

        self._info_bar = InfoBar()
        content_layout.addWidget(self._info_bar)

        self.setCentralWidget(root)

        # --- pages ------------------------------------------------------------
        self._bands = BandsPage(self._loader)
        self._register_page(self._bands)
        self._register_placeholder("Structure")
        self._register_placeholder("Inputs")

        self._nav.setItemEnabled(1, False, "The Structure page is not built yet.")
        self._nav.setItemEnabled(2, False, "The Inputs page is not built yet.")

        self._connect_toolbar()

    def _register_page(self, page) -> None:
        self._page_stack.addWidget(page)
        self._sidebar_stack.addWidget(page.sidebar())
        if hasattr(page, "issueRaised"):
            page.issueRaised.connect(self._show_issue_details)

    def _register_placeholder(self, name: str) -> None:
        from .widgets.empty_state import EmptyState
        from .widgets.primitives import Card

        card = Card()
        card.body().addWidget(
            EmptyState(
                title=f"{name} page",
                message="This page is part of a later phase.",
                glyph="◇",
            )
        )
        self._page_stack.addWidget(card)
        self._sidebar_stack.addWidget(QWidget())

    def _connect_toolbar(self) -> None:
        bar = self._toolbar
        bar.resetRequested.connect(self._reset_current_page)
        bar.copyRequested.connect(self._copy_current_page)
        bar.exportRequested.connect(self._export_current_page)
        bar.openRequested.connect(self.open_directory_dialog)
        bar.reloadRequested.connect(self._reload)
        bar.revealRequested.connect(self._reveal_in_finder)
        bar.logRequested.connect(self._reveal_log)
        self._info_bar.detailsRequested.connect(self._show_calculation_details)

    def _build_menus(self) -> None:
        menu = self.menuBar()

        file_menu = menu.addMenu("File")
        open_action = QAction("Open Calculation…", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_directory_dialog)
        file_menu.addAction(open_action)

        self._recent_menu = file_menu.addMenu("Open Recent")
        self._rebuild_recent_menu()

        file_menu.addSeparator()
        reload_action = QAction("Reload Calculation", self)
        reload_action.setShortcut(QKeySequence.Refresh)
        reload_action.triggered.connect(self._reload)
        file_menu.addAction(reload_action)

        view_menu = menu.addMenu("View")
        for index, name in enumerate(PAGES):
            action = QAction(name, self)
            action.setShortcut(QKeySequence(f"Ctrl+{index + 1}"))
            action.triggered.connect(lambda _=False, i=index: self._nav.setCurrentIndex(i))
            view_menu.addAction(action)

        help_menu = menu.addMenu("Help")
        log_action = QAction("Show Log File", self)
        log_action.triggered.connect(self._reveal_log)
        help_menu.addAction(log_action)

    def _rebuild_recent_menu(self) -> None:
        self._recent_menu.clear()
        recent = self._settings.recent_directories()
        if not recent:
            empty = QAction("No Recent Calculations", self)
            empty.setEnabled(False)
            self._recent_menu.addAction(empty)
            return
        for path in recent:
            action = QAction(path.name, self)
            action.setToolTip(str(path))
            action.triggered.connect(lambda _=False, p=path: self.open_directory(p))
            self._recent_menu.addAction(action)
        self._recent_menu.addSeparator()
        clear = QAction("Clear Menu", self)
        clear.triggered.connect(
            lambda: (self._settings.clear_recent(), self._rebuild_recent_menu())
        )
        self._recent_menu.addAction(clear)

    # -- calculation lifecycle --------------------------------------------------

    def open_directory_dialog(self) -> None:
        start = self._settings.last_directory()
        directory = QFileDialog.getExistingDirectory(
            self,
            "Open Calculation Directory",
            str(start) if start else str(Path.home()),
        )
        if directory:
            self.open_directory(Path(directory))

    def open_directory(self, directory: Path) -> None:
        log.info("Opening %s", directory)
        calculation = discover(directory)
        self._calculation = calculation
        self._loader.set_calculation(calculation)

        self.setWindowTitle(f"VASP Analyzer — {directory.name}")
        self._settings.set_last_directory(directory)
        self._rebuild_recent_menu()

        self._info_bar.update_from(calculation)
        self._toolbar.set_actions_enabled(not calculation.is_empty)

        for index in range(self._page_stack.count()):
            page = self._page_stack.widget(index)
            if hasattr(page, "set_calculation"):
                page.set_calculation(calculation)

        for issue in calculation.issues:
            log.info("Discovery issue: %s | %s", issue.message, issue.detail)

    def _open_last_directory(self) -> None:
        last = self._settings.last_directory()
        if last is not None:
            self.open_directory(last)
        else:
            self._info_bar.clear()
            self._toolbar.set_actions_enabled(False)
            for index in range(self._page_stack.count()):
                page = self._page_stack.widget(index)
                if hasattr(page, "set_calculation"):
                    page.set_calculation(None)

    def _reload(self) -> None:
        if self._calculation is not None:
            self._loader.clear_cache()
            self.open_directory(self._calculation.directory)

    # -- navigation ---------------------------------------------------------------

    def _on_page_changed(self, index: int) -> None:
        self._page_stack.setCurrentIndex(index)
        self._sidebar_stack.setCurrentIndex(index)

    def _current_page(self):
        return self._page_stack.currentWidget()

    # -- toolbar actions -----------------------------------------------------------

    def _reset_current_page(self) -> None:
        page = self._current_page()
        if hasattr(page, "reset"):
            page.reset()

    def _copy_current_page(self) -> None:
        page = self._current_page()
        # A page that can render its own figure produces a far better image
        # than a screen grab; falling back keeps the action always useful.
        if hasattr(page, "copy_to_clipboard") and page.copy_to_clipboard():
            self.statusBar().showMessage("Copied to clipboard", 2000)
            return
        QGuiApplication.clipboard().setPixmap(page.grab())
        self.statusBar().showMessage("Copied to clipboard", 2000)

    def _export_current_page(self) -> None:
        page = self._current_page()
        if hasattr(page, "export"):
            page.export()

    def _reveal_in_finder(self) -> None:
        if self._calculation is None:
            return
        if sys.platform == "darwin":
            subprocess.run(["open", str(self._calculation.directory)], check=False)

    def _reveal_log(self) -> None:
        path = log_file()
        if sys.platform == "darwin" and path.exists():
            subprocess.run(["open", "-R", str(path)], check=False)

    # -- details ---------------------------------------------------------------------

    def _show_issue_details(self, issue: Issue) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("Details")
        box.setIcon(QMessageBox.Information)
        box.setText(issue.message)
        box.setDetailedText(issue.detail or "No further detail was recorded.")
        box.exec()

    def _show_calculation_details(self) -> None:
        calculation = self._calculation
        if calculation is None:
            return
        lines = [f"Directory: {calculation.directory}", ""]
        for entry in calculation.index.entries.values():
            if entry.role.hidden:
                continue
            lines.append(
                f"{entry.role.filename:<20} {entry.status.value:<14} {entry.size_label}"
                + (f"  — {entry.note}" if entry.note else "")
            )
        if calculation.issues:
            lines.append("")
            lines.append("Notes:")
            lines.extend(f"  • {issue.message}" for issue in calculation.issues)

        box = QMessageBox(self)
        box.setWindowTitle("Calculation Details")
        box.setIcon(QMessageBox.Information)
        box.setText(f"{calculation.directory.name}")
        box.setDetailedText("\n".join(lines))
        box.exec()

    # -- drag and drop -------------------------------------------------------------------

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        urls = event.mimeData().urls()
        if urls and Path(urls[0].toLocalFile()).is_dir():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt naming
        urls = event.mimeData().urls()
        if not urls:
            return
        path = Path(urls[0].toLocalFile())
        if path.is_dir():
            self.open_directory(path)
            event.acceptProposedAction()

    # -- geometry ----------------------------------------------------------------------

    def _restore_geometry(self) -> None:
        geometry = self._settings.window_geometry()
        if geometry is not None:
            self.restoreGeometry(geometry)
        else:
            self.resize(1440, 900)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._settings.set_window_geometry(self.saveGeometry())
        self._settings.sync()
        super().closeEvent(event)
