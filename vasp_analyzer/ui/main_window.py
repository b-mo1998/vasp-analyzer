"""The application shell: sidebar, navigation, page stack and information bar.

The shell owns navigation and the current calculation; pages own their content.
Opening a directory runs discovery synchronously (it is a stat-level scan) and
then hands the result to every page, which decides for itself what it can show.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
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
from ..services.discovery import discover, discover_file, discover_with_layout
from ..services.layout import ProjectLayout, classify_directory
from ..services.loader import DataLoader
from ..services.settings import Settings
from .pages.bands.page import BandsPage
from .pages.charge.page import ChargePage
from .pages.inputs.page import InputsPage
from .pages.md.page import MDPage
from .pages.neb.page import NEBPage
from .pages.editor.page import StructureEditorPage
from .pages.phonons.page import PhononsPage
from .pages.structure.page import StructurePage
from .shell.toolbar import PageToolbar
from .widgets.scroll_guard import ScrollGuard
from .theme import METRICS, SPACING
from .widgets.info_bar import InfoBar
from .widgets.segmented import SegmentedControl

log = get_logger("ui.main_window")

PAGES = ["Structure", "Editor", "Bands", "Charge", "MD", "NEB", "Phonons", "Inputs"]
# Which bottom-bar field set each page wants. Pages not listed keep the
# electronic-structure default.
PAGE_INFO_LAYOUT = {
    "Editor": "editor",
    "MD": "md",
    "NEB": "neb",
    "Charge": "charge",
    "Phonons": "phonon",
}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VASP Analyzer")
        # Wide enough that the Structure page's toolbar (title, five icon
        # buttons, the a/b/c orientation bar) never gets compressed
        # below its text width once the two fixed-width side panels are
        # subtracted — that compression is what clipped the old projection
        # pill's labels to "rspect" / "hograp" at the previous 1180px floor.
        # Also wide enough
        # for the 8-tab nav (Editor and Phonons added since) plus the full
        # toolbar to coexist without either one being squeezed.
        self.setMinimumSize(1500, 760)
        self.setAcceptDrops(True)

        self._settings = Settings()
        self._loader = DataLoader(self)
        self._calculation: Calculation | None = None

        # Application-wide, and held as an attribute because an event filter
        # is not owned by the object it is installed on — dropping the last
        # reference would silently uninstall it.
        self._scroll_guard = ScrollGuard(self)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self._scroll_guard)

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

        # Nav and toolbar are separate layout items — never sharing a cell —
        # so they cannot physically overlap regardless of how many pages or
        # toolbar actions get added later. Equal stretch on both sides of the
        # nav keeps it visually centred; the toolbar sits at the true right
        # edge rather than fighting the nav pill for the same space.
        #
        # This replaced an earlier "overlap two widgets in one grid cell,
        # aligned center and right respectively" layout that only worked by
        # coincidence while the centered pill happened to stay narrower than
        # the gap to the toolbar. Adding the Editor and Phonons tabs (6 -> 8)
        # pushed the pill's right edge into the toolbar's buttons.
        header = QWidget()
        header_row = QHBoxLayout(header)
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(SPACING.md)

        self._nav = SegmentedControl(PAGES)
        self._nav.currentChanged.connect(self._on_page_changed)
        header_row.addStretch(1)
        header_row.addWidget(self._nav)
        header_row.addStretch(1)

        self._toolbar = PageToolbar()
        header_row.addWidget(self._toolbar)
        content_layout.addWidget(header)

        self._page_stack = QStackedWidget()
        content_layout.addWidget(self._page_stack, 1)

        # One InfoBar, moved between homes: the 3D pages offer a slot at the
        # foot of their viewer column so the bar ends level with the viewer
        # card and the side panel runs on past it; every other page shows it
        # here, in a full-width wrapper under the page stack.
        self._info_bar = InfoBar()
        self._gap: tuple[str, str] = ("—", "")
        self._info_bar_wrap = QWidget()
        self._info_bar_wrap_layout = QHBoxLayout(self._info_bar_wrap)
        self._info_bar_wrap_layout.setContentsMargins(0, 0, 0, 0)
        self._info_bar_wrap_layout.setSpacing(0)
        self._info_bar_wrap_layout.addWidget(self._info_bar)
        content_layout.addWidget(self._info_bar_wrap)

        self.setCentralWidget(root)

        # --- pages ------------------------------------------------------------
        self._structure = StructurePage(self._loader)
        self._register_page(self._structure)
        self._editor = StructureEditorPage()
        self._register_page(self._editor)
        self._bands = BandsPage(self._loader)
        self._register_page(self._bands)
        self._bands.gapComputed.connect(self._on_gap_computed)
        self._charge = ChargePage(self._loader)
        self._register_page(self._charge)
        self._editor.infoRequested.connect(self._on_page_info)
        self._charge.infoRequested.connect(self._on_page_info)
        self._md = MDPage(self._loader)
        self._register_page(self._md)
        self._md.infoRequested.connect(self._on_page_info)
        self._neb = NEBPage(self._loader)
        self._register_page(self._neb)
        self._neb.infoRequested.connect(self._on_page_info)
        self._phonons = PhononsPage(self._loader)
        self._register_page(self._phonons)
        self._phonons.infoRequested.connect(self._on_page_info)
        self._inputs = InputsPage()
        self._register_page(self._inputs)

        # Clicking a file on the Structure page opens it on the Inputs page.
        self._structure.navigationRequested.connect(self._open_input_file)

        self._connect_toolbar()

    def _register_page(self, page) -> None:
        self._page_stack.addWidget(page)
        self._sidebar_stack.addWidget(page.sidebar())
        if hasattr(page, "issueRaised"):
            page.issueRaised.connect(self._show_issue_details)
        if hasattr(page, "infoBarSlotChanged"):
            page.infoBarSlotChanged.connect(self._place_info_bar)

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
        # The starting page never emits currentChanged, so its label has to be
        # applied directly or the button would read "Reset" until the first
        # navigation.
        self._apply_page_action(self._nav.currentIndex())

    def _build_menus(self) -> None:
        menu = self.menuBar()

        file_menu = menu.addMenu("File")
        open_action = QAction("Open Calculation…", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_directory_dialog)
        file_menu.addAction(open_action)

        open_file_action = QAction("Open File…", self)
        open_file_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
        open_file_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_file_action)

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

    def open_file_dialog(self) -> None:
        start = self._settings.last_directory()
        path, _selected = QFileDialog.getOpenFileName(
            self,
            "Open a VASP File",
            str(start) if start else str(Path.home()),
            "VASP files (POSCAR* CONTCAR* CHGCAR* CHG* ELFCAR* LOCPOT* PARCHG*"
            " XDATCAR* EIGENVAL* DOSCAR* INCAR* KPOINTS* OUTCAR* vasprun.xml);;"
            "All files (*)",
        )
        if path:
            self.open_file(Path(path))

    def open_directory(self, directory: Path) -> None:
        log.info("Opening %s", directory)
        calculation = discover(directory)
        self._apply_calculation(calculation, opened_path=directory, record_recent=True)

    def _apply_calculation(
        self, calculation: Calculation, opened_path: Path, record_recent: bool
    ) -> None:
        """Make ``calculation`` the current one and propagate it everywhere.

        Shared by a full open and by attaching a single md/neb/band sub-root
        to whatever is already open — the latter still runs every page's
        ``set_calculation`` again, which looks wasteful but isn't: the other
        two roots are unchanged, so their pages just re-request data the
        loader already has cached and settle back to the same view.
        """
        self._calculation = calculation
        self._loader.set_calculation(calculation)

        self.setWindowTitle(f"VASP Analyzer — {opened_path.name}")
        if record_recent:
            self._settings.set_last_directory(opened_path)
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
        # Restoring the previous session is deferred to the first event-loop
        # turn so the window can paint first -- which means anything that
        # opens a calculation *before* that turn (a path on the command line,
        # a Finder "Open With", a drop onto a window that has only just
        # appeared) would be silently thrown away here a moment later. If
        # something is already open, this has nothing left to restore.
        if self._calculation is not None:
            return

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
        """Reload Calculation restarts the whole process rather than just
        re-reading files from disk. A stale in-memory Qt layout or widget
        state (the kind of bug this app has had — a pop-out window slowly
        growing its host card on every use, say) is not fixed by re-parsing
        the calculation, because nothing about that touches the accumulated
        state that caused it; only a genuinely fresh process is. The
        currently open directory, if any, is passed to the new process so it
        reopens where this one left off rather than landing on a blank
        window."""
        # Asked before anything is torn down, because a restart is exactly as
        # destructive as closing the window: unsaved structure edits, the
        # current selection and every view setting go with the old process.
        # "Reload" sounds like a cheap refresh, so the dialog has to say what
        # it actually does rather than let the label imply otherwise.
        confirm = QMessageBox(self)
        confirm.setWindowTitle("Reload Calculation")
        confirm.setIcon(QMessageBox.Warning)
        confirm.setText("Reload this calculation?")
        confirm.setInformativeText(
            "VASP Analyzer will close and reopen. Any unsaved changes — "
            "including edits in the Structure Editor that have not been "
            "saved to a file — will be lost."
        )
        reload_button = confirm.addButton("Reload", QMessageBox.DestructiveRole)
        confirm.addButton("Cancel", QMessageBox.RejectRole)
        confirm.setDefaultButton(reload_button)
        confirm.exec()
        if confirm.clickedButton() is not reload_button:
            return

        directory = (
            self._calculation.display_directory if self._calculation is not None else None
        )
        self._settings.set_window_geometry(self.saveGeometry())
        self._settings.sync()
        self._restart_process(directory)

    @staticmethod
    def restart_argv(directory: Path | None) -> tuple[str, list[str]]:
        """The (executable, argv) a restart should exec, built from scratch.

        Rebuilt rather than derived from the live ``sys.argv``, which was
        wrong in all three of the ways it could be:

        * Under ``python -m vasp_analyzer`` — how the app is normally run in
          development — ``sys.argv[0]`` is the *path to* ``__main__.py``.
          Re-running that as a plain script gives it no package context, so
          it dies immediately on its own relative imports. The restart has
          to go back through ``-m``.
        * In a PyInstaller bundle ``sys.executable`` and ``sys.argv[0]`` are
          both the app binary, so passing the latter through as an argument
          pushed the directory to ``argv[2]`` where nothing looked for it,
          and the restarted app came up blank.
        * Either way, appending the directory to the *previous* argv meant a
          second reload appended it again, and a third a third time.
        """
        executable = sys.executable
        if getattr(sys, "frozen", False):
            args = [executable]
        else:
            args = [executable, "-m", "vasp_analyzer"]
        if directory is not None:
            args.append(str(directory))
        return executable, args

    def _restart_process(self, directory: Path | None) -> None:  # pragma: no cover - replaces the process
        executable, args = self.restart_argv(directory)
        log.info("Restarting: %s", " ".join(args))
        os.execv(executable, args)

    # -- navigation ---------------------------------------------------------------

    def _on_page_changed(self, index: int) -> None:
        self._page_stack.setCurrentIndex(index)
        self._sidebar_stack.setCurrentIndex(index)
        self._apply_info_layout(index)
        self._apply_page_action(index)
        self._place_info_bar()

    def _place_info_bar(self) -> None:
        """Put the bottom bar where the visible page wants it.

        A 3D page showing content hands back the layout at the foot of its
        viewer column and the bar moves in there; any other page, or a 3D
        page on its full-width empty card, gets it back in the wrapper.
        Re-run on every page switch and whenever a page flips between its
        content and empty states.
        """
        page = self._current_page()
        slot = getattr(page, "info_bar_slot", None)
        target = slot() if callable(slot) else None
        if target is not None:
            target.addWidget(self._info_bar)
            self._info_bar_wrap.hide()
        else:
            self._info_bar_wrap_layout.addWidget(self._info_bar)
            self._info_bar_wrap.show()
        # addWidget reparents, and a reparented widget comes back hidden
        # until the next event-loop turn; show it now so a page switch never
        # paints a frame with no bar at all.
        self._info_bar.show()

    def _apply_page_action(self, index: int) -> None:
        """Let the visible page name the toolbar's first button.

        A page that holds unsaved work wants that slot to say what it does to
        that work — "Reset" is fine for a set of view controls and far too
        vague for a structure with edits in it.
        """
        page = self._page_stack.widget(index)
        action = getattr(page, "TOOLBAR_ACTION", None)
        label, tooltip = action if action else PageToolbar.DEFAULT_ACTION
        self._toolbar.set_page_action(label, tooltip)

    def _apply_info_layout(self, index: int) -> None:
        """Give the bottom bar the field set the visible page describes."""
        name = PAGES[index] if 0 <= index < len(PAGES) else ""
        layout = PAGE_INFO_LAYOUT.get(name, "default")
        self._info_bar.set_layout(layout)
        if layout == "default":
            self._info_bar.update_from(self._calculation)
        else:
            page = self._page_stack.widget(index)
            # Re-ask the page for its fields: it may have loaded its data while
            # another page was in front, and the strip was showing something
            # else at the time.
            refresh = getattr(page, "refresh_info", None)
            if callable(refresh):
                refresh()

    def _on_page_info(self, fields: dict, colors: dict | None = None) -> None:
        """A page reporting the values for its own bottom-bar layout."""
        sender = self.sender()
        if sender is not self._current_page():
            return
        self._info_bar.set_fields(fields, colors)
        # The gap is the Bands page's to know, not the reporting page's:
        # re-apply the last value it announced so a layout that carries a
        # Gap field does not show "—" merely because a different page
        # filled in the rest.
        self._info_bar.set_gap(*self._gap)

    def _on_gap_computed(self, text: str, tooltip: str) -> None:
        self._gap = (text, tooltip)
        self._info_bar.set_gap(text, tooltip)

    def _current_page(self):
        return self._page_stack.currentWidget()

    def _open_input_file(self, role_key: str) -> None:
        """Jump to the Inputs page with a particular file selected."""
        self._nav.setCurrentIndex(PAGES.index("Inputs"))
        self._inputs.show_file(role_key)

    # -- toolbar actions -----------------------------------------------------------

    def _reset_current_page(self) -> None:
        page = self._current_page()
        # A page that renamed the button also supplies what it now does; the
        # label and the behaviour must not be able to drift apart.
        action = getattr(page, "toolbar_action", None)
        if callable(action):
            action()
            return
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
            subprocess.run(
                ["open", str(self._calculation.display_directory)], check=False
            )

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
        if calculation.is_multi_root:
            lines = [
                f"Project folder: {calculation.display_directory}",
                f"  Structure / Bands / Charge / Inputs: {calculation.directory}",
                f"  MD: {calculation.md_directory}",
                f"  NEB: {calculation.neb_directory}",
                "",
            ]
        else:
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
        box.setText(calculation.name)
        box.setDetailedText("\n".join(lines))
        box.exec()

    # -- drag and drop -------------------------------------------------------------------

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # Files as well as folders: dropping a lone POSCAR or CHGCAR is a
        # real thing to want, and refusing the drag outright gave no hint
        # that the app had an opinion about it at all.
        urls = event.mimeData().urls()
        if urls and Path(urls[0].toLocalFile()).exists():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt naming
        urls = event.mimeData().urls()
        if not urls:
            return
        path = Path(urls[0].toLocalFile())
        if path.is_dir():
            self._handle_dropped_directory(path)
            event.acceptProposedAction()
        elif path.is_file():
            self.open_file(path)
            event.acceptProposedAction()

    def open_file(self, path: Path) -> None:
        """Open one file on its own, as much of a calculation as it is.

        Everything downstream binds to capabilities rather than filenames, so
        a POSCAR lights up Structure, Editor and Inputs while a CHGCAR adds
        Charge Density on top — and every other page shows its ordinary
        "not found" state naming what is missing, which is honest and, more
        to the point, does not crash.
        """
        log.info("Opening file %s", path)
        calculation = discover_file(path)
        self._apply_calculation(calculation, opened_path=path, record_recent=False)
        # The window title and Reveal in Finder work off display_directory,
        # which for a single file is its folder; name the file itself instead
        # so it is obvious what is actually open.
        self.setWindowTitle(f"VASP Analyzer — {path.name}")

        unrecognised = [i for i in calculation.issues if i.code == "file_unrecognised"]
        if unrecognised:
            self._show_issue_details(unrecognised[0])

    def _handle_dropped_directory(self, path: Path) -> None:
        """Open a dropped folder, or attach it as one sub-root of the project
        that's already open.

        With nothing open yet, or when the folder doesn't look like it
        belongs to one specific page family, this is an ordinary full open —
        which may itself resolve into a multi-root project. Otherwise only
        the matching sub-root is replaced, so re-dropping (say) an updated
        "neb" folder onto an open project refreshes just the NEB page and
        leaves Structure/Bands/Charge/MD exactly where they were.
        """
        current = self._calculation
        if current is None:
            self.open_directory(path)
            return

        kind = classify_directory(path)
        if kind is None:
            self.open_directory(path)
            return

        layout = ProjectLayout(
            electronic=path if kind == "band" else current.directory,
            md=path if kind == "md" else current.md_directory,
            neb=path if kind == "neb" else current.neb_directory,
        )
        log.info("Attaching %s (%s) to the open project", path, kind)
        calculation = discover_with_layout(current.display_directory, layout)
        self._apply_calculation(
            calculation, opened_path=current.display_directory, record_recent=False
        )

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
