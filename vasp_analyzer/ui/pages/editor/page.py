"""The Editor page: a lightweight structure-editing workspace.

The page owns the only mutable structure in the application, and every change
funnels through :meth:`_commit`. That single choke point is what makes the
editor trustworthy: it is where the undo entry is recorded, where the views
are refreshed, and where the Modified flag is set, so no operation can change
the structure without also being undoable and visible.

Three structures are kept apart on purpose:

``_original``   what was parsed from disk — never written to, so "Reset to
                Source" always has somewhere to go back to.
``_saved``      the last state the user explicitly saved or applied.
``_working``    what is being edited and drawn right now.

Editing is not saving. Nothing here writes to disk except the explicit
Save/Export actions, and those never silently overwrite the file the
calculation was read from.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ....core.errors import Issue, Severity
from ....core.logging_setup import get_logger
from ....models.calculation import Calculation
from ....models.editable import EditableStructure
from ....services.editing import CommandStack, EditCommand
from ....services.loader import STRUCTURE_SOURCE_KEYS, read_structure
from ....services.supercell import build_supercell
from ...style import icon_url
from ...theme import METRICS, PALETTE, RADIUS, SPACING, TYPE
from ...widgets.axis_bar import AxisBar
from ...widgets.element_swatch import ElementSwatch
from ...widgets.empty_state import EmptyState
from ...widgets.notice import Notice
from ...widgets.popout import PopoutButton
from ...widgets.primitives import Card, Divider
from ...widgets.segmented import SegmentedControl
from ..base import Page
from ..structure.export import save_cif, save_poscar
from ..structure.viewer import ViewOptions
from .inspector import (
    AddAtomCard,
    OperationsCard,
    SaveCard,
    SelectionCard,
    SupercellCard,
)
from .sidebar import EditorSidebar
from .table import COL_X, AtomTable
from .viewer import MAX_NEIGHBOR_HANDLES, NEIGHBOR_DISTANCE_RATIO, EditorView

log = get_logger("ui.editor")

# Snap increments. Fractional coordinates are dimensionless, so the two modes
# need different steps to feel the same on screen.
SNAP_FRACTIONAL = 0.01
SNAP_CARTESIAN = 0.05

# Below this separation two atoms are almost certainly a duplicate rather than
# a real bond — no element pair sits this close.
OVERLAP_THRESHOLD = 0.8

class StructureEditorPage(Page):
    """Edit a periodic structure and export it as a valid VASP POSCAR."""

    # Claims the toolbar's first slot. This page is the only one whose state
    # is unsaved *work* rather than view settings, so the global button does
    # the same thing as the Revert Changes button beside Save Structure —
    # two routes to one behaviour, rather than a "Reset" that would sit next
    # to Copy/Export looking like it only touched the display.
    TOOLBAR_ACTION = ("Revert Changes", "Discard edits and return to the last saved structure")

    title = "Editor"
    issueRaised = Signal(object)
    infoRequested = Signal(dict, dict)   # values, colours -- for the bottom bar
    infoBarSlotChanged = Signal()       # where the bottom bar belongs changed

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sidebar = EditorSidebar()

        self._original: EditableStructure | None = None
        self._saved: EditableStructure | None = None
        self._working: EditableStructure | None = None
        self._selection: set[int] = set()          # stable atom ids
        # Atoms hidden from the viewer, also by stable id — so an id survives
        # rows shifting under it when other atoms are added or deleted, and a
        # hidden atom that is later deleted simply drops out of the set. This
        # is purely a view state: hidden atoms are still in the structure,
        # still in the table, and still written out on save.
        self._hidden: set[int] = set()
        self._history = CommandStack()
        self._modified = False
        self._fractional = True
        self._filter: set[str] = set()
        self._sources: dict[str, Path] = {}
        self._drag_origin: EditableStructure | None = None

        self._build()
        self._connect()
        self._install_shortcuts()
        # Rotate is the default, and it is set through the same path a click
        # takes so the viewer's real mode and the highlighted button cannot
        # disagree. Previously the toolbar merely called setChecked(True) on
        # its first button, which paints a button active without emitting
        # `clicked` -- so _set_tool() never ran and the strip showed "Select"
        # highlighted while the viewer was actually still on
        # StructureView.__init__'s own MODE_ROTATE default. Orbiting was the
        # behaviour people expected on opening the page; the bug was only
        # ever that the toolbar lied about which tool that was.
        self._set_tool("rotate")

    # -- construction ------------------------------------------------------

    def _build(self) -> None:
        layout = self.body()
        layout.setSpacing(SPACING.md)

        self._notice = Notice()
        self._notice.detailsRequested.connect(self.issueRaised)
        layout.addWidget(self._notice)

        self._stack = QStackedWidget()
        layout.addWidget(self._stack, 1)

        content = QWidget()
        row = QHBoxLayout(content)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING.md)

        centre = QWidget()
        column = QVBoxLayout(centre)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(SPACING.md)
        # The viewer is the workspace and keeps the lion's share of the height;
        # the table is a companion to it, not a peer.
        column.addWidget(self._build_viewer_card(), 5)
        column.addWidget(self._build_table_card(), 3)
        self._viewer_column = column
        row.addWidget(centre, 1)
        self._right_column = self._build_inspector()
        row.addWidget(self._right_column, 0)
        self._stack.addWidget(content)
        self._stack.currentChanged.connect(lambda _i: self.infoBarSlotChanged.emit())

        empty_card = Card()
        self._empty = EmptyState(
            title="No structure to edit",
            message="Open a calculation containing a POSCAR or CONTCAR to begin editing.",
            glyph="◇",
        )
        empty_card.body().addWidget(self._empty)
        self._stack.addWidget(empty_card)
        self._stack.setCurrentIndex(1)


    # -- bottom info bar -------------------------------------------------------

    def info_bar_slot(self):
        """Where the window's info bar should sit while this page is in front.

        The foot of the viewer column: the bar then ends level with the viewer
        card, and the panel on the right runs on past it to the bottom of the
        page instead of stopping short with a blank strip beneath it. None on
        the empty card, which spans the full width and wants the bar to match.
        """
        return self._viewer_column if self._stack.currentIndex() == 0 else None

    def _build_viewer_card(self) -> QWidget:
        card = Card(padding=SPACING.lg)
        body = card.body()
        body.setSpacing(SPACING.sm)

        holder = QWidget()
        holder_layout = QVBoxLayout(holder)
        holder_layout.setContentsMargins(0, 0, 0, 0)
        self._view = EditorView()
        self._view.setMinimumHeight(300)
        holder_layout.addWidget(self._view)

        self._toolbar = self._build_floating_toolbar(holder)
        # Orientation presets in place of the projection/along-axis dropdown:
        # the same six directions, but one click each and readable at a glance
        # rather than hidden behind a closed combo box.
        self._axis_bar = AxisBar(holder)
        self._axis_bar.orientationRequested.connect(self._view.orient)
        self._popout = PopoutButton(
            holder, "Structure Editor", holder, shortcuts=self._shortcut_specs()
        )
        self._legend = self._build_floating_legend(holder)

        holder.installEventFilter(self)
        self._holder = holder
        body.addWidget(holder, 1)
        return card

    def _build_floating_toolbar(self, parent: QWidget) -> QWidget:
        bar = QWidget(parent)
        bar.setObjectName("ViewerToolbar")
        bar.setAttribute(Qt.WA_StyledBackground, True)
        row = QHBoxLayout(bar)
        # No margins: the strip's background is transparent, so a margin here
        # only pushed its buttons 4px below the orientation bar and pop-out
        # button, which every page places at the same offset. All three now
        # share one top edge and one 28px height.
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING.xs)

        entries = [
            ("select.svg", "Select  (S)", lambda: self._set_tool("select")),
            ("rotate.svg", "Rotate the camera  (R)", lambda: self._set_tool("rotate")),
            ("pan.svg", "Pan the camera  (T)", lambda: self._set_tool("pan")),
            # Measure belongs here and on the Structure page only: a distance
            # or angle describes one fixed geometry, which is exactly what
            # these two pages show. On a trajectory or reaction path the
            # geometry changes frame to frame, so a measurement would only be
            # true of whichever frame happened to be on screen.
            ("measure.svg", "Measure — two atoms for a distance, three for an "
                            "angle  (M)", lambda: self._set_tool("measure")),
        ]
        self._camera_group = QButtonGroup(self)
        self._camera_group.setExclusive(True)
        for index, (icon_name, tip, slot) in enumerate(entries):
            button = QPushButton()
            button.setIcon(QIcon(icon_url(icon_name)))
            button.setIconSize(QSize(15, 15))
            button.setObjectName("ToolButton")
            button.setCheckable(True)
            button.setFixedSize(32, METRICS.viewer_control_height)
            button.setToolTip(tip)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(slot)
            self._camera_group.addButton(button, index)
            row.addWidget(button)

        for icon_name, tip, slot in (
            ("zoom-in.svg", "Zoom in", lambda: self._view.zoom_by(1.2)),
            ("zoom-out.svg", "Zoom out", lambda: self._view.zoom_by(1 / 1.2)),
            ("fit.svg", "Fit the structure in view", self._fit_view),
        ):
            button = QPushButton()
            button.setIcon(QIcon(icon_url(icon_name)))
            button.setIconSize(QSize(15, 15))
            button.setObjectName("ToolButton")
            button.setFixedSize(32, METRICS.viewer_control_height)
            button.setToolTip(tip)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(slot)
            row.addWidget(button)

        return bar

    def _build_floating_legend(self, parent: QWidget) -> QWidget:
        legend = QWidget(parent)
        legend.setObjectName("ViewerLegend")
        legend.setAttribute(Qt.WA_StyledBackground, True)
        self._legend_layout = QVBoxLayout(legend)
        self._legend_layout.setContentsMargins(
            SPACING.md, SPACING.sm, SPACING.md, SPACING.sm
        )
        self._legend_layout.setSpacing(SPACING.xs)
        legend.setStyleSheet(
            f"#ViewerLegend {{ background: rgba(38,39,43,0.88);"
            f" border: 1px solid rgba(255,255,255,0.14);"
            f" border-radius: {RADIUS.md}px; }}"
        )
        legend.hide()
        return legend

    def _build_table_card(self) -> QWidget:
        card = Card(padding=SPACING.lg)
        body = card.body()
        body.setSpacing(SPACING.sm)

        header = QHBoxLayout()
        header.setSpacing(SPACING.md)
        self._table_title = QLabel("Atom List")
        self._table_title.setObjectName("CardTitle")
        header.addWidget(self._table_title)

        self._search = QLineEdit()
        self._search.setObjectName("ValueField")
        self._search.setPlaceholderText("Search atoms…")
        self._search.setFixedWidth(190)
        self._search.setFixedHeight(28)
        self._search.textChanged.connect(self._apply_table_filter)
        header.addWidget(self._search)

        self._element_filter = QComboBox()
        self._element_filter.setFixedHeight(28)
        self._element_filter.setFixedWidth(96)
        self._element_filter.currentTextChanged.connect(
            lambda _t: self._apply_table_filter(self._search.text())
        )
        header.addWidget(self._element_filter)
        header.addStretch(1)

        self._save_structure = QPushButton("Save Structure")
        self._save_structure.setObjectName("PrimaryButton")
        self._save_structure.setCursor(Qt.PointingHandCursor)
        self._save_structure.setFixedHeight(30)
        self._save_structure.setToolTip(
            "Commit the working structure for this session — does not write a file"
        )
        self._save_structure.clicked.connect(self._apply_changes)
        header.addWidget(self._save_structure)

        self._revert = QPushButton("Revert Changes")
        self._revert.setObjectName("ToolButton")
        self._revert.setCursor(Qt.PointingHandCursor)
        self._revert.setFixedHeight(30)
        self._revert.clicked.connect(self._revert_changes)
        header.addWidget(self._revert)

        self._overflow = QPushButton("⋮")
        self._overflow.setObjectName("ToolButton")
        self._overflow.setFixedSize(34, 30)
        self._overflow.setCursor(Qt.PointingHandCursor)
        self._overflow.clicked.connect(self._show_overflow)
        header.addWidget(self._overflow)
        body.addLayout(header)

        self._table = AtomTable()
        body.addWidget(self._table, 1)
        return card

    def _build_inspector(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        # Wide enough for three coordinate fields plus their row label without
        # a horizontal scrollbar, which would put the Delete button off-screen.
        scroll.setFixedWidth(392)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        column = QVBoxLayout(content)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(SPACING.md)

        self._selection_card = SelectionCard()
        self._add_card = AddAtomCard()
        self._supercell_card = SupercellCard()
        self._operations_card = OperationsCard()
        self._save_card = SaveCard()
        for card in (
            self._selection_card,
            self._add_card,
            self._supercell_card,
            self._operations_card,
            self._save_card,
        ):
            column.addWidget(card)
        column.addStretch(1)

        scroll.setWidget(content)
        return scroll

    # -- wiring ------------------------------------------------------------

    def _connect(self) -> None:
        side = self._sidebar
        side.toolChanged.connect(self._on_tool_changed)
        side.styleChanged.connect(lambda _s: self._apply_view_options())
        side.coloringChanged.connect(lambda _s: self._apply_view_options())
        side.coordinateModeChanged.connect(self._set_coordinate_mode)
        side.snapChanged.connect(lambda _v: self._apply_snap())
        side.constrainChanged.connect(lambda _v: None)
        side.filterChanged.connect(self._on_element_filter)
        side.undoRequested.connect(self._undo)
        side.redoRequested.connect(self._redo)
        side.clearSelectionRequested.connect(lambda: self._set_selection(set()))
        side.hideSelectionRequested.connect(self._hide_selection)
        side.isolateSelectionRequested.connect(self._hide_unselected)
        side.showAllAtomsRequested.connect(self._show_all_atoms)
        side.applyRequested.connect(self._apply_changes)
        side.sourceChanged.connect(self._on_source_changed)

        self._view.selectionChanged.connect(self._on_viewer_selection)
        self._view.dragMoved.connect(self._on_drag_moved)
        self._view.dragFinished.connect(self._on_drag_finished)

        self._table.selectionSet.connect(self._on_table_selection)
        model = self._table.model_()
        model.coordinateEdited.connect(self._on_table_coordinate)

        card = self._selection_card
        card.coordinateEdited.connect(self._on_inspector_coordinate)
        card.moveRequested.connect(lambda: self._set_tool("move"))
        card.deleteRequested.connect(self._delete_selection)
        card.duplicateRequested.connect(self._duplicate_selection)
        card.fixRequested.connect(self._toggle_fixed)
        card.coordinateModeChanged.connect(self._set_coordinate_mode)

        self._add_card.insertRequested.connect(self._insert_atom)
        self._supercell_card.buildRequested.connect(self._build_supercell)

        ops = self._operations_card
        ops.wrapRequested.connect(self._wrap_into_cell)
        ops.centreRequested.connect(self._centre_structure)
        ops.sortRequested.connect(self._sort_by_element)
        ops.overlapsRequested.connect(self._check_overlaps)

        self._save_card.savePoscarRequested.connect(self._save_poscar)
        self._save_card.exportCifRequested.connect(self._export_cif)
        self._save_card.exportXyzRequested.connect(self._export_xyz)

    def _shortcut_specs(self) -> list[tuple[QKeySequence | str, object]]:
        """(key sequence, slot) pairs shared by the page's own shortcuts and
        the ones re-installed on a popped-out viewer window (see
        PopoutButton's ``shortcuts`` parameter) — one list, so the two can
        never quietly drift apart."""
        return [
            ("S", lambda: self._set_tool("select")),
            ("R", lambda: self._set_tool("rotate")),
            ("T", lambda: self._set_tool("pan")),
            ("G", lambda: self._set_tool("move")),
            ("D", self._duplicate_selection),
            ("M", lambda: self._set_tool("measure")),
            ("H", self._hide_selection),
            ("Shift+H", self._hide_unselected),
            ("Alt+H", self._show_all_atoms),
            ("Esc", lambda: self._set_tool("select")),
            (QKeySequence.Undo, self._undo),
            (QKeySequence.Redo, self._redo),
            (QKeySequence.Delete, self._delete_selection),
            (QKeySequence("Backspace"), self._delete_selection),
        ]

    def _install_shortcuts(self) -> None:
        # Qt.WindowShortcut, not WidgetWithChildrenShortcut: these are
        # whole-page actions ("undo the last edit", "switch tools"), not
        # actions scoped to one specific child widget, and they must fire no
        # matter which widget inside the page currently holds keyboard focus.
        # WidgetWithChildrenShortcut requires the focused widget to be a
        # descendant of `self` specifically, which silently broke two ways
        # that are both completely ordinary things to have happened: (1)
        # MainWindow keeps a page's sidebar in a separate _sidebar_stack
        # rather than nesting it inside the page (see _register_page), so
        # EditorSidebar is not a descendant of this page at all -- confirmed
        # with `self.isAncestorOf(self._sidebar) is False` -- meaning every
        # one of these shortcuts went dead the instant focus was anywhere in
        # the sidebar, including on the sidebar's own Undo/Redo buttons right
        # after clicking them; (2) every edit rebuilds the selection
        # inspector's coordinate grid (SelectionCard._build_grid), and if a
        # field in the grid that is about to be torn down happens to hold
        # focus, Qt's own fallback focus handling can land focus on an
        # unrelated, newly-created field elsewhere in the page instead of
        # dropping it cleanly -- also outside what WidgetWithChildrenShortcut
        # was scoped to notice as "the viewport."  Qt.WindowShortcut sidesteps
        # both: it fires whenever this page's window is the active one,
        # regardless of which specific descendant has focus, while still
        # correctly refusing to fire while a *different* page is showing
        # (Qt's shortcut dispatch requires the owning widget itself to be
        # visible, and QStackedWidget hides every page but the current one).
        # This also matches the popped-out window's own shortcuts (see
        # widgets/popout.py), which already use WindowShortcut for this exact
        # spec list -- the two had simply drifted onto different contexts.
        # Single-letter entries (S/R/T/G/D/M/H) stay safe with a text field
        # focused because QLineEdit sends its own ShortcutOverride for plain
        # typed characters, independent of a QShortcut's context, which is
        # what stops ordinary typing from being hijacked as a tool switch.
        for keys, slot in self._shortcut_specs():
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(slot)

    # -- Page interface ----------------------------------------------------

    def sidebar(self) -> QWidget:
        return self._sidebar

    def set_calculation(self, calculation: Calculation | None) -> None:
        super().set_calculation(calculation)
        self._history.clear()
        self._selection.clear()
        # Ids are only meaningful within one structure, so carrying a hidden
        # set across a new calculation would hide arbitrary atoms in it.
        self._hidden.clear()
        self._notice.clear()
        self._sources = {}

        if calculation is None or calculation.is_empty:
            self._load_structure(None)
            self._sidebar.set_sources([])
            return

        # The same ordered list the loader uses, so the Editor and the
        # Structure page never disagree about what counts as a structure —
        # which includes a CHGCAR's embedded POSCAR block when a calculation
        # has nothing better.
        for key in STRUCTURE_SOURCE_KEYS:
            path = calculation.index.path_for(key)
            if path is not None:
                self._sources[path.name] = path
        if not self._sources:
            self._load_structure(None)
            self._sidebar.set_sources([])
            self._empty.set_text(
                "No structure to edit",
                "This calculation has no POSCAR, CONTCAR or volumetric file "
                "to open in the editor.",
            )
            return

        names = list(self._sources)
        self._sidebar.set_sources(names, names[0])
        self._open_source(names[0])

    def _on_source_changed(self, name: str) -> None:
        if name and name in self._sources:
            if self._confirm_discard() is False:
                return
            self._open_source(name)

    def _open_source(self, name: str) -> None:
        path = self._sources.get(name)
        if path is None:
            return
        result = read_structure(path)
        if not result.ok or result.value is None:
            self._load_structure(None)
            if result.issues:
                self._notice.show_issue(result.issues[0])
            return
        structure = EditableStructure.from_structure(result.value)
        self._original = structure.copy()
        self._saved = structure.copy()
        self._history.clear()
        self._selection.clear()
        # A different source file means different atoms behind the same ids.
        self._hidden.clear()
        self._load_structure(structure)
        log.info("Editing %s: %d atoms", path.name, structure.n_atoms)

    def _load_structure(self, structure: EditableStructure | None) -> None:
        self._working = structure
        self._modified = False
        if structure is None:
            self._stack.setCurrentIndex(1)
            self._sidebar.set_structure_caption("")
            self._sidebar.set_elements([])
            self._table.model_().set_structure(None)
            self._apply_hidden()
            self._refresh_status()
            return

        self._stack.setCurrentIndex(0)
        self._sidebar.set_elements(structure.elements)
        self._add_card.set_elements(structure.elements)
        self._rebuild_element_filter(structure)
        self._refresh_all()
        QTimer.singleShot(0, self._fit_view)

    # -- refresh -----------------------------------------------------------

    def _refresh_all(self) -> None:
        structure = self._working
        if structure is None:
            return
        plain = structure.to_structure()
        # An edit changes the atoms, not where the user was looking from.
        # Reframing here threw away the orientation and zoom after every
        # single move, undo or delete; _load_structure fits the camera
        # explicitly when a genuinely new structure arrives.
        self._view.set_structure(plain, reset_camera=False)
        self._apply_hidden()
        self._apply_view_options()
        self._table.model_().set_structure(structure)
        self._apply_table_filter(self._search.text())
        self._sidebar.set_structure_caption(structure.formula)
        self._supercell_card.set_structure(structure)
        self._update_legend(structure)
        self._sync_selection_views()
        self._refresh_status()

    def _refresh_status(self) -> None:
        structure = self._working
        if structure is None:
            self._table_title.setText("Atom List")
            # Otherwise these kept reporting whatever the previous structure's
            # history looked like — a stale "Undo" enabled for edits that no
            # longer apply to anything on screen.
            self._sidebar.set_history_state(False, False)
        else:
            self._table_title.setText(f"Atom List ({structure.n_atoms:,} atoms)")
            self._sidebar.set_history_state(self._history.can_undo, self._history.can_redo)
            self._revert.setEnabled(self._modified)
        self.refresh_info()

    # -- bottom info bar -------------------------------------------------------

    def info_fields(self) -> tuple[dict[str, str], dict[str, str]]:
        """Values and colours for the window's bottom bar in its "editor"
        layout. The bar shows what used to be this page's own status strip
        (atoms, selection, mode, save state) alongside the calculation's
        k-mesh and cutoff. The gap is filled in by the window, which hears
        it from the Bands page whenever EIGENVAL has been read."""
        structure = self._working
        values: dict[str, str] = {}
        colours: dict[str, str] = {}
        calculation = self.calculation()
        meta = None if calculation is None or calculation.is_empty else calculation.metadata
        if meta is not None:
            values["kpoints"] = "—" if meta.kmesh is None else str(meta.kmesh.value)
            values["encut"] = "—" if meta.encut is None else f"{meta.encut.value:g} eV"
        if structure is not None:
            values["system"] = structure.formula or "—"
            values["atoms"] = f"{structure.n_atoms:,}"
            values["selected"] = str(len(self._selection))
            values["mode"] = "Fractional" if self._fractional else "Cartesian"
            if self._modified:
                values["status"], colours["status"] = "● Modified", PALETTE.warning
            else:
                values["status"], colours["status"] = "● Saved", PALETTE.success
        return values, colours

    def refresh_info(self) -> None:
        values, colours = self.info_fields()
        self.infoRequested.emit(values, colours)

    def _update_legend(self, structure: EditableStructure) -> None:
        while self._legend_layout.count():
            item = self._legend_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        for element in structure.elements:
            entry = QWidget()
            row = QHBoxLayout(entry)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(SPACING.sm)
            swatch = ElementSwatch(element, size=10, dark=True)
            swatch.changed.connect(self._on_element_color_changed)
            row.addWidget(swatch)
            label = QLabel(element)
            label.setStyleSheet(f"font-size: {TYPE.small}px; color: #f0f0f2;")
            row.addWidget(label)
            self._legend_layout.addWidget(entry)
        self._legend.show()
        QTimer.singleShot(0, self._position_overlays)

    def _on_element_color_changed(self) -> None:
        self._view.update()
        if self._working is not None:
            self._sidebar.set_elements(self._working.elements)
        self._table.viewport().update()

    # -- layout ------------------------------------------------------------

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt naming
        if watched is self._holder and event.type() == event.Type.Resize:
            self._position_overlays()
        return super().eventFilter(watched, event)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # The legend is built while the page is still hidden, when the holder
        # has no meaningful size yet and every overlay would clamp to the
        # top-left corner. Re-place them once real geometry exists.
        super().showEvent(event)
        QTimer.singleShot(0, self._position_overlays)

    def _position_overlays(self) -> None:
        holder = self._holder
        margin = SPACING.md
        self._toolbar.adjustSize()
        self._toolbar.move(margin, margin)
        self._toolbar.raise_()

        self._axis_bar.adjustSize()
        self._axis_bar.move(
            max(0, holder.width() - self._axis_bar.width() - margin), margin
        )
        self._axis_bar.raise_()

        self._popout.move(
            max(0, self._axis_bar.x() - self._popout.width() - SPACING.xs), margin
        )
        self._popout.raise_()

        # Positioned whenever it has entries rather than when it reports being
        # visible: a widget inside a page that has not been shown yet answers
        # isVisible() with False, which would leave the legend parked in the
        # top-left corner underneath the toolbar until something else forced a
        # resize.
        if self._legend_layout.count():
            self._legend.adjustSize()
            self._legend.move(
                max(0, holder.width() - self._legend.width() - margin),
                max(0, holder.height() - self._legend.height() - margin),
            )
            self._legend.raise_()

    def _apply_view_options(self) -> None:
        self._view.set_options(
            ViewOptions(
                style=self._sidebar.style(),
                coloring=self._sidebar.coloring(),
                show_cell=True,
                show_axes=True,
                perspective=True,
            )
        )
        # A per-element key beside atoms that are all one colour is not a
        # legend, it is a contradiction — so it goes away with the colours it
        # describes.
        self._legend.setVisible(self._sidebar.coloring() != "Uniform")

    def _fit_view(self) -> None:
        if self._working is None:
            return
        self._view.fit_view()

    # -- tools -------------------------------------------------------------

    def _set_tool(self, key: str) -> None:
        self._sidebar.set_tool(key) if key in (
            "select", "delete", "move", "duplicate", "measure",
        ) else None
        # The floating tool buttons only check themselves when clicked
        # directly. A keyboard shortcut (R for rotate, M for measure, …)
        # bypasses that, so without this the icon strip would keep showing
        # the previous tool highlighted while the view had already switched.
        # Order must match _build_floating_toolbar's `entries`.
        camera_keys = ["select", "rotate", "pan", "measure"]
        if key in camera_keys:
            button = self._camera_group.button(camera_keys.index(key))
            if button is not None:
                button.setChecked(True)
        self._on_tool_changed(key)

    def _on_tool_changed(self, key: str) -> None:
        # The one-shot tools act immediately rather than becoming a mode:
        # "Delete Atom" is a verb, not a state to be left in.
        if key == "delete":
            self._delete_selection()
            self._sidebar.set_tool("select")
            key = "select"
        elif key == "duplicate":
            self._duplicate_selection()
            self._sidebar.set_tool("move")
            key = "move"

        modes = {
            "select": EditorView.MODE_SELECT,
            "move": EditorView.MODE_MOVE,
            "measure": EditorView.MODE_MEASURE,
            "rotate": EditorView.MODE_ROTATE,
            "pan": EditorView.MODE_PAN,
        }
        self._view.set_mode(modes.get(key, EditorView.MODE_SELECT))
        self._apply_snap()
        self._view.update()

    def _apply_snap(self) -> None:
        if not self._sidebar.snap_enabled:
            self._view.set_snap(None)
            return
        # The gizmo works in Å, so the fractional step is converted through
        # the shortest lattice vector — the axis where a given fractional
        # step moves the atom least.
        if self._fractional and self._working is not None:
            shortest = min(self._working.lattice_parameters)
            self._view.set_snap(SNAP_FRACTIONAL * shortest)
        else:
            self._view.set_snap(SNAP_CARTESIAN)

    def _set_coordinate_mode(self, fractional: bool) -> None:
        """Switching representation must never move an atom — only the
        numbers shown change, not the structure behind them."""
        if fractional == self._fractional:
            return
        self._fractional = fractional
        self._table.model_().set_fractional(fractional)
        self._selection_card.set_coordinate_mode(fractional)
        self._add_card.set_coordinate_mode(fractional)
        self._sidebar._coords.setCurrentIndex(1 if fractional else 0)
        self._apply_snap()
        self._refresh_status()

    # -- selection ---------------------------------------------------------

    def _rows_for_selection(self) -> list[int]:
        if self._working is None:
            return []
        return self._working.indices_of(self._selection)

    def _set_selection(self, atom_ids: set[int]) -> None:
        self._selection = set(atom_ids)
        self._sync_selection_views()
        self._refresh_status()

    def _sync_selection_views(self) -> None:
        rows = set(self._rows_for_selection())
        self._view.set_selection(rows)
        self._view.set_neighbor_handles(self._neighbor_handles_for(rows))
        self._apply_pair_handle(rows)
        self._table.apply_selection(rows)
        self._selection_card.set_state(self._working, rows)
        self._sidebar.set_selection_present(bool(rows))

    def _apply_pair_handle(self, rows: set[int]) -> None:
        """Offer a "move toward the other selected atom" handle when exactly
        two atoms are selected, anchored to whichever was selected first —
        it is the only one that moves."""
        structure = self._working
        order = self._view.selection_order()
        if structure is None or len(rows) != 2 or len(order) != 2:
            self._view.set_pair_handle(None, None, None)
            return
        first, second = order
        direction, distance = structure.direction_to(first, second)
        if distance <= 0:
            self._view.set_pair_handle(None, None, None)
            return
        self._view.set_pair_handle(direction, distance, second)

    def _neighbor_handles_for(self, rows: set[int]) -> list[tuple[int, np.ndarray, float]]:
        """Nearest-neighbour bond directions to offer as extra move-gizmo
        handles — only when the selection is unambiguous (exactly one atom)
        and only the ones close enough to the nearest to plausibly be a real
        bond, so a sparsely-coordinated atom (an adsorbate, a vacancy edge)
        does not sprout handles pointing at atoms far away.
        """
        structure = self._working
        if structure is None or len(rows) != 1:
            return []
        row = next(iter(rows))
        neighbors = structure.nearest_neighbors(row, k=MAX_NEIGHBOR_HANDLES)
        if not neighbors:
            return []
        nearest_distance = neighbors[0][2]
        cutoff = nearest_distance * NEIGHBOR_DISTANCE_RATIO
        return [n for n in neighbors if n[2] <= cutoff]

    # -- visibility --------------------------------------------------------

    def _apply_hidden(self) -> None:
        """Push the hidden set to the viewer, translated to row indices.

        Ids are the durable handle; the viewer speaks rows. Translating on
        every refresh rather than storing rows is what keeps the set correct
        across edits that renumber the structure underneath it.
        """
        structure = self._working
        rows = set(structure.indices_of(self._hidden)) if structure else set()
        self._view.set_hidden_atoms(rows)
        self._sidebar.set_hidden_count(len(rows))

    def _hide_selection(self) -> None:
        """Hide the selected atoms, and drop them from the selection.

        Leaving them selected would keep the move gizmo and the inspector
        pointed at atoms that are no longer on screen, so a drag would move
        something invisible.
        """
        if not self._selection:
            return
        self._hidden |= set(self._selection)
        self._set_selection(set())
        self._apply_hidden()

    def _hide_unselected(self) -> None:
        """Isolate the selection: hide everything else.

        The common case for hiding at all — reading one adsorbate or one
        layer out of a slab without deleting the rest of the cell.
        """
        structure = self._working
        if structure is None or not self._selection:
            return
        self._hidden = set(structure.ids) - set(self._selection)
        self._apply_hidden()

    def _show_all_atoms(self) -> None:
        if not self._hidden:
            return
        self._hidden = set()
        self._apply_hidden()

    def _ids_for_rows(self, rows: set[int]) -> set[int]:
        structure = self._working
        if structure is None:
            return set()
        return {structure.ids[r] for r in rows if 0 <= r < structure.n_atoms}

    def _on_viewer_selection(self, rows: set[int]) -> None:
        structure = self._working
        if structure is None:
            return
        # An active element filter restricts what a click in the viewer can
        # select, which is the whole point of the chips.
        if self._filter:
            rows = {r for r in rows if structure.species[r] in self._filter}
        self._set_selection(self._ids_for_rows(rows))

    def _on_table_selection(self, rows: set[int]) -> None:
        self._set_selection(self._ids_for_rows(rows))

    def _on_element_filter(self, allowed: set[str]) -> None:
        self._filter = set(allowed)

    def _rebuild_element_filter(self, structure: EditableStructure) -> None:
        self._element_filter.blockSignals(True)
        self._element_filter.clear()
        self._element_filter.addItems(["All", *structure.elements])
        self._element_filter.blockSignals(False)

    def _apply_table_filter(self, text: str) -> None:
        structure = self._working
        if structure is None:
            return
        needle = text.strip().lower()
        element = self._element_filter.currentText()
        rows = []
        for index in range(structure.n_atoms):
            symbol = structure.species[index]
            if element not in ("", "All") and symbol != element:
                continue
            if needle:
                # Match either the element or the displayed 1-based index.
                if needle not in symbol.lower() and needle != str(index + 1):
                    continue
            rows.append(index)
        self._table.model_().set_visible_rows(rows)
        self._table.apply_selection(set(self._rows_for_selection()))

    # -- the one place the structure changes --------------------------------

    def _commit(self, label: str, structure: EditableStructure, selection: set[int]) -> None:
        """Apply a change, record it in history, and refresh everything."""
        if self._working is None:
            return
        command = EditCommand(
            label=label,
            before=self._working,
            after=structure,
            selection_before=set(self._selection),
            selection_after=set(selection),
        )
        self._history.push(command)
        self._working = structure
        self._selection = set(selection)
        self._modified = True
        self._after_structure_change()

    def _after_structure_change(self) -> None:
        structure = self._working
        if structure is None:
            return
        self._sidebar.set_elements(structure.elements)
        self._add_card.set_elements(structure.elements)
        current_element = self._element_filter.currentText()
        self._rebuild_element_filter(structure)
        index = self._element_filter.findText(current_element)
        if index >= 0:
            self._element_filter.setCurrentIndex(index)
        self._refresh_all()
        # Deferred: the coordinate grid this just rebuilt (see
        # SelectionCard._clear_grid) tears down its old QLineEdit fields via
        # deleteLater, which Qt only actually processes on the next event
        # loop turn. Checking focus synchronously here would still see the
        # about-to-be-destroyed field. See _reclaim_lost_focus for why this
        # matters at all.
        QTimer.singleShot(0, self._reclaim_lost_focus)

    def _reclaim_lost_focus(self) -> None:
        """Give the viewport focus back when a refresh has silently dropped
        it to nothing.

        Every edit rebuilds the selection inspector's coordinate grid, which
        destroys its old QLineEdit fields. When one of them happened to hold
        keyboard focus at that moment (a very ordinary thing to have
        happened -- e.g. the field a coordinate was just typed into),
        destroying it can leave the whole application with no focus widget
        at all rather than handing focus to anything else. With nothing
        focused, every keyboard shortcut in the app — S/R/T/G/D/M, H, and
        Undo/Redo among them — has nothing to dispatch to and silently does
        nothing, which is exactly the "the shortcuts don't work" symptom.
        Restoring focus to the viewport only when focus has gone missing
        entirely, never when the user is legitimately focused on something
        else (a search field, a table row), keeps this from stealing focus
        the user still wants.
        """
        if QApplication.focusWidget() is None and self._working is not None:
            self._view.setFocus()

    def _undo(self) -> None:
        command = self._history.undo()
        if command is None:
            return
        self._working = command.before
        self._selection = set(command.selection_before)
        self._modified = self._saved is None or command.before is not self._saved
        self._after_structure_change()

    def _redo(self) -> None:
        command = self._history.redo()
        if command is None:
            return
        self._working = command.after
        self._selection = set(command.selection_after)
        self._modified = True
        self._after_structure_change()

    # -- editing operations -------------------------------------------------

    def _delete_selection(self) -> None:
        rows = self._rows_for_selection()
        if not rows or self._working is None:
            return
        self._commit(
            f"Delete {len(rows)} atom{'s' if len(rows) != 1 else ''}",
            self._working.with_atoms_removed(rows),
            set(),
        )

    def _duplicate_selection(self) -> None:
        rows = self._rows_for_selection()
        if not rows or self._working is None:
            return
        structure, new_ids = self._working.with_atoms_duplicated(rows)
        self._commit(
            f"Duplicate {len(rows)} atom{'s' if len(rows) != 1 else ''}",
            structure,
            set(new_ids),
        )
        # The copies land on top of the originals, so the natural next action
        # is to move them — switch to the tool that does it.
        self._set_tool("move")

    def _insert_atom(self, element: str, coords: np.ndarray, fractional: bool) -> None:
        structure = self._working
        if structure is None:
            return
        frac = coords if fractional else structure.to_fractional(coords)
        new_structure, atom_id = structure.with_atom_added(element, frac)
        self._commit(f"Add {element}", new_structure, {atom_id})

    def _on_table_coordinate(self, row: int, axis: int, value: float) -> None:
        self._edit_coordinate(row, axis, value)

    def _on_inspector_coordinate(self, row: int, axis: int, value: float) -> None:
        self._edit_coordinate(row, axis, value)

    def _edit_coordinate(self, row: int, axis: int, value: float) -> None:
        structure = self._working
        if structure is None or not (0 <= row < structure.n_atoms):
            return
        if self._fractional:
            frac = structure.frac_coords[row].copy()
            frac[axis] = value
        else:
            cart = structure.cart_coords[row].copy()
            cart[axis] = value
            frac = structure.to_fractional(cart)
        if np.allclose(frac, structure.frac_coords[row]):
            return
        self._commit(
            f"Edit atom {row + 1} coordinate",
            structure.with_coords([row], frac.reshape(1, 3)),
            set(self._selection),
        )

    def _toggle_fixed(self, _flags) -> None:
        """Toggle the whole selection between fully free and fully fixed."""
        rows = self._rows_for_selection()
        structure = self._working
        if not rows or structure is None:
            return
        currently_fixed = not structure.selective[rows].any()
        free = np.tile(np.array([currently_fixed] * 3), (len(rows), 1))
        self._commit(
            f"{'Free' if currently_fixed else 'Fix'} {len(rows)} atom(s)",
            structure.with_selective(rows, free),
            set(self._selection),
        )

    # -- gizmo dragging -----------------------------------------------------

    def _drag_rows(self, structure: EditableStructure) -> list[int]:
        """Which rows a drag in progress should move.

        Ordinarily the whole current selection. The one exception is the
        pair-move handle, which is anchored to (and only ever moves) the
        first of exactly two selected atoms — the second is the fixed
        reference its direction was measured toward, reported by the view via
        :meth:`EditorView.drag_target_rows`.
        """
        target = self._view.drag_target_rows()
        if target is not None:
            return sorted(r for r in target if 0 <= r < structure.n_atoms)
        return structure.indices_of(self._selection)

    def _on_drag_moved(self, delta_cart: np.ndarray) -> None:
        """Live preview during a drag.

        The preview is applied to a scratch copy taken from the drag's start,
        so the many intermediate positions never enter the undo history — only
        the completed move does.
        """
        structure = self._working
        if structure is None:
            return
        if self._drag_origin is None:
            self._drag_origin = structure
        rows = self._drag_rows(self._drag_origin)
        if not rows:
            return
        delta_frac = self._drag_origin.to_fractional(delta_cart)
        moved = self._drag_origin.translated(rows, delta_frac)
        if self._sidebar.constrain_enabled:
            moved = moved.wrapped(rows)
        self._working = moved
        # reset_camera=False, emphatically: the default refits and re-orients
        # the camera, so every mouse-move during a drag snapped the view back
        # to the default orientation and the structure appeared to jump out
        # from under the cursor. A drag has to happen against a frozen view.
        self._view.set_structure(moved.to_structure(), reset_camera=False)
        self._view.set_selection(set(self._selection))
        self._view.refresh_gizmo()
        self._table.model_().set_structure(moved)
        self._selection_card.set_state(moved, set(self._rows_for_selection()))

    def _on_drag_finished(self, delta_cart: np.ndarray) -> None:
        origin = self._drag_origin
        self._drag_origin = None
        if origin is None or self._working is None:
            return
        rows = self._drag_rows(origin)
        if not rows:
            return
        final = self._working
        # Roll the live preview back so the command records the true before
        # state, then commit the completed translation as one entry.
        self._working = origin
        distance = float(np.linalg.norm(delta_cart))
        label = (
            f"Move {len(rows)} atom{'s' if len(rows) != 1 else ''} ({distance:.3f} Å)"
            if self._view.drag_target_rows() is None
            else f"Move atom {rows[0] + 1} toward its neighbour ({distance:.3f} Å)"
        )
        self._commit(label, final, set(self._selection))

    # -- structure operations ----------------------------------------------

    def _build_supercell(self, na: int, nb: int, nc: int) -> None:
        structure = self._working
        if structure is None or (na, nb, nc) == (1, 1, 1):
            return
        total = structure.n_atoms * na * nb * nc
        if total > 20000:
            QMessageBox.warning(
                self,
                "Supercell too large",
                f"{na}×{nb}×{nc} would produce {total:,} atoms, which this "
                "editor cannot display usefully. Try a smaller replication.",
            )
            return
        plain = build_supercell(structure.to_structure(), na, nb, nc)
        new_structure = EditableStructure.from_structure(plain)
        new_structure.has_selective_dynamics = structure.has_selective_dynamics
        self._commit(f"Build {na}×{nb}×{nc} supercell", new_structure, set())

    def _wrap_into_cell(self) -> None:
        if self._working is None:
            return
        self._commit("Wrap into cell", self._working.wrapped(), set(self._selection))

    def _centre_structure(self) -> None:
        if self._working is None:
            return
        self._commit("Center structure", self._working.centred(), set(self._selection))

    def _sort_by_element(self) -> None:
        if self._working is None:
            return
        self._commit(
            "Sort by element", self._working.sorted_by_element(), set(self._selection)
        )

    def _check_overlaps(self) -> None:
        structure = self._working
        if structure is None:
            return
        pairs = structure.close_pairs(OVERLAP_THRESHOLD)
        if not pairs:
            self._notice.show_issue(
                Issue(
                    code="no_overlaps",
                    message=f"No atoms are closer than {OVERLAP_THRESHOLD} Å.",
                    severity=Severity.INFO,
                )
            )
            return
        # Detection selects the offenders and reports; it never deletes on its
        # own, because "remove overlaps" silently discarding a real atom is
        # exactly the kind of change that ruins a calculation unnoticed.
        ids = {structure.ids[j] for _i, j, _d in pairs}
        self._set_selection(ids)
        self._notice.show_issue(
            Issue(
                code="overlaps_found",
                message=(
                    f"{len(pairs)} overlapping pair(s) closer than "
                    f"{OVERLAP_THRESHOLD} Å. The later atom of each pair is now "
                    "selected — press Delete to remove them."
                ),
                detail="\n".join(
                    f"atoms {i + 1} and {j + 1}: {d:.3f} Å" for i, j, d in pairs[:20]
                ),
                severity=Severity.WARNING,
            )
        )

    # -- session state ------------------------------------------------------

    def _apply_changes(self) -> None:
        if self._working is None:
            return
        self._saved = self._working.copy()
        self._modified = False
        self._refresh_status()
        self._notice.show_issue(
            Issue(
                code="structure_applied",
                message=(
                    "Working structure committed for this session. Use "
                    "Save POSCAR to write it to disk."
                ),
                severity=Severity.INFO,
            )
        )

    def _revert_changes(self) -> None:
        if self._saved is None:
            return
        if self._confirm_discard() is False:
            return
        self._working = self._saved.copy()
        self._selection.clear()
        self._history.clear()
        self._modified = False
        self._after_structure_change()

    def _reset_to_source(self) -> None:
        if self._original is None:
            return
        if self._confirm_discard() is False:
            return
        self._working = self._original.copy()
        self._saved = self._original.copy()
        self._selection.clear()
        self._history.clear()
        self._modified = False
        self._after_structure_change()

    def _confirm_discard(self) -> bool | None:
        if not self._modified:
            return None
        box = QMessageBox(self)
        box.setWindowTitle("Discard changes?")
        box.setIcon(QMessageBox.Warning)
        box.setText("The edited structure has unsaved changes.")
        box.setInformativeText("Discarding cannot be undone.")
        box.setStandardButtons(QMessageBox.Discard | QMessageBox.Cancel)
        return None if box.exec() == QMessageBox.Discard else False

    def _show_overflow(self) -> None:
        menu = QMenu(self)
        menu.addAction("Validate Structure", self._validate)
        menu.addAction("Copy Coordinates", self._copy_coordinates)
        menu.addSeparator()
        reset = menu.addAction("Reset to Source", self._reset_to_source)
        reset.setEnabled(self._original is not None)
        menu.exec(self._overflow.mapToGlobal(self._overflow.rect().bottomLeft()))

    # -- validation and export ----------------------------------------------

    def _validation_errors(self) -> list[str]:
        """Problems that make a structure genuinely unwritable."""
        structure = self._working
        if structure is None:
            return ["No structure is open."]
        errors: list[str] = []
        if structure.n_atoms == 0:
            errors.append("The structure has no atoms.")
        if structure.volume < 1e-6:
            errors.append("The lattice is degenerate (zero volume).")
        if not np.isfinite(structure.frac_coords).all():
            errors.append("Some coordinates are not finite numbers.")
        if len(structure.species) != len(structure.frac_coords):
            errors.append("Species and coordinate counts disagree.")
        if structure.selective.shape != (structure.n_atoms, 3):
            errors.append("Constraint flags do not match the atom count.")
        return errors

    def _validation_warnings(self) -> list[str]:
        structure = self._working
        if structure is None:
            return []
        warnings: list[str] = []
        pairs = structure.close_pairs(OVERLAP_THRESHOLD)
        if pairs:
            warnings.append(f"{len(pairs)} atom pair(s) closer than {OVERLAP_THRESHOLD} Å.")
        outside = int((~((structure.frac_coords >= 0) & (structure.frac_coords < 1))).any(axis=1).sum())
        if outside:
            warnings.append(f"{outside} atom(s) lie outside the primary cell.")
        if not structure.is_element_grouped():
            warnings.append(
                "Atoms of the same element are not contiguous; saving will "
                "reorder them so the POSCAR is valid."
            )
        return warnings

    def _validate(self) -> None:
        errors = self._validation_errors()
        warnings = self._validation_warnings()
        if errors:
            self._notice.show_issue(
                Issue(
                    code="structure_invalid",
                    message="This structure cannot be saved.",
                    detail="\n".join(errors),
                    severity=Severity.ERROR,
                )
            )
            return
        self._notice.show_issue(
            Issue(
                code="structure_valid",
                message=(
                    "Structure is valid."
                    if not warnings
                    else f"Structure is valid, with {len(warnings)} warning(s)."
                ),
                detail="\n".join(warnings),
                severity=Severity.INFO if not warnings else Severity.WARNING,
            )
        )

    def _guard_export(self) -> bool:
        errors = self._validation_errors()
        if errors:
            self._notice.show_issue(
                Issue(
                    code="structure_invalid",
                    message="This structure cannot be saved.",
                    detail="\n".join(errors),
                    severity=Severity.ERROR,
                )
            )
            return False
        return True

    def _default_directory(self) -> Path:
        calculation = self.calculation()
        return calculation.directory if calculation else Path.home()

    def _save_poscar(self) -> None:
        if self._working is None or not self._guard_export():
            return
        structure = self._working
        # POSCAR cannot express interleaved species, so grouping happens here
        # rather than silently inside the writer.
        if not structure.is_element_grouped():
            structure = structure.sorted_by_element()
        suggested = self._default_directory() / "POSCAR_edited"
        path, _filter = QFileDialog.getSaveFileName(
            self, "Save POSCAR", str(suggested), "VASP structure (POSCAR* CONTCAR*);;All files (*)"
        )
        if not path:
            return
        target = Path(path)
        source = self._sources.get(self._sidebar._source.currentText())
        if source is not None and target.resolve() == source.resolve():
            confirm = QMessageBox.warning(
                self,
                "Overwrite the source file?",
                f"{target.name} is the file this calculation was read from.\n\n"
                "Overwriting it replaces the original structure.",
                QMessageBox.Save | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if confirm != QMessageBox.Save:
                return
        try:
            save_poscar(structure.to_structure(), target)
        except OSError as exc:
            self._report_write_error(target, exc)
            return
        self._saved = self._working.copy()
        self._modified = False
        self._refresh_status()
        log.info("Saved %s (%d atoms)", target, structure.n_atoms)
        self._notice.show_issue(
            Issue(
                code="poscar_saved",
                message=f"Saved {target.name}.",
                severity=Severity.INFO,
            )
        )

    def _export_cif(self) -> None:
        if self._working is None or not self._guard_export():
            return
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "Export CIF",
            str(self._default_directory() / f"{self._working.formula or 'structure'}.cif"),
            "CIF (*.cif)",
        )
        if not path:
            return
        try:
            save_cif(self._working.to_structure(), Path(path))
        except OSError as exc:
            self._report_write_error(Path(path), exc)
            return
        self._notice.show_issue(
            Issue(code="cif_saved", message=f"Exported {Path(path).name}.", severity=Severity.INFO)
        )

    def _export_xyz(self) -> None:
        if self._working is None or not self._guard_export():
            return
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "Export XYZ",
            str(self._default_directory() / f"{self._working.formula or 'structure'}.xyz"),
            "XYZ (*.xyz)",
        )
        if not path:
            return
        structure = self._working
        lattice = structure.lattice
        # extended-XYZ Lattice header, so the periodicity survives the export
        # rather than the file becoming a bare molecule.
        lattice_text = " ".join(f"{v:.8f}" for v in lattice.reshape(-1))
        lines = [
            str(structure.n_atoms),
            f'Lattice="{lattice_text}" Properties=species:S:1:pos:R:3',
        ]
        for symbol, cart in zip(structure.species, structure.cart_coords):
            lines.append(f"{symbol:<3s} {cart[0]:16.8f} {cart[1]:16.8f} {cart[2]:16.8f}")
        try:
            Path(path).write_text("\n".join(lines) + "\n")
        except OSError as exc:
            self._report_write_error(Path(path), exc)
            return
        self._notice.show_issue(
            Issue(code="xyz_saved", message=f"Exported {Path(path).name}.", severity=Severity.INFO)
        )

    def _report_write_error(self, path: Path, exc: OSError) -> None:
        log.exception("Could not write %s", path)
        self._notice.show_issue(
            Issue(
                code="write_failed",
                message=f"{path.name} could not be written.",
                detail=str(exc),
                severity=Severity.ERROR,
            )
        )

    def _copy_coordinates(self) -> bool:
        structure = self._working
        if structure is None:
            return False
        rows = self._rows_for_selection() or list(range(structure.n_atoms))
        coords = structure.frac_coords if self._fractional else structure.cart_coords
        lines = [
            f"{structure.species[r]:<3s} {coords[r][0]:14.8f} {coords[r][1]:14.8f} {coords[r][2]:14.8f}"
            for r in rows
        ]
        QGuiApplication.clipboard().setText("\n".join(lines))
        return True

    # -- toolbar hooks ------------------------------------------------------

    def reset(self) -> None:
        self._sidebar.reset()
        self._fractional = True
        self._table.model_().set_fractional(True)
        self._search.clear()
        self._set_selection(set())
        self._fit_view()

    def toolbar_action(self) -> None:
        self._revert_changes()

    def copy_to_clipboard(self) -> bool:
        return self._copy_coordinates()

    def export(self) -> None:
        self._save_poscar()
