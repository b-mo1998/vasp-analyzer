"""The Structure page: interactive crystal viewer with lattice information."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QColorDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ....core.errors import Issue, ParseResult, Severity
from ....core.logging_setup import get_logger
from ....models.calculation import Calculation
from ....models.capabilities import Capability
from ....models.structure import StructureData
from ....services import appearance
from ....services.bonds import MAX_ATOMS_FOR_BONDS
from ....services.loader import DATASET_STRUCTURE, DataLoader
from ....services.supercell import build_supercell, cell_count
from ....services.symmetry import analyse
from ...style import icon_url
from ...theme import METRICS, PALETTE, RADIUS, SPACING, TYPE
from ...widgets.empty_state import EmptyState
from ...widgets.notice import Notice
from ...widgets.primitives import Card
from ...widgets.segmented import SegmentedControl
from ..base import Page
from .export import save_cif, save_poscar
from .info_panel import StructureInfoPanel
from .sidebar import StructureSidebar
from .viewer import StructureView, ViewOptions

log = get_logger("ui.structure")


class _ElementSwatch(QLabel):
    """A clickable colour dot in the element legend.

    Click opens a colour picker; right-click offers to reset to the default
    Jmol-style colour. The choice is applied immediately and persists across
    sessions and calculations via :mod:`vasp_analyzer.services.appearance` —
    it's a display preference for "carbon", not a fact about one structure.
    """

    changed = Signal()

    def __init__(self, element: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._element = element
        self.setFixedSize(12, 12)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(f"{element}: click to change colour, right-click to reset")
        self._apply_style()

    def _apply_style(self) -> None:
        colour = appearance.effective_color(self._element)
        self.setStyleSheet(
            f"background: {colour}; border-radius: 6px;"
            f" border: 1px solid rgba(0,0,0,0.15);"
        )

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.button() == Qt.LeftButton:
            self._pick_color()
        elif event.button() == Qt.RightButton:
            self._show_menu(event.globalPosition().toPoint())
        super().mousePressEvent(event)

    def _pick_color(self) -> None:
        current = QColor(appearance.effective_color(self._element))
        chosen = QColorDialog.getColor(current, self, f"Colour for {self._element}")
        if chosen.isValid():
            appearance.set_override(self._element, chosen.name())
            self._apply_style()
            self.changed.emit()

    def _show_menu(self, global_pos) -> None:
        menu = QMenu(self)
        reset = menu.addAction("Reset to Default Colour")
        reset.setEnabled(appearance.has_override(self._element))
        if menu.exec(global_pos) is reset:
            appearance.clear_override(self._element)
            self._apply_style()
            self.changed.emit()

# Glyph characters render inconsistently across fonts, so the tools use
# bundled icons instead.
TOOLS = [
    ("select", "select.svg", "Select atoms", StructureView.MODE_SELECT),
    ("rotate", "rotate.svg", "Rotate", StructureView.MODE_ROTATE),
    ("pan", "pan.svg", "Pan", StructureView.MODE_PAN),
    (
        "measure",
        "measure.svg",
        "Measure — click two atoms for a distance, three for an angle",
        StructureView.MODE_MEASURE,
    ),
]


class StructurePage(Page):
    """Viewer, display controls and structure metadata."""

    title = "Structure"
    issueRaised = Signal(object)
    navigationRequested = Signal(str)   # role key to open on the Inputs page

    def __init__(self, loader: DataLoader, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loader = loader
        self._sidebar = StructureSidebar()
        self._base_structure: StructureData | None = None  # as parsed, untiled
        self._structure: StructureData | None = None        # currently rendered
        self._supercell: tuple[int, int, int] = (1, 1, 1)

        self._build()
        self._connect()
        loader.data_ready.connect(self._on_data_ready)

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
        row.addWidget(self._build_viewer_card(), 1)

        self._info = StructureInfoPanel()
        self._info.setFixedWidth(METRICS.dos_panel_width)
        row.addWidget(self._info, 0)
        self._stack.addWidget(content)

        empty_card = Card()
        self._empty = EmptyState(
            title="Structure unavailable",
            message="No POSCAR or CONTCAR was found in this calculation.",
            glyph="⬡",
        )
        empty_card.body().addWidget(self._empty)
        self._stack.addWidget(empty_card)
        self._stack.setCurrentIndex(1)

    def _build_viewer_card(self) -> QWidget:
        card = Card(padding=SPACING.lg)
        body = card.body()
        body.setSpacing(SPACING.sm)

        body.addWidget(self._build_toolbar())

        # The viewer and its floating legend share a stacked container so the
        # legend sits over the bottom-right corner of the canvas.
        canvas_holder = QWidget()
        holder_layout = QVBoxLayout(canvas_holder)
        holder_layout.setContentsMargins(0, 0, 0, 0)
        self._view = StructureView()
        holder_layout.addWidget(self._view)

        self._legend = QWidget(canvas_holder)
        # An id selector, so the panel's border does not cascade onto every
        # swatch and label inside it.
        self._legend.setObjectName("ViewerLegend")
        self._legend.setAttribute(Qt.WA_StyledBackground, True)
        self._legend_layout = QVBoxLayout(self._legend)
        self._legend_layout.setContentsMargins(
            SPACING.md, SPACING.sm, SPACING.md, SPACING.sm
        )
        self._legend_layout.setSpacing(SPACING.xs)
        self._legend.setStyleSheet(
            f"#ViewerLegend {{ background: rgba(255,255,255,0.92);"
            f" border: 1px solid {PALETTE.border};"
            f" border-radius: {RADIUS.md}px; }}"
        )
        self._legend.hide()
        canvas_holder.installEventFilter(self)
        self._canvas_holder = canvas_holder
        body.addWidget(canvas_holder, 1)

        body.addWidget(self._build_status_strip())
        return card

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(38)
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING.sm)

        title = QLabel("Structure Viewer")
        title.setObjectName("CardTitle")
        row.addWidget(title)
        row.addSpacing(SPACING.lg)

        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)
        for index, (key, icon_name, tip, mode) in enumerate(TOOLS):
            button = QPushButton()
            button.setIcon(QIcon(icon_url(icon_name)))
            button.setIconSize(QSize(15, 15))
            button.setObjectName("ToolButton")
            button.setCheckable(True)
            button.setFixedSize(34, 30)
            button.setToolTip(tip)
            button.setCursor(Qt.PointingHandCursor)
            button.setProperty("mode", mode)
            self._tool_group.addButton(button, index)
            row.addWidget(button)
            if key == "rotate":
                button.setChecked(True)
        self._tool_group.idClicked.connect(self._on_tool_selected)

        zoom_in = QPushButton("+")
        zoom_in.setObjectName("ToolButton")
        zoom_in.setFixedSize(34, 30)
        zoom_in.setToolTip("Zoom in")
        zoom_in.clicked.connect(lambda: self._view.zoom_by(1.2))
        row.addWidget(zoom_in)

        zoom_out = QPushButton("−")
        zoom_out.setObjectName("ToolButton")
        zoom_out.setFixedSize(34, 30)
        zoom_out.setToolTip("Zoom out")
        zoom_out.clicked.connect(lambda: self._view.zoom_by(1 / 1.2))
        row.addWidget(zoom_out)

        row.addStretch(1)

        self._projection = SegmentedControl(["Perspective", "Orthographic"])
        self._projection.currentChanged.connect(self._on_projection_changed)
        row.addWidget(self._projection)
        return bar

    def _build_status_strip(self) -> QWidget:
        strip = QWidget()
        strip.setFixedHeight(24)
        row = QHBoxLayout(strip)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING.xl)
        row.addStretch(1)

        self._status = QLabel("")
        self._status.setStyleSheet(
            f"color: {PALETTE.text_secondary}; font-size: {TYPE.small}px;"
        )
        row.addWidget(self._status)
        row.addStretch(1)
        return strip

    def _connect(self) -> None:
        side = self._sidebar
        side.styleChanged.connect(self._on_options_changed)
        side.coloringChanged.connect(self._on_options_changed)
        side.showCellChanged.connect(self._on_options_changed)
        side.showAxesChanged.connect(self._on_options_changed)
        self._view.atomSelected.connect(self._on_atom_selected)
        self._view.measurementChanged.connect(self._info.show_measurement)
        self._info.fileRequested.connect(self.navigationRequested)
        self._sidebar.supercellChanged.connect(self._on_supercell_changed)

    def eventFilter(self, watched, event):  # noqa: N802 - Qt naming
        if watched is self._canvas_holder and event.type() in (
            QEvent.Resize,
            QEvent.Show,
        ):
            self._position_legend()
        return super().eventFilter(watched, event)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # Data can arrive while this page is off-screen, where Qt has not yet
        # given the legend's children a size. Re-place it once visible.
        super().showEvent(event)
        QTimer.singleShot(0, self._position_legend)

    def _position_legend(self) -> None:
        if self._legend.isHidden():
            return
        # adjustSize() runs before the freshly added children report a size and
        # leaves the legend a few pixels wide; resizing to the size hint that
        # the activated layout produces is reliable.
        self._legend_layout.activate()
        self._legend.resize(self._legend.sizeHint())
        self._legend.raise_()
        holder = self._canvas_holder
        self._legend.move(
            max(0, holder.width() - self._legend.width() - SPACING.md),
            max(0, holder.height() - self._legend.height() - SPACING.md),
        )

    # -- Page interface --------------------------------------------------------

    def sidebar(self) -> QWidget:
        return self._sidebar

    def set_calculation(self, calculation: Calculation | None) -> None:
        super().set_calculation(calculation)
        self._base_structure = None
        self._structure = None
        self._supercell = (1, 1, 1)
        for box in self._sidebar.supercell_boxes.values():
            box.blockSignals(True)
            box.setValue(1)
            box.blockSignals(False)
        self._sidebar.set_supercell_hint("")
        self._notice.clear()
        self._view.set_structure(None)
        self._info.clear()
        self._info.update_files(calculation)
        self._status.setText("")

        if calculation is None:
            self._sidebar.set_enabled(False)
            self._empty.set_text(
                "No calculation open",
                "Open a calculation directory to view its structure.",
            )
            self._stack.setCurrentIndex(1)
            return

        if not calculation.can(Capability.STRUCTURE):
            self._sidebar.set_enabled(False)
            self._empty.set_text(
                "Structure unavailable", calculation.why_not(Capability.STRUCTURE)
            )
            self._stack.setCurrentIndex(1)
            return

        self._sidebar.set_enabled(True)
        self._loader.request(DATASET_STRUCTURE)

    def _on_data_ready(self, dataset: str, result: ParseResult) -> None:
        if dataset != DATASET_STRUCTURE:
            return
        if not result.ok or result.value is None:
            message = (
                result.issues[0].message
                if result.issues
                else "The structure file could not be read."
            )
            self._empty.set_text("Structure unavailable", message)
            self._stack.setCurrentIndex(1)
            return

        for issue in result.issues:
            if issue.severity in (Severity.WARNING, Severity.ERROR):
                self._notice.show_issue(issue)
                break

        structure = result.value
        # spglib is only worth the import once a structure is actually shown.
        # It always runs on the base cell: a supercell has the same symmetry
        # family, and re-deriving it from a larger, non-primitive cell is both
        # slower and can report a different setting than the one the
        # calculation actually used.
        analyse(structure)

        self._base_structure = structure
        self._info.update_structure(structure, self.calculation())
        self._apply_supercell()
        self._stack.setCurrentIndex(0)

    def _on_supercell_changed(self, na: int, nb: int, nc: int) -> None:
        self._supercell = (na, nb, nc)
        if self._base_structure is not None:
            self._apply_supercell()

    def _apply_supercell(self) -> None:
        assert self._base_structure is not None
        na, nb, nc = self._supercell
        structure = build_supercell(self._base_structure, na, nb, nc)

        self._structure = structure
        self._view.set_structure(structure)
        self._update_legend(structure)
        self._update_status(structure)

        cells = cell_count(na, nb, nc)
        if cells == 1:
            self._sidebar.set_supercell_hint("")
        else:
            hint = f"{structure.n_atoms} atoms across {cells} cells"
            if structure.n_atoms > MAX_ATOMS_FOR_BONDS:
                hint += " — too many to draw bonds"
            self._sidebar.set_supercell_hint(hint)

    def _update_status(self, structure: StructureData) -> None:
        parts = [f"Atoms: {structure.n_atoms}"]
        for element, count in sorted(structure.composition.items()):
            parts.append(f"{element}: {count}")
        base = self._base_structure
        if base is not None and base.space_group_number is not None:
            parts.append(
                f"Space Group: {base.space_group_symbol} ({base.space_group_number})"
            )
        if self._supercell != (1, 1, 1):
            na, nb, nc = self._supercell
            parts.append(f"{na}×{nb}×{nc} supercell")
        vacuum = self._view.vacuum_info()
        if vacuum is not None:
            parts.append(f"Vacuum: {vacuum.vacuum_length:.1f} Å ({vacuum.axis_name})")
        self._status.setText("      ".join(parts))

    def _update_legend(self, structure: StructureData) -> None:
        while self._legend_layout.count():
            item = self._legend_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        for element in sorted(structure.composition):
            entry = QWidget()
            row = QHBoxLayout(entry)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(SPACING.sm)

            swatch = _ElementSwatch(element)
            swatch.changed.connect(self._on_color_changed)
            row.addWidget(swatch)

            label = QLabel(element)
            label.setStyleSheet(f"font-size: {TYPE.small}px;")
            row.addWidget(label)
            self._legend_layout.addWidget(entry)

        self._legend.show()
        QTimer.singleShot(0, self._position_legend)

    # -- controls ----------------------------------------------------------------

    def _on_options_changed(self, _value=None) -> None:
        options = ViewOptions(
            style=self._sidebar.style_box.currentText(),
            coloring=self._sidebar.coloring_box.currentText(),
            show_cell=self._sidebar.show_cell.isChecked(),
            show_axes=self._sidebar.show_axes.isChecked(),
            perspective=self._projection.currentIndex() == 0,
        )
        self._view.set_options(options)

    def _on_projection_changed(self, _index: int) -> None:
        self._on_options_changed()

    def _on_tool_selected(self, index: int) -> None:
        button = self._tool_group.button(index)
        if button is not None:
            self._view.set_mode(button.property("mode"))

    def _on_atom_selected(self, index: int | None) -> None:
        self._info.show_atom(self._structure, index)

    def _on_color_changed(self) -> None:
        self._view.update()

    # -- toolbar actions -----------------------------------------------------------

    def reset(self) -> None:
        self._sidebar.reset()
        self._projection.setCurrentIndex(0)
        self._on_options_changed()
        self._view.reset_view()
        self._view._clear_measurement()
        self._info.show_atom(self._structure, None)

    def _render_pixmap(self, scale: int = 2) -> QPixmap | None:
        if self._structure is None:
            return None
        size = self._view.size() * scale
        pixmap = QPixmap(size)
        pixmap.setDevicePixelRatio(scale)
        pixmap.fill(QColor(PALETTE.surface))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        self._view.render(painter)
        painter.end()
        return pixmap

    def copy_to_clipboard(self) -> bool:
        pixmap = self._view.grab()
        if pixmap.isNull():
            return False
        QGuiApplication.clipboard().setPixmap(pixmap)
        return True

    def export(self) -> None:
        if self._structure is None:
            return
        calculation = self.calculation()
        stem = calculation.directory.name if calculation is not None else "structure"
        suggested = str(
            (calculation.directory if calculation else Path.home()) / f"{stem}-structure.png"
        )

        filters = "PNG image (*.png);;CIF file (*.cif);;VASP POSCAR (*.vasp)"
        path_text, selected = QFileDialog.getSaveFileName(
            self, "Export Structure", suggested, filters
        )
        if not path_text:
            return

        path = Path(path_text)
        try:
            if selected.startswith("CIF"):
                save_cif(self._structure, path.with_suffix(".cif"))
            elif selected.startswith("VASP"):
                save_poscar(self._structure, path.with_suffix(".vasp"))
            else:
                if not self._view.grab().save(str(path)):
                    raise OSError("QPixmap.save returned False")
            log.info("Exported structure to %s", path)
        except OSError as exc:
            log.exception("Structure export failed")
            self._notice.show_issue(
                Issue(
                    code="export_failed",
                    message="The file could not be written.",
                    detail=str(exc),
                    severity=Severity.ERROR,
                )
            )
