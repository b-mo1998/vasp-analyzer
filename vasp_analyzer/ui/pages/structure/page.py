"""The Structure page: interactive crystal viewer with lattice information."""

from __future__ import annotations

from pathlib import Path

import numpy as np

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
from ....services.bonds import nearest_neighbors as find_nearest_neighbors
from ....parsers.outcar import MagneticMoments, parse_magnetic_moments
from ....services import relaxation
from ....services.loader import DATASET_STRUCTURE, DataLoader, read_structure
from ....services.supercell import build_supercell, cell_count
from ....services.symmetry import SymmetryEquivalence, analyse, equivalence
from ...style import icon_url
from ...theme import METRICS, PALETTE, RADIUS, SPACING, TYPE
from ...widgets.axis_bar import AxisBar
from ...widgets.element_swatch import ElementSwatch
from ...widgets.empty_state import EmptyState
from ...widgets.notice import Notice
from ...widgets.popout import PopoutButton
from ...widgets.viewer_toolbar import ViewerToolbar
from ...widgets.primitives import Card
from ..base import Page
from .export import save_cif, save_poscar
from .info_panel import StructureInfoPanel
from .sidebar import MAX_NEIGHBOR_ROWS, StructureSidebar
from .viewer import StructureView, VectorOverlay, ViewOptions

log = get_logger("ui.structure")

# Below these, a field is noise rather than data: VASP prints "-0.000" for a
# converged non-magnetic atom, and CONTCAR round-trips POSCAR to ~1e-6 Å.
MIN_DRAWABLE_MOMENT = 1e-3            # μB
MIN_MEANINGFUL_DISPLACEMENT = 1e-3    # Å


# Glyph characters render inconsistently across fonts, so the tools use
# bundled icons instead.


class StructurePage(Page):
    """Viewer, display controls and structure metadata."""

    title = "Structure"
    issueRaised = Signal(object)
    navigationRequested = Signal(str)   # role key to open on the Inputs page
    infoBarSlotChanged = Signal()       # where the bottom bar belongs changed

    def __init__(self, loader: DataLoader, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loader = loader
        self._sidebar = StructureSidebar()
        self._base_structure: StructureData | None = None  # as parsed, untiled
        self._structure: StructureData | None = None        # currently rendered
        self._supercell: tuple[int, int, int] = (1, 1, 1)
        # Per-atom annotations of the *base* cell, tiled to match whatever
        # supercell is displayed when they are drawn.
        self._moments: MagneticMoments | None = None
        self._relaxation: relaxation.RelaxationResult | None = None
        # Symmetry orbits of the displayed structure, computed on first use
        # and remembered for as long as that structure object is the one on
        # screen -- spglib on a supercell is not free.
        self._equivalence_cache: tuple[int, SymmetryEquivalence | None] | None = None

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
        viewer_column = QWidget()
        self._viewer_column = QVBoxLayout(viewer_column)
        self._viewer_column.setContentsMargins(0, 0, 0, 0)
        self._viewer_column.setSpacing(SPACING.md)
        self._viewer_column.addWidget(self._build_viewer_card(), 1)
        row.addWidget(viewer_column, 1)

        self._info = StructureInfoPanel()
        self._info.setFixedWidth(METRICS.dos_panel_width)
        row.addWidget(self._info, 0)
        self._right_column = self._info
        self._stack.addWidget(content)
        self._stack.currentChanged.connect(lambda _i: self.infoBarSlotChanged.emit())

        empty_card = Card()
        self._empty = EmptyState(
            title="Structure unavailable",
            message="No POSCAR or CONTCAR was found in this calculation.",
            glyph="⬡",
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

        # The same arrangement every other 3D page uses: the viewer fills the
        # holder, and the toolbar, orientation bar, pop-out button and legend
        # float over its corners as children of that holder. This page used to
        # be the odd one out, with an in-flow toolbar strip above the canvas
        # carrying a redundant "Structure Viewer" title — redundant because
        # the card it sits in is already unambiguous, and it pushed the tools
        # away from the thing they act on. Floating them also means popping
        # out the holder takes the whole set with it, which is what the
        # in-flow version needed a second wrapper widget to achieve.
        holder = QWidget()
        holder_layout = QVBoxLayout(holder)
        holder_layout.setContentsMargins(0, 0, 0, 0)
        self._view = StructureView()
        holder_layout.addWidget(self._view)

        self._toolbar = self._build_floating_toolbar(holder)
        self._toolbar.install_shortcuts(self)
        # Orientation presets in place of a projection toggle: which way the
        # cell faces is the question that actually comes up while reading a
        # structure, and perspective is the right answer for all of them.
        self._axis_bar = AxisBar(holder)
        self._axis_bar.orientationRequested.connect(self._view.orient)
        self._popout = PopoutButton(holder, "Structure Viewer", holder)

        self._legend = QWidget(holder)
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

        holder.installEventFilter(self)
        self._holder = holder
        body.addWidget(holder, 1)

        body.addWidget(self._build_status_strip())
        return card

    def _build_floating_toolbar(self, parent: QWidget) -> QWidget:
        # measure=True: a distance or angle is a property of *this* structure,
        # which is well defined here and on the Editor, and ambiguous on a
        # trajectory or reaction path where the geometry changes frame to
        # frame.
        return ViewerToolbar(self._view, parent, measure=True)

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
        side.momentsToggled.connect(lambda _v: self._apply_overlays())
        side.momentComponentChanged.connect(lambda _v: self._apply_overlays())
        side.displacementsToggled.connect(lambda _v: self._apply_overlays())
        side.displacementScaleChanged.connect(lambda _v: self._apply_overlays())
        side.equivalentToggled.connect(
            lambda _v: self._on_atom_selected(self._view.selected_index())
        )

    def eventFilter(self, watched, event):  # noqa: N802 - Qt naming
        if watched is self._holder and event.type() in (
            QEvent.Resize,
            QEvent.Show,
        ):
            self._position_overlays()
        return super().eventFilter(watched, event)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # Data can arrive while this page is off-screen, where Qt has not yet
        # given the legend's children a size. Re-place it once visible.
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

        self._position_legend()

    def _position_legend(self) -> None:
        if self._legend.isHidden():
            return
        # adjustSize() runs before the freshly added children report a size and
        # leaves the legend a few pixels wide; resizing to the size hint that
        # the activated layout produces is reliable.
        self._legend_layout.activate()
        self._legend.resize(self._legend.sizeHint())
        self._legend.raise_()
        holder = self._holder
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
        self._moments = None
        self._relaxation = None
        self._equivalence_cache = None
        self._sidebar.set_moments_available(False)
        self._sidebar.set_displacements_available(False)
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
        self._load_annotations(structure)
        self._apply_supercell()
        self._stack.setCurrentIndex(0)

    def _load_annotations(self, structure: StructureData) -> None:
        """Magnetic moments and relaxation displacements for the base cell.

        Both are optional extras: a non-magnetic run has no moments and a
        single-point run has no CONTCAR, and either absence just leaves its
        control disabled with the reason in the tooltip.
        """
        calculation = self.calculation()
        index = calculation.index if calculation is not None else None

        # -- moments ----------------------------------------------------------
        self._moments = None
        outcar = index.path_for("outcar") if index is not None else None
        reason = "No OUTCAR in this calculation."
        drawable = False
        if outcar is not None:
            result = parse_magnetic_moments(outcar)
            if result.ok and result.value is not None:
                if result.value.n_atoms != structure.n_atoms:
                    reason = (
                        f"OUTCAR reports moments for {result.value.n_atoms} atoms "
                        f"but the structure has {structure.n_atoms}."
                    )
                else:
                    self._moments = result.value
                    # A spin-polarised run can still converge to zero moment
                    # everywhere -- MAGMOM = 0 on every atom does exactly
                    # that, and VASP writes the tables regardless. The card
                    # reports the zero honestly; the overlay is withheld,
                    # because an enabled toggle that draws nothing reads as
                    # broken rather than as "there is no moment".
                    largest = float(self._moments.magnitudes.max())
                    drawable = largest >= MIN_DRAWABLE_MOMENT
                    if not drawable:
                        reason = (
                            "Every atom's moment is zero -- this run converged "
                            "non-magnetic (was MAGMOM set to a non-zero initial "
                            "guess?). Nothing to draw."
                        )
            elif result.issues:
                reason = result.issues[0].message
        non_collinear = self._moments is not None and not self._moments.collinear
        self._sidebar.set_moments_available(drawable, reason, non_collinear=non_collinear)
        if drawable and non_collinear:
            # Open on whichever spin axis actually carries weight, rather
            # than always on the total vector -- for a run where the moment
            # lies almost entirely along one axis, that view is the same
            # picture with less to read.
            self._sidebar.set_moment_component("vector")
        self._info.show_magnetization(self._moments)

        # -- relaxation ---------------------------------------------------------
        self._relaxation = None
        reason = "Needs both a POSCAR and a CONTCAR."
        contcar = index.path_for("contcar") if index is not None else None
        showing_poscar = structure.source.upper().startswith("POSCAR")
        if contcar is not None and showing_poscar:
            final = read_structure(contcar)
            if final.ok and final.value is not None:
                compared = relaxation.compare(structure, final.value)
                if compared is None:
                    reason = "POSCAR and CONTCAR do not describe the same atoms."
                elif compared.max_displacement < MIN_MEANINGFUL_DISPLACEMENT:
                    # A static (SCF, bands, DOS) run copies POSCAR to CONTCAR
                    # unchanged, so "what did the relaxation do" has no
                    # answer there. Withholding the overlay on that test --
                    # rather than on the folder's name -- also keeps it
                    # working for a relaxation, an MD run, or an NEB image,
                    # wherever they happen to live.
                    reason = (
                        "CONTCAR is identical to POSCAR: no atom moved in this "
                        "calculation (a static run, not a relaxation or MD)."
                    )
                else:
                    self._relaxation = compared
            elif final.issues:
                reason = final.issues[0].message
        elif contcar is not None and not showing_poscar:
            reason = "The structure shown is already the relaxed CONTCAR."
        self._sidebar.set_displacements_available(self._relaxation is not None, reason)
        self._info.show_relaxation(self._relaxation)

    def _apply_overlays(self) -> None:
        """Push whichever arrow fields are switched on to the viewer, tiled
        to the displayed supercell."""
        side = self._sidebar
        n_cells = cell_count(*self._supercell)

        overlay = None
        if self._moments is not None and side.show_moments.isChecked():
            component = side.moment_component_key()
            if component == "vector" or self._moments.collinear:
                vectors = self._moments.vectors
            else:
                vectors = self._moments.component_overlay_vectors(component)
            magnitudes = np.linalg.norm(vectors, axis=1)
            largest = float(magnitudes.max()) if magnitudes.size else 0.0
            # The biggest moment draws about 1.2 Å long -- readable beside a
            # ball without spearing its neighbours; the rest scale with it.
            # Normalised per component, so isolating a small S_x is still
            # legible rather than a row of specks.
            scale = 1.2 / largest if largest > 0 else 1.0
            # Sign colouring only means something along one axis: for a full
            # 3D vector "negative" is not a property the arrow direction
            # leaves ambiguous, so the second colour is dropped there.
            axis_index = {"x": 0, "y": 1, "z": 2}.get(component, 2)
            overlay = VectorOverlay(
                vectors=relaxation.tile(vectors, n_cells),
                color="#e11d48",          # along +axis
                signed_color=(
                    "#2563eb"
                    if (self._moments.collinear or component != "vector")
                    else None
                ),
                signed_axis=axis_index,
                scale=scale,
                max_length=1.5,
            )
        self._view.set_vector_overlay("moments", overlay)

        overlay = None
        if self._relaxation is not None and side.show_displacements.isChecked():
            overlay = VectorOverlay(
                vectors=relaxation.tile(self._relaxation.vectors, n_cells),
                color="#7c3aed",
                scale=float(side.displacement_scale.value()),
                max_length=4.0,
            )
        self._view.set_vector_overlay("displacements", overlay)

    def _equivalence_for(self, structure: StructureData | None):
        if structure is None:
            return None
        if self._equivalence_cache is not None and self._equivalence_cache[0] == id(structure):
            return self._equivalence_cache[1]
        # A cap, because spglib's cost grows with atom count and a large
        # supercell would freeze the click that asked for it.
        result = equivalence(structure) if structure.n_atoms <= 2000 else None
        self._equivalence_cache = (id(structure), result)
        return result

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
        # set_structure dropped the previous cell's overlays and highlight;
        # both are indexed per atom and have to be rebuilt for this tiling.
        self._equivalence_cache = None
        self._apply_overlays()
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

            swatch = ElementSwatch(element, size=12)
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
            perspective=True,
        )
        self._view.set_options(options)
        # A per-element key beside atoms that are all one colour is not a
        # legend, it is a contradiction — so it goes away with the colours it
        # describes.
        self._legend.setVisible(options.coloring != "Uniform")


    def _on_atom_selected(self, index: int | None) -> None:
        structure = self._structure
        extras: dict = {}
        highlighted: set[int] = set()

        if structure is not None and index is not None and index < structure.n_atoms:
            if self._sidebar.highlight_equivalent.isChecked():
                orbits = self._equivalence_for(structure)
                if orbits is not None:
                    members = orbits.equivalents_of(index)
                    extras["wyckoff"] = (
                        f"{orbits.wyckoff_label(index)}   site symmetry "
                        f"{orbits.site_symmetry[index]}"
                    )
                    extras["equivalents"] = members
                    highlighted = set(members) - {index}

            # Annotations belong to the base cell; a supercell repeats them
            # cell-major / atom-minor, so the base atom is index mod n_base.
            base_n = self._base_structure.n_atoms if self._base_structure else 0
            base_index = index % base_n if base_n else index
            if self._moments is not None and base_index < self._moments.n_atoms:
                extras["moment"] = float(self._moments.scalar[base_index])
                if not self._moments.collinear:
                    # The three components individually: for a non-collinear
                    # run the magnitude alone hides the direction, which is
                    # the whole reason the calculation was non-collinear.
                    vector = self._moments.vectors[base_index]
                    extras["spin"] = (
                        float(vector[0]), float(vector[1]), float(vector[2])
                    )
            if self._relaxation is not None and base_index < self._relaxation.n_atoms:
                extras["displacement"] = float(self._relaxation.magnitudes[base_index])

        self._view.set_highlighted_atoms(highlighted)
        self._info.show_atom(structure, index, **extras)
        self._update_neighbor_table(index)

    def _update_neighbor_table(self, index: int | None) -> None:
        structure = self._structure
        if structure is None or index is None:
            self._sidebar.set_neighbor_table([])
            return
        neighbors = find_nearest_neighbors(
            structure.frac_coords, structure.lattice, index, k=MAX_NEIGHBOR_ROWS
        )
        self._sidebar.set_neighbor_table(
            [
                (structure.species[row], row + 1, distance)
                for row, _direction, distance in neighbors
            ]
        )

    def _on_color_changed(self) -> None:
        self._view.update()

    # -- toolbar actions -----------------------------------------------------------

    def reset(self) -> None:
        self._sidebar.reset()
        self._on_options_changed()
        self._view.reset_view()
        self._view._clear_measurement()
        self._view.set_highlighted_atoms(set())
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
        stem = calculation.name if calculation is not None else "structure"
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
