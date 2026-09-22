"""The NEB page: reaction-path viewer, energy profile and image navigation.

The three views — thumbnails, energy profile and 3D viewer — are one selection
seen three ways. Clicking a point on the profile, a thumbnail, or the previous/
next arrows all funnel through :meth:`_select_image`, so they can never drift
out of step.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ....core.errors import Issue, ParseResult, Severity
from ....core.logging_setup import get_logger
from ....models.calculation import Calculation
from ....models.capabilities import Capability
from ....models.neb import NEBPath
from ....models.structure import StructureData
from ....parsers.incar import as_int, parse_incar
from ....services.loader import DATASET_NEB, DataLoader
from ....services.trajectory import unwrap_fractional, wrap_fractional
from ...style import icon_url
from ...theme import METRICS, PALETTE, RADIUS, SPACING, TYPE
from ...widgets.axis_bar import AxisBar
from ...widgets.element_swatch import ElementSwatch
from ...widgets.empty_state import EmptyState
from ...widgets.notice import Notice
from ...widgets.flex_column import FlexColumn
from ...widgets.popout import PopoutButton
from ...widgets.viewer_toolbar import ViewerToolbar
from ...widgets.primitives import Card, Divider
from ...widgets.range_slider import RangeRow
from ..base import Page
from ..sequence_view import SequenceView, TrailOptions, sequence_extent
from ..structure.viewer import StructureView, ViewOptions
from .info_panel import KeyValueCard, SummaryStrip
from .profile import EnergyProfilePlot
from .sidebar import NEBSidebar
from .thumbnails import ImageStrip

log = get_logger("ui.neb")


# VTST's IOPT values; IBRION alone describes VASP's own optimisers.
_IOPT_NAMES = {
    0: "VASP optimiser (IBRION)",
    1: "LBFGS (VTST)",
    2: "Conjugate Gradient (VTST)",
    3: "Quick-Min (VTST)",
    4: "SciPy optimiser (VTST)",
    7: "FIRE (VTST)",
}
_IBRION_NAMES = {
    1: "RMM-DIIS quasi-Newton",
    2: "Conjugate gradient",
    3: "Damped molecular dynamics",
}


class NEBPage(Page):
    """Reaction path: images, energies and the structures behind them."""

    title = "NEB"
    issueRaised = Signal(object)
    infoRequested = Signal(dict)
    infoBarSlotChanged = Signal()       # where the bottom bar belongs changed

    def __init__(self, loader: DataLoader, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loader = loader
        self._sidebar = NEBSidebar()
        self._path: NEBPath | None = None
        self._unwrapped: np.ndarray | None = None   # (n_images, n_atoms, 3) Å
        self._selected = 0

        self._build()
        self._connect()
        loader.data_ready.connect(self._on_data_ready)

    # -- construction ---------------------------------------------------------

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
        self._right_column = self._build_info_column()
        row.addWidget(self._right_column, 0)
        self._stack.addWidget(content)

        empty_card = Card()
        self._empty = EmptyState(
            title="No NEB calculation found",
            message="This directory does not contain numbered image folders.",
            glyph="⤳",
        )
        empty_card.body().addWidget(self._empty)
        self._stack.addWidget(empty_card)
        self._stack.setCurrentIndex(1)
        self._stack.currentChanged.connect(lambda _i: self.infoBarSlotChanged.emit())

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
        self._view = SequenceView()
        self._view.setMinimumHeight(320)
        holder_layout.addWidget(self._view)

        self._toolbar = self._build_floating_toolbar(holder)
        self._toolbar.install_shortcuts(self)
        # Orientation presets in place of a projection toggle: which way the
        # cell faces is the question that actually comes up while reading a
        # structure, and perspective is the right answer for all of them.
        self._axis_bar = AxisBar(holder)
        self._axis_bar.orientationRequested.connect(self._view.orient)
        self._popout = PopoutButton(holder, "NEB Path Viewer", holder)
        self._legend = self._build_floating_legend(holder)
        holder.installEventFilter(self)
        self._holder = holder
        body.addWidget(holder, 1)

        body.addWidget(Divider())
        self._strip = ImageStrip()
        body.addWidget(self._strip)
        return card

    def _build_floating_toolbar(self, parent: QWidget) -> QWidget:
        # Fit frames the whole reaction path rather than the current image,
        # so stepping between images does not reframe the view each time.
        return ViewerToolbar(self._view, parent, on_fit=self._fit_sequence)

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

    def _build_info_column(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setFixedWidth(METRICS.dos_panel_width + 60)

        container = QWidget()
        # FlexColumn rather than QVBoxLayout: the wrapped "Source Files"
        # caption below would otherwise make the scroll area lay this column
        # out at its preferred height and scroll, instead of letting the
        # energy-profile card (stretch 1) give up the difference.
        column = FlexColumn(container)
        column.setContentsMargins(0, 0, SPACING.xs, 0)
        column.setSpacing(SPACING.md)
        scroll.setWidget(container)

        # --- energy profile ---------------------------------------------------
        profile_card = Card(padding=SPACING.md)
        profile_body = profile_card.body()
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Energy Profile")
        title.setObjectName("CardTitle")
        title_row.addWidget(title)
        title_row.addStretch(1)
        profile_body.addLayout(title_row)

        self._profile = EnergyProfilePlot()
        title_row.addWidget(PopoutButton(self._profile, "NEB Energy Profile", flat=True))
        profile_body.addWidget(self._profile, 1)

        self._energy_range = RangeRow("Energy", "eV")
        self._energy_range.rangeChanged.connect(
            lambda _l, _h: self._on_profile_range_changed()
        )
        profile_body.addWidget(self._energy_range)
        self._coord_range = RangeRow("Coord", "Å")
        self._coord_range.rangeChanged.connect(
            lambda _l, _h: self._on_profile_range_changed()
        )
        profile_body.addWidget(self._coord_range)

        self._summary = SummaryStrip()
        profile_body.addWidget(self._summary)
        column.addWidget(profile_card, 1)

        # --- selected image ----------------------------------------------------
        self._image_card = KeyValueCard("Image —")
        self._image_card.build(
            [
                ("energy", "Energy"),
                ("force", "Force (max)"),
                ("coordinate", "Reaction Coordinate"),
                ("index", "Image Index"),
            ]
        )
        column.addWidget(self._image_card)

        # --- model information ---------------------------------------------------
        # Four fields in a 2x2 grid. Pseudopotentials and XC functional used
        # to sit here too; they made the panel taller than the page and are
        # provenance rather than something read while following a path.
        self._model_card = KeyValueCard("Model Information")
        self._model_card.build(
            [
                ("method", "Method"),
                ("encut", "Cutoff Energy"),
                ("kpoints", "K-Points"),
                ("convergence", "Convergence"),
            ]
        )
        column.addWidget(self._model_card)

        # --- lattice ---------------------------------------------------------------
        self._lattice_card = KeyValueCard("Lattice Parameters")
        self._lattice_card.build(
            [
                ("a", "a"), ("b", "b"), ("c", "c"),
                ("alpha", "α"), ("beta", "β"), ("gamma", "γ"),
            ],
            columns=3,
        )
        column.addWidget(self._lattice_card)

        # --- provenance ---------------------------------------------------------------
        source_card = Card(padding=SPACING.md)
        source_body = source_card.body()
        source_title = QLabel("Source Files")
        source_title.setStyleSheet(
            f"font-size: {TYPE.body}px; font-weight: 600; color: {PALETTE.text};"
        )
        source_body.addWidget(source_title)
        self._source_label = QLabel("—")
        self._source_label.setWordWrap(True)
        self._source_label.setStyleSheet(
            f"font-size: {TYPE.small}px; color: {PALETTE.accent};"
        )
        source_body.addWidget(self._source_label)
        column.addWidget(source_card)

        column.addStretch(1)
        return scroll

    # -- wiring -------------------------------------------------------------------

    def _connect(self) -> None:
        side = self._sidebar
        for signal in (
            side.styleChanged,
            side.coloringChanged,
            side.showCellChanged,
            side.showAxesChanged,
        ):
            signal.connect(lambda _v=None: self._apply_view_options())
        side.trailsChanged.connect(self._apply_trail_options)
        side.wrapChanged.connect(lambda _v: self._show_image(self._selected))
        side.centreChanged.connect(lambda _v: self._show_image(self._selected))
        side.selectionCleared.connect(self._clear_selection)

        self._strip.imageSelected.connect(self._select_image)
        self._profile.imageSelected.connect(self._select_image)

    def eventFilter(self, watched, event):  # noqa: N802 - Qt naming
        if watched is self._holder and event.type() in (QEvent.Resize, QEvent.Show):
            self._position_overlays()
        return super().eventFilter(watched, event)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().showEvent(event)
        QTimer.singleShot(0, self._position_overlays)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.key() == Qt.Key_Left:
            self._select_image(self._selected - 1)
        elif event.key() == Qt.Key_Right:
            self._select_image(self._selected + 1)
        else:
            super().keyPressEvent(event)

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

        if not self._legend.isHidden():
            self._legend_layout.activate()
            self._legend.resize(self._legend.sizeHint())
            self._legend.move(
                max(0, holder.width() - self._legend.width() - margin),
                max(0, holder.height() - self._legend.height() - margin),
            )
            self._legend.raise_()

    # -- Page interface -------------------------------------------------------------

    def sidebar(self) -> QWidget:
        return self._sidebar

    def set_calculation(self, calculation: Calculation | None) -> None:
        super().set_calculation(calculation)
        self._reset_state()

        if calculation is None:
            self._sidebar.set_enabled(False)
            self._empty.set_text(
                "No calculation open", "Open a calculation directory to begin."
            )
            self._stack.setCurrentIndex(1)
            self.refresh_info()
            return

        if not calculation.can(Capability.NEB_PATH):
            self._sidebar.set_enabled(False)
            self._empty.set_text(
                "No NEB calculation found", calculation.why_not(Capability.NEB_PATH)
            )
            self._stack.setCurrentIndex(1)
            self.refresh_info()
            return

        self._sidebar.set_enabled(True)
        self._loader.request(DATASET_NEB)

    def _reset_state(self) -> None:
        self._path = None
        self._unwrapped = None
        self._selected = 0
        self._notice.clear()
        self._view.set_structure(None)
        self._view.set_path(None)
        self._strip.clear()
        self._profile.show_unavailable("No NEB path loaded")
        self._summary.update_from(None)
        self._image_card.clear()
        self._image_card.set_title("Image —")
        self._model_card.clear()
        self._lattice_card.clear()
        self._source_label.setText("—")
        self._legend.hide()

    # -- data ----------------------------------------------------------------------

    def _on_data_ready(self, dataset: str, result: ParseResult) -> None:
        if dataset != DATASET_NEB:
            return
        if not result.ok or result.value is None:
            message = (
                result.issues[0].message
                if result.issues
                else "The NEB path could not be read."
            )
            self._empty.set_text("No NEB calculation found", message)
            self._stack.setCurrentIndex(1)
            self.refresh_info()
            return

        for issue in result.issues:
            self._notice.show_issue(issue)
            break

        path = result.value
        self._path = path
        self._stack.setCurrentIndex(0)

        self._prepare_viewer(path)
        self._sidebar.describe_path(
            path.method,
            self._optimiser_name(),
            path.force_threshold,
            path.is_climbing,
        )
        self._populate_profile(path)
        self._populate_model_card(path)
        self._source_label.setText(self._describe_sources(path))

        self._strip.set_path(path, self._view.camera())
        self._select_image(path.climbing_image if path.climbing_image is not None else 0)
        self.refresh_info()

    def _fit_sequence(self) -> None:
        """Reframe on the whole band, for the toolbar's Fit button."""
        if self._path is not None:
            self._prepare_viewer(self._path)
            self._show_image(self._selected)

    def _prepare_viewer(self, path: NEBPath) -> None:
        """Fit the camera to the whole band and build the trail geometry."""
        structures = [image.structure for image in path.images]
        valid = [s for s in structures if s is not None]
        if not valid:
            self._view.set_structure(None)
            self._unwrapped = None
            return

        reference = valid[0]
        self._view.set_structure(reference, reset_camera=True)

        # Trails along the path need continuous coordinates for the same reason
        # a trajectory does: an atom that crosses a face between two images
        # would otherwise trail a line straight across the cell.
        compatible = [
            s for s in structures if s is not None and s.n_atoms == reference.n_atoms
        ]
        if len(compatible) >= 2:
            frac = np.array([s.frac_coords for s in compatible])
            self._unwrapped = unwrap_fractional(frac) @ reference.lattice
        else:
            self._unwrapped = None

        # Fit to the coordinates that will actually be drawn — wrapping and
        # centring are display transforms, and framing the untransformed ones
        # leaves the structure sitting outside its own view.
        displayed = np.array(
            [
                self._display_frac(index)
                for index in range(len(path.images))
                if path.images[index].structure is not None
            ]
        )
        sample = displayed @ reference.lattice if displayed.size else None
        extent = sequence_extent(
            reference.lattice,
            sample if sample is not None else reference.cart_coords,
            self._view.vacuum_info(),
        )
        self._view.fit_to_sequence(extent)
        self._apply_view_options()
        self._apply_trail_options()
        self._update_legend(reference.species)

    def _on_profile_range_changed(self) -> None:
        self._profile.set_view_limits(
            self._coord_range.window(), self._energy_range.window()
        )

    def _populate_profile(self, path: NEBPath) -> None:
        coordinate = path.reaction_coordinate
        use_index = coordinate is None
        if coordinate is None:
            coordinate = np.arange(path.n_images, dtype=float)
        self._profile.set_path(
            coordinate,
            path.relative_energies(),
            path.climbing_image,
            use_index,
            spline=path.spline,
        )
        (x_low, x_high), (y_low, y_high) = self._profile.data_limits()
        self._coord_range.set_bounds(x_low, x_high)
        self._energy_range.set_bounds(y_low, y_high)
        self._summary.update_from(path)

    def _populate_model_card(self, path: NEBPath) -> None:
        calculation = self.calculation()
        metadata = calculation.metadata if calculation else None

        def sourced(value) -> str:
            return "—" if value is None else str(value.value)

        convergence = "—"
        tooltip = ""
        converged = path.converged
        if converged is True:
            convergence = "Converged ✓"
        elif converged is False:
            convergence = "Not converged"
            force, threshold = path.max_force, path.force_threshold
            if force is not None and threshold is not None:
                tooltip = (
                    f"Max force {force:.3f} eV/Å against a target of "
                    f"{threshold:.3f} eV/Å."
                )
        elif path.force_threshold is None:
            tooltip = "EDIFFG does not state a force criterion for this run."
        else:
            tooltip = "Some movable images never reported a force."

        self._model_card.set_values(
            {
                "method": path.method,
                "encut": (
                    "—"
                    if metadata is None or metadata.encut is None
                    else f"{metadata.encut.value:g} eV"
                ),
                "kpoints": "—" if metadata is None else sourced(metadata.kmesh),
                "convergence": convergence,
            },
            tooltips={"convergence": tooltip},
        )

    def _optimiser_name(self) -> str:
        """Describe the ionic optimiser the INCAR selected.

        Read from the NEB directory directly rather than
        ``calculation.index`` — a project layout can point that index at an
        unrelated electronic-structure directory with its own, different
        INCAR.
        """
        calculation = self.calculation()
        if calculation is None:
            return "—"
        incar_path = calculation.neb_directory / "INCAR"
        if not incar_path.is_file():
            return "—"
        parsed = parse_incar(incar_path)
        if not parsed.ok or parsed.value is None:
            return "—"
        tags = parsed.value

        iopt = as_int(tags.get("IOPT"))
        if iopt is not None and iopt in _IOPT_NAMES and iopt != 0:
            return _IOPT_NAMES[iopt]
        ibrion = as_int(tags.get("IBRION"))
        if ibrion is not None:
            return _IBRION_NAMES.get(ibrion, f"IBRION = {ibrion}")
        return "—"

    def _describe_sources(self, path: NEBPath) -> str:
        """A compact provenance line rather than one entry per image file."""
        top_level = [name for name in ("INCAR", "KPOINTS") if name in path.source_files]
        vtst = [n for n in ("neb.dat", "nebef.dat") if n in path.source_files]

        per_image: set[str] = set()
        for image in path.images:
            for entry in image.source_files:
                if "/" in entry:
                    per_image.add(entry.split("/", 1)[1])

        parts = list(top_level)
        if per_image and path.images:
            first = path.images[0].folder
            last = path.images[-1].folder
            for name in sorted(per_image):
                parts.append(f"{first}–{last}/{name}")
        parts.extend(vtst)
        return ", ".join(parts) if parts else "—"

    def _update_legend(self, species: list[str]) -> None:
        while self._legend_layout.count():
            item = self._legend_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        seen: list[str] = []
        for symbol in species:
            if symbol not in seen:
                seen.append(symbol)

        for element in seen:
            entry = QWidget()
            row = QHBoxLayout(entry)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(SPACING.sm)
            dot = ElementSwatch(element, size=10, dark=True)
            dot.changed.connect(self._on_element_color_changed)
            row.addWidget(dot)
            label = QLabel(element)
            label.setStyleSheet(f"font-size: {TYPE.small}px; color: #f0f0f2;")
            row.addWidget(label)
            self._legend_layout.addWidget(entry)

        self._legend.show()
        QTimer.singleShot(0, self._position_overlays)

    def _on_element_color_changed(self) -> None:
        """The viewer resolves colours at paint time, but the image strip's
        thumbnails are pre-rendered pixmaps and have to be drawn again."""
        self._view.update()
        if self._path is not None:
            self._strip.set_path(self._path, self._view.camera())
            self._strip.select(self._selected, notify=False)

    # -- selection ---------------------------------------------------------------------

    def _select_image(self, index: int) -> None:
        path = self._path
        if path is None or path.n_images == 0:
            return
        index = int(np.clip(index, 0, path.n_images - 1))
        self._selected = index

        self._strip.select(index, notify=False)
        self._profile.set_selected(index)
        self._show_image(index)
        self._populate_image_card(index)
        self._populate_lattice_card(index)

    def _display_frac(self, index: int) -> np.ndarray:
        """The fractional coordinates actually drawn for one image."""
        path = self._path
        assert path is not None
        structure = path.images[index].structure
        assert structure is not None

        frac = structure.frac_coords
        if self._sidebar.wrap_atoms.isChecked():
            frac = wrap_fractional(frac)
        if self._sidebar.centre_structure.isChecked():
            frac = frac - frac.mean(axis=0) + np.array([0.5, 0.5, 0.5])
        return frac

    def _show_image(self, index: int) -> None:
        path = self._path
        if path is None:
            return
        source = path.images[index].structure
        if source is None:
            return

        structure = StructureData(
            lattice=source.lattice,
            species=list(source.species),
            frac_coords=self._display_frac(index),
            comment=source.comment,
            source=source.source,
        )
        self._view.show_frame(structure, index)
        self._view.set_frame_index(index)

    def _populate_image_card(self, index: int) -> None:
        path = self._path
        if path is None:
            return
        image = path.images[index]

        title = f"Image {image.folder}"
        if index == path.climbing_image:
            title += " (Climbing Image)"
        elif image.is_endpoint:
            title += " (Initial)" if index == 0 else " (Final)"
        self._image_card.set_title(title)

        coordinate = path.coordinate_of(index)
        force_text = "—"
        force_tip = ""
        if image.max_force is not None:
            force_text = f"{abs(image.max_force):.3f} eV/Å"
            force_tip = (
                "Projected NEB force — the quantity the optimiser converges."
                if image.force_kind == "NEB projected"
                else "Largest atomic force from this image's OUTCAR."
            )

        climb_tip = ""
        if index == path.climbing_image:
            climb_tip = (
                "Deduced as the highest-energy intermediate image; VASP does "
                "not record which image climbed."
                if path.climbing_image_origin == "inferred"
                else ""
            )

        self._image_card.set_values(
            {
                "energy": "—" if image.energy is None else f"{image.energy:.2f} eV",
                "force": force_text,
                "coordinate": "—" if coordinate is None else f"{coordinate:.2f} Å",
                "index": f"{index + 1} / {path.n_images}",
            },
            tooltips={
                "energy": f"Read from {image.energy_source}" if image.energy_source else "",
                "force": force_tip,
                "index": climb_tip,
            },
        )

    def _populate_lattice_card(self, index: int) -> None:
        path = self._path
        if path is None:
            return
        structure = path.images[index].structure
        if structure is None:
            self._lattice_card.clear()
            return
        a, b, c = structure.lattice_parameters
        alpha, beta, gamma = structure.lattice_angles
        self._lattice_card.set_values(
            {
                "a": f"{a:.2f} Å",
                "b": f"{b:.2f} Å",
                "c": f"{c:.2f} Å",
                "alpha": f"{alpha:.1f}°",
                "beta": f"{beta:.1f}°",
                "gamma": f"{gamma:.1f}°",
            }
        )

    def _clear_selection(self) -> None:
        self._view.set_tracked_atom(None)
        self._view.clear_selection()
        self._view.update()

    # -- options -----------------------------------------------------------------------

    def _apply_view_options(self) -> None:
        self._view.set_options(
            ViewOptions(
                style=self._sidebar.style_box.currentText(),
                coloring=self._sidebar.coloring_box.currentText(),
                show_cell=self._sidebar.show_cell.isChecked(),
                show_axes=self._sidebar.show_axes.isChecked(),
                perspective=True,
            )
        )

    def _apply_trail_options(self) -> None:
        self._view.set_trail_options(
            TrailOptions(
                enabled=self._sidebar.atom_trails.isChecked(),
                length=self._sidebar.trail_images(),
                selected_only=False,
            )
        )
        self._view.set_path(self._unwrapped)


    # -- bottom bar ---------------------------------------------------------------------

    def _info_fields(self) -> dict[str, str]:
        path = self._path
        if path is None:
            return {}

        species = path.species
        formula = "—"
        if species:
            from collections import Counter

            formula = "".join(f"{el}{n}" for el, n in Counter(species).items())

        converged = path.converged
        convergence = "—"
        if converged is True:
            convergence = "Converged ✓"
        elif converged is False:
            convergence = "Not converged"

        barrier = path.forward_barrier
        force = path.max_force
        return {
            "system": formula,
            "images": f"{path.n_images}",
            "method": path.method,
            "barrier": "—" if barrier is None else f"{barrier:.2f} eV",
            "force": "—" if force is None else f"{force:.3f} eV/Å",
            "convergence": convergence,
        }

    def refresh_info(self) -> None:
        self.infoRequested.emit(self._info_fields())

    # -- toolbar actions -------------------------------------------------------------------

    def reset(self) -> None:
        self._sidebar.reset()
        self._view.orient("default")
        self._clear_selection()
        self._apply_view_options()
        self._apply_trail_options()
        if self._path is not None:
            self._prepare_viewer(self._path)
            self._select_image(0)

    def copy_to_clipboard(self) -> bool:
        pixmap = self._view.grab()
        if pixmap.isNull():
            return False
        QGuiApplication.clipboard().setPixmap(pixmap)
        return True

    def export(self) -> None:
        path = self._path
        if path is None:
            return
        calculation = self.calculation()
        directory = calculation.neb_directory if calculation else Path.home()
        stem = calculation.name if calculation else "neb"

        filters = (
            "PNG image (*.png);;Energy profile (*.csv);;"
            "Selected image as POSCAR (*.vasp);;Path (*.xyz)"
        )
        path_text, selected = QFileDialog.getSaveFileName(
            self, "Export NEB", str(directory / f"{stem}-neb.png"), filters
        )
        if not path_text:
            return

        from ..structure.export import save_poscar
        from .export import save_path_xyz, save_profile_csv

        target = Path(path_text)
        try:
            if selected.startswith("Energy profile"):
                save_profile_csv(path, target.with_suffix(".csv"))
            elif selected.startswith("Selected image"):
                structure = path.images[self._selected].structure
                if structure is None:
                    raise ValueError("the selected image has no structure to export")
                save_poscar(structure, target.with_suffix(".vasp"))
            elif selected.startswith("Path"):
                save_path_xyz(path, target.with_suffix(".xyz"))
            elif not self._view.grab().save(str(target)):
                raise OSError("QPixmap.save returned False")
            log.info("Exported NEB data to %s", target)
        except (OSError, ValueError) as exc:
            log.exception("NEB export failed")
            self._notice.show_issue(
                Issue(
                    code="export_failed",
                    message="The file could not be written.",
                    detail=str(exc),
                    severity=Severity.ERROR,
                )
            )
