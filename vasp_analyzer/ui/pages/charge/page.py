"""The Charge Density page: isosurface, sectional slices and field analysis.

One selected scalar field drives everything on the page — the 3D surface, the
2D cut, the planar average, the statistics and the integral. Switching source
or data type re-derives all of them together, so no panel can be left showing
a previous field's numbers.

The page is careful about two things the mockup gets wrong, because both would
misstate physics: a *total* electron density is non-negative and gets a
sequential colour map rather than a signed one, and an ELF has no units and no
electron count, so its integration card is replaced rather than relabelled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QObject, QRunnable, QSize, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
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
from ....models.volumetric import FieldKind, VolumetricField, VolumetricFile
from ....services import volumetric as vol
from ....services.atom_selector import parse_atom_selector
from ....services.isosurface import (
    extract_isosurface,
    smooth_mesh,
    suggested_subsample,
)
from ....services.loader import DATASET_VOLUMETRIC, DataLoader
from ...style import icon_url
from ...theme import METRICS, PALETTE, RADIUS, SPACING, TYPE
from ...widgets.axis_bar import AxisBar
from ...widgets.element_swatch import ElementSwatch
from ...widgets.empty_state import EmptyState
from ...widgets.notice import Notice
from ...widgets.popout import PopoutButton
from ...widgets.viewer_toolbar import ViewerToolbar
from ...widgets.primitives import Card
from ...widgets.range_slider import RangeRow
from ...widgets.segmented import SegmentedControl
from ..base import Page
from ..neb.info_panel import KeyValueCard
from ..sequence_view import sequence_extent
from ..structure.viewer import StructureView, ViewOptions
from .plots import PlanarAveragePlot, SliceHeatmap
from .sidebar import DIVERGING_MAPS, SEQUENTIAL_MAPS, ChargeSidebar
from .viewer import DensityView, neighbour_shifts, slice_plane_corners

log = get_logger("ui.charge")


PLANES = ["a-b", "b-c", "a-c"]
PLANE_TOOLTIPS = [
    "The a-b plane, cut along c",
    "The b-c plane, cut along a",
    "The a-c plane, cut along b",
]
# Which lattice direction each plane's normal runs along.
PLANE_AXIS = {0: 2, 1: 0, 2: 1}


class _CompareWorkerSignals(QObject):
    finished = Signal(object)  # ParseResult[VolumetricFile]


class _CompareWorker(QRunnable):
    """Parses a comparison volumetric file off the GUI thread.

    Kept separate from the page's normal DataLoader-driven loads: those are
    keyed only by dataset name on a shared signal, with no way to tell "the
    primary source arrived" from "the comparison file arrived" when both
    happen to be named CHGCAR — the ordinary case for a before/after
    comparison. A dedicated worker with its own signal sidesteps that
    ambiguity entirely rather than trying to disambiguate it.
    """

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path
        self.signals = _CompareWorkerSignals()

    def run(self) -> None:  # pragma: no cover - exercised via the GUI
        from ....parsers.chgcar import parse_volumetric  # noqa: PLC0415

        self.signals.finished.emit(parse_volumetric(self._path))


# A long atom selection (e.g. "all") producing hundreds of rows would turn
# this card into an unusable scroll; capped consistently with the Structure
# page's own neighbour table.
MAX_INTEGRATION_ROWS = 20


class AtomIntegrationTable(QWidget):
    """Per-atom integrated charge, one row per atom, largest first."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._rows_layout = QVBoxLayout()
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)
        layout.addLayout(self._rows_layout)

        self._placeholder = QLabel(
            "Localize to atoms, then Integrate Per-Atom."
        )
        self._placeholder.setWordWrap(True)
        self._placeholder.setStyleSheet(
            f"color: {PALETTE.text_tertiary}; font-size: {TYPE.small}px;"
        )
        layout.addWidget(self._placeholder)

        self._more_label = QLabel("")
        self._more_label.setStyleSheet(
            f"color: {PALETTE.text_tertiary}; font-size: {TYPE.small}px;"
        )
        self._more_label.hide()
        layout.addWidget(self._more_label)

        self._row_widgets: list[QWidget] = []

    def clear(self) -> None:
        for widget in self._row_widgets:
            self._rows_layout.removeWidget(widget)
            # Hidden before the reparent: setParent(None) turns a
            # still-visible widget into a momentary top-level window of its
            # own until deleteLater() runs -- the same hazard that let the
            # Editor's inspector fields steal window activation. These rows
            # can't take focus, but a burst of transient native windows
            # during a reset is exactly the kind of thing that destabilises
            # the platform plugin, so they are taken off screen first.
            widget.hide()
            widget.setParent(None)
            widget.deleteLater()
        self._row_widgets.clear()
        self._more_label.hide()
        self._placeholder.show()

    def set_results(self, results: list, signed: bool) -> None:
        """``results`` is a list of :class:`~....services.volumetric.AtomIntegration`.

        ``signed`` reflects the underlying field's kind (a charge-difference
        integral can be negative, an ordinary density integral cannot), and
        picks the +/- formatting; rows are ordered by |value| so the atoms
        with the most charge either way surface first.
        """
        self.clear()
        if not results:
            return
        self._placeholder.hide()
        ordered = sorted(results, key=lambda r: abs(r.value), reverse=True)
        for entry in ordered[:MAX_INTEGRATION_ROWS]:
            row = self._build_row(entry, signed)
            self._rows_layout.addWidget(row)
            self._row_widgets.append(row)
        if len(ordered) > MAX_INTEGRATION_ROWS:
            self._more_label.setText(
                f"…and {len(ordered) - MAX_INTEGRATION_ROWS} more atom"
                f"{'s' if len(ordered) - MAX_INTEGRATION_ROWS != 1 else ''}"
            )
            self._more_label.show()

    def _build_row(self, entry, signed: bool) -> QWidget:
        row = QWidget()
        row.setObjectName("AtomIntegrationRow")
        row.setStyleSheet(
            f"#AtomIntegrationRow {{ background: {PALETTE.surface_sunken};"
            f" border-radius: {RADIUS.sm}px; }}"
        )
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(SPACING.sm, SPACING.xs, SPACING.sm, SPACING.xs)
        row_layout.setSpacing(SPACING.xs)

        dot = ElementSwatch(entry.symbol, size=10, dark=True)
        row_layout.addWidget(dot)

        name = QLabel(f"{entry.symbol} #{entry.row + 1}")
        name.setStyleSheet(f"font-size: {TYPE.small}px; color: {PALETTE.text};")
        row_layout.addWidget(name)
        row_layout.addStretch(1)

        text = f"{entry.value:+.3f} e⁻" if signed else f"{entry.value:.3f} e⁻"
        value = QLabel(text)
        value.setStyleSheet(
            f"font-size: {TYPE.small}px; font-weight: 600; color: {PALETTE.text};"
        )
        row_layout.addWidget(value)
        return row


class ChargePage(Page):
    """Volumetric field visualisation and analysis."""

    title = "Charge"
    issueRaised = Signal(object)
    infoRequested = Signal(dict)
    infoBarSlotChanged = Signal()       # where the bottom bar belongs changed

    def __init__(self, loader: DataLoader, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loader = loader
        self._sidebar = ChargeSidebar()
        self._sources: list[vol.VolumetricSource] = []
        self._file: VolumetricFile | None = None
        # self._field is what every renderer/statistic reads; self._field_raw
        # is the true parsed data it is derived from. Localizing to atoms
        # replaces self._field with a masked copy and leaves _raw untouched,
        # so every existing consumer of self._field picks up the mask with
        # no changes of its own, and turning localization back off is just
        # restoring self._field = self._field_raw.
        self._field: VolumetricField | None = None
        self._field_raw: VolumetricField | None = None
        # The comparison field loaded via "Charge Difference"; kept separate
        # from _field_raw's own source since it comes from an arbitrary,
        # independently-picked file rather than the current calculation.
        self._compare_field: VolumetricField | None = None
        self._compare_worker: _CompareWorker | None = None
        # Tracks which colour-map family (sequential/diverging) is currently
        # populated in the sidebar, so _apply_localization only rebuilds it
        # when a field's sign convention actually changes (e.g. toggling
        # Charge Difference on), not on every unrelated call.
        self._field_is_signed: bool | None = None
        self._source_index = 0
        self._plane = 0
        self._slice_fraction = 0.5
        self._colormap = "viridis"
        self._pending = False

        # Meshing is not instantaneous, so the isovalue slider schedules it
        # rather than remeshing on every mouse-move event.
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(180)
        self._render_timer.timeout.connect(self._rebuild_surface)

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

        centre = QWidget()
        column = QVBoxLayout(centre)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(SPACING.md)
        column.addWidget(self._build_viewer_card(), 3)
        column.addWidget(self._build_slices_card(), 2)
        self._viewer_column = column
        row.addWidget(centre, 1)
        self._right_column = self._build_analysis_column()
        row.addWidget(self._right_column, 0)
        self._stack.addWidget(content)

        empty_card = Card()
        self._empty = EmptyState(
            title="No volumetric density data found",
            message=(
                "Add a CHGCAR, CHG, PARCHG or ELFCAR file to visualize "
                "real-space electronic data."
            ),
            glyph="◈",
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
        self._view = DensityView()
        self._view.setMinimumHeight(320)
        holder_layout.addWidget(self._view)

        self._toolbar = self._build_floating_toolbar(holder)
        self._toolbar.install_shortcuts(self)
        # Orientation presets in place of a projection toggle: which way the
        # cell faces is the question that actually comes up while reading a
        # structure, and perspective is the right answer for all of them.
        self._axis_bar = AxisBar(holder)
        self._axis_bar.orientationRequested.connect(self._view.orient)
        self._popout = PopoutButton(holder, "Charge Density Viewer", holder)
        self._legend = self._build_floating_legend(holder)
        self._scale_bar = self._build_scale_legend(holder)

        holder.installEventFilter(self)
        self._holder = holder
        body.addWidget(holder, 1)
        return card

    def _on_average_range_changed(self) -> None:
        self._average_plot.set_view_limits(
            self._average_xrange.window(), self._average_yrange.window()
        )

    def _build_floating_toolbar(self, parent: QWidget) -> QWidget:
        return ViewerToolbar(self._view, parent)

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

    def _build_scale_legend(self, parent: QWidget) -> QWidget:
        """The isovalue readout floating in the viewer's top-right.

        Deliberately *not* a colour ramp for a single isosurface: every point
        on one surface has the same value by construction, so a ramp beside it
        would imply variation that does not exist.
        """
        panel = QWidget(parent)
        panel.setObjectName("ScaleLegend")
        panel.setAttribute(Qt.WA_StyledBackground, True)
        column = QVBoxLayout(panel)
        column.setContentsMargins(SPACING.md, SPACING.sm, SPACING.md, SPACING.sm)
        column.setSpacing(2)

        self._scale_title = QLabel("")
        self._scale_title.setStyleSheet(
            f"font-size: {TYPE.small}px; color: #e8e8ea; font-weight: 600;"
        )
        column.addWidget(self._scale_title)

        self._scale_value = QLabel("")
        self._scale_value.setStyleSheet(f"font-size: {TYPE.small}px; color: #c9c9ce;")
        column.addWidget(self._scale_value)

        self._scale_range = QLabel("")
        self._scale_range.setStyleSheet(f"font-size: {TYPE.tiny}px; color: #9a9aa0;")
        column.addWidget(self._scale_range)

        panel.setStyleSheet(
            f"#ScaleLegend {{ background: rgba(38,39,43,0.88);"
            f" border: 1px solid rgba(255,255,255,0.14);"
            f" border-radius: {RADIUS.md}px; }}"
        )
        panel.hide()
        return panel

    def _build_slices_card(self) -> QWidget:
        card = Card(padding=SPACING.lg)
        body = card.body()
        body.setSpacing(SPACING.sm)

        slices_header = QHBoxLayout()
        slices_header.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Sectional Slices")
        title.setObjectName("CardTitle")
        slices_header.addWidget(title)
        slices_header.addStretch(1)
        body.addLayout(slices_header)

        subtitle = QLabel("Explore charge density on different planes through the cell.")
        subtitle.setStyleSheet(
            f"font-size: {TYPE.small}px; color: {PALETTE.text_secondary};"
        )
        body.addWidget(subtitle)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING.lg)

        controls = QWidget()
        controls.setFixedWidth(250)
        control_column = QVBoxLayout(controls)
        control_column.setContentsMargins(0, 0, 0, 0)
        control_column.setSpacing(SPACING.sm)

        plane_caption = QLabel("Plane")
        plane_caption.setStyleSheet(
            f"font-size: {TYPE.small}px; color: {PALETTE.text_secondary};"
        )
        control_column.addWidget(plane_caption)

        self._plane_control = SegmentedControl(PLANES)
        for index, tip in enumerate(PLANE_TOOLTIPS):
            self._plane_control._buttons[index].setToolTip(tip)
        self._plane_control.currentChanged.connect(self._on_plane_changed)
        control_column.addWidget(self._plane_control)

        self._position_label = QLabel("")
        self._position_label.setAlignment(Qt.AlignCenter)
        self._position_label.setStyleSheet(
            f"font-size: {TYPE.body}px; font-weight: 500;"
        )
        control_column.addWidget(self._position_label)

        from ...widgets.painted_slider import PaintedSlider

        self._position_slider = PaintedSlider(0.0, 1.0, 0.5, single_step=0.002)
        self._position_slider.valueChanged.connect(self._on_position_changed)
        control_column.addWidget(self._position_slider)

        ends = QHBoxLayout()
        ends.setContentsMargins(0, 0, 0, 0)
        self._range_low = QLabel("")
        self._range_low.setObjectName("RangeLabel")
        self._range_high = QLabel("")
        self._range_high.setObjectName("RangeLabel")
        ends.addWidget(self._range_low)
        ends.addStretch(1)
        ends.addWidget(self._range_high)
        control_column.addLayout(ends)

        self._show_slice = QCheckBox("Show Slice")
        self._show_slice.setChecked(True)
        self._show_slice.setToolTip(
            "Also draw the cutting plane inside the 3D viewer."
        )
        self._show_slice.toggled.connect(self._on_show_slice)
        control_column.addWidget(self._show_slice)

        self._show_atoms_on_slice = QCheckBox("Atoms Near Plane")
        self._show_atoms_on_slice.setChecked(True)
        self._show_atoms_on_slice.setToolTip(
            "Mark atoms within 1 Å of the cut, so features in the map can be "
            "attributed to the atoms responsible."
        )
        self._show_atoms_on_slice.toggled.connect(lambda _v: self._update_slice())
        control_column.addWidget(self._show_atoms_on_slice)

        control_column.addStretch(1)
        row.addWidget(controls, 0)

        self._slice_plot = SliceHeatmap()
        slices_header.addWidget(PopoutButton(self._slice_plot, "Charge Density Slice", flat=True))
        row.addWidget(self._slice_plot, 1)
        body.addLayout(row, 1)
        return card

    def _build_analysis_column(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setFixedWidth(METRICS.dos_panel_width + 70)

        container = QWidget()
        column = QVBoxLayout(container)
        column.setContentsMargins(0, 0, SPACING.xs, 0)
        column.setSpacing(SPACING.md)
        scroll.setWidget(container)

        # One card for everything that describes the loaded field: its
        # statistics and its grid. These used to be split across this card
        # and an "Information" card at the foot of the column, which said
        # the same range twice and cost the panel ~110px.
        self._stats_card = KeyValueCard("Field Statistics")
        self._stats_card.build(
            [
                ("min", "Min"), ("max", "Max"), ("mean", "Average"),
                ("grid", "Grid Size"), ("spacing", "Grid Spacing"), ("file", "File"),
            ],
            columns=3,
        )
        column.addWidget(self._stats_card)

        average_card = Card(padding=SPACING.md)
        average_body = average_card.body()
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        average_title = QLabel("Planar Average")
        average_title.setStyleSheet(
            f"font-size: {TYPE.body}px; font-weight: 600; color: {PALETTE.text};"
        )
        header.addWidget(average_title)
        header.addStretch(1)
        along = QLabel("Along")
        along.setStyleSheet(
            f"font-size: {TYPE.small}px; color: {PALETTE.text_secondary};"
        )
        header.addWidget(along)
        self._average_axis = QComboBox()
        self._average_axis.addItems(["X (a-axis)", "Y (b-axis)", "Z (c-axis)"])
        self._average_axis.setCurrentIndex(2)
        self._average_axis.setFixedWidth(112)
        self._average_axis.currentIndexChanged.connect(lambda _i: self._update_average())
        header.addWidget(self._average_axis)
        average_body.addLayout(header)

        self._average_plot = PlanarAveragePlot()
        header.addWidget(PopoutButton(self._average_plot, "Planar Average", flat=True))
        self._average_plot.positionPicked.connect(self._on_average_picked)
        average_body.addWidget(self._average_plot, 1)

        self._average_xrange = RangeRow("X limits", "Å")
        self._average_xrange.rangeChanged.connect(
            lambda _l, _h: self._on_average_range_changed()
        )
        average_body.addWidget(self._average_xrange)
        self._average_yrange = RangeRow("Y limits")
        self._average_yrange.rangeChanged.connect(
            lambda _l, _h: self._on_average_range_changed()
        )
        average_body.addWidget(self._average_yrange)
        column.addWidget(average_card)

        self._integration_card = KeyValueCard("Integration")
        self._integration_card.build(
            [
                ("integral", "Integrated Density"),
                ("expected", "Expected Electrons"),
                ("difference", "Difference"),
            ],
            columns=2,
        )
        column.addWidget(self._integration_card)

        atom_integration_card = Card(padding=SPACING.md)
        atom_body = atom_integration_card.body()
        atom_title = QLabel("Per-Atom Charge")
        atom_title.setStyleSheet(
            f"font-size: {TYPE.body}px; font-weight: 600; color: {PALETTE.text};"
        )
        atom_body.addWidget(atom_title)
        self._atom_integration_table = AtomIntegrationTable()
        atom_body.addWidget(self._atom_integration_table)
        column.addWidget(atom_integration_card)

        column.addStretch(1)
        return scroll

    # -- wiring ----------------------------------------------------------------

    def _connect(self) -> None:
        side = self._sidebar
        side.sourceChanged.connect(self._on_source_changed)
        side.componentChanged.connect(self._on_component_changed)
        side.isovalueChanged.connect(self._on_isovalue_changed)
        side.opacityChanged.connect(self._on_opacity_changed)
        side.colormapChanged.connect(self._on_colormap_changed)
        side.displayChanged.connect(self._on_display_changed)
        side.renderRequested.connect(self._rebuild_surface)
        side.exportRequested.connect(self.export)
        side.localizeChanged.connect(self._apply_localization)
        side.compareBrowseRequested.connect(self._browse_compare_file)
        side.compareClearRequested.connect(self._clear_compare)
        side.compareChanged.connect(self._apply_localization)
        side.integratePerAtomRequested.connect(self._on_integrate_per_atom)

    def eventFilter(self, watched, event):  # noqa: N802 - Qt naming
        if watched is self._holder and event.type() in (QEvent.Resize, QEvent.Show):
            self._position_overlays()
        return super().eventFilter(watched, event)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt naming
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

        if not self._scale_bar.isHidden():
            self._scale_bar.adjustSize()
            self._scale_bar.move(
                max(0, holder.width() - self._scale_bar.width() - margin),
                margin + self._axis_bar.height() + SPACING.sm,
            )
            self._scale_bar.raise_()

        if not self._legend.isHidden():
            self._legend_layout.activate()
            self._legend.resize(self._legend.sizeHint())
            self._legend.move(
                max(0, holder.width() - self._legend.width() - margin),
                max(0, holder.height() - self._legend.height() - margin),
            )
            self._legend.raise_()

    # -- Page interface ----------------------------------------------------------

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

        if not calculation.can(Capability.CHARGE_DENSITY):
            self._sidebar.set_enabled(False)
            self._empty.set_text(
                "No volumetric density data found",
                "Add a CHGCAR, CHG, PARCHG or ELFCAR file to visualize "
                "real-space electronic data.",
            )
            self._stack.setCurrentIndex(1)
            self.refresh_info()
            return

        # From discovery rather than a fresh directory scan: opening a single
        # dropped CHGCAR must offer that file alone, not everything sitting
        # beside it. Falls back to a scan only for a Calculation built by
        # older code paths that predate the field.
        self._sources = list(calculation.volumetric_sources) or (
            vol.find_volumetric_sources(calculation.directory)
        )
        self._sidebar.set_enabled(True)
        self._sidebar.set_sources([source.label for source in self._sources])
        self._source_index = 0
        self._request_source(0)

    def _reset_state(self) -> None:
        self._file = None
        self._field = None
        self._field_raw = None
        self._compare_field = None
        self._compare_worker = None
        self._sources = []
        self._notice.clear()
        self._view.set_structure(None)
        self._view.set_mesh(None)
        self._view.set_slice_plane(None)
        self._legend.hide()
        self._scale_bar.hide()
        self._slice_plot.show_message("No field loaded")
        self._average_plot.show_message("No field loaded")
        for card in (self._stats_card, self._integration_card):
            card.clear()
        self._atom_integration_table.clear()

    def _request_source(self, index: int) -> None:
        if not (0 <= index < len(self._sources)):
            return
        self._source_index = index
        source = self._sources[index]
        self._pending = True
        self._notice.clear()
        self._slice_plot.show_message(f"Loading {source.name}…")
        self._average_plot.show_message(f"Loading {source.name}…")
        self._loader.request(DATASET_VOLUMETRIC, key=str(source.path))

    # -- data ---------------------------------------------------------------------

    def _on_data_ready(self, dataset: str, result: ParseResult) -> None:
        if dataset != DATASET_VOLUMETRIC:
            return
        self._pending = False

        if not result.ok or result.value is None:
            message = (
                result.issues[0].message
                if result.issues
                else "The volumetric file could not be read."
            )
            self._slice_plot.show_message("Unavailable")
            self._average_plot.show_message("Unavailable")
            self._notice.show_issue(
                result.issues[0]
                if result.issues
                else Issue(
                    code="volumetric_unreadable",
                    message=message,
                    severity=Severity.ERROR,
                )
            )
            self._stack.setCurrentIndex(0)
            return

        for issue in result.issues:
            if issue.severity in (Severity.WARNING, Severity.ERROR):
                self._notice.show_issue(issue)
                break

        self._file = result.value
        self._stack.setCurrentIndex(0)
        self._sidebar.set_components(
            [self._component_label(f) for f in self._file.fields]
        )
        self._warn_if_md_density()
        self._select_component(0)

    def _component_label(self, field: VolumetricField) -> str:
        if field.component == "total":
            return field.kind.label
        if field.component == "magnetization":
            return FieldKind.MAGNETIZATION.label
        if field.component.startswith("m_"):
            return f"{FieldKind.MAGNETIZATION.label} ({field.component[-1]})"
        return field.kind.label

    def _warn_if_md_density(self) -> None:
        """A CHGCAR written during MD is an extrapolated density.

        VASP writes a density set up to *continue* the trajectory rather than
        the converged density of the final geometry, so presenting it as an
        ordinary static result would be misleading. A quiet note, not a dialog.
        """
        calculation = self.calculation()
        if calculation is None or not calculation.can(Capability.MD_TRAJECTORY):
            return
        self._notice.show_issue(
            Issue(
                code="chgcar_from_md",
                message=(
                    "This directory is a molecular-dynamics run. VASP writes an "
                    "extrapolated CHGCAR during MD, so it is not the converged "
                    "density of any single geometry."
                ),
                severity=Severity.INFO,
                source=self._file.path_name if self._file else "CHGCAR",
            )
        )

    def _select_component(self, index: int) -> None:
        if self._file is None or not self._file.fields:
            return
        index = int(np.clip(index, 0, len(self._file.fields) - 1))
        self._field_raw = self._file.fields[index]
        self._field = self._field_raw
        field = self._field

        # Colour maps follow the field's physics: a signed quantity gets a
        # diverging family, a non-negative one a sequential family.
        self._field_is_signed = field.kind.is_signed
        names = DIVERGING_MAPS if self._field_is_signed else SEQUENTIAL_MAPS
        default = vol.colormap_for(field)
        self._colormap = default
        self._sidebar.set_colormaps(names, self._display_name(default))

        self._slice_fraction = self._default_slice_fraction(field)
        self._position_slider.setValue(self._slice_fraction, notify=False)

        self._prepare_viewer(field)
        self._update_statistics()
        # Reads/writes self._field, configures the isovalue against whichever
        # field ends up active, and internally refreshes statistics,
        # integration, the slice, the average and the surface — so it
        # replaces the direct calls to all of those that used to sit here.
        self._apply_localization()
        self.refresh_info()

    # -- charge difference -------------------------------------------------------

    def _browse_compare_file(self) -> None:
        if self._field_raw is None:
            return
        path_text, _selected = QFileDialog.getOpenFileName(
            self,
            "Compare Against",
            str(Path.home()),
            "Volumetric files (CHGCAR* CHG* ELFCAR* LOCPOT* PARCHG*);;All files (*)",
        )
        if not path_text:
            return
        self._start_compare_load(Path(path_text))

    def _start_compare_load(self, path: Path) -> None:
        from ....parsers.chgcar import peek_volumetric_grid  # noqa: PLC0415

        assert self._field_raw is not None
        grid = peek_volumetric_grid(path)
        if grid is not None and grid != self._field_raw.shape:
            nx, ny, nz = self._field_raw.shape
            self._sidebar.set_compare_status(
                f"Grid mismatch: {path.name} is {grid[0]}×{grid[1]}×{grid[2]}, "
                f"the loaded field is {nx}×{ny}×{nz}."
            )
            return

        self._compare_field = None
        self._sidebar.set_compare_file(path.name)
        self._sidebar.set_compare_status(f"Loading {path.name}…")
        worker = _CompareWorker(path)
        worker.signals.finished.connect(self._on_compare_loaded)
        self._compare_worker = worker
        QThreadPool.globalInstance().start(worker)

    def _on_compare_loaded(self, result: ParseResult) -> None:
        self._compare_worker = None
        if self._field_raw is None:
            return
        if not result.ok or result.value is None or not result.value.fields:
            message = (
                result.issues[0].message if result.issues else "Could not read this file."
            )
            self._sidebar.set_compare_status(message)
            return

        field = result.value.field_for("total") or result.value.fields[0]
        if field.shape != self._field_raw.shape:
            nx, ny, nz = self._field_raw.shape
            fnx, fny, fnz = field.shape
            self._sidebar.set_compare_status(
                f"Grid mismatch: {result.value.path_name} is {fnx}×{fny}×{fnz}, "
                f"the loaded field is {nx}×{ny}×{nz}."
            )
            return

        self._compare_field = field
        self._sidebar.set_compare_status(f"Ready — {result.value.path_name}")
        self._sidebar.compare_enabled.setChecked(True)
        self._apply_localization()

    def _clear_compare(self) -> None:
        self._compare_field = None
        self._compare_worker = None
        self._sidebar.set_compare_file(None)
        self._sidebar.set_compare_status("")
        self._apply_localization()

    def _current_base_field(self) -> VolumetricField | None:
        """``self._field_raw``, or its difference against the comparison
        field when Charge Difference is active -- the single input every
        later display step (localization, isovalue, slice, average) works
        from, so none of them need to know a comparison is involved."""
        if self._field_raw is None:
            return None
        if self._compare_field is not None and self._sidebar.compare_enabled.isChecked():
            try:
                return vol.difference(self._field_raw, self._compare_field)
            except ValueError as exc:
                self._sidebar.set_compare_status(str(exc))
                return self._field_raw
        return self._field_raw

    def _apply_localization(self) -> None:
        """Rebuild self._field from the current base field (see
        :meth:`_current_base_field`), masked to the selected atoms if
        localization is on, then refresh everything that depends on it. The
        single point every renderer's plain ``field = self._field`` goes
        through, so none of them need to know localization or comparison
        exist."""
        base = self._current_base_field()
        if base is None:
            return

        if self._sidebar.localize_enabled.isChecked():
            text = self._sidebar.localize_selector_text()
            selection = parse_atom_selector(text, base.structure.species)
            if selection.is_empty:
                self._field = base
                if not text.strip():
                    self._sidebar.set_localize_status("")
                elif selection.invalid_tokens:
                    self._sidebar.set_localize_status(
                        f"Not recognized: {', '.join(selection.invalid_tokens)}"
                    )
                else:
                    self._sidebar.set_localize_status("No atoms matched")
            else:
                radius = self._sidebar.localize_radius.value()
                self._field = vol.localize_to_atoms(base, selection.rows, radius)
                count = len(selection.rows)
                status = f"{count} atom{'s' if count != 1 else ''} within {radius:.1f} Å"
                if selection.invalid_tokens:
                    status += f" — not recognized: {', '.join(selection.invalid_tokens)}"
                self._sidebar.set_localize_status(status)
        else:
            self._field = base
            self._sidebar.set_localize_status("")

        # A comparison field's sign convention (signed, diverging) differs
        # from an ordinary density's (non-negative, sequential); refreshing
        # the colour map choices here — not only on component switch — keeps
        # the map honest through a Charge Difference toggle too. Guarded so
        # an unrelated call (e.g. dragging the localization radius) doesn't
        # reset a colour map the user already chose.
        if self._field.kind.is_signed != self._field_is_signed:
            self._field_is_signed = self._field.kind.is_signed
            names = DIVERGING_MAPS if self._field_is_signed else SEQUENTIAL_MAPS
            self._colormap = vol.colormap_for(self._field)
            self._sidebar.set_colormaps(names, self._display_name(self._colormap))

        # Zeroing most of the grid can drop the field's own value range by
        # orders of magnitude (a Te dopant's immediate neighbourhood is a
        # small fraction of the whole cell's charge) — an isovalue chosen
        # for the full field is then often above everything remaining,
        # rendering an empty surface with no indication why. Re-suggesting
        # the isovalue against whichever field is now active keeps the
        # surface meaningful through a localization change instead of
        # silently going blank.
        assert self._field is not None
        low, high = self._field.percentile_range(50.0, 99.5)
        self._sidebar.configure_isovalue(
            low, high, self._field.suggested_isovalue(), self._field.kind.units
        )

        self._update_statistics()
        self._update_integration()
        self._update_slice()
        self._update_average()
        self._rebuild_surface()
        # A per-atom breakdown from before this change (a different
        # selection, radius, comparison state, ...) no longer describes what
        # is now on screen — cleared rather than left stale until the button
        # is pressed again.
        self._atom_integration_table.clear()

    def _on_integrate_per_atom(self) -> None:
        base = self._current_base_field()
        if base is None or not self._sidebar.localize_enabled.isChecked():
            return
        if base.kind.units != "e/Å³":
            self._sidebar.set_localize_status(
                f"Per-atom integration needs a density-like field, not {base.kind.label}."
            )
            return

        selection = parse_atom_selector(
            self._sidebar.localize_selector_text(), base.structure.species
        )
        if selection.is_empty:
            return

        radius = self._sidebar.localize_radius.value()
        try:
            results = vol.integrate_per_atom(base, selection.rows, radius)
        except Exception as exc:  # noqa: BLE001 - surfaced, never fatal
            # A numerical analysis step must never take the application down:
            # whatever went wrong belongs in the status line the user is
            # already looking at, with the log carrying the detail.
            log.exception("Per-atom integration failed")
            self._sidebar.set_localize_status(f"Integration failed: {exc}")
            return
        self._atom_integration_table.set_results(results, signed=base.kind.is_signed)

    def _default_slice_fraction(self, field: VolumetricField) -> float:
        """Open the cut where the atoms are, not at the middle of the cell.

        For a slab the geometric centre is usually vacuum, so a default of 0.5
        would open the page on a uniformly empty map. The mean atomic position
        along the plane normal lands on the structure instead.
        """
        axis = PLANE_AXIS[self._plane]
        coords = field.structure.frac_coords
        if coords.size == 0:
            return 0.5
        # Circular mean, so atoms straddling the periodic boundary average to
        # the sheet between them rather than to the opposite side of the cell.
        angles = 2.0 * np.pi * (coords[:, axis] % 1.0)
        mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
        return float((mean_angle / (2.0 * np.pi)) % 1.0)

    @staticmethod
    def _display_name(name: str) -> str:
        return {"viridis": "Viridis", "plasma": "Plasma", "RdBu_r": "RdBu"}.get(
            name, name.capitalize()
        )

    def _prepare_viewer(self, field: VolumetricField) -> None:
        structure = field.structure
        self._view.set_structure(structure, reset_camera=True)
        extent = sequence_extent(
            structure.lattice,
            (structure.frac_coords % 1.0) @ structure.lattice,
            self._view.vacuum_info(),
        )
        if extent.size:
            self._view.camera().fit(extent)
            self._view.camera().extent *= 1.1
            self._view.camera().reset()
        self._apply_view_options()
        self._update_legend(structure.species)

    # -- rendering -------------------------------------------------------------------

    def _rebuild_surface(self) -> None:
        field = self._field
        if field is None:
            return
        isovalue = self._sidebar.isovalue.value()
        # "Full Resolution" still respects the automatic per-axis stride: on a
        # finely sampled slab grid, meshing every point yields sub-pixel facets
        # that cost time and render as speckle. An explicit coarser choice
        # multiplies that baseline rather than replacing it, so the control
        # always makes the mesh coarser and never unexpectedly finer.
        chosen = self._sidebar.subsample_factor()
        automatic = suggested_subsample(field)
        subsample = tuple(stride * chosen for stride in automatic)

        mesh = extract_isosurface(field, isovalue, subsample=subsample)
        smoothing = self._sidebar.smoothing.value()
        if smoothing > 0.0 and not mesh.is_empty:
            mesh = smooth_mesh(mesh, smoothing)

        self._view.set_mesh(mesh)
        self._view.set_opacity(self._sidebar.opacity.value())
        self._update_scale_legend(mesh.n_triangles)

    def _update_scale_legend(self, triangles: int) -> None:
        field = self._field
        if field is None:
            self._scale_bar.hide()
            return
        units = field.kind.units
        isovalue = self._sidebar.isovalue.value()

        self._scale_title.setText(f"{field.kind.label}{f'  ({units})' if units else ''}")
        self._scale_value.setText(
            f"Isosurface at {isovalue:.4g}{f' {units}' if units else ''}"
        )
        self._scale_range.setText(
            f"field {field.minimum:.4g} … {field.maximum:.4g}   ·   "
            f"{triangles:,} triangles"
        )
        self._scale_bar.show()
        QTimer.singleShot(0, self._position_overlays)

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
        """The 3D viewer resolves colours at paint time; the slice plot draws
        atom markers into a matplotlib figure and has to be rebuilt."""
        self._view.update()
        self._update_slice()

    # -- analysis panels --------------------------------------------------------------

    def _update_statistics(self) -> None:
        field = self._field
        if field is None:
            self._stats_card.clear()
            return
        units = field.kind.units
        suffix = f" {units}" if units else ""
        self._stats_card.set_title(
            f"{field.kind.label} Statistics" if not units
            else f"{field.kind.label} ({units})"
        )
        self._stats_card.set_values(
            {
                "min": f"{field.minimum:+.4f}{suffix}" if field.kind.is_signed
                else f"{field.minimum:.4f}{suffix}",
                "max": f"{field.maximum:+.4f}{suffix}" if field.kind.is_signed
                else f"{field.maximum:.4f}{suffix}",
                "mean": f"{field.mean:+.4f}{suffix}" if field.kind.is_signed
                else f"{field.mean:.4f}{suffix}",
                "grid": vol.describe_grid(field),
                "spacing": vol.describe_spacing(field),
                "file": field.source_file,
            },
            tooltips={
                "spacing": "Along a, b and c. The grid is uniform in fractional "
                "coordinates, so unequal lattice vectors give unequal spacings.",
            },
        )

    def _update_integration(self) -> None:
        field = self._field
        if field is None:
            self._integration_card.clear()
            return

        # An ELF integral is not an electron count and has no useful meaning;
        # the card reports the field's mean instead of a fabricated charge.
        if not field.kind.integrates_to_electrons:
            self._integration_card.set_title("Field Summary")
            self._integration_card.build(
                [("mean", "Volume Average"), ("span", "Range")], columns=1
            )
            self._integration_card.set_values(
                {
                    "mean": f"{field.mean:.4f}",
                    "span": f"{field.minimum:.4f} … {field.maximum:.4f}",
                }
            )
            return

        calculation = self.calculation()
        expected = None
        source = ""
        if calculation is not None and calculation.metadata.n_electrons is not None:
            expected = float(calculation.metadata.n_electrons.value)
            source = calculation.metadata.n_electrons.source

        result = vol.integrate(field, expected, source)
        self._integration_card.set_title("Integration")
        self._integration_card.build(
            [
                ("integral", "Integrated Density"),
                ("expected", "Expected Electrons"),
                ("difference", "Difference"),
            ],
            columns=2,
        )

        difference = "—"
        tooltip = ""
        if result.difference is not None:
            error = result.relative_error or 0.0
            mark = " ✓" if result.is_consistent else ""
            difference = f"{result.difference:+.4f} e  ({error:+.3f} %){mark}"
            tooltip = (
                "A grid integral is a finite-resolution approximation, so a "
                "fraction of a percent is expected. A large difference points "
                "to a normalisation or grid-order problem."
            )

        self._integration_card.set_values(
            {
                "integral": f"{result.integral:.4f} e",
                "expected": "—" if expected is None else f"{expected:.4f} e",
                "difference": difference,
            },
            tooltips={
                "expected": f"Read from {source}" if source else "",
                "difference": tooltip,
            },
        )

    def _update_slice(self) -> None:
        field = self._field
        if field is None:
            return
        axis = PLANE_AXIS[self._plane]
        data = vol.slice_geometry(field, axis, self._slice_fraction)
        markers = (
            vol.atoms_near_plane(field, axis, self._slice_fraction, tolerance=1.0)
            if self._show_atoms_on_slice.isChecked()
            else None
        )
        self._slice_plot.set_slice(
            data,
            self._colormap,
            vol.display_range(field),
            field.kind.units,
            markers,
        )

        length = field.structure.lattice_parameters[axis]
        name = vol.AXIS_NAMES[axis]
        self._position_label.setText(f"{name} = {data.distance:.3f} Å")
        self._range_low.setText("0.000 Å")
        self._range_high.setText(f"{length:.3f} Å")

        self._view.set_slice_plane(
            slice_plane_corners(field.lattice, axis, self._slice_fraction)
            if self._show_slice.isChecked()
            else None
        )
        self._sync_average_cursor()

    def _update_average(self) -> None:
        field = self._field
        if field is None:
            return
        axis = self._average_axis.currentIndex()
        distance, values = vol.planar_average(field, axis)
        units = field.kind.units
        label = f"{field.kind.label} ({units})" if units else field.kind.label
        self._average_plot.set_curve(
            distance,
            values,
            f"{vol.AXIS_NAMES[axis]} (Å)",
            label,
            field.kind,
            "ρ̄" if field.kind.integrates_to_electrons else "mean",
        )
        # New data means a new extent; the sliders reset to it rather than
        # keeping a window that referred to the previous field's numbers.
        (x_low, x_high), (y_low, y_high) = self._average_plot.data_limits()
        self._average_xrange.set_bounds(x_low, x_high)
        self._average_yrange.set_bounds(y_low, y_high)
        self._sync_average_cursor()

    def _sync_average_cursor(self) -> None:
        """Show where the sectional slice sits, when the axes agree."""
        field = self._field
        if field is None:
            return
        axis = PLANE_AXIS[self._plane]
        same = axis == self._average_axis.currentIndex()
        length = field.structure.lattice_parameters[axis]
        self._average_plot.set_cursor(self._slice_fraction * length, visible=same)

    # -- control handlers ----------------------------------------------------------------

    def _on_source_changed(self, index: int) -> None:
        self._request_source(index)

    def _on_component_changed(self, index: int) -> None:
        self._select_component(index)

    def _on_isovalue_changed(self, _value: float) -> None:
        # Debounced: dragging the slider emits continuously, and remeshing on
        # every event would make the control feel stuck.
        self._render_timer.start()

    def _on_opacity_changed(self, value: float) -> None:
        # Cheap — no remeshing needed, so it applies immediately.
        self._view.set_opacity(value)

    def _on_colormap_changed(self, name: str) -> None:
        self._colormap = name.lower().replace(" ", "")
        if self._colormap == "rdbu":
            self._colormap = "RdBu_r"
        elif self._colormap == "coolwarm":
            self._colormap = "coolwarm"
        elif self._colormap == "bwr":
            self._colormap = "bwr_r"
        self._sidebar.colormap_strip.set_colormap(self._colormap)
        self._update_slice()

    def _on_display_changed(self) -> None:
        checks = self._sidebar.checks
        self._view.set_show_surface(checks["surface"].isChecked())
        self._view.set_show_atoms(checks["atoms"].isChecked())
        self._view.set_periodic_images(
            neighbour_shifts(self._field.lattice)
            if self._field is not None and checks["periodic"].isChecked()
            else None
        )
        self._apply_view_options()

    def _apply_view_options(self) -> None:
        checks = self._sidebar.checks
        self._view.set_options(
            ViewOptions(
                style="Ball-and-Stick",
                coloring="By Element",
                show_cell=checks["cell"].isChecked(),
                show_axes=checks["axes"].isChecked(),
                perspective=True,
            )
        )

    def _on_plane_changed(self, index: int) -> None:
        self._plane = index
        # Each plane has its own useful default, since the atoms sit at a
        # different fraction along each lattice direction.
        if self._field is not None:
            self._slice_fraction = self._default_slice_fraction(self._field)
            self._position_slider.setValue(self._slice_fraction, notify=False)
        self._update_slice()

    def _on_position_changed(self, value: float) -> None:
        self._slice_fraction = float(np.clip(value, 0.0, 1.0))
        self._update_slice()

    def _on_show_slice(self, _checked: bool) -> None:
        self._update_slice()

    def _on_average_picked(self, fraction: float) -> None:
        """Clicking the planar average moves the sectional cut there."""
        field = self._field
        if field is None:
            return
        axis = self._average_axis.currentIndex()
        # Only meaningful when the two panels refer to the same direction.
        for plane, mapped in PLANE_AXIS.items():
            if mapped == axis:
                self._plane_control.setCurrentIndex(plane)
                self._plane = plane
                break
        self._slice_fraction = float(np.clip(fraction, 0.0, 1.0))
        self._position_slider.setValue(self._slice_fraction, notify=False)
        self._update_slice()


    # -- bottom bar -----------------------------------------------------------------------

    def _info_fields(self) -> dict[str, str]:
        field = self._field
        if field is None:
            return {}
        integral = "—"
        if field.kind.integrates_to_electrons:
            integral = f"{field.integral():.2f} e"
        return {
            "system": field.structure.formula or "—",
            "source": field.source_file,
            "datatype": field.kind.label,
            "grid": vol.describe_grid(field),
            "volume": f"{field.volume:.2f} Å³",
            "integrated": integral,
            "spin": self._file.spin_mode if self._file else "—",
        }

    def refresh_info(self) -> None:
        self.infoRequested.emit(self._info_fields())

    # -- toolbar actions ---------------------------------------------------------------------

    def reset(self) -> None:
        self._sidebar.reset()
        self._view.orient("default")
        self._plane_control.setCurrentIndex(0)
        self._plane = 0
        self._slice_fraction = 0.5
        self._position_slider.setValue(0.5, notify=False)
        self._average_axis.setCurrentIndex(2)
        self._show_slice.setChecked(True)
        if self._field_raw is not None:
            # One pipeline, from the raw base field. The sidebar reset above
            # is signal-quiet, so localization/comparison state changed
            # without recomputing self._field -- _apply_localization is the
            # single place that recomputes it and refreshes everything
            # downstream (isovalue suggestion, statistics, slice, average,
            # surface). Anything less left self._field pointing at the
            # now-stale localized/compared field after a reset.
            self._apply_localization()
            self._prepare_viewer(self._field)
            self._on_display_changed()

    def copy_to_clipboard(self) -> bool:
        pixmap = self._view.grab()
        if pixmap.isNull():
            return False
        QGuiApplication.clipboard().setPixmap(pixmap)
        return True

    def export(self) -> None:
        field = self._field
        if field is None:
            return
        calculation = self.calculation()
        directory = calculation.directory if calculation else Path.home()
        stem = calculation.name if calculation else "density"

        filters = (
            "3D view (*.png);;Slice image (*.png);;Planar average (*.csv);;"
            "Slice data (*.csv)"
        )
        path_text, selected = QFileDialog.getSaveFileName(
            self, "Export Charge Density", str(directory / f"{stem}-density.png"), filters
        )
        if not path_text:
            return

        from .export import save_average_csv, save_slice_csv

        target = Path(path_text)
        try:
            if selected.startswith("Planar average"):
                save_average_csv(
                    field, self._average_axis.currentIndex(), target.with_suffix(".csv")
                )
            elif selected.startswith("Slice data"):
                save_slice_csv(
                    field,
                    PLANE_AXIS[self._plane],
                    self._slice_fraction,
                    target.with_suffix(".csv"),
                )
            elif selected.startswith("Slice image"):
                self._slice_plot.figure.savefig(
                    target, dpi=300, bbox_inches="tight", facecolor="white"
                )
            elif not self._view.grab().save(str(target)):
                raise OSError("QPixmap.save returned False")
            log.info("Exported charge density to %s", target)
        except (OSError, ValueError) as exc:
            log.exception("Charge density export failed")
            self._notice.show_issue(
                Issue(
                    code="export_failed",
                    message="The file could not be written.",
                    detail=str(exc),
                    severity=Severity.ERROR,
                )
            )
