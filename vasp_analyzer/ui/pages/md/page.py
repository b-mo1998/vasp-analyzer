"""The Molecular Dynamics page: trajectory playback with synchronised telemetry.

Layout follows the reference design: display controls on the left, the 3D
trajectory viewer and its transport in the middle, and the temperature / energy
/ MSD charts on the right. The charts and the viewer are two views of one
state — the current frame — and every control that changes it updates both.
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
    QRadioButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ....core.errors import Issue, ParseResult, Severity
from ....core.logging_setup import get_logger
from ....models.calculation import Calculation
from ....models.capabilities import Capability
from ....models.trajectory import MDTrajectory
from ....services import md as md_service
from ....services import md_stats
from ....services.appearance import effective_color
from ....services.loader import DATASET_MD, DataLoader
from ....services.trajectory import (
    angle_series,
    mean_square_displacement,
    pair_distance_series,
    unwrapped_cartesian,
    wrap_fractional,
)
from ...style import icon_url
from ...theme import METRICS, PALETTE, RADIUS, SPACING, TYPE
from ...widgets.axis_bar import AxisBar
from ...widgets.element_swatch import ElementSwatch
from ...widgets.empty_state import EmptyState
from ...widgets.notice import Notice
from ...widgets.popout import PopoutButton
from ...widgets.viewer_toolbar import ViewerToolbar
from ...widgets.primitives import Card, Divider, SectionHeader
from ..base import Page
from ..neb.info_panel import KeyValueCard
from ..sequence_view import SequenceView, TrailOptions, sequence_extent
from ..structure.viewer import StructureView, ViewOptions
from .playback import PlaybackBar
from .plots import EnergyPlot, GeometryPlot, MSDPlot, TemperaturePlot
from .sidebar import UNWRAPPED, MDSidebar

log = get_logger("ui.md")



class MDPage(Page):
    """Trajectory viewer, playback transport and MD telemetry."""

    title = "MD"
    issueRaised = Signal(object)
    infoRequested = Signal(dict)   # bottom-bar fields for this calculation

    def __init__(self, loader: DataLoader, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loader = loader
        self._sidebar = MDSidebar()
        self._trajectory: MDTrajectory | None = None
        self._unwrapped: np.ndarray | None = None    # (n_frames, n_atoms, 3) Å
        self._msd: dict[str, np.ndarray] | None = None
        self._frame = 0
        self._energy_key = "total_energy"

        self._build()
        self._connect()
        loader.data_ready.connect(self._on_data_ready)

    # -- construction --------------------------------------------------------

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
        row.addWidget(self._build_data_panel(), 0)
        self._stack.addWidget(content)

        empty_card = Card()
        self._empty = EmptyState(
            title="No MD trajectory found",
            message="A compatible XDATCAR or trajectory source was not detected.",
            glyph="◎",
        )
        empty_card.body().addWidget(self._empty)
        self._stack.addWidget(empty_card)
        self._stack.setCurrentIndex(1)

    def _build_viewer_card(self) -> QWidget:
        card = Card(padding=SPACING.lg)
        body = card.body()
        body.setSpacing(SPACING.sm)

        # --- viewer with floating overlays --------------------------------
        holder = QWidget()
        holder_layout = QVBoxLayout(holder)
        holder_layout.setContentsMargins(0, 0, 0, 0)

        self._view = SequenceView()
        self._view.setMinimumHeight(340)
        holder_layout.addWidget(self._view)

        self._toolbar = self._build_floating_toolbar(holder)
        self._toolbar.install_shortcuts(self)
        # Orientation presets in place of a projection toggle: which way the
        # cell faces is the question that actually comes up while reading a
        # structure, and perspective is the right answer for all of them.
        self._axis_bar = AxisBar(holder)
        self._axis_bar.orientationRequested.connect(self._view.orient)
        self._popout = PopoutButton(holder, "MD Trajectory Viewer", holder)
        self._legend = self._build_floating_legend(holder)

        holder.installEventFilter(self)
        self._holder = holder
        body.addWidget(holder, 1)

        body.addWidget(Divider())
        self._playback = PlaybackBar()
        body.addWidget(self._playback)
        return card

    def _build_floating_toolbar(self, parent: QWidget) -> QWidget:
        # Fit frames the whole trajectory, not the current frame: fitting to
        # one frame would make the structure jump as playback moved through
        # frames whose extents differ.
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

    def _build_data_panel(self) -> QWidget:
        # Scrollable: three plots plus the statistics card need more height
        # than a laptop window has, and without this Qt squeezes the plots
        # until their axis titles collide with the controls beneath them.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setFixedWidth(METRICS.dos_panel_width + 56)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        card = Card(padding=SPACING.lg)
        body = card.body()
        body.setSpacing(SPACING.sm)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        title = QLabel("MD Data")
        title.setObjectName("CardTitle")
        header.addWidget(title)
        header.addStretch(1)
        self._source_label = QLabel("")
        self._source_label.setStyleSheet(
            f"font-size: {TYPE.small}px; color: {PALETTE.text_secondary};"
        )
        header.addWidget(self._source_label)
        body.addLayout(header)

        self._temperature_plot = TemperaturePlot()
        body.addWidget(self._wrap_plot("Temperature (K)", self._temperature_plot), 1)

        self._energy_plot = EnergyPlot()
        energy_card = self._wrap_plot("Energy (eV)", self._energy_plot)
        energy_card.layout().addWidget(self._build_energy_selector())
        body.addWidget(energy_card, 1)

        self._msd_plot = MSDPlot()
        body.addWidget(
            self._wrap_plot("Mean Square Displacement (Å²)", self._msd_plot), 1
        )

        # Shown only while two or three atoms are selected: a distance or an
        # angle over time. Hidden rather than showing "select atoms" text,
        # because an empty block that size is height the three plots above
        # can use.
        self._geometry_plot = GeometryPlot()
        self._geometry_block = self._wrap_plot("Selection Geometry", self._geometry_plot)
        self._geometry_title = self._geometry_block.findChild(QLabel)
        self._geometry_block.hide()
        body.addWidget(self._geometry_block, 1)

        # Two columns, four fields: half the vertical footprint of the
        # single-column version, which is height the three plots above need
        # more than this card does. The instantaneous min-max range moved
        # into the mean temperature's tooltip rather than being dropped —
        # it is context for the mean, not a headline number, and a wide
        # swing around a correct mean is normal in a small cell.
        self._stats_card = KeyValueCard("Run Statistics")
        self._stats_card.build(
            [
                ("mean_t", "Mean Temperature"),
                ("drift", "Energy Drift"),
                ("target_t", "Target"),
                ("window", "Averaged Over"),
            ],
            columns=2,
        )
        body.addWidget(self._stats_card)

        # Formerly a permanent caption line. It is a one-time discovery aid,
        # and the row it occupied is worth more as plot height, so it moved to
        # a tooltip on the panel itself.
        card.setToolTip("Click a plot to jump to that time.")
        scroll.setWidget(card)
        return scroll

    def _wrap_plot(self, title: str, plot: QWidget) -> QWidget:
        holder = QWidget()
        holder.setObjectName("PlotBlock")
        holder.setAttribute(Qt.WA_StyledBackground, True)
        column = QVBoxLayout(holder)
        column.setContentsMargins(SPACING.xs, SPACING.xs, SPACING.xs, SPACING.xs)
        column.setSpacing(2)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        label = QLabel(title)
        label.setStyleSheet(
            f"font-size: {TYPE.small}px; font-weight: 600; color: {PALETTE.text};"
        )
        header.addWidget(label)
        header.addStretch(1)
        header.addWidget(PopoutButton(plot, title, flat=True))
        column.addLayout(header)
        column.addWidget(plot, 1)
        holder.setStyleSheet(
            f"#PlotBlock {{ background: {PALETTE.surface_sunken};"
            f" border: 1px solid {PALETTE.border};"
            f" border-radius: {RADIUS.md}px; }}"
        )
        return holder

    def _build_energy_selector(self) -> QWidget:
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING.md)

        self._energy_group = QButtonGroup(self)
        self._energy_group.setExclusive(True)
        self._energy_buttons: dict[str, QRadioButton] = {}
        for index, (key, label, _colour) in enumerate(EnergyPlot.SERIES):
            # "Total" / "Potential" / "Kinetic" — the panel is too narrow for
            # three full "… Energy" labels and the card title already says
            # what is being measured.
            button = QRadioButton(label.replace(" Energy", ""))
            button.setToolTip(label)
            button.setStyleSheet(f"font-size: {TYPE.tiny}px;")
            self._energy_group.addButton(button, index)
            self._energy_buttons[key] = button
            row.addWidget(button)
            if key == self._energy_key:
                button.setChecked(True)
        row.addStretch(1)
        self._energy_group.idToggled.connect(self._on_energy_selected)
        return holder

    # -- wiring ---------------------------------------------------------------

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
        side.showTrailChanged.connect(lambda _v: self._apply_trail_options())
        side.boundaryModeChanged.connect(lambda _m: self._show_frame(self._frame))
        side.centreChanged.connect(lambda _v: self._show_frame(self._frame))
        side.trackChanged.connect(self._on_track_changed)
        side.selectionCleared.connect(self._clear_selection)

        self._playback.frameChanged.connect(self._on_frame_changed)
        self._view.atomSelected.connect(self._on_atom_selected)
        self._view.selectionChanged.connect(self._on_selection_geometry)

        for plot in (
            self._temperature_plot, self._energy_plot, self._msd_plot,
            self._geometry_plot,
        ):
            plot.seekRequested.connect(self._on_seek)

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

        if not self._legend.isHidden():
            self._legend_layout.activate()
            self._legend.resize(self._legend.sizeHint())
            self._legend.move(
                max(0, holder.width() - self._legend.width() - margin),
                max(0, holder.height() - self._legend.height() - margin),
            )
            self._legend.raise_()

    # -- Page interface --------------------------------------------------------

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

        if not calculation.can(Capability.MD_TRAJECTORY):
            self._sidebar.set_enabled(False)
            self._empty.set_text(
                "No MD trajectory found",
                calculation.why_not(Capability.MD_TRAJECTORY),
            )
            self._stack.setCurrentIndex(1)
            self.refresh_info()
            return

        self._sidebar.set_enabled(True)
        self._loader.request(DATASET_MD)

    def _reset_state(self) -> None:
        self._trajectory = None
        self._unwrapped = None
        self._msd = None
        self._frame = 0
        self._notice.clear()
        self._view.set_structure(None)
        self._view.set_path(None)
        self._view.set_tracked_atom(None)
        self._playback.set_frame_count(0)
        self._sidebar.set_selected_atom(None, "")
        self._source_label.setText("")
        self._legend.hide()

    # -- data -------------------------------------------------------------------

    def _on_data_ready(self, dataset: str, result: ParseResult) -> None:
        if dataset != DATASET_MD:
            return
        if not result.ok or result.value is None:
            message = (
                result.issues[0].message
                if result.issues
                else "The trajectory could not be read."
            )
            self._empty.set_text("No MD trajectory found", message)
            self._stack.setCurrentIndex(1)
            self.refresh_info()
            return

        for issue in result.issues:
            if issue.severity in (Severity.WARNING, Severity.ERROR):
                self._notice.show_issue(issue)
                break
            if issue.severity is Severity.INFO:
                self._notice.show_issue(issue)
                break

        trajectory = result.value
        self._trajectory = trajectory
        self._unwrapped = unwrapped_cartesian(trajectory)
        self._msd = mean_square_displacement(trajectory, self._unwrapped)

        self._stack.setCurrentIndex(0)
        self._prepare_viewer(trajectory)
        self._populate_plots(trajectory)
        self._populate_statistics(trajectory)
        self._update_legend(trajectory)

        times = trajectory.times().tolist()
        self._playback.set_frame_count(
            trajectory.n_frames,
            times,
            "ps" if trajectory.has_time_axis else "",
        )
        self._show_frame(0)
        self._emit_info(trajectory)

    def _display_frac(self, frame: int) -> np.ndarray:
        """The fractional coordinates actually drawn for one frame.

        Wrapping and centring are display transforms, so the camera has to be
        fitted to their output rather than to the raw file coordinates — fitting
        to one and drawing the other is what pushes a centred structure outside
        its own framing.
        """
        trajectory = self._trajectory
        assert trajectory is not None
        if self._sidebar.boundary_mode() == UNWRAPPED and self._unwrapped is not None:
            lattice = trajectory.lattices[frame]
            frac = self._unwrapped[frame] @ np.linalg.inv(lattice)
        else:
            frac = wrap_fractional(trajectory.frac_coords[frame])

        if self._sidebar.centre_structure.isChecked():
            frac = frac - frac.mean(axis=0) + np.array([0.5, 0.5, 0.5])
        return frac

    def _prepare_viewer(self, trajectory: MDTrajectory) -> None:
        """Frame the camera on the whole run, not just its first configuration."""
        self._view.set_structure(trajectory.structure(0), reset_camera=True)
        # Sample rather than use every frame: fifty configurations already
        # bound the motion, and a 100k-frame run should not pay for the rest.
        step = max(1, trajectory.n_frames // 50)
        frames = range(0, trajectory.n_frames, step)
        sample = np.array([self._display_frac(f) for f in frames]) @ trajectory.lattices[0]
        extent = sequence_extent(
            trajectory.lattices[0], sample, self._view.vacuum_info()
        )
        self._view.fit_to_sequence(extent)
        self._apply_view_options()
        self._apply_trail_options()

    def _fit_sequence(self) -> None:
        """Reframe on the whole run, for the toolbar's Fit button.

        Deliberately not the view's own fit_view(), which frames only the
        frame currently on screen -- on a trajectory that makes Fit mean
        something different depending on when it is pressed.
        """
        if self._trajectory is not None:
            self._prepare_viewer(self._trajectory)
            self._show_frame(self._frame)

    def _populate_statistics(self, trajectory: MDTrajectory) -> None:
        """Summarise how the run behaved, not just what it looked like.

        A temperature trace swinging between 90 K and 750 K around a 300 K
        target looks alarming and is completely normal for a small cell —
        only the mean and its spread say whether the thermostat held, and
        only the energy drift says whether the timestep was sane.
        """
        stats = md_stats.analyse(trajectory)
        if stats is None:
            self._stats_card.clear()
            self._stats_card.setVisible(False)
            return
        self._stats_card.setVisible(True)

        values: dict[str, str] = {}
        tooltips: dict[str, str] = {}

        temperature = stats.temperature
        if temperature is not None:
            values["mean_t"] = temperature.summary
            # The instantaneous spread rides along as the mean's tooltip
            # rather than its own row; see the card's layout comment.
            tooltips["mean_t"] = (
                f"Instantaneous range over the same window: "
                f"{temperature.minimum:.0f} – {temperature.maximum:.0f} K.\n"
                "Instantaneous temperature fluctuates as 1/√N; a wide range in "
                "a small cell is expected and is not itself a problem."
            )
        else:
            values["mean_t"] = "—"

        offset = stats.temperature_offset
        if stats.target_temperature is None:
            values["target_t"] = "—"
        elif offset is None:
            values["target_t"] = f"{stats.target_temperature:.0f} K"
        else:
            values["target_t"] = f"{stats.target_temperature:.0f} K  ({offset:+.1f} K)"
            tooltips["target_t"] = (
                "Mean minus target. A well-behaved thermostat sits within a "
                "few kelvin of the requested temperature."
            )

        energy = stats.total_energy
        if energy is None:
            values["drift"] = "—"
        else:
            values["drift"] = f"{md_stats.format_value(energy.drift)} {energy.unit}"
            tooltips["drift"] = (
                "End-to-end change of a straight-line fit to the total energy. "
                "A large drift points at too long a timestep or too loose an "
                "electronic convergence."
            )

        skipped = stats.equilibration_steps
        values["window"] = f"steps {skipped + 1}–{stats.n_steps}"
        tooltips["window"] = (
            "The leading fraction of a run is equilibration; averaging over it "
            "would bias every number taken from the trajectory."
        )

        self._stats_card.set_values(values, tooltips=tooltips)

    def _populate_plots(self, trajectory: MDTrajectory) -> None:
        x = trajectory.times()
        xlabel = "Time (ps)" if trajectory.has_time_axis else "Frame"

        temperature = trajectory.series_for("temperature")
        if temperature is not None:
            self._temperature_plot.set_data(
                x, temperature.values, xlabel, trajectory.target_temperature
            )
        else:
            self._temperature_plot.show_unavailable(
                "Temperature unavailable\nno OSZICAR temperature history"
            )

        energies = {
            key: series.values
            for key, series in trajectory.series.items()
            if key.endswith("energy")
        }
        available = [key for key, _l, _c in EnergyPlot.SERIES if key in energies]
        for key, button in self._energy_buttons.items():
            button.setEnabled(key in energies)
        if available:
            if self._energy_key not in energies:
                self._energy_key = available[0]
                self._energy_buttons[self._energy_key].setChecked(True)
            self._energy_plot.set_data(x, energies, self._energy_key, xlabel)
        else:
            self._energy_plot.show_unavailable(
                "Energies unavailable\nno OSZICAR energy history"
            )

        if self._msd:
            self._msd_plot.set_data(x, self._msd, trajectory.elements, xlabel)
        else:
            self._msd_plot.show_unavailable("Not enough frames for MSD")

        self._source_label.setText(
            "Source: " + " + ".join(trajectory.source_files)
            if trajectory.source_files
            else ""
        )

    def _update_legend(self, trajectory: MDTrajectory) -> None:
        while self._legend_layout.count():
            item = self._legend_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        for element in trajectory.elements:
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
        """Repaint everything that draws an element in its own colour. The
        viewer resolves colours at paint time, so an update() is enough there;
        the sidebar's selected-atom dot holds a colour it was handed, so it is
        re-sent."""
        self._view.update()
        self._on_atom_selected(self._view.selected_index())

    # -- frame handling -----------------------------------------------------------

    def _on_frame_changed(self, frame: int) -> None:
        self._show_frame(frame)

    def _show_frame(self, frame: int) -> None:
        trajectory = self._trajectory
        if trajectory is None:
            return
        self._frame = int(np.clip(frame, 0, trajectory.n_frames - 1))

        structure = trajectory.structure(self._frame)
        # Wrapping (or unwrapping) and centring are applied by _display_frac, so
        # exactly the coordinates the camera was fitted to are what gets drawn.
        structure.frac_coords = self._display_frac(self._frame)

        self._view.show_frame(structure, self._frame)
        self._view.set_frame_index(self._frame)

        x = trajectory.times()
        value = float(x[self._frame]) if x.size else 0.0
        for plot in (
            self._temperature_plot, self._energy_plot, self._msd_plot,
            self._geometry_plot,
        ):
            plot.set_cursor(value)

        self._playback.set_frame(self._frame, notify=False)

    def _on_seek(self, value: float) -> None:
        trajectory = self._trajectory
        if trajectory is None:
            return
        self._playback.stop()
        frame = md_service.frame_for_time(trajectory, value)
        self._playback.set_frame(frame, notify=False)
        self._show_frame(frame)

    # -- options -----------------------------------------------------------------

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
        side = self._sidebar
        # Any selection (multi or tracked) qualifies for the selected-only
        # trail, not just the single tracked atom.
        has_selection = (
            self._view.tracked_atom() is not None or bool(self._view.selection())
        )
        selected_only = has_selection and side.show_trail.isChecked() and (
            not side.atom_trails.isChecked()
        )
        self._view.set_trail_options(
            TrailOptions(
                enabled=side.atom_trails.isChecked() or selected_only,
                length=side.trail_frames(),
                selected_only=selected_only,
            )
        )
        self._view.set_path(self._unwrapped if self._unwrapped is not None else None)


    def _on_energy_selected(self, index: int, checked: bool) -> None:
        if not checked or self._trajectory is None:
            return
        self._energy_key = EnergyPlot.SERIES[index][0]
        self._populate_plots(self._trajectory)
        self._show_frame(self._frame)

    # -- selection ----------------------------------------------------------------

    def _on_atom_selected(self, index: int | None) -> None:
        trajectory = self._trajectory
        if trajectory is None or index is None:
            self._clear_selection()
            return
        selection = self._view.selection()
        if len(selection) > 1:
            # The chip names one atom; with several selected it reports the
            # count instead, coloured neutrally -- picking one element's
            # colour for a mixed selection would misdescribe it.
            self._sidebar.set_selected_atom(
                f"{len(selection)} atoms selected", PALETTE.accent
            )
        else:
            element = trajectory.species[index]
            self._sidebar.set_selected_atom(
                f"{element} #{index + 1}", effective_color(element)
            )
        if self._sidebar.track_atom.isChecked():
            # Tracking follows the atom clicked last -- the camera can only
            # centre on one.
            self._view.set_tracked_atom(index)
        self._apply_trail_options()
        self._show_frame(self._frame)

    def _on_selection_geometry(self, selection: list) -> None:
        """Distance for two selected atoms, angle at the middle one for
        three; nothing otherwise. Ordered by click, so for an angle the
        *second* atom clicked is the vertex, matching the measure tool."""
        trajectory = self._trajectory
        atoms = [int(i) for i in selection]
        if trajectory is None or len(atoms) not in (2, 3):
            self._geometry_block.hide()
            return

        x = trajectory.times()
        xlabel = "Time (ps)" if trajectory.has_time_axis else "Frame"
        names = [f"{trajectory.species[i]}{i + 1}" for i in atoms]
        if len(atoms) == 2:
            values = pair_distance_series(trajectory, atoms[0], atoms[1])
            label = f"{names[0]}–{names[1]}"
            title, ylabel = f"Distance {label} (Å)", "Å"
        else:
            values = angle_series(trajectory, atoms[0], atoms[1], atoms[2])
            label = f"{names[0]}–{names[1]}–{names[2]}"
            title, ylabel = f"Angle {label} (°)", "°"

        self._geometry_plot.set_data(x, values, label, ylabel, xlabel)
        if self._geometry_title is not None:
            self._geometry_title.setText(title)
        self._geometry_block.show()
        value = float(x[self._frame]) if x.size else 0.0
        self._geometry_plot.set_cursor(value)

    def _clear_selection(self) -> None:
        self._view.set_tracked_atom(None)
        self._view.clear_selection()
        self._geometry_block.hide()
        self._sidebar.set_selected_atom(None, "")
        self._apply_trail_options()
        self._view.update()

    def _on_track_changed(self, enabled: bool) -> None:
        selected = self._view.selected_index()
        self._view.set_tracked_atom(selected if enabled else None)
        self._show_frame(self._frame)

    # -- bottom bar -----------------------------------------------------------------

    def _info_fields(self) -> dict[str, str]:
        trajectory = self._trajectory
        if trajectory is None:
            return {}

        def temperature_text() -> str:
            target = trajectory.target_temperature
            if target is None:
                # With no thermostat setpoint, the run's own mean is the only
                # honest summary — and it is labelled as such.
                series = trajectory.series_for("temperature")
                if series is None:
                    return "—"
                return f"{float(np.mean(series.values)):.0f} K (mean)"
            begin, end = target
            if abs(end - begin) < 1e-9:
                return f"{begin:g} K"
            return f"{begin:g} → {end:g} K"

        duration = trajectory.duration_ps
        return {
            "system": trajectory.formula or "—",
            "frames": f"{trajectory.n_frames}",
            "timestep": (
                f"{trajectory.timestep:g} fs" if trajectory.timestep is not None else "—"
            ),
            "duration": f"{duration:.2f} ps" if duration is not None else "—",
            "ensemble": trajectory.ensemble or "—",
            "temperature": temperature_text(),
        }

    def refresh_info(self) -> None:
        """Re-report the bottom-bar fields, e.g. after the page comes forward."""
        self.infoRequested.emit(self._info_fields())

    def _emit_info(self, trajectory: MDTrajectory) -> None:
        _ = trajectory
        self.refresh_info()

    # -- toolbar actions ----------------------------------------------------------

    def reset(self) -> None:
        self._sidebar.reset()
        self._view.orient("default")
        self._playback.reset()
        self._energy_key = "total_energy"
        if "total_energy" in self._energy_buttons:
            self._energy_buttons["total_energy"].setChecked(True)
        self._clear_selection()
        self._apply_view_options()
        self._apply_trail_options()
        if self._trajectory is not None:
            self._prepare_viewer(self._trajectory)
            self._populate_plots(self._trajectory)
            self._show_frame(0)

    def copy_to_clipboard(self) -> bool:
        pixmap = self._view.grab()
        if pixmap.isNull():
            return False
        QGuiApplication.clipboard().setPixmap(pixmap)
        return True

    def export(self) -> None:
        trajectory = self._trajectory
        if trajectory is None:
            return
        calculation = self.calculation()
        directory = calculation.md_directory if calculation else Path.home()
        stem = calculation.name if calculation else "trajectory"

        filters = (
            "PNG image (*.png);;Current frame as POSCAR (*.vasp);;"
            "Trajectory (*.xyz);;MSD data (*.csv);;Temperature data (*.csv);;"
            "Energy data (*.csv)"
        )
        path_text, selected = QFileDialog.getSaveFileName(
            self, "Export Trajectory", str(directory / f"{stem}-md.png"), filters
        )
        if not path_text:
            return

        from .export import (
            save_energy_csv,
            save_msd_csv,
            save_temperature_csv,
            save_trajectory_xyz,
        )
        from ..structure.export import save_poscar

        path = Path(path_text)
        try:
            if selected.startswith("Current frame"):
                save_poscar(trajectory.structure(self._frame), path.with_suffix(".vasp"))
            elif selected.startswith("Trajectory"):
                save_trajectory_xyz(trajectory, path.with_suffix(".xyz"))
            elif selected.startswith("MSD"):
                save_msd_csv(trajectory, self._msd or {}, path.with_suffix(".csv"))
            elif selected.startswith("Temperature"):
                save_temperature_csv(trajectory, path.with_suffix(".csv"))
            elif selected.startswith("Energy"):
                save_energy_csv(trajectory, path.with_suffix(".csv"))
            elif not self._view.grab().save(str(path)):
                raise OSError("QPixmap.save returned False")
            log.info("Exported MD data to %s", path)
        except (OSError, ValueError) as exc:
            log.exception("MD export failed")
            self._notice.show_issue(
                Issue(
                    code="export_failed",
                    message="The file could not be written.",
                    detail=str(exc),
                    severity=Severity.ERROR,
                )
            )


__all__ = ["MDPage", "SectionHeader"]
