"""The Phonons page.

One piece of state — which (q-point, branch) is selected — drives the whole
page. The dispersion plot, the DOS cursor, the mode information card, the 3D
animation and the mode explorer all render *from* it and all report changes
*to* it, so no two of them can disagree about what is being shown. Nothing here
writes to the calculation: this page reads a finished phonon run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, Signal
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
from ....models.phonon import (
    THZ_TO_CM1,
    THZ_TO_MEV,
    PhononDataset,
    SelectedMode,
    convert_frequency,
)
from ....services import phonon as phonon_service
from ....services.loader import DataLoader
from ...style import icon_url
from ...theme import METRICS, PALETTE, RADIUS, SPACING, TYPE
from ...widgets.axis_bar import AxisBar
from ...widgets.element_swatch import ElementSwatch
from ...widgets.empty_state import EmptyState
from ...widgets.notice import Notice
from ...widgets.painted_slider import PaintedSlider
from ...widgets.popout import PopoutButton
from ...widgets.primitives import Card, Divider
from ..base import Page
from ..neb.info_panel import KeyValueCard
from ..structure.viewer import StructureView, ViewOptions
from .explorer import ModeExplorer
from .plots import DispersionPlot, PhononDosPlot
from .sidebar import (
    BRANCH_ACOUSTIC,
    BRANCH_ALL,
    BRANCH_OPTICAL,
    DOS_ELEMENT,
    PhononSidebar,
)
from .viewer import PhononView

log = get_logger("ui.phonons")

DATASET_PHONON = "phonon"
# Phases sampled over one oscillation. 24 is smooth at 30 fps and keeps the
# precomputed frame array small even for a drawn supercell.
N_PHASES = 24
BASE_INTERVAL_MS = 40


class PhononsPage(Page):
    """Phonon dispersion, DOS and vibrational mode animation."""

    title = "Phonons"
    issueRaised = Signal(object)
    infoRequested = Signal(dict)

    def __init__(self, loader: DataLoader, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loader = loader
        self._sidebar = PhononSidebar()
        self._dataset: PhononDataset | None = None
        self._selected = SelectedMode()
        self._playing = False
        self._pending = False

        self._timer = QTimer(self)
        self._timer.setInterval(BASE_INTERVAL_MS)
        self._timer.timeout.connect(self._on_tick)

        self._build()
        self._connect()

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
        column = QVBoxLayout(content)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(SPACING.md)

        upper = QHBoxLayout()
        upper.setSpacing(SPACING.md)
        upper.addWidget(self._build_dispersion_card(), 5)
        upper.addWidget(self._build_dos_card(), 2)
        upper.addWidget(self._build_right_column(), 0)
        column.addLayout(upper, 5)
        column.addWidget(self._build_explorer_card(), 2)
        self._stack.addWidget(content)

        empty_card = Card()
        self._empty = EmptyState(
            title="No phonon data found",
            message=(
                "Load a Phonopy or VASP phonon calculation to visualize phonon "
                "dispersion, DOS, and vibrational modes."
            ),
            glyph="∿",
        )
        empty_card.body().addWidget(self._empty)
        self._stack.addWidget(empty_card)
        self._stack.setCurrentIndex(1)

    def _build_dispersion_card(self) -> QWidget:
        card = Card(padding=SPACING.lg)
        body = card.body()
        body.setSpacing(SPACING.sm)

        header = QHBoxLayout()
        title = QLabel("Phonon Dispersion")
        title.setObjectName("CardTitle")
        header.addWidget(title)
        header.addStretch(1)
        for text, colour, dashed in (
            ("Real", PALETTE.accent, False),
            ("Imaginary", PALETTE.danger, True),
        ):
            sample = QLabel("––" if dashed else "——")
            sample.setStyleSheet(f"color: {colour}; font-size: {TYPE.small}px;")
            header.addWidget(sample)
            caption = QLabel(text)
            caption.setStyleSheet(
                f"color: {PALETTE.text_secondary}; font-size: {TYPE.small}px;"
            )
            header.addWidget(caption)
        body.addLayout(header)

        self._dispersion = DispersionPlot()
        header.addWidget(PopoutButton(self._dispersion, "Phonon Dispersion", flat=True))
        body.addWidget(self._dispersion, 1)
        return card

    def _build_dos_card(self) -> QWidget:
        card = Card(padding=SPACING.lg)
        card.setMinimumWidth(240)
        body = card.body()
        body.setSpacing(SPACING.sm)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Phonon DOS")
        title.setObjectName("CardTitle")
        header.addWidget(title)
        header.addStretch(1)
        body.addLayout(header)

        self._dos_plot = PhononDosPlot()
        header.addWidget(PopoutButton(self._dos_plot, "Phonon DOS", flat=True))
        body.addWidget(self._dos_plot, 1)
        return card

    def _build_right_column(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setFixedWidth(METRICS.dos_panel_width + 76)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        column = QVBoxLayout(content)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(SPACING.md)

        # Two columns rather than seven stacked rows: the same seven facts in
        # roughly half the height, and the height saved goes straight to the
        # eigenvector viewer below, which is the part of this column that
        # actually benefits from being larger.
        self._mode_card = KeyValueCard("Mode Information")
        self._mode_card.build(
            [
                ("qpoint", "Q-point"),
                ("branch", "Branch"),
                ("frequency", "Frequency"),
                ("wavelength", "Wavelength"),
                ("kind", "Mode Type"),
                ("velocity", "Group Velocity"),
                ("degeneracy", "Degeneracy"),
            ],
            columns=2,
        )
        column.addWidget(self._mode_card, 0)
        # Stretch on the animation card, not on a trailing spacer: any spare
        # height in this column should grow the viewer.
        column.addWidget(self._build_animation_card(), 1)

        scroll.setWidget(content)
        return scroll

    def _build_animation_card(self) -> QWidget:
        card = Card(padding=SPACING.lg)
        body = card.body()
        body.setSpacing(SPACING.sm)

        title = QLabel("Eigenvector Animation")
        title.setObjectName("CardTitle")
        body.addWidget(title)

        controls = QHBoxLayout()
        controls.setSpacing(SPACING.sm)
        # Icons rather than "▶" / "❚❚" glyph text: the pause glyph in
        # particular renders as two hairlines (or a tofu box) in several of
        # the fonts this app can fall back to, which is what made the pair
        # unreadable as transport controls.
        self._play = QPushButton()
        self._play.setObjectName("PrimaryButton")
        self._play.setFixedSize(34, 30)
        self._play.setIconSize(QSize(14, 14))
        self._play.setCursor(Qt.PointingHandCursor)
        self._play.clicked.connect(self._toggle_play)
        controls.addWidget(self._play)

        # Loop, not stop. One button that both starts and stops the animation
        # is the whole transport most users need here, and a separate "stop"
        # beside it read as a pause that surprisingly rewound to frame 0.
        # Whether the animation repeats is the genuinely useful second
        # control, and it mirrors the sidebar's Repeat checkbox rather than
        # introducing a second, independent source of truth.
        # Its own object name, not the shared "ToolButton" one: a blue
        # checked-state background only reads correctly against this
        # button's own icon, which switches to a white stroke to match (see
        # _apply_loop_icon) — giving every #ToolButton that treatment as a
        # side effect would have made the dark-stroke Select/Rotate/Pan tool
        # icons unreadable the moment they were checked.
        self._loop = QPushButton()
        self._loop.setObjectName("LoopButton")
        self._loop.setFixedSize(34, 30)
        self._loop.setIconSize(QSize(14, 14))
        self._loop.setCheckable(True)
        self._loop.setChecked(True)
        self._loop.setToolTip("Loop the animation continuously")
        self._loop.setCursor(Qt.PointingHandCursor)
        self._loop.toggled.connect(self._on_loop_toggled)
        controls.addWidget(self._loop)
        self._apply_play_icon()
        self._apply_loop_icon(self._loop.isChecked())

        self._phase = PaintedSlider(0.0, float(N_PHASES - 1), 0.0, 1.0)
        self._phase.valueChanged.connect(self._on_phase_changed)
        controls.addWidget(self._phase, 1)

        self._speed_label = QLabel("1×")
        self._speed_label.setStyleSheet(
            f"font-size: {TYPE.small}px; color: {PALETTE.text_secondary};"
        )
        controls.addWidget(self._speed_label)
        body.addLayout(controls)

        holder = QWidget()
        holder_layout = QVBoxLayout(holder)
        holder_layout.setContentsMargins(0, 0, 0, 0)
        self._view = PhononView()
        self._view.setMinimumHeight(200)
        holder_layout.addWidget(self._view)
        # Orientation presets in place of a projection toggle: which way the
        # cell faces is the question that actually comes up while reading a
        # structure, and perspective is the right answer for all of them.
        self._axis_bar = AxisBar(holder)
        self._axis_bar.orientationRequested.connect(self._view.orient)
        self._popout = PopoutButton(holder, "Phonon Mode Viewer", holder)
        self._legend = self._build_legend(holder)
        holder.installEventFilter(self)
        self._holder = holder
        body.addWidget(holder, 1)

        body.addWidget(Divider())
        options = QLabel("Animation Options")
        options.setObjectName("CardTitle")
        body.addWidget(options)

        scale_row = QHBoxLayout()
        scale_row.addWidget(self._caption("Displacement Scale"))
        self._scale = PaintedSlider(0.1, 3.0, 1.0, 0.1)
        self._scale.valueChanged.connect(self._on_scale_changed)
        scale_row.addWidget(self._scale, 1)
        self._scale_value = QLabel("1.0 Å")
        self._scale_value.setStyleSheet(
            f"font-size: {TYPE.small}px; color: {PALETTE.text_secondary};"
        )
        scale_row.addWidget(self._scale_value)
        body.addLayout(scale_row)

        style_row = QHBoxLayout()
        style_row.addWidget(self._caption("Vector Style"))
        self._vector_style = QComboBox()
        self._vector_style.addItems(["Arrows", "Lines", "Hidden"])
        self._vector_style.setFixedHeight(28)
        self._vector_style.currentTextChanged.connect(lambda _t: self._apply_vectors())
        style_row.addWidget(self._vector_style, 1)
        body.addLayout(style_row)

        self._show_vectors = QCheckBox("Show Vectors")
        self._show_vectors.setChecked(True)
        self._show_vectors.toggled.connect(lambda _v: self._apply_vectors())
        body.addWidget(self._show_vectors)
        return card

    def _build_legend(self, parent: QWidget) -> QWidget:
        legend = QWidget(parent)
        legend.setObjectName("ViewerLegend")
        legend.setAttribute(Qt.WA_StyledBackground, True)
        self._legend_layout = QVBoxLayout(legend)
        self._legend_layout.setContentsMargins(
            SPACING.sm, SPACING.xs, SPACING.sm, SPACING.xs
        )
        self._legend_layout.setSpacing(2)
        legend.setStyleSheet(
            f"#ViewerLegend {{ background: rgba(38,39,43,0.88);"
            f" border: 1px solid rgba(255,255,255,0.14);"
            f" border-radius: {RADIUS.md}px; }}"
        )
        legend.hide()
        return legend

    def _build_explorer_card(self) -> QWidget:
        card = Card(padding=SPACING.lg)
        self._explorer = ModeExplorer()
        card.body().addWidget(self._explorer)
        return card

    @staticmethod
    def _caption(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(
            f"color: {PALETTE.text_secondary}; font-size: {TYPE.small}px;"
        )
        return label

    # -- wiring ------------------------------------------------------------

    def _connect(self) -> None:
        side = self._sidebar
        side.branchFilterChanged.connect(lambda _t: self._redraw_dispersion())
        side.imaginaryToggled.connect(lambda _v: self._redraw_dispersion())
        side.coloringChanged.connect(lambda _t: self._redraw_dispersion())
        side.unitsChanged.connect(self._on_units_changed)
        side.dosTypeChanged.connect(lambda _t: self._redraw_dos())
        side.projectedToggled.connect(lambda _v: self._redraw_dos())
        side.smearingChanged.connect(lambda _v: self._redraw_dos())
        side.amplitudeChanged.connect(self._on_amplitude_changed)
        side.speedChanged.connect(self._on_speed_changed)
        side.repeatToggled.connect(self._on_repeat_toggled)
        side.displayChanged.connect(self._apply_view_options)

        self._dispersion.modePicked.connect(self.select_mode)
        self._explorer.modeSelected.connect(self.select_mode)

    # -- Page interface ----------------------------------------------------

    def sidebar(self) -> QWidget:
        return self._sidebar

    def set_calculation(self, calculation: Calculation | None) -> None:
        super().set_calculation(calculation)
        self._stop_animation()
        self._dataset = None
        self._notice.clear()

        if calculation is None or calculation.is_empty:
            self._show_empty(
                "No phonon data found",
                "Load a Phonopy or VASP phonon calculation to visualize phonon "
                "dispersion, DOS, and vibrational modes.",
            )
            return

        root = phonon_service.phonon_root(calculation.display_directory)
        if root is None:
            self._show_empty(
                "No phonon data found",
                "This calculation contains no band.yaml, mesh.yaml or "
                "FORCE_CONSTANTS. Run Phonopy to produce them.",
            )
            return

        result = phonon_service.load_phonons(root)
        for issue in result.issues:
            if issue.severity in (Severity.WARNING, Severity.ERROR):
                self._notice.show_issue(issue)
                break
        if not result.ok or result.value is None:
            message = result.errors[0].message if result.errors else "Unreadable."
            self._show_empty("Phonon data incomplete", message)
            return

        self._apply_dataset(result.value)

    def _show_empty(self, title: str, message: str) -> None:
        self._empty.set_text(title, message)
        self._stack.setCurrentIndex(1)
        self.infoRequested.emit({})

    def _apply_dataset(self, dataset: PhononDataset) -> None:
        self._dataset = dataset
        self._stack.setCurrentIndex(0)

        paths = ["High Symmetry Path"] if dataset.labelled_qpoints() else ["Raw Calculation Path"]
        self._sidebar.set_qpaths(paths)
        self._sidebar.set_acoustic_sum_state(self._acoustic_sum_satisfied(dataset))

        available = dataset.has_eigenvectors and dataset.structure is not None
        self._sidebar.set_animation_available(
            available,
            "" if available else "This calculation carries no eigenvectors, so "
            "modes cannot be animated.",
        )
        for widget in (self._play, self._loop, self._phase, self._scale,
                       self._vector_style, self._show_vectors):
            widget.setEnabled(available)

        # A one-element cell has nothing to project onto: every mode is that
        # element's, so the "projection" is the total curve drawn twice.
        # Saying so is better than leaving a control that appears to do
        # nothing when clicked.
        elements = (
            sorted(set(dataset.structure.species)) if dataset.structure else []
        )
        has_projection_data = dataset.has_eigenvectors or (
            dataset.dos is not None and dataset.dos.has_projection
        )
        if len(elements) < 2:
            reason = (
                f"This structure contains only {elements[0]}, so a per-element "
                "projection is identical to the total DOS."
                if elements
                else "No structure was loaded, so the DOS cannot be projected."
            )
            self._sidebar.set_projection_available(False, reason)
        elif not has_projection_data:
            self._sidebar.set_projection_available(
                False, "Projected DOS needs eigenvectors or a projected_dos.dat."
            )
        else:
            self._sidebar.set_projection_available(True)

        if dataset.structure is not None:
            self._view.set_structure(dataset.structure)
            self._update_legend(dataset)
        self._explorer.set_dataset(dataset)

        self._redraw_dispersion()
        self._redraw_dos()
        # Start on the first branch at the first q-point, which is Γ for a
        # standard path — the most useful default there is.
        self.select_mode(0, 0)
        self.refresh_info()

    @staticmethod
    def _acoustic_sum_satisfied(dataset: PhononDataset) -> bool | None:
        """Whether the three acoustic branches actually reach zero at Γ."""
        for index, band in enumerate(dataset.bands):
            if np.allclose(band.qpoint, 0.0, atol=1e-6):
                lowest = np.sort(band.frequencies)[:3]
                return bool(np.all(np.abs(lowest) < 0.15))
        return None

    def _on_units_changed(self, units: str) -> None:
        """Display-only: the data stays in THz, only the axes and labels move."""
        self._dispersion.set_units(units)
        self._dos_plot.set_units(units)
        self._explorer.set_units(units)
        self._redraw_dispersion()
        self._update_mode_card()

    def _units(self) -> str:
        return self._sidebar.units_box.currentText()

    # -- selection ---------------------------------------------------------

    def select_mode(self, qpoint_index: int, branch_index: int) -> None:
        """The single entry point for changing what is shown."""
        dataset = self._dataset
        if dataset is None or dataset.n_qpoints == 0:
            return
        qpoint_index = max(0, min(int(qpoint_index), dataset.n_qpoints - 1))
        branch_index = max(0, min(int(branch_index), dataset.n_branches - 1))
        self._selected = SelectedMode(qpoint_index, branch_index)

        self._dispersion.set_selection(qpoint_index, branch_index)
        frequency = float(dataset.bands[qpoint_index].frequencies[branch_index])
        self._dos_plot.set_frequency_cursor(frequency)
        self._explorer.set_selection(qpoint_index, branch_index)
        self._update_mode_card()
        self._rebuild_animation()

    def _update_mode_card(self) -> None:
        dataset = self._dataset
        if dataset is None:
            self._mode_card.clear()
            return
        q, b = self._selected.qpoint_index, self._selected.branch_index
        band = dataset.bands[q]
        frequency = float(band.frequencies[b])

        coords = ", ".join(f"{v:.3f}" for v in band.qpoint)
        qpoint_text = f"{band.label} ({coords})" if band.label else f"({coords})"

        values = {
            "qpoint": qpoint_text,
            "branch": f"{b + 1} / {dataset.n_branches}",
            "frequency": (
                f"{convert_frequency(frequency, self._units()):.2f} {self._units()}"
            ),
            "kind": dataset.mode_type(q, b),
            "degeneracy": str(dataset.degeneracy(q, b)),
        }
        tooltips = {
            "frequency": (
                f"{frequency:.3f} THz  ·  {frequency * THZ_TO_CM1:.1f} cm⁻¹  ·  "
                f"{frequency * THZ_TO_MEV:.2f} meV"
            ),
            "kind": (
                "Acoustic and optical are only well defined at Γ; away from it "
                "branches mix and no claim is made."
            ),
        }

        # Wavelength is only meaningful for a wave with a direction: at Γ the
        # wavevector is zero and the wavelength is not finite.
        structure = dataset.structure
        if structure is not None and not np.allclose(band.qpoint, 0.0, atol=1e-9):
            reciprocal = structure.reciprocal_lattice
            q_cartesian = band.qpoint @ reciprocal
            magnitude = float(np.linalg.norm(q_cartesian))
            values["wavelength"] = (
                f"{2 * np.pi / magnitude:.2f} Å" if magnitude > 1e-9 else "—"
            )
        else:
            values["wavelength"] = "—"
            tooltips["wavelength"] = "At Γ the wavevector is zero, so there is no finite wavelength."

        velocity = dataset.group_velocity(q, b)
        values["velocity"] = "—" if velocity is None else f"{velocity:.2f} km/s"
        tooltips["velocity"] = (
            "|dω/dq| along the plotted path — a directional derivative, not the "
            "full gradient."
        )
        self._mode_card.set_values(values, tooltips=tooltips)

    # -- plots -------------------------------------------------------------

    def _branch_filter(self) -> list[int] | None:
        dataset = self._dataset
        if dataset is None:
            return None
        choice = self._sidebar.branch_box.currentText()
        if choice == BRANCH_ALL:
            return None
        if choice == BRANCH_ACOUSTIC:
            return [0, 1, 2]
        if choice == BRANCH_OPTICAL:
            return list(range(3, dataset.n_branches))
        return None

    def _redraw_dispersion(self) -> None:
        dataset = self._dataset
        if dataset is None:
            return
        self._dispersion._branch_filter = self._branch_filter()
        # Without this the Coloring combo was a dead control: it triggered
        # this redraw but the plot never learned what was chosen.
        self._dispersion.set_coloring(self._sidebar.coloring_box.currentText())
        self._dispersion.set_dataset(
            dataset, show_imaginary=self._sidebar.show_imaginary.isChecked()
        )
        self._dispersion.set_selection(
            self._selected.qpoint_index, self._selected.branch_index
        )
        self._redraw_dos()

    def _redraw_dos(self) -> None:
        """Draw the DOS the sidebar is actually asking for.

        Source order matters scientifically. phonopy's own ``total_dos.dat`` /
        ``projected_dos.dat`` are computed over a uniform mesh (usually with
        the tetrahedron method) and are the best available answer, so they
        win when present — re-broadening them here would convolve a second
        Gaussian over an already-broadened curve and widen every peak past
        the width the slider claims. Only when no such file exists is a DOS
        built here, and then from ``mesh.yaml``'s uniform sampling rather
        than the band path, which is a set of high-symmetry lines and a
        biased sample of the zone.
        """
        dataset = self._dataset
        if dataset is None:
            return
        limits = self._dispersion.frequency_limits()
        smearing = float(self._sidebar.smearing.value())
        # The Type dropdown selects the mode; the checkbox only decides
        # whether the per-element curves are overlaid on it. Previously the
        # two were OR-ed together, so with the checkbox on by default the
        # dropdown could never change anything.
        by_element = self._sidebar.dos_box.currentText() == DOS_ELEMENT
        overlay_elements = by_element and self._sidebar.show_projected.isChecked()

        from_file = dataset.dos
        dos = None

        if by_element:
            if from_file is not None and from_file.has_projection:
                dos = from_file
            elif dataset.has_eigenvectors:
                dos = phonon_service.project_dos_by_element(dataset, smearing)
        if dos is None:
            if from_file is not None:
                dos = from_file
            else:
                frequencies = (
                    dataset.mesh_frequencies
                    if dataset.mesh_frequencies is not None
                    else dataset.frequency_matrix
                )
                weights = (
                    dataset.mesh_weights
                    if dataset.mesh_frequencies is not None
                    else None
                )
                if frequencies.size:
                    dos = phonon_service.gaussian_dos(
                        frequencies, smearing, weights=weights
                    )

        if dos is None:
            self._dos_plot.show_message("No DOS available")
            return

        # Smearing only means something when this page is the one doing the
        # broadening; say so rather than leaving a live-looking slider that
        # cannot move the curve it sits under.
        recomputed = dos is not from_file
        self._sidebar.set_smearing_active(
            recomputed,
            "" if recomputed else (
                "This DOS comes from phonopy's own total_dos.dat / "
                "projected_dos.dat, already broadened when it was computed. "
                "Re-broadening it here would widen every peak by more than "
                "the width shown."
            ),
        )
        self._dos_plot.set_dos(dos, overlay_elements, limits)
        if self._dataset is not None:
            band = self._dataset.bands[self._selected.qpoint_index]
            self._dos_plot.set_frequency_cursor(
                float(band.frequencies[self._selected.branch_index])
            )

    def _update_legend(self, dataset: PhononDataset) -> None:
        while self._legend_layout.count():
            item = self._legend_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        if dataset.structure is None:
            return
        seen: list[str] = []
        for symbol in dataset.structure.species:
            if symbol not in seen:
                seen.append(symbol)
        for element in seen:
            entry = QWidget()
            row = QHBoxLayout(entry)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(SPACING.xs)
            swatch = ElementSwatch(element, size=9, dark=True)
            swatch.changed.connect(self._view.update)
            row.addWidget(swatch)
            label = QLabel(element)
            label.setStyleSheet(f"font-size: {TYPE.tiny}px; color: #f0f0f2;")
            row.addWidget(label)
            self._legend_layout.addWidget(entry)
        self._legend.show()
        QTimer.singleShot(0, self._position_overlays)

    # -- animation ---------------------------------------------------------

    def _rebuild_animation(self) -> None:
        dataset = self._dataset
        if dataset is None or not dataset.has_eigenvectors or dataset.structure is None:
            self._view.set_frames(None)
            self._view.set_displacements(None)
            return

        q, b = self._selected.qpoint_index, self._selected.branch_index
        amplitude = float(self._sidebar.amplitude.value())
        supercell = phonon_service.animation_supercell(dataset, q)

        phases = np.linspace(0.0, 2.0 * np.pi, N_PHASES, endpoint=False)
        frames = phonon_service.mode_displacements(
            dataset, q, b, phases, amplitude=amplitude, supercell=supercell
        )
        if frames is None:
            self._view.set_frames(None)
            return

        # The drawn cell changes with the q-point, so the viewer's base
        # structure is rebuilt only when the supercell does — not per frame.
        from ....services.supercell import build_supercell

        drawn = (
            dataset.structure
            if supercell == (1, 1, 1)
            else build_supercell(dataset.structure, *supercell)
        )
        self._view.set_structure(drawn)
        self._view.set_frames(frames)
        self._view.set_displacements(frames[0] - drawn.cart_coords)
        self._apply_vectors()
        self._apply_view_options()

    def _apply_vectors(self) -> None:
        self._view.set_vector_options(
            self._show_vectors.isChecked(),
            self._vector_style.currentText(),
            float(self._scale.value()),
        )

    def _apply_view_options(self) -> None:
        side = self._sidebar
        self._view.set_options(
            ViewOptions(
                style="Ball-and-Stick" if side.show_atoms.isChecked() else "Wireframe",
                coloring="By Element",
                show_cell=side.show_cell.isChecked(),
                show_axes=side.show_axes.isChecked(),
                perspective=True,
            )
        )

    def _apply_play_icon(self) -> None:
        playing = self._playing
        self._play.setIcon(QIcon(icon_url("pause.svg" if playing else "play.svg")))
        self._play.setToolTip("Pause the animation" if playing else "Play the animation")

    def _on_repeat_toggled(self, checked: bool) -> None:
        """The other half of the loop button / Repeat checkbox pairing."""
        if self._loop.isChecked() != checked:
            self._loop.setChecked(checked)

    def _on_loop_toggled(self, checked: bool) -> None:
        """Keep the toolbar's loop button and the sidebar's Repeat checkbox
        as one setting seen from two places, rather than two settings that
        can disagree."""
        self._apply_loop_icon(checked)
        if self._sidebar.repeat.isChecked() != checked:
            self._sidebar.repeat.setChecked(checked)

    def _apply_loop_icon(self, checked: bool) -> None:
        """White stroke on the accent-blue checked background, dark stroke
        on the plain button otherwise — a single fixed-color icon would lose
        all contrast in one of the two states."""
        self._loop.setIcon(QIcon(icon_url("loop-on.svg" if checked else "loop.svg")))

    def _toggle_play(self) -> None:
        if self._playing:
            self._pause()
        else:
            if self._view.n_phases == 0:
                return
            self._playing = True
            self._apply_play_icon()
            self._timer.start()

    def _pause(self) -> None:
        """Stop where it is. Deliberately does not rewind: pausing mid-cycle
        to look at a displacement pattern and having the view snap back to
        the undisplaced frame is the opposite of what pausing is for."""
        self._playing = False
        self._apply_play_icon()
        self._timer.stop()

    def _stop_animation(self) -> None:
        self._pause()
        self._view.set_phase_index(0)
        self._phase.setValue(0.0, notify=False)

    def _on_tick(self) -> None:
        if self._view.n_phases == 0:
            return
        previous = self._view.phase_index
        self._view.advance()
        self._phase.setValue(float(self._view.phase_index), notify=False)
        # A completed cycle stops here unless Repeat is on.
        if not self._sidebar.repeat.isChecked() and self._view.phase_index < previous:
            self._stop_animation()

    def _on_phase_changed(self, value: float) -> None:
        self._view.set_phase_index(int(round(value)))

    def _on_speed_changed(self, speed: float) -> None:
        self._speed_label.setText(f"{speed:g}×")
        self._timer.setInterval(max(8, int(BASE_INTERVAL_MS / max(speed, 0.01))))

    def _on_amplitude_changed(self, _value: float) -> None:
        self._rebuild_animation()

    def _on_scale_changed(self, value: float) -> None:
        self._scale_value.setText(f"{value:.1f} Å")
        self._apply_vectors()

    # -- layout ------------------------------------------------------------

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt naming
        if watched is self._holder and event.type() == event.Type.Resize:
            self._position_overlays()
        return super().eventFilter(watched, event)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().showEvent(event)
        QTimer.singleShot(0, self._position_overlays)

    def _position_overlays(self) -> None:
        margin = SPACING.sm
        holder = self._holder

        self._axis_bar.adjustSize()
        self._axis_bar.move(
            max(0, holder.width() - self._axis_bar.width() - margin), margin
        )
        self._axis_bar.raise_()

        self._popout.move(
            max(0, self._axis_bar.x() - self._popout.width() - SPACING.xs), margin
        )
        self._popout.raise_()

        if self._legend_layout.count() == 0:
            return
        self._legend.adjustSize()
        self._legend.move(
            max(0, holder.width() - self._legend.width() - margin),
            max(0, holder.height() - self._legend.height() - margin),
        )
        self._legend.raise_()

    # -- footer ------------------------------------------------------------

    def refresh_info(self) -> None:
        self.infoRequested.emit(self._info_fields())

    def _info_fields(self) -> dict:
        dataset = self._dataset
        if dataset is None:
            return {}
        structure = dataset.structure
        return {
            "system": structure.formula if structure is not None else "—",
            "calculation": dataset.calculation or "—",
            "supercell": (
                "×".join(str(v) for v in dataset.supercell)
                if dataset.supercell
                else "—"
            ),
            "atoms": str(dataset.n_atoms),
            "qgrid": "×".join(str(v) for v in dataset.mesh) if dataset.mesh else "—",
            "branches": str(dataset.n_branches),
            "imaginary": str(dataset.imaginary_count()),
            "maxfreq": f"{dataset.max_frequency:.2f} THz",
        }

    # -- toolbar -----------------------------------------------------------

    def reset(self) -> None:
        self._sidebar.reset()
        self._stop_animation()
        self._scale.setValue(1.0)
        self._vector_style.setCurrentIndex(0)
        self._show_vectors.setChecked(True)
        self._view.reset_view()
        self._redraw_dispersion()
        self.select_mode(0, 0)

    def copy_to_clipboard(self) -> bool:
        if self._dataset is None:
            return False
        QGuiApplication.clipboard().setPixmap(self._dispersion.grab())
        return True

    def export(self) -> None:
        dataset = self._dataset
        if dataset is None:
            return
        calculation = self.calculation()
        stem = calculation.name if calculation is not None else "phonons"
        directory = calculation.display_directory if calculation else Path.home()
        filters = (
            "Dispersion image (*.png);;Frequencies (*.csv);;DOS data (*.csv);;"
            "Selected mode (*.png)"
        )
        path_text, selected = QFileDialog.getSaveFileName(
            self, "Export Phonons", str(directory / f"{stem}-phonons.png"), filters
        )
        if not path_text:
            return
        target = Path(path_text)
        try:
            if selected.startswith("Frequencies"):
                self._write_frequencies(target)
            elif selected.startswith("DOS"):
                self._write_dos(target)
            elif selected.startswith("Selected"):
                self._view.grab().save(str(target))
            else:
                self._dispersion.figure.savefig(
                    target, dpi=200, facecolor=PALETTE.surface
                )
        except OSError as exc:
            log.exception("Phonon export failed")
            self._notice.show_issue(
                Issue(
                    code="phonon_export_failed",
                    message=f"{target.name} could not be written.",
                    detail=str(exc),
                    severity=Severity.ERROR,
                )
            )
            return
        log.info("Exported phonons to %s", target)

    def _write_frequencies(self, path: Path) -> None:
        dataset = self._dataset
        assert dataset is not None
        lines = ["distance,qx,qy,qz,label," + ",".join(
            f"branch_{i + 1}_THz" for i in range(dataset.n_branches)
        )]
        for band in dataset.bands:
            lines.append(
                f"{band.distance:.8f},"
                + ",".join(f"{v:.8f}" for v in band.qpoint)
                + f",{band.label},"
                + ",".join(f"{v:.8f}" for v in band.frequencies)
            )
        path.write_text("\n".join(lines) + "\n")

    def _write_dos(self, path: Path) -> None:
        dataset = self._dataset
        assert dataset is not None
        smearing = float(self._sidebar.smearing.value())
        dos = phonon_service.project_dos_by_element(dataset, smearing)
        if dos is None:
            dos = phonon_service.gaussian_dos(dataset.frequency_matrix, smearing)
        columns = ["frequency_THz", "total"] + list(dos.projected)
        lines = [",".join(columns)]
        for index, frequency in enumerate(dos.frequencies):
            row = [f"{frequency:.8f}", f"{dos.total[index]:.8f}"]
            row.extend(f"{values[index]:.8f}" for values in dos.projected.values())
            lines.append(",".join(row))
        path.write_text("\n".join(lines) + "\n")
