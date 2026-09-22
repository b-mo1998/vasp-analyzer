"""The Bands page: band structure with an optional aligned DOS panel."""

from __future__ import annotations

import io
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication, QImage
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ....core.errors import Issue, ParseResult, Severity
from ....core.logging_setup import get_logger
from ....models.bands import BandData
from ....models.calculation import Calculation
from ....models.capabilities import Capability
from ....models.dos import DosData
from ....parsers.procar import ProcarData
from ....parsers.wannier import WannierBands
from ....services.atom_selector import parse_atom_selector
from ....services.loader import (
    DATASET_BANDS,
    DATASET_DOS,
    DATASET_PROCAR,
    DATASET_STRUCTURE,
    DATASET_WANNIER_BANDS,
    DataLoader,
)
from ...theme import METRICS, PALETTE, SPACING, TYPE
from ...widgets.colorbar import DivergingColorbar
from ...widgets.empty_state import EmptyState
from ...widgets.legend import PlotLegend
from ...widgets.notice import Notice
from ...widgets.popout import PopoutButton
from ...widgets.primitives import Card
from ...widgets.slider_row import SliderRow
from ...widgets.toggle import LabelledToggle
from ..base import Page
from .band_plot import BandPlot
from .dos_plot import DosPlot, nice_upper_bound
from .export import (
    ExportRequest,
    build_dos_figure,
    build_figure,
    save_band_csv,
    save_dos_csv,
    save_dos_figure,
    save_figure,
    save_paper_figure,
)
from .projection import (
    ATOM_PROJECTION_COLOR,
    AXIS_LABELS,
    build_atom_overlay,
    build_element_overlay,
    build_orbital_overlay,
    build_spin_overlay,
    legend_elements,
    legend_orbitals,
)
from .sidebar import BandsSidebar

log = get_logger("ui.bands")

# Both plot cards reserve exactly this much height above their canvas. The band
# card centres its legend and DOS switch in it; the DOS card stacks its caption
# and slider into it. Equal header height + equal pixel margins is what puts the
# two Fermi lines on the same screen row.
PLOT_HEADER_HEIGHT = 58


class BandsPage(Page):
    """Band structure, DOS, and the controls that drive them."""

    title = "Bands"
    issueRaised = Signal(object)
    gapComputed = Signal(str, str)  # display text, tooltip

    def __init__(self, loader: DataLoader, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loader = loader
        self._sidebar = BandsSidebar()
        self._bands: BandData | None = None
        self._dos: DosData | None = None
        self._procar: ProcarData | None = None
        self._species: list[str] | None = None
        self._orbital_selection: dict[str, list[str]] | None = None
        self._element_projection_requested = False
        self._wannier_bands: WannierBands | None = None
        self._wannier_projection_requested = False
        self._spin_projection_requested = False
        self._atom_projection_requested = False
        self._spin_axis = 2
        self._dos_visible = True
        self._auto_xmax = True

        self._build()
        self._connect()
        loader.data_ready.connect(self._on_data_ready)

    # -- construction --------------------------------------------------------

    def _build(self) -> None:
        layout = self.body()
        layout.setSpacing(SPACING.md)

        self._notice = Notice()
        layout.addWidget(self._notice)

        self._stack = QStackedWidget()
        layout.addWidget(self._stack, 1)

        # --- plots -----------------------------------------------------------
        plots = QWidget()
        row = QHBoxLayout(plots)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING.md)
        row.addWidget(self._build_band_card(), 1)
        row.addWidget(self._build_dos_card(), 0)
        self._stack.addWidget(plots)

        # --- unavailable ------------------------------------------------------
        self._empty = EmptyState(
            title="Band structure unavailable",
            message="EIGENVAL was not found in this calculation.",
            glyph="◍",
        )
        empty_card = Card()
        empty_card.body().addWidget(self._empty)
        self._stack.addWidget(empty_card)

    def _build_band_card(self) -> QWidget:
        card = Card(padding=SPACING.lg)
        body = card.body()
        body.setSpacing(SPACING.sm)

        header = QWidget()
        header.setFixedHeight(PLOT_HEADER_HEIGHT)
        header_row = QHBoxLayout(header)
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setAlignment(Qt.AlignVCenter)

        self._legend = PlotLegend()
        self._legend.add_entry("vasp", "VASP", PALETTE.accent)
        self._legend.add_entry("wannier", "Wannier90", PALETTE.wannier, dashed=True)
        self._legend.set_entry_visible("wannier", False)
        header_row.addWidget(self._legend, 0, Qt.AlignVCenter)
        header_row.addStretch(1)

        self._dos_toggle = LabelledToggle("DOS", checked=True)
        header_row.addWidget(self._dos_toggle, 0, Qt.AlignVCenter)
        body.addWidget(header)

        plot_row = QHBoxLayout()
        plot_row.setContentsMargins(0, 0, 0, 0)
        plot_row.setSpacing(SPACING.sm)

        self._band_plot = BandPlot()
        header_row.addWidget(
            PopoutButton(self._band_plot, "Band Structure", flat=True), 0, Qt.AlignVCenter
        )
        plot_row.addWidget(self._band_plot, 1)

        self._spin_colorbar = DivergingColorbar()
        self._spin_colorbar.setVisible(False)
        plot_row.addWidget(self._spin_colorbar, 0)

        body.addLayout(plot_row, 1)
        self._band_card = card
        return card

    def _build_dos_card(self) -> QWidget:
        card = Card(padding=SPACING.lg)
        card.setFixedWidth(METRICS.dos_panel_width)
        body = card.body()
        body.setSpacing(SPACING.sm)

        header = QWidget()
        header.setFixedHeight(PLOT_HEADER_HEIGHT)
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(SPACING.xs)

        caption_row = QHBoxLayout()
        caption_row.setContentsMargins(0, 0, 0, 0)
        caption = QLabel("X Max (states/eV)")
        caption.setAlignment(Qt.AlignCenter)
        caption.setStyleSheet(
            f"font-size: {TYPE.small}px; color: {PALETTE.text_secondary};"
        )
        caption_row.addWidget(caption, 1)
        header_layout.addLayout(caption_row)

        self._xmax = SliderRow("", 0.0, 16.0, 8.0, 0.1, "", decimals=2, inline=True)
        header_layout.addWidget(self._xmax)
        header_layout.addStretch(1)
        body.addWidget(header)

        self._dos_plot = DosPlot()
        caption_row.addWidget(PopoutButton(self._dos_plot, "Density of States", flat=True))
        body.addWidget(self._dos_plot, 1)
        self._dos_card = card
        return card

    def _connect(self) -> None:
        side = self._sidebar
        side.yMinChanged.connect(self._on_limits_changed)
        side.yMaxChanged.connect(self._on_limits_changed)
        side.fermiChanged.connect(self._on_reference_changed)
        side.dosToggled.connect(self._set_dos_visible)
        side.projectionToggled.connect(self._on_projection_toggled)
        side.orbitalsChanged.connect(self._on_orbitals_changed)
        side.spinAxisChanged.connect(self._on_spin_axis_changed)
        side.atomProjectionChanged.connect(self._try_build_atom_projection)
        side.bandGapMarkersToggled.connect(self._band_plot.set_band_gap_visible)
        side.hoverInfoToggled.connect(self._band_plot.set_hover_enabled)

        self._dos_toggle.toggled.connect(self._set_dos_visible)
        self._xmax.valueChanged.connect(self._on_xmax_changed)
        self._notice.detailsRequested.connect(self.issueRaised)

    # -- Page interface --------------------------------------------------------

    def sidebar(self) -> QWidget:
        return self._sidebar

    def set_calculation(self, calculation: Calculation | None) -> None:
        super().set_calculation(calculation)
        self._bands = None
        self._dos = None
        self._procar = None
        self._species = None
        self._orbital_selection = None
        self._element_projection_requested = False
        self._wannier_bands = None
        self._wannier_projection_requested = False
        self._spin_projection_requested = False
        self._atom_projection_requested = False
        self._auto_xmax = True
        self._notice.clear()
        self._band_plot.set_bands(None)
        self._band_plot.set_projection(None)
        self._band_plot.set_wannier(None, None)
        self._band_plot.set_band_gap(None)
        self._band_plot.set_orbital_source(None, None)
        self._dos_plot.set_dos(None)
        self._legend.remove_entries_with_prefix("element:")
        self._legend.remove_entries_with_prefix("orbital:")
        self._legend.set_entry_visible("wannier", False)
        self._spin_colorbar.setVisible(False)
        self._sidebar.projections["element"].setChecked(False)
        self._sidebar.projections["wannier"].setChecked(False)
        self._sidebar.projections["spin"].setChecked(False)
        self._sidebar.clear_orbitals()
        self.gapComputed.emit("—", "")

        if calculation is None:
            self._sidebar.set_controls_enabled(False)
            self._sidebar.set_dos_enabled(False, "No calculation is open.")
            self._dos_toggle.setEnabled(False)
            self._show_unavailable(
                "No calculation open", "Open a calculation directory to begin."
            )
            return

        self._apply_capabilities(calculation)

        # DOS (DOSCAR) needs nothing from EIGENVAL and stands alone fine, so
        # it's requested independent of whether bands are available — it
        # used to be nested inside the BANDS branch below, which meant a
        # DOSCAR-without-EIGENVAL calculation never loaded DOS data at all,
        # silently defeating DOS export/copy for that whole scenario.
        if calculation.can(Capability.DOS):
            self._loader.request(DATASET_DOS)

        if calculation.can(Capability.BANDS):
            self._stack.setCurrentIndex(0)
            self._loader.request(DATASET_BANDS)
        elif calculation.can(Capability.DOS):
            # No band structure, but DOS works on its own — show the plot
            # area (with an empty band plot) rather than blocking the whole
            # page just because EIGENVAL specifically is missing.
            self._stack.setCurrentIndex(0)
            self._notice.show_issue(
                Issue(
                    code="bands_unavailable_dos_ok",
                    message=(
                        f"Band structure unavailable: {calculation.why_not(Capability.BANDS)}"
                    ),
                    severity=Severity.INFO,
                )
            )
        else:
            self._show_unavailable(
                "Band structure unavailable",
                calculation.why_not(Capability.BANDS),
            )

    def _apply_capabilities(self, calculation: Calculation) -> None:
        side = self._sidebar
        # The Y-axis/Fermi sliders drive the energy axis shared by both the
        # band plot and the DOS panel, so they stay usable whenever either
        # one has data — not just when bands specifically are available.
        side.set_controls_enabled(
            calculation.can(Capability.BANDS) or calculation.can(Capability.DOS)
        )

        fermi = calculation.metadata.e_fermi
        side.set_fermi_range(fermi.value if fermi is not None else None)
        reference = fermi.value if fermi is not None else 0.0
        self._band_plot.set_reference(reference)
        self._dos_plot.set_reference(reference)

        dos_ok = calculation.can(Capability.DOS)
        side.set_dos_enabled(dos_ok, calculation.why_not(Capability.DOS))
        self._dos_toggle.setEnabled(dos_ok)
        self._dos_toggle.setToolTip(
            "" if dos_ok else calculation.why_not(Capability.DOS)
        )
        if not dos_ok:
            self._set_dos_visible(False)
        else:
            self._set_dos_visible(side.show_dos.isChecked())

        for key, capability in (
            ("spin", Capability.SPIN_PROJECTION),
            ("element", Capability.ELEMENT_PROJECTION),
            # Same prerequisite as Element: a PROCAR matching the loaded
            # bands — atom projection reads the identical per-ion weights,
            # just restricted to specific rows instead of a species.
            ("atom", Capability.ELEMENT_PROJECTION),
            ("wannier", Capability.WANNIER_BANDS),
        ):
            side.set_projection_enabled(
                key, calculation.can(capability), calculation.why_not(capability)
            )

        for issue in calculation.issues:
            if issue.severity in (Severity.WARNING, Severity.ERROR):
                self._notice.show_issue(issue)
                break

    def reset(self) -> None:
        calculation = self.calculation()
        fermi = None
        if calculation is not None and calculation.metadata.e_fermi is not None:
            fermi = calculation.metadata.e_fermi.value
        self._sidebar.reset(fermi)
        self._auto_xmax = True
        reference = fermi if fermi is not None else 0.0
        self._band_plot.set_reference(reference)
        self._dos_plot.set_reference(reference)
        self._on_limits_changed()
        if calculation is not None and calculation.can(Capability.DOS):
            self._sidebar.show_dos.setChecked(True)

    # -- data ------------------------------------------------------------------

    def _on_data_ready(self, dataset: str, result: ParseResult) -> None:
        if dataset == DATASET_BANDS:
            self._apply_bands(result)
        elif dataset == DATASET_DOS:
            self._apply_dos(result)
        elif dataset == DATASET_PROCAR:
            self._apply_procar(result)
        elif dataset == DATASET_STRUCTURE:
            self._apply_species(result)
        elif dataset == DATASET_WANNIER_BANDS:
            self._apply_wannier_bands(result)

    def _apply_bands(self, result: ParseResult) -> None:
        for issue in result.issues:
            if issue.severity in (Severity.WARNING, Severity.ERROR):
                self._notice.show_issue(issue)
                break
        if not result.ok:
            self._show_unavailable(
                "Band structure unavailable",
                result.issues[0].message if result.issues else "EIGENVAL could not be read.",
            )
            return
        self._bands = result.value
        self._stack.setCurrentIndex(0)
        self._band_plot.set_bands(self._bands)
        self._on_limits_changed()
        self._try_build_element_projection()
        self._try_build_spin_projection()
        self._update_gap_display()

    def _update_gap_display(self) -> None:
        calculation = self.calculation()
        # The gap is a property of the calculation, resolved once against its
        # own determined Fermi level — not the sidebar's interactive Fermi
        # slider, which is a display reference the user can drag anywhere and
        # shouldn't make this readout jump around.
        if self._bands is None or calculation is None or calculation.metadata.e_fermi is None:
            self.gapComputed.emit(
                "—",
                "A band gap needs both EIGENVAL and a determined Fermi level "
                "(from DOSCAR, OUTCAR or vasprun.xml).",
            )
            self._band_plot.set_band_gap(None)
            return

        e_fermi = calculation.metadata.e_fermi
        gap = self._bands.band_gap(e_fermi.value)
        if gap is None:
            self.gapComputed.emit(
                "Metallic",
                f"No band gap at E_F = {e_fermi.value:.4f} eV ({e_fermi.source}): "
                "the valence and conduction states overlap there.",
            )
            self._band_plot.set_band_gap(None)
            return

        kind = "direct" if gap.is_direct else "indirect"
        self.gapComputed.emit(
            f"{gap.gap:.2f} eV ({kind})",
            f"VBM {gap.vbm:.4f} eV, CBM {gap.cbm:.4f} eV, relative to "
            f"E_F = {e_fermi.value:.4f} eV ({e_fermi.source}).",
        )
        # Plotted at the *absolute* VBM/CBM energies (matching the gap's own
        # eV values above), and BandPlot subtracts its own reference the same
        # way it does for every band line — so the markers stay correctly
        # placed as the user drags the Fermi/reference control, not fixed to
        # wherever E_F happened to be when the gap was computed.
        self._band_plot.set_band_gap(gap)

    def _apply_dos(self, result: ParseResult) -> None:
        if not result.ok:
            for issue in result.issues:
                self._notice.show_issue(issue)
                break
            self._sidebar.set_dos_enabled(
                False, "DOSCAR was found but could not be parsed."
            )
            self._dos_toggle.setEnabled(False)
            self._set_dos_visible(False)
            return
        self._dos = result.value
        self._dos_plot.set_dos(self._dos)
        self._rescale_dos()

    # -- control handlers --------------------------------------------------------

    def _limits(self) -> tuple[float, float]:
        low = self._sidebar.y_min.value()
        high = self._sidebar.y_max.value()
        if high <= low:
            high = low + 0.1
        return low, high

    def _on_limits_changed(self, _value: float | None = None) -> None:
        low, high = self._limits()
        self._band_plot.set_ylimits(low, high)
        self._dos_plot.set_ylimits(low, high)
        self._rescale_dos()

    def _on_reference_changed(self, value: float) -> None:
        self._band_plot.set_reference(value)
        self._dos_plot.set_reference(value)
        self._rescale_dos()

    def _on_xmax_changed(self, value: float) -> None:
        self._auto_xmax = False
        self._dos_plot.set_xmax(value)

    def _rescale_dos(self) -> None:
        """Fit the DOS x-axis to the visible energy window, until the user
        chooses a value themselves."""
        if self._dos is None or not self._auto_xmax:
            return
        peak = self._dos_plot.suggest_xmax()
        if peak is None:
            return
        upper = nice_upper_bound(peak * 1.1)
        self._xmax.setRange(0.0, max(upper * 2, 1.0))
        self._xmax.setValue(upper, notify=False)
        self._dos_plot.set_xmax(upper)

    def _set_dos_visible(self, visible: bool) -> None:
        self._dos_visible = bool(visible)
        self._dos_card.setVisible(self._dos_visible)
        # Keep the two DOS controls in step without echoing signals back.
        for widget, setter in (
            (self._sidebar.show_dos, self._sidebar.show_dos.setChecked),
            (self._dos_toggle, self._dos_toggle.setChecked),
        ):
            widget.blockSignals(True)
            setter(self._dos_visible)
            widget.blockSignals(False)

    def _on_projection_toggled(self, key: str, checked: bool) -> None:
        if key == "element":
            self._set_element_projection(checked)
            return
        if key == "wannier":
            self._set_wannier_projection(checked)
            return
        if key == "spin":
            self._set_spin_projection(checked)
            return
        if key == "atom":
            self._set_atom_projection(checked)
            return

    def _set_element_projection(self, checked: bool) -> None:
        if not checked:
            self._element_projection_requested = False
            self._band_plot.set_projection(None)
            self._legend.remove_entries_with_prefix("element:")
            self._legend.remove_entries_with_prefix("orbital:")
            return

        calculation = self.calculation()
        if calculation is None or not calculation.can(Capability.ELEMENT_PROJECTION):
            return

        self._element_projection_requested = True
        # request() returns None exactly when it had to start a background
        # load rather than deliver an already-cached result — that's the
        # signal to show the indicator; a cached PROCAR (already used for
        # spin projection, say) delivers through data_ready synchronously
        # below and never leaves anything "loading" to show.
        if self._loader.request(DATASET_PROCAR) is None:
            self._sidebar.set_projection_loading(True)
        self._loader.request(DATASET_STRUCTURE)
        self._try_build_element_projection()

    def _set_spin_projection(self, checked: bool) -> None:
        if not checked:
            self._spin_projection_requested = False
            self._band_plot.set_projection(None)
            self._spin_colorbar.setVisible(False)
            return

        calculation = self.calculation()
        if calculation is None or not calculation.can(Capability.SPIN_PROJECTION):
            return

        self._spin_projection_requested = True
        if self._loader.request(DATASET_PROCAR) is None:
            self._sidebar.set_projection_loading(True)
        self._try_build_spin_projection()

    def _on_spin_axis_changed(self, axis: int) -> None:
        self._spin_axis = axis
        self._try_build_spin_projection()

    def _set_atom_projection(self, checked: bool) -> None:
        if not checked:
            self._atom_projection_requested = False
            self._band_plot.set_projection(None)
            self._legend.remove_entries_with_prefix("atom:")
            self._sidebar.set_atom_status("")
            return

        calculation = self.calculation()
        if calculation is None or not calculation.can(Capability.ELEMENT_PROJECTION):
            return

        self._atom_projection_requested = True
        if self._loader.request(DATASET_PROCAR) is None:
            self._sidebar.set_projection_loading(True)
        self._loader.request(DATASET_STRUCTURE)
        self._try_build_atom_projection()

    def _try_build_spin_projection(self) -> None:
        if not self._spin_projection_requested:
            return
        if self._bands is None or self._procar is None:
            return

        # Fixed to the physical range of a single electron's spin expectation
        # value (+-1/2), not auto-scaled to whatever this calculation's data
        # happens to span — a fixed scale is what makes the colour comparable
        # across different calculations and axis choices.
        result = build_spin_overlay(self._bands, self._procar, axis=self._spin_axis, vmax=0.5)
        if result is None:
            self._notice.show_issue(
                Issue(
                    code="spin_projection_unavailable",
                    message=(
                        "PROCAR doesn't carry usable spin data for this "
                        "calculation, or its dimensions don't match EIGENVAL."
                    ),
                    severity=Severity.WARNING,
                )
            )
            self._sidebar.projections["spin"].setChecked(False)
            return

        overlay, _vmax = result
        axis_label = AXIS_LABELS[self._spin_axis] if self._procar.is_noncollinear else "up-down"
        self._band_plot.set_projection(overlay)
        self._legend.remove_entries_with_prefix("element:")
        self._legend.remove_entries_with_prefix("orbital:")
        self._spin_colorbar.set_label(f"Spin ({axis_label})")
        self._spin_colorbar.setVisible(True)

    def _set_wannier_projection(self, checked: bool) -> None:
        if not checked:
            self._wannier_projection_requested = False
            self._band_plot.set_wannier(None, None)
            self._legend.set_entry_visible("wannier", False)
            return

        calculation = self.calculation()
        if calculation is None or not calculation.can(Capability.WANNIER_BANDS):
            return

        self._wannier_projection_requested = True
        self._loader.request(DATASET_WANNIER_BANDS)
        self._apply_wannier_overlay()

    def _apply_wannier_bands(self, result: ParseResult) -> None:
        if not result.ok:
            for issue in result.issues:
                self._notice.show_issue(issue)
                break
            self._sidebar.projections["wannier"].setChecked(False)
            return
        self._wannier_bands = result.value
        for issue in result.issues:
            if issue.severity in (Severity.WARNING, Severity.ERROR):
                self._notice.show_issue(issue)
                break
        self._apply_wannier_overlay()

    def _apply_wannier_overlay(self) -> None:
        if not self._wannier_projection_requested or self._wannier_bands is None:
            return
        self._band_plot.set_wannier(self._wannier_bands.x, self._wannier_bands.energies)
        self._legend.set_entry_visible("wannier", True)

    def _apply_procar(self, result: ParseResult) -> None:
        self._sidebar.set_projection_loading(False)
        if not result.ok:
            for issue in result.issues:
                self._notice.show_issue(issue)
                break
            self._sidebar.projections["element"].setChecked(False)
            self._sidebar.projections["spin"].setChecked(False)
            self._sidebar.projections["atom"].setChecked(False)
            return
        self._procar = result.value
        self._band_plot.set_orbital_source(self._procar, self._species)
        self._refresh_orbital_matrix()
        for issue in result.issues:
            if issue.severity in (Severity.WARNING, Severity.ERROR):
                self._notice.show_issue(issue)
                break
        self._try_build_element_projection()
        self._try_build_spin_projection()
        self._try_build_atom_projection()

    def _apply_species(self, result: ParseResult) -> None:
        if result.ok and result.value is not None:
            self._species = list(result.value.species)
            self._band_plot.set_orbital_source(self._procar, self._species)
            self._refresh_orbital_matrix()
            self._try_build_element_projection()
            self._try_build_atom_projection()

    def _refresh_orbital_matrix(self) -> None:
        """The matrix needs both PROCAR's orbital columns and the structure's
        element list, which arrive from two independent loads."""
        if self._procar is None or self._species is None:
            return
        elements: list[str] = []
        for symbol in self._species:
            if symbol not in elements:
                elements.append(symbol)
        self._sidebar.set_orbitals(self._procar.orbitals, elements)

    def _on_orbitals_changed(self, selection: dict) -> None:
        self._orbital_selection = dict(selection)
        self._try_build_element_projection()

    def _try_build_element_projection(self) -> None:
        if not self._element_projection_requested:
            return
        if self._bands is None or self._procar is None or self._species is None:
            return

        selection = self._orbital_selection
        full = self._sidebar.orbital_selection_is_full()

        if selection is not None and not selection:
            # Every chip is off: nothing contributes, so show nothing rather
            # than an overlay that implies a result.
            self._band_plot.set_projection(None)
            self._legend.remove_entries_with_prefix("element:")
            self._legend.remove_entries_with_prefix("orbital:")
            return

        # A single-element cell has a flat by-element view — every point is
        # the same colour — so a partial selection there is far more useful
        # coloured by orbital. With more than one element present, keeping
        # the element colours is what makes a per-element comparison legible.
        distinct = len(set(self._species))
        by_orbital = (
            not full and selection is not None and distinct <= 1
        )
        selected = (
            sorted({o for orbitals in selection.values() for o in orbitals})
            if selection
            else None
        )

        if by_orbital:
            ordered = [o for o in self._procar.orbitals if o in set(selected or [])]
            overlay = build_orbital_overlay(self._bands, self._procar, ordered)
            selected = ordered
        else:
            overlay = build_element_overlay(
                self._bands,
                self._procar,
                self._species,
                orbitals_by_element=None if full else selection,
            )
        if overlay is None:
            self._notice.show_issue(
                Issue(
                    code="procar_mismatched_dimensions",
                    message=(
                        "PROCAR's k-points or bands don't match EIGENVAL, so "
                        "element projection can't be shown."
                    ),
                    severity=Severity.WARNING,
                )
            )
            self._sidebar.projections["element"].setChecked(False)
            return

        self._band_plot.set_projection(overlay)
        self._legend.remove_entries_with_prefix("element:")
        self._legend.remove_entries_with_prefix("orbital:")
        if by_orbital:
            for orbital, color in legend_orbitals(selected):
                self._legend.add_dot_entry(f"orbital:{orbital}", orbital, color)
        else:
            for element, color in legend_elements(self._species):
                self._legend.add_dot_entry(f"element:{element}", element, color)

    def _try_build_atom_projection(self) -> None:
        if not self._atom_projection_requested:
            return
        if self._bands is None or self._procar is None or self._species is None:
            return

        # The orbital chips need PROCAR's own orbital list, which isn't
        # known until PROCAR itself has arrived — the same moment this can
        # first run at all, so populating here rather than in _apply_procar
        # keeps "where the chips get their orbitals from" in one place.
        if not self._sidebar.atom_orbitals.has_orbitals():
            self._sidebar.set_atom_orbitals(self._procar.orbitals)

        text = self._sidebar.atom_selector_text()
        selection = parse_atom_selector(text, self._species)
        orbital_selection = self._sidebar.atom_orbital_selection()
        full_orbitals = self._sidebar.atom_orbitals.is_full_selection()

        if selection.is_empty or not orbital_selection:
            self._band_plot.set_projection(None)
            self._legend.remove_entries_with_prefix("atom:")
            if not text.strip():
                self._sidebar.set_atom_status("")
            elif selection.invalid_tokens:
                self._sidebar.set_atom_status(
                    f"Not recognized: {', '.join(selection.invalid_tokens)}"
                )
            elif selection.is_empty:
                self._sidebar.set_atom_status("No atoms matched")
            else:
                self._sidebar.set_atom_status("No orbitals selected")
            return

        rows = sorted(selection.rows)
        orbitals = None if full_orbitals else orbital_selection
        overlay = build_atom_overlay(self._bands, self._procar, rows, orbitals)
        if overlay is None:
            self._notice.show_issue(
                Issue(
                    code="procar_mismatched_dimensions",
                    message=(
                        "PROCAR's k-points or bands don't match EIGENVAL, so "
                        "atom projection can't be shown."
                    ),
                    severity=Severity.WARNING,
                )
            )
            self._sidebar.projections["atom"].setChecked(False)
            return

        self._band_plot.set_projection(overlay)
        self._legend.remove_entries_with_prefix("atom:")
        count = len(rows)
        label = f"Atom{'s' if count != 1 else ''} " + ", ".join(
            str(r + 1) for r in rows[:6]
        )
        if count > 6:
            label += f" (+{count - 6})"
        self._legend.add_dot_entry("atom:selection", label, ATOM_PROJECTION_COLOR)

        status = f"{count} atom{'s' if count != 1 else ''} selected"
        if selection.invalid_tokens:
            status += f" — not recognized: {', '.join(selection.invalid_tokens)}"
        self._sidebar.set_atom_status(status)

    def _show_unavailable(self, title: str, message: str) -> None:
        self._empty.set_text(title, message)
        self._stack.setCurrentIndex(1)

    # -- export ---------------------------------------------------------------

    def _export_request(self) -> ExportRequest:
        low, high = self._limits()
        calculation = self.calculation()
        return ExportRequest(
            bands=self._bands,
            dos=self._dos,
            reference=self._sidebar.fermi.value(),
            y_min=low,
            y_max=high,
            dos_xmax=self._xmax.value(),
            include_dos=self._dos_visible and self._dos is not None,
            title=calculation.name if calculation is not None else "",
        )

    def copy_to_clipboard(self) -> bool:
        if self._bands is None and self._dos is None:
            return False
        request = self._export_request()
        if self._bands is not None:
            figure = build_figure(request, width=8.0, height=5.5)
        else:
            # Bands-less DOS (DOSCAR without EIGENVAL): copy what's actually
            # on screen instead of a figure with nothing in it.
            figure = build_dos_figure(request, width=4.0, height=5.5)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=200, bbox_inches="tight",
                       facecolor="white")
        image = QImage.fromData(buffer.getvalue(), "PNG")
        QGuiApplication.clipboard().setImage(image)
        log.info("Copied figure to clipboard")
        return True

    def export(self) -> None:
        # Bands-less DOS (DOSCAR without EIGENVAL) is a real, supported case —
        # exporting shouldn't require bands to exist, only whichever of the
        # two the user is actually trying to save.
        if self._bands is None and self._dos is None:
            return
        calculation = self.calculation()
        stem = calculation.name if calculation is not None else "bands"
        suggested = str(Path.home() / f"{stem}-bands.png")

        filters = (
            "PNG image (*.png);;SVG vector (*.svg);;PDF document (*.pdf);;"
            "DOS image (*.png);;Band data (*.csv);;DOS data (*.csv);;"
            "Paper figure — single column (*.pdf);;Paper figure — single column (*.svg)"
        )
        path_text, selected = QFileDialog.getSaveFileName(
            self, "Export Bands", suggested, filters
        )
        if not path_text:
            return

        path = Path(path_text)
        request = self._export_request()
        try:
            if selected.startswith("Band data"):
                if request.bands is None:
                    self._notice.show_issue(
                        Issue(
                            code="band_export_unavailable",
                            message="There is no band data to export.",
                            severity=Severity.INFO,
                        )
                    )
                    return
                save_band_csv(request, path.with_suffix(".csv"))
            elif selected.startswith("DOS data"):
                if request.dos is None:
                    self._notice.show_issue(
                        Issue(
                            code="dos_export_unavailable",
                            message="There is no DOS data to export.",
                            severity=Severity.INFO,
                        )
                    )
                    return
                save_dos_csv(request, path.with_suffix(".csv"))
            elif selected.startswith("DOS image"):
                if request.dos is None:
                    self._notice.show_issue(
                        Issue(
                            code="dos_export_unavailable",
                            message="There is no DOS data to export.",
                            severity=Severity.INFO,
                        )
                    )
                    return
                save_dos_figure(request, path)
            elif selected.startswith("Paper figure"):
                if request.bands is None:
                    self._notice.show_issue(
                        Issue(
                            code="band_export_unavailable",
                            message="There is no band data to export.",
                            severity=Severity.INFO,
                        )
                    )
                    return
                save_paper_figure(request, path)
            elif request.bands is not None:
                save_figure(request, path)
            else:
                # No bands at all: the combined figure has nothing to show,
                # so fall back to exactly what the DOS image option would do.
                save_dos_figure(request, path)
        except Exception as exc:  # noqa: BLE001 - report, never crash the GUI
            log.exception("Export failed")
            self._notice.show_issue(
                Issue(
                    code="export_failed",
                    message="The file could not be written.",
                    detail=f"{type(exc).__name__}: {exc}",
                    severity=Severity.ERROR,
                )
            )
