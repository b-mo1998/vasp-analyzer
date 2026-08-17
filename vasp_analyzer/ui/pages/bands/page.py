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
from ....services.loader import (
    DATASET_BANDS,
    DATASET_DOS,
    DATASET_PROCAR,
    DATASET_STRUCTURE,
    DataLoader,
)
from ...theme import METRICS, PALETTE, SPACING, TYPE
from ...widgets.empty_state import EmptyState
from ...widgets.legend import PlotLegend
from ...widgets.notice import Notice
from ...widgets.primitives import Card
from ...widgets.slider_row import SliderRow
from ...widgets.toggle import LabelledToggle
from ..base import Page
from .band_plot import BandPlot
from .dos_plot import DosPlot, nice_upper_bound
from .export import ExportRequest, build_figure, save_band_csv, save_dos_csv, save_figure
from .projection import build_element_overlay, legend_elements
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

    def __init__(self, loader: DataLoader, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loader = loader
        self._sidebar = BandsSidebar()
        self._bands: BandData | None = None
        self._dos: DosData | None = None
        self._procar: ProcarData | None = None
        self._species: list[str] | None = None
        self._element_projection_requested = False
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

        self._band_plot = BandPlot()
        body.addWidget(self._band_plot, 1)
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

        caption = QLabel("X Max (states/eV)")
        caption.setAlignment(Qt.AlignCenter)
        caption.setStyleSheet(
            f"font-size: {TYPE.small}px; color: {PALETTE.text_secondary};"
        )
        header_layout.addWidget(caption)

        self._xmax = SliderRow("", 0.0, 16.0, 8.0, 0.1, "", decimals=2, inline=True)
        header_layout.addWidget(self._xmax)
        header_layout.addStretch(1)
        body.addWidget(header)

        self._dos_plot = DosPlot()
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
        self._element_projection_requested = False
        self._auto_xmax = True
        self._notice.clear()
        self._band_plot.set_bands(None)
        self._band_plot.set_projection(None)
        self._dos_plot.set_dos(None)
        self._legend.remove_entries_with_prefix("element:")
        self._sidebar.projections["element"].setChecked(False)

        if calculation is None:
            self._sidebar.set_controls_enabled(False)
            self._sidebar.set_dos_enabled(False, "No calculation is open.")
            self._dos_toggle.setEnabled(False)
            self._show_unavailable(
                "No calculation open", "Open a calculation directory to begin."
            )
            return

        self._apply_capabilities(calculation)

        if calculation.can(Capability.BANDS):
            self._stack.setCurrentIndex(0)
            self._loader.request(DATASET_BANDS)
            if calculation.can(Capability.DOS):
                self._loader.request(DATASET_DOS)
        else:
            self._show_unavailable(
                "Band structure unavailable",
                calculation.why_not(Capability.BANDS),
            )

    def _apply_capabilities(self, calculation: Calculation) -> None:
        side = self._sidebar
        side.set_controls_enabled(calculation.can(Capability.BANDS))

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
            ("wannier", Capability.WANNIER_BANDS),
        ):
            side.set_projection_enabled(
                key, calculation.can(capability), calculation.why_not(capability)
            )

        self._legend.set_entry_visible(
            "wannier", calculation.can(Capability.WANNIER_BANDS)
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
        self._try_build_projection()

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

        # Spin and Wannier projections aren't implemented yet; the checkboxes
        # exist and correctly respect capability state, but nothing is drawn
        # until each data path lands.
        log.info("Projection %s toggled -> %s (not yet implemented)", key, checked)
        if checked:
            self._notice.show_issue(
                Issue(
                    code="projection_not_implemented",
                    message=f"{key.title()} projection is not implemented yet.",
                    severity=Severity.INFO,
                )
            )
        else:
            self._notice.clear()

    def _set_element_projection(self, checked: bool) -> None:
        if not checked:
            self._element_projection_requested = False
            self._band_plot.set_projection(None)
            self._legend.remove_entries_with_prefix("element:")
            return

        calculation = self.calculation()
        if calculation is None or not calculation.can(Capability.ELEMENT_PROJECTION):
            return

        self._element_projection_requested = True
        self._loader.request(DATASET_PROCAR)
        self._loader.request(DATASET_STRUCTURE)
        self._try_build_projection()

    def _apply_procar(self, result: ParseResult) -> None:
        if not result.ok:
            for issue in result.issues:
                self._notice.show_issue(issue)
                break
            self._sidebar.projections["element"].setChecked(False)
            return
        self._procar = result.value
        for issue in result.issues:
            if issue.severity in (Severity.WARNING, Severity.ERROR):
                self._notice.show_issue(issue)
                break
        self._try_build_projection()

    def _apply_species(self, result: ParseResult) -> None:
        if result.ok and result.value is not None:
            self._species = list(result.value.species)
            self._try_build_projection()

    def _try_build_projection(self) -> None:
        if not self._element_projection_requested:
            return
        if self._bands is None or self._procar is None or self._species is None:
            return

        overlay = build_element_overlay(self._bands, self._procar, self._species)
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
        for element, color in legend_elements(self._species):
            self._legend.add_dot_entry(f"element:{element}", element, color)

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
            title=calculation.directory.name if calculation is not None else "",
        )

    def copy_to_clipboard(self) -> bool:
        if self._bands is None:
            return False
        figure = build_figure(self._export_request(), width=8.0, height=5.5)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=200, bbox_inches="tight",
                       facecolor="white")
        image = QImage.fromData(buffer.getvalue(), "PNG")
        QGuiApplication.clipboard().setImage(image)
        log.info("Copied band figure to clipboard")
        return True

    def export(self) -> None:
        if self._bands is None:
            return
        calculation = self.calculation()
        stem = calculation.directory.name if calculation is not None else "bands"
        suggested = str(Path.home() / f"{stem}-bands.png")

        filters = (
            "PNG image (*.png);;SVG vector (*.svg);;PDF document (*.pdf);;"
            "Band data (*.csv);;DOS data (*.csv)"
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
            else:
                save_figure(request, path)
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
