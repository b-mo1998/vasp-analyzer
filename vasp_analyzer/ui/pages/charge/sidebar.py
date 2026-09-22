"""Controls for the Charge Density page."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...theme import PALETTE, RADIUS, SPACING, TYPE
from ...widgets.primitives import Divider, Label, SectionHeader
from ...widgets.slider_row import SliderRow

# Restrained rather than every matplotlib map: sequential maps for
# non-negative fields, diverging for signed ones, chosen automatically.
SEQUENTIAL_MAPS = ["Viridis", "Plasma", "Inferno", "Magma", "Cividis"]
DIVERGING_MAPS = ["Coolwarm", "RdBu", "Seismic", "BWR"]

SUBSAMPLING = [
    ("1 (Full Resolution)", 1),
    ("2× Coarser", 2),
    ("4× Coarser", 4),
    ("8× Coarser", 8),
]


class ChargeSidebar(QWidget):
    """Source and type selection, isosurface controls and display options."""

    sourceChanged = Signal(int)
    componentChanged = Signal(int)
    isovalueChanged = Signal(float)
    opacityChanged = Signal(float)
    colormapChanged = Signal(str)
    displayChanged = Signal()
    renderRequested = Signal()
    exportRequested = Signal()
    localizeChanged = Signal()
    compareBrowseRequested = Signal()
    compareClearRequested = Signal()
    compareChanged = Signal()
    integratePerAtomRequested = Signal()

    _resetting = False

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        # A sidebar is a fixed-width column; anything too wide for it is a
        # layout bug to fix at the source, not something to scroll sideways.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        content = QWidget()
        layout = QVBoxLayout(content)
        # Side padding is lg rather than xl so the widest row here — the
        # isovalue slider with its value box and unit — fits the fixed
        # sidebar width instead of being clipped by it.
        layout.setContentsMargins(SPACING.lg, SPACING.xl, SPACING.lg, SPACING.xl)
        layout.setSpacing(SPACING.lg)
        scroll.setWidget(content)

        layout.addWidget(Label("Charge Density", "TitleLabel"))
        layout.addWidget(Label("Visualize and analyze electron density.", "SubtitleLabel"))
        layout.addSpacing(SPACING.xs)
        layout.addWidget(Divider())

        # --- what to show -----------------------------------------------------
        #
        # Source and data type are *choices* only when there is more than one
        # of them: a run with a lone CHGCAR that is not spin polarised has
        # exactly one source and one component, and a one-item dropdown is a
        # control that looks interactive while doing nothing. Each collapses
        # to an inline "label: value" line in that case and reappears as a
        # real dropdown when the calculation actually offers alternatives
        # (CHGCAR alongside ELFCAR, or a spin-polarised total/magnetization
        # pair).
        self.source_row = QHBoxLayout()
        self.source_row.setContentsMargins(0, 0, 0, 0)
        self.source_row.setSpacing(SPACING.sm)
        self.source_caption = self._caption("Data Source")
        self.source_row.addWidget(self.source_caption)
        self.source_value = QLabel("—")
        self.source_value.setStyleSheet(f"font-size: {TYPE.body}px; font-weight: 500;")
        self.source_row.addWidget(self.source_value, 1)
        layout.addLayout(self.source_row)

        self.source_box = QComboBox()
        self.source_box.currentIndexChanged.connect(self.sourceChanged)
        self.source_box.hide()
        layout.addWidget(self.source_box)

        self.component_caption = self._caption("Data Type")
        layout.addWidget(self.component_caption)
        self.component_box = QComboBox()
        self.component_box.currentIndexChanged.connect(self.componentChanged)
        layout.addWidget(self.component_box)
        self.component_caption.hide()
        self.component_box.hide()

        layout.addWidget(Divider())

        # --- visualisation ------------------------------------------------------
        layout.addWidget(SectionHeader("Visualization"))
        self.isovalue = SliderRow(
            "Isovalue", 0.0, 1.0, 0.5, 0.001, "", decimals=4
        )
        self.isovalue.valueChanged.connect(self.isovalueChanged)
        layout.addWidget(self.isovalue)

        layout.addWidget(self._caption("Color Map"))
        self.colormap_box = QComboBox()
        self.colormap_box.currentTextChanged.connect(self.colormapChanged)
        layout.addWidget(self.colormap_box)
        self.colormap_strip = _ColormapStrip()
        layout.addWidget(self.colormap_strip)

        self.opacity = SliderRow("Opacity", 0.05, 1.0, 0.7, 0.01, "", decimals=2)
        self.opacity.valueChanged.connect(self.opacityChanged)
        layout.addWidget(self.opacity)

        layout.addWidget(Divider())

        # --- display options -------------------------------------------------------
        layout.addWidget(SectionHeader("Display Options"))
        self.checks: dict[str, QCheckBox] = {}
        for key, text, default in (
            ("cell", "Show Unit Cell", True),
            ("atoms", "Show Atoms", True),
            ("surface", "Show Isosurface", True),
            ("axes", "Show Axes", True),
            ("periodic", "Periodic Images", False),
        ):
            box = QCheckBox(text)
            box.setChecked(default)
            box.toggled.connect(lambda _v: self.displayChanged.emit())
            layout.addWidget(box)
            self.checks[key] = box
        self.checks["periodic"].setToolTip(
            "Repeat the isosurface into the six neighbouring cells. Kept to "
            "face neighbours only — a full shell would cost 27× the drawing "
            "for little extra insight."
        )

        layout.addWidget(Divider())

        # --- local density -------------------------------------------------------------
        #
        # A CHGCAR carries only the total density on its grid — nothing that
        # decomposes it by atom or orbital. What's offered here instead is a
        # spatial mask: zero the density beyond a radius of the chosen atoms,
        # which is a real and commonly useful way to isolate a local bonding
        # environment (a dopant, a defect, one species) even though it is not
        # a quantum-mechanical projection. The isovalue, statistics and plots
        # all update to reflect only the localized region while this is on.
        layout.addWidget(SectionHeader("Local Density"))
        self.localize_enabled = QCheckBox("Localize to Atoms")
        self.localize_enabled.setToolTip(
            "Zero the density beyond the radius below, measured from the "
            "nearest of the listed atoms. Not a per-atom decomposition — "
            "CHGCAR doesn't carry one — a spatial mask around them."
        )
        self.localize_enabled.toggled.connect(self._on_localize_changed)
        layout.addWidget(self.localize_enabled)

        self.localize_field = QLineEdit()
        self.localize_field.setObjectName("ValueField")
        self.localize_field.setPlaceholderText("Atom IDs or elements, e.g. Te, 12, 45")
        self.localize_field.setEnabled(False)
        self.localize_field.hide()
        self.localize_field.textChanged.connect(self._on_localize_changed)
        layout.addWidget(self.localize_field)

        self.localize_radius = SliderRow("Radius", 0.5, 8.0, 2.0, 0.1, "Å", decimals=1)
        self.localize_radius.setEnabled(False)
        self.localize_radius.hide()
        self.localize_radius.valueChanged.connect(lambda _v: self.localizeChanged.emit())
        layout.addWidget(self.localize_radius)

        self.localize_status = Label("", "TertiaryLabel")
        self.localize_status.setWordWrap(True)
        self.localize_status.hide()
        layout.addWidget(self.localize_status)

        # Not a Bader (zero-flux-surface) analysis -- reuses the same
        # per-atom sphere the Localize controls above already draw, just
        # integrated individually per atom rather than shown as one mask.
        self.integrate_button = QPushButton("Integrate Per-Atom")
        self.integrate_button.setToolTip(
            "∫f dV within the radius above, around each listed atom "
            "individually. Overlapping spheres double-count that region — "
            "keep the radius under half the interatomic spacing."
        )
        self.integrate_button.setEnabled(False)
        self.integrate_button.hide()
        self.integrate_button.clicked.connect(self.integratePerAtomRequested)
        layout.addWidget(self.integrate_button)

        layout.addWidget(Divider())

        # --- charge difference -------------------------------------------------------
        #
        # Subtracting a second density from the loaded one shows where charge
        # was actually gained or lost -- e.g. before vs. after an adsorption
        # event on the same slab -- rather than only inspecting each density
        # on its own. Both files need the same FFT grid; a mismatch is
        # reported through the status line rather than silently resampled.
        layout.addWidget(SectionHeader("Charge Difference"))
        compare_row = QHBoxLayout()
        compare_row.setContentsMargins(0, 0, 0, 0)
        compare_row.setSpacing(SPACING.sm)
        self.compare_browse = QPushButton("Browse…")
        self.compare_browse.clicked.connect(self.compareBrowseRequested)
        compare_row.addWidget(self.compare_browse, 1)
        self.compare_clear = QPushButton("Clear")
        self.compare_clear.setEnabled(False)
        self.compare_clear.clicked.connect(self.compareClearRequested)
        compare_row.addWidget(self.compare_clear, 0)
        layout.addLayout(compare_row)

        self.compare_file_label = Label("No comparison file chosen", "TertiaryLabel")
        self.compare_file_label.setWordWrap(True)
        layout.addWidget(self.compare_file_label)

        self.compare_enabled = QCheckBox("Show Difference (Current − Compare)")
        self.compare_enabled.setEnabled(False)
        self.compare_enabled.toggled.connect(lambda _v: self.compareChanged.emit())
        layout.addWidget(self.compare_enabled)

        self.compare_status = Label("", "TertiaryLabel")
        self.compare_status.setWordWrap(True)
        self.compare_status.hide()
        layout.addWidget(self.compare_status)

        layout.addWidget(Divider())

        # --- advanced ----------------------------------------------------------------
        layout.addWidget(SectionHeader("Advanced"))
        self.smoothing = SliderRow("Mesh Smoothing", 0.0, 1.0, 0.5, 0.05, "", decimals=2)
        self.smoothing.setToolTip(
            "Smooths the extracted surface mesh only. The density itself is "
            "left untouched, so every reported number is unaffected."
        )
        layout.addWidget(self.smoothing)

        layout.addWidget(self._caption("Grid Subsampling"))
        self.subsample_box = QComboBox()
        for label, _factor in SUBSAMPLING:
            self.subsample_box.addItem(label)
        self.subsample_box.setToolTip(
            "Rendering resolution only. Statistics, the planar average and the "
            "integral are always computed from the full grid."
        )
        layout.addWidget(self.subsample_box)

        layout.addSpacing(SPACING.xs)
        self.render_button = QPushButton("⟳  Render")
        self.render_button.setFixedHeight(36)
        self.render_button.setObjectName("RenderButton")
        self.render_button.setCursor(Qt.PointingHandCursor)
        self.render_button.clicked.connect(self.renderRequested)
        self.render_button.setStyleSheet(
            f"#RenderButton {{ border: none; border-radius: {RADIUS.md}px;"
            f" background: {PALETTE.accent}; color: #ffffff;"
            f" font-size: {TYPE.body}px; font-weight: 600; }}"
            f"#RenderButton:hover:enabled {{ background: {PALETTE.accent_hover}; }}"
            f"#RenderButton:disabled {{ background: {PALETTE.border};"
            f" color: {PALETTE.disabled_text}; }}"
        )
        layout.addWidget(self.render_button)

        self.export_button = QPushButton("Export Image")
        self.export_button.clicked.connect(self.exportRequested)
        layout.addWidget(self.export_button)

        layout.addStretch(1)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _caption(text: str) -> Label:
        label = Label(text)
        label.setStyleSheet(
            f"color: {PALETTE.text_secondary}; font-size: {TYPE.body}px;"
        )
        return label

    # -- population -------------------------------------------------------------

    def set_sources(self, labels: list[str]) -> None:
        self.source_box.blockSignals(True)
        self.source_box.clear()
        self.source_box.addItems(labels)
        self.source_box.blockSignals(False)

        single = len(labels) <= 1
        self.source_value.setText(labels[0] if labels else "—")
        self.source_value.setVisible(single)
        self.source_box.setVisible(not single)

    def set_components(self, labels: list[str]) -> None:
        self.component_box.blockSignals(True)
        self.component_box.clear()
        self.component_box.addItems(labels)
        self.component_box.blockSignals(False)

        # Hidden entirely for a single component — there is nothing to pick.
        multiple = len(labels) > 1
        self.component_caption.setVisible(multiple)
        self.component_box.setVisible(multiple)

    def set_colormaps(self, names: list[str], current: str) -> None:
        self.colormap_box.blockSignals(True)
        self.colormap_box.clear()
        self.colormap_box.addItems(names)
        index = self.colormap_box.findText(current)
        self.colormap_box.setCurrentIndex(max(index, 0))
        self.colormap_box.blockSignals(False)
        self.colormap_strip.set_colormap(self.colormap_box.currentText())

    def configure_isovalue(
        self, low: float, high: float, value: float, units: str
    ) -> None:
        """Set the slider's range from the data rather than a fixed 0.001-0.1.

        Densities differ by orders of magnitude between systems, so a hard-coded
        range would be unusable for most of them.
        """
        step = max((high - low) / 500.0, 1e-6)
        self.isovalue.blockSignals(True)
        self.isovalue.setRange(low, high)
        self.isovalue.set_unit(units)
        self.isovalue.set_single_step(step)
        self.isovalue.setValue(value, notify=False)
        self.isovalue.blockSignals(False)
        # The unit already sits in its own label beside the field, so
        # repeating it in the title would show it twice on one row.
        self.isovalue.set_title("Isovalue")

    def subsample_factor(self) -> int:
        index = max(0, self.subsample_box.currentIndex())
        return SUBSAMPLING[index][1]

    def _on_localize_changed(self, *_args) -> None:
        active = self.localize_enabled.isEnabled() and self.localize_enabled.isChecked()
        # Shown only while the box is on: the atom field, radius and
        # per-atom button are one feature, and greyed they only add length.
        for widget in (self.localize_field, self.localize_radius, self.integrate_button):
            widget.setEnabled(active)
            widget.setVisible(active)
        if not self._resetting:
            self.localizeChanged.emit()

    def localize_selector_text(self) -> str:
        return self.localize_field.text()

    def set_localize_status(self, text: str) -> None:
        self.localize_status.setText(text)
        self.localize_status.setVisible(bool(text))

    def set_compare_file(self, name: str | None) -> None:
        self.compare_file_label.setText(name or "No comparison file chosen")
        self.compare_enabled.setEnabled(bool(name))
        self.compare_clear.setEnabled(bool(name))
        if not name:
            # Silently during a reset -- see reset() for why.
            self.compare_enabled.blockSignals(self._resetting)
            self.compare_enabled.setChecked(False)
            self.compare_enabled.blockSignals(False)

    def set_compare_status(self, text: str) -> None:
        self.compare_status.setText(text)
        self.compare_status.setVisible(bool(text))

    def set_enabled(self, enabled: bool) -> None:
        for widget in (
            self.source_box,
            self.component_box,
            self.isovalue,
            self.opacity,
            self.colormap_box,
            self.smoothing,
            self.subsample_box,
            self.render_button,
            self.export_button,
            self.localize_enabled,
            self.compare_browse,
            *self.checks.values(),
        ):
            widget.setEnabled(enabled)
        if not enabled:
            for widget in (self.localize_field, self.localize_radius, self.integrate_button):
                widget.setEnabled(False)
                widget.setVisible(False)
            self.compare_enabled.setEnabled(False)
            self.compare_clear.setEnabled(False)

    def reset(self) -> None:
        # Silent: returning each control to its default must not fire that
        # control's own change signal. Before this guard, one Reset click
        # emitted localizeChanged twice (the checkbox and the cleared text
        # field) and compareChanged once, each triggering the page's full
        # rebuild pipeline -- statistics, slice, planar average and a fresh
        # isosurface extraction -- before the page's own reset ran the same
        # pipeline a fourth time. On a large CHGCAR that stacked several
        # seconds of redundant marching-cubes work into one click and froze
        # the UI long enough to read as a crash. The page's reset performs
        # exactly one rebuild after this returns.
        self._resetting = True
        try:
            self._reset_controls()
        finally:
            self._resetting = False

    def _reset_controls(self) -> None:
        self.opacity.setValue(0.7, notify=False)
        self.smoothing.setValue(0.5, notify=False)
        self.subsample_box.setCurrentIndex(0)
        self.localize_enabled.setChecked(False)
        self.localize_field.clear()
        self.localize_radius.setValue(2.0, notify=False)
        self.integrate_button.setEnabled(False)
        self.set_localize_status("")
        self.set_compare_file(None)
        self.set_compare_status("")
        for key, default in (
            ("cell", True), ("atoms", True), ("surface", True),
            ("axes", True), ("periodic", False),
        ):
            self.checks[key].blockSignals(True)
            self.checks[key].setChecked(default)
            self.checks[key].blockSignals(False)


class _ColormapStrip(QWidget):
    """A preview bar showing the selected colour map."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(12)
        self._name = "viridis"

    def set_colormap(self, name: str) -> None:
        self._name = (name or "viridis").lower()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        from matplotlib import colormaps
        from PySide6.QtGui import QColor, QLinearGradient, QPainter

        painter = QPainter(self)
        try:
            cmap = colormaps[self._name]
        except KeyError:
            cmap = colormaps["viridis"]
        gradient = QLinearGradient(0, 0, self.width(), 0)
        for step in range(17):
            position = step / 16.0
            red, green, blue, _ = cmap(position)
            gradient.setColorAt(
                position,
                QColor(int(red * 255), int(green * 255), int(blue * 255)),
            )
        painter.fillRect(self.rect(), gradient)
        painter.end()
