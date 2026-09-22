"""The MD telemetry charts: temperature, energy and mean-square displacement.

All three share one x-axis (simulation time, or frame index when POTIM is
unknown) and one movable cursor. Clicking anywhere on any of them seeks the
trajectory, which is what turns the page from three static graphs into an
instrument for inspecting a movie.

The cursor is a single ``axvline`` moved by ``set_xdata`` rather than a redrawn
artist: dragging the timeline updates three charts per frame, and rebuilding
them at 20 fps would not keep up.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal
from matplotlib.ticker import MaxNLocator

from ...plotting import GUIDE_COLOR, PixelMargins, PlotCanvas
from ...theme import PALETTE, SERIES_COLORS

CURSOR_COLOR = PALETTE.accent
MSD_FLOOR = 1e-4          # Å², the smallest value a log axis will show
# The bottom inset has to clear the tick labels *and* the axis title; at 34 and
# again at 42 the title was cropped by the canvas edge on the shorter panels.
PLOT_MARGINS = PixelMargins(left=52, right=10, top=8, bottom=50)


class MDPlot(PlotCanvas):
    """A time-series chart with a seekable playback cursor."""

    seekRequested = Signal(float)   # x-axis value the user clicked

    def __init__(self, ylabel: str = "") -> None:
        super().__init__(PLOT_MARGINS)
        self._cursor = None
        self._xlabel = "Time (ps)"
        self._ylabel = ylabel
        self._has_data = False
        # A *floor*, not the working height: each plot is added with a stretch
        # factor, so on any normal window the three of them share out whatever
        # the panel has left after the statistics card and grow well past this.
        # Setting the floor to the desired height instead is what forced a
        # scrollbar — three 215px minimums plus the card exceeded the panel,
        # so the layout could only overflow rather than fit. Low enough that
        # the panel still fits an unusually short window without clipping, at
        # which point the scrollbar is the correct fallback.
        self.setMinimumHeight(140)
        self.mpl_connect("button_press_event", self._on_click)

    # -- interaction ---------------------------------------------------------

    def _on_click(self, event) -> None:
        if event.inaxes is not self.axes or event.xdata is None or not self._has_data:
            return
        self.seekRequested.emit(float(event.xdata))

    # -- drawing ---------------------------------------------------------------

    def _begin(self, xlabel: str) -> None:
        self.axes.clear()
        self._style_axes()
        self._cursor = None
        self._xlabel = xlabel
        self.axes.set_xlabel(xlabel)
        if self._ylabel:
            self.axes.set_ylabel(self._ylabel)
        self.axes.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        self.axes.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))

    def _finish(self, x: np.ndarray) -> None:
        if x.size:
            self.axes.set_xlim(float(x[0]), float(x[-1]) if x[-1] > x[0] else float(x[0]) + 1)
        self._cursor = self.axes.axvline(
            float(x[0]) if x.size else 0.0,
            color=CURSOR_COLOR,
            linewidth=1.1,
            zorder=6,
        )
        self._has_data = True
        self.redraw()

    def show_unavailable(self, message: str) -> None:
        """Replace the chart with an explanation instead of an empty frame."""
        self.axes.clear()
        self._style_axes()
        self._cursor = None
        self._has_data = False
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.axes.text(
            0.5,
            0.5,
            message,
            transform=self.axes.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color=PALETTE.text_tertiary,
            wrap=True,
        )
        self.redraw()

    def set_cursor(self, value: float) -> None:
        if self._cursor is None:
            return
        self._cursor.set_xdata([value, value])
        self.draw_idle()

    def _legend(self) -> None:
        legend = self.axes.legend(
            loc="upper right",
            fontsize=7.5,
            frameon=False,
            handlelength=1.4,
            borderaxespad=0.2,
            labelspacing=0.25,
        )
        if legend is not None:
            for text in legend.get_texts():
                text.set_color(PALETTE.text_secondary)


class TemperaturePlot(MDPlot):
    """Instantaneous ionic temperature, with the thermostat's target."""

    def __init__(self) -> None:
        super().__init__()

    def set_data(
        self,
        x: np.ndarray,
        temperature: np.ndarray,
        xlabel: str,
        target: tuple[float, float] | None = None,
    ) -> None:
        self._begin(xlabel)
        self.axes.plot(
            x, temperature, color=PALETTE.accent, linewidth=0.9, label="Inst. Temp"
        )

        if target is not None:
            begin, end = target
            if abs(end - begin) < 1e-9:
                self.axes.axhline(
                    begin,
                    color=PALETTE.fermi,
                    linewidth=1.0,
                    linestyle=(0, (5, 3)),
                    label=f"Target ({begin:g} K)",
                )
            else:
                # A TEBEG -> TEEND ramp is a schedule, not a single setpoint;
                # drawing one flat line would misrepresent the run.
                schedule = np.linspace(begin, end, len(x))
                self.axes.plot(
                    x,
                    schedule,
                    color=PALETTE.fermi,
                    linewidth=1.0,
                    linestyle=(0, (5, 3)),
                    label=f"Target ({begin:g}→{end:g} K)",
                )

        self.axes.set_ylim(0.0, max(float(np.max(temperature)) * 1.15, 1.0))
        self._legend()
        self._finish(x)


class EnergyPlot(MDPlot):
    """Total, potential and kinetic energy, one selected at a time."""

    SERIES = [
        ("total_energy", "Total Energy", SERIES_COLORS[0]),
        ("potential_energy", "Potential Energy", SERIES_COLORS[3]),
        ("kinetic_energy", "Kinetic Energy", SERIES_COLORS[2]),
    ]

    def set_data(
        self, x: np.ndarray, series: dict[str, np.ndarray], selected: str, xlabel: str
    ) -> None:
        self._begin(xlabel)
        values = series.get(selected)
        if values is None:
            self.show_unavailable("Selected energy unavailable")
            return

        label, color = next(
            ((name, colour) for key, name, colour in self.SERIES if key == selected),
            ("Energy", SERIES_COLORS[0]),
        )
        self.axes.plot(x, values, color=color, linewidth=0.9, label=label)

        span = float(np.max(values) - np.min(values))
        pad = span * 0.12 if span > 1e-9 else max(abs(float(np.mean(values))) * 1e-4, 0.01)
        self.axes.set_ylim(float(np.min(values)) - pad, float(np.max(values)) + pad)
        self._legend()
        self._finish(x)


class MSDPlot(MDPlot):
    """Per-element mean-square displacement on a logarithmic axis.

    Log scaling is the convention for MSD: the interesting behaviour spans
    orders of magnitude between a vibrating lattice atom and a diffusing one,
    and on a linear axis the former is a flat line at zero.
    """

    def set_data(
        self, x: np.ndarray, curves: dict[str, np.ndarray], elements: list[str], xlabel: str
    ) -> None:
        self._begin(xlabel)
        if not curves:
            self.show_unavailable("Not enough frames for MSD")
            return

        drawn = False
        for position, element in enumerate(elements):
            values = curves.get(element)
            if values is None:
                continue
            # A perfect zero at t=0 cannot be drawn on a log axis; clamping to a
            # floor keeps the first point visible without distorting the curve.
            self.axes.plot(
                x,
                np.maximum(values, MSD_FLOOR),
                color=SERIES_COLORS[position % len(SERIES_COLORS)],
                linewidth=0.9,
                label=element,
            )
            drawn = True

        if not drawn:
            self.show_unavailable("No displacement data")
            return

        self.axes.set_yscale("log")
        self._legend()
        self._finish(x)


class GeometryPlot(MDPlot):
    """One distance or angle from the current atom selection, over time.

    Two selected atoms give their separation; three give the angle at the
    middle one. It answers "does this bond break?" on the same time axis as
    the temperature and energy above it, with the same seekable cursor.
    """

    def set_data(
        self, x: np.ndarray, values: np.ndarray, label: str, ylabel: str, xlabel: str
    ) -> None:
        self._ylabel = ylabel
        self._begin(xlabel)
        self.axes.plot(x, values, color=SERIES_COLORS[0], linewidth=0.9, label=label)
        self._legend()
        self._finish(x)
