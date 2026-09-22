"""The Charge Density page's 2D plots: the slice heat map and the planar average.

Both draw the field in real in-plane cartesian coordinates rather than array
indices. That matters for any cell that is not orthogonal: a slice through a
hexagonal cell is a rhombus, and stretching it into a rectangle would put every
atom in the wrong place relative to the density around it.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal
from matplotlib.ticker import MaxNLocator

from ....models.volumetric import FieldKind
from ....services.appearance import effective_color
from ....services.volumetric import AtomMarker, SliceData
from ...plotting import GUIDE_COLOR, PixelMargins, PlotCanvas
from ...theme import PALETTE

# The bottom inset has to clear the tick labels *and* the axis label beneath
# them; at 40px the "c (Å)" caption was being cut in half by the canvas edge.
# The slice also needs headroom at the top for the colorbar's unit caption,
# which sits above the bar rather than beside it.
SLICE_MARGINS = PixelMargins(left=52, right=82, top=26, bottom=54)
AVERAGE_MARGINS = PixelMargins(left=66, right=18, top=12, bottom=56)


class SliceHeatmap(PlotCanvas):
    """A 2D cut through the field, drawn on its true lattice geometry."""

    def __init__(self) -> None:
        super().__init__(SLICE_MARGINS)
        self._colorbar = None
        self.setMinimumHeight(230)

    def show_message(self, message: str) -> None:
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        self._style_axes()
        self._colorbar = None
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.axes.text(
            0.5, 0.5, message, transform=self.axes.transAxes,
            ha="center", va="center", fontsize=9, color=PALETTE.text_tertiary,
        )
        self.redraw()

    def set_slice(
        self,
        data: SliceData,
        colormap: str,
        limits: tuple[float, float],
        units: str,
        markers: list[AtomMarker] | None = None,
    ) -> None:
        # A colorbar adds its own axes, so the figure is rebuilt rather than
        # cleared — otherwise each update would stack another one alongside.
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        self._style_axes()

        mesh = self.axes.pcolormesh(
            data.x_edges,
            data.y_edges,
            data.values,
            cmap=colormap,
            vmin=limits[0],
            vmax=limits[1],
            shading="flat",
            rasterized=True,
        )

        if markers:
            # Atoms near the plane, faded by how far off it they sit, so a
            # ring that is barely in view reads as barely in view.
            for marker in markers:
                fade = float(np.clip(1.0 - marker.distance, 0.15, 1.0))
                self.axes.plot(
                    marker.x,
                    marker.y,
                    marker="o",
                    markersize=5.0,
                    markerfacecolor=effective_color(marker.symbol),
                    markeredgecolor="#ffffff",
                    markeredgewidth=0.7,
                    alpha=fade,
                    zorder=5,
                )

        self.axes.set_xlabel(data.x_label)
        self.axes.set_ylabel(data.y_label)
        self.axes.set_aspect("equal", adjustable="box")
        self.axes.xaxis.set_major_locator(MaxNLocator(nbins=4))
        self.axes.yaxis.set_major_locator(MaxNLocator(nbins=4))

        bar = self.figure.colorbar(mesh, ax=self.axes, fraction=0.05, pad=0.03)
        bar.ax.tick_params(labelsize=7.5)
        bar.outline.set_linewidth(0.6)
        if units:
            bar.ax.set_title(units, fontsize=7.5, color=PALETTE.text_secondary, pad=4)
        self._colorbar = bar
        self.redraw()


class PlanarAveragePlot(PlotCanvas):
    """The field averaged over the two directions perpendicular to one axis."""

    positionPicked = Signal(float)   # fractional position along the axis

    def __init__(self) -> None:
        super().__init__(AVERAGE_MARGINS)
        self._cursor = None
        self._length = 1.0
        self._has_data = False
        # The full extent of the data, kept so the axis-limit sliders know
        # what range they are allowed to span and can be reset to it.
        self._data_xlim = (0.0, 1.0)
        self._data_ylim = (0.0, 1.0)
        self._view_xlim: tuple[float, float] | None = None
        self._view_ylim: tuple[float, float] | None = None
        self.setMinimumHeight(230)
        self.mpl_connect("button_press_event", self._on_click)

    def data_limits(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return self._data_xlim, self._data_ylim

    def set_view_limits(
        self,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
    ) -> None:
        """Restrict the visible window without touching the underlying data.

        Passing None for an axis returns it to the data's own extent.
        """
        self._view_xlim = xlim
        self._view_ylim = ylim
        if not self._has_data:
            return
        self._apply_view_limits()
        self.draw_idle()

    def _apply_view_limits(self) -> None:
        low, high = self._view_xlim or self._data_xlim
        if high > low:
            self.axes.set_xlim(low, high)
        low, high = self._view_ylim or self._data_ylim
        if high > low:
            self.axes.set_ylim(low, high)

    def _on_click(self, event) -> None:
        if event.inaxes is not self.axes or event.xdata is None or not self._has_data:
            return
        self.positionPicked.emit(float(event.xdata) / max(self._length, 1e-9))

    def show_message(self, message: str) -> None:
        self.axes.clear()
        self._style_axes()
        self._cursor = None
        self._has_data = False
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.axes.text(
            0.5, 0.5, message, transform=self.axes.transAxes,
            ha="center", va="center", fontsize=9, color=PALETTE.text_tertiary,
        )
        self.redraw()

    def set_curve(
        self,
        distance: np.ndarray,
        values: np.ndarray,
        axis_label: str,
        value_label: str,
        kind: FieldKind,
        series_label: str,
    ) -> None:
        self.axes.clear()
        self._style_axes()

        self.axes.plot(
            distance,
            values,
            color=PALETTE.accent,
            linewidth=1.0,
            marker="o",
            markersize=2.2,
            label=series_label,
        )

        if kind.is_signed:
            # Zero is a meaningful level for a signed field, so it gets a rule.
            self.axes.axhline(
                0.0, color=GUIDE_COLOR, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1
            )

        self.axes.set_xlabel(axis_label)
        self.axes.set_ylabel(value_label)
        self.axes.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        self.axes.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))

        self._length = float(distance[-1]) if distance.size else 1.0
        if distance.size:
            self._data_xlim = (float(distance[0]), self._length)
        if values.size:
            # A little padding, so the curve does not touch the frame.
            low, high = float(np.min(values)), float(np.max(values))
            pad = (high - low) * 0.05 or max(abs(high), 1.0) * 0.05
            self._data_ylim = (low - pad, high + pad)
        self._apply_view_limits()

        legend = self.axes.legend(
            loc="upper right", fontsize=7.5, frameon=False, handlelength=1.4,
            borderaxespad=0.2,
        )
        if legend is not None:
            for text in legend.get_texts():
                text.set_color(PALETTE.text_secondary)

        self._cursor = self.axes.axvline(
            0.0, color=PALETTE.accent, linewidth=1.0, alpha=0.0, zorder=6
        )
        self._has_data = True
        self.redraw()

    def set_cursor(self, distance: float, visible: bool = True) -> None:
        """Mark where the sectional slice currently sits."""
        if self._cursor is None:
            return
        self._cursor.set_xdata([distance, distance])
        self._cursor.set_alpha(0.9 if visible else 0.0)
        self.draw_idle()
