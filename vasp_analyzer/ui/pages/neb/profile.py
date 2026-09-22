"""The NEB energy profile: relative energy against reaction coordinate.

The discrete images are the data; the curve through them is an aid to reading
it, kept subordinate to the markers because a smooth line is the part a reader
is most likely to mistake for computed information.

The curve comes from VTST's own ``spline.dat`` whenever the run produced one.
That spline is fitted to the image energies *and* their tangential forces, so
it is not reproducible from the energies alone — interpolating here instead
would draw a visibly different curve, and quote a different barrier, from the
one VTST reports for the same calculation.

Clicking a point selects that image — the fastest way to get from "what is that
peak?" to the structure responsible for it.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal
from matplotlib.ticker import MaxNLocator

from ...plotting import PixelMargins, PlotCanvas
from ...theme import PALETTE

PROFILE_MARGINS = PixelMargins(left=58, right=14, top=14, bottom=44)

POINT_COLOR = PALETTE.accent
CLIMB_COLOR = "#dc2626"
SELECTED_RING = "#1d4ed8"
MISSING_COLOR = "#c8c8cc"


class EnergyProfilePlot(PlotCanvas):
    """Discrete NEB image energies and the interpolated path through them."""

    imageSelected = Signal(int)

    def __init__(self) -> None:
        super().__init__(PROFILE_MARGINS)
        self._x: np.ndarray | None = None
        self._known: list[int] = []
        self._selected: int | None = None
        self._highlight = None
        self._data_xlim = (0.0, 1.0)
        self._data_ylim = (0.0, 1.0)
        self._view_xlim: tuple[float, float] | None = None
        self._view_ylim: tuple[float, float] | None = None
        self.setMinimumHeight(230)
        self.mpl_connect("button_press_event", self._on_click)

    # -- axis windows ----------------------------------------------------------

    def data_limits(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return self._data_xlim, self._data_ylim

    def set_view_limits(
        self,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
    ) -> None:
        """Restrict the visible window. None returns an axis to its data extent."""
        self._view_xlim = xlim
        self._view_ylim = ylim
        self._apply_view_limits()
        self.draw_idle()

    def _apply_view_limits(self) -> None:
        low, high = self._view_xlim or self._data_xlim
        if high > low:
            self.axes.set_xlim(low, high)
        low, high = self._view_ylim or self._data_ylim
        if high > low:
            self.axes.set_ylim(low, high)

    # -- interaction -----------------------------------------------------------

    def _on_click(self, event) -> None:
        if event.inaxes is not self.axes or event.xdata is None or self._x is None:
            return
        if not self._known:
            return
        # Nearest *computed* image along x: clicking near a gap should select
        # the image that exists, not the one that has no energy to show.
        candidates = np.array([self._x[i] for i in self._known])
        nearest = self._known[int(np.argmin(np.abs(candidates - float(event.xdata))))]
        self.imageSelected.emit(nearest)

    # -- drawing ------------------------------------------------------------------

    def show_unavailable(self, message: str) -> None:
        self.axes.clear()
        self._style_axes()
        self._x = None
        self._known = []
        self._highlight = None
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
        )
        self.redraw()

    def set_path(
        self,
        coordinate: np.ndarray,
        relative: list[float | None],
        climbing_image: int | None = None,
        use_index_axis: bool = False,
        spline: "SplineCurve | None" = None,
    ) -> None:
        """Draw the profile.

        ``relative`` may contain None for images that never reported an energy;
        those are shown as hollow markers on the axis rather than omitted, so a
        half-finished band reads as incomplete instead of merely short.

        ``climbing_image`` is accepted but no longer marked on the plot: the
        climbing image is named in the summary strip and ringed when selected,
        and a second permanent glyph on the peak crowded the curve.
        """
        _ = climbing_image
        self.axes.clear()
        self._style_axes()
        self._highlight = None

        self._x = np.asarray(coordinate, dtype=float)
        self._known = [i for i, value in enumerate(relative) if value is not None]
        if not self._known:
            self.show_unavailable("No image energies available")
            return

        known_x = np.array([self._x[i] for i in self._known])
        known_y = np.array([relative[i] for i in self._known], dtype=float)
        missing = [i for i, value in enumerate(relative) if value is None]

        # A curve is only drawn through a *complete* band. Joining the two ends
        # of a half-finished one would draw a straight climb the calculation
        # never computed, which is exactly the shape a reader would take for
        # the reaction path.
        curve_x: np.ndarray | None = None
        curve_y: np.ndarray | None = None
        if spline is not None and spline.n_points > 1 and not use_index_axis:
            curve_x = np.asarray(spline.reaction_coordinate, dtype=float)
            curve_y = np.asarray(spline.relative_energy, dtype=float)
            self.axes.plot(
                curve_x, curve_y, color=POINT_COLOR, linewidth=1.0, alpha=0.55,
                zorder=2, label="Spline (VTST)",
            )
        elif not missing:
            # No spline.dat: join the computed points directly rather than
            # inventing a smooth curve. A straight segment is honest about
            # being a guide between two data points; a fitted spline would
            # look like the calculated path without being it.
            self.axes.plot(
                known_x, known_y, color=POINT_COLOR, linewidth=1.0, alpha=0.5,
                zorder=2,
            )

        self.axes.plot(
            known_x,
            known_y,
            linestyle="none",
            marker="o",
            markersize=5.0,
            color=POINT_COLOR,
            zorder=4,
            label="Path",
        )

        if missing:
            # Marked as vertical guides rather than points: an image with no
            # energy has no y value, and drawing it at zero would read as
            # "degenerate with the initial state" rather than "not computed".
            for position, index in enumerate(missing):
                self.axes.axvline(
                    self._x[index],
                    color=MISSING_COLOR,
                    linewidth=1.0,
                    linestyle=(0, (2, 3)),
                    zorder=1,
                    label="Not computed" if position == 0 else None,
                )

        self.axes.set_xlabel("Image Index" if use_index_axis else "Reaction Coordinate (Å)")
        self.axes.set_ylabel("Energy (eV)")
        self.axes.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        self.axes.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))

        # The natural extent covers the spline too, which can overshoot the
        # image energies between points — clipping it would hide the very
        # maximum the barrier is read from.
        low_y = float(min(known_y.min(), curve_y.min() if curve_y is not None else known_y.min()))
        high_y = float(max(known_y.max(), curve_y.max() if curve_y is not None else known_y.max()))
        span = high_y - low_y
        pad = span * 0.18 if span > 1e-9 else 0.5
        self._data_ylim = (low_y - pad, high_y + pad)
        if self._x.size:
            width = float(self._x.max() - self._x.min()) or 1.0
            self._data_xlim = (
                float(self._x.min()) - width * 0.05,
                float(self._x.max()) + width * 0.05,
            )
        self._apply_view_limits()

        legend = self.axes.legend(
            loc="best", fontsize=7.5, frameon=False, handlelength=1.2,
            labelspacing=0.25, borderaxespad=0.2,
        )
        if legend is not None:
            for text in legend.get_texts():
                text.set_color(PALETTE.text_secondary)

        self._relative = relative
        self.set_selected(self._selected)
        self.redraw()

    def set_selected(self, index: int | None) -> None:
        """Ring the currently selected image without redrawing the profile."""
        self._selected = index
        if self._highlight is not None:
            self._highlight.remove()
            self._highlight = None

        if (
            index is not None
            and self._x is not None
            and index < len(self._x)
            and index in self._known
        ):
            value = self._relative[index]
            if value is not None:
                (self._highlight,) = self.axes.plot(
                    [self._x[index]],
                    [value],
                    linestyle="none",
                    marker="o",
                    markersize=10.0,
                    markerfacecolor="none",
                    markeredgecolor=SELECTED_RING,
                    markeredgewidth=1.8,
                    zorder=6,
                )
        self.draw_idle()
