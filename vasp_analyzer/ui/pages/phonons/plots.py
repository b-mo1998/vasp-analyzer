"""The phonon dispersion and DOS plots.

The two share a frequency axis and are drawn side by side, so their vertical
insets must match exactly or a feature in the DOS would sit at a different
height from the branch that produced it.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal
from matplotlib.ticker import MaxNLocator

from ....models.phonon import (
    IMAGINARY_TOLERANCE_THZ,
    PhononDataset,
    convert_frequency,
)
from ....services.appearance import effective_color
from ...plotting import GUIDE_COLOR, PixelMargins, PlotCanvas
from ...theme import PALETTE

# Top and bottom must agree between the two plots for the shared frequency
# axis to line up.
DISPERSION_MARGINS = PixelMargins(left=64, right=14, top=14, bottom=46)
DOS_MARGINS = PixelMargins(left=16, right=14, top=14, bottom=46)

STABLE_COLOR = PALETTE.accent
IMAGINARY_COLOR = "#dc2626"
TOTAL_DOS_COLOR = "#1a1a1a"


# Per-branch colours, deliberately excluding every red and warm orange.
#
# Red means one thing on this plot -- imaginary, i.e. dynamically unstable --
# and the legend says so. Spreading branch colours around a full hue wheel
# (which is what a cyclic colormap does) inevitably lands one *stable* branch
# on red, at which point the plot contradicts its own legend and the single
# most important thing it has to communicate becomes ambiguous. These hues
# are spaced far enough apart to tell neighbouring branches apart while
# leaving the red end of the spectrum reserved.
BRANCH_COLORS = (
    "#2563eb",  # blue
    "#0891b2",  # cyan
    "#059669",  # emerald
    "#65a30d",  # lime
    "#ca8a04",  # gold
    "#7c3aed",  # violet
    "#c026d3",  # fuchsia
    "#0f766e",  # teal
    "#4338ca",  # indigo
    "#475569",  # slate
)


def _branch_colour(branch: int) -> str:
    """A distinct, never-red colour for one branch.

    Cycles rather than interpolating: a fixed palette keeps every branch a
    colour a person can actually name and match against, and past ten
    branches the eye has long since stopped tracking individual curves by
    hue anyway.
    """
    return BRANCH_COLORS[branch % len(BRANCH_COLORS)]


class DispersionPlot(PlotCanvas):
    """Phonon branches along the q-path, with click-to-select."""

    modePicked = Signal(int, int)      # qpoint index, branch index

    def __init__(self) -> None:
        super().__init__(DISPERSION_MARGINS)
        self._dataset: PhononDataset | None = None
        self._distances: np.ndarray | None = None
        self._selection_line = None
        self._selection_point = None
        self._branch_filter: list[int] | None = None
        self._coloring = "By Stability"
        self._units = "THz"
        self.setMinimumHeight(320)
        self.mpl_connect("button_press_event", self._on_click)

    def show_message(self, message: str) -> None:
        self.axes.clear()
        self._style_axes()
        self._dataset = None
        self._selection_line = None
        self._selection_point = None
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.axes.text(
            0.5, 0.5, message, transform=self.axes.transAxes,
            ha="center", va="center", fontsize=9, color=PALETTE.text_tertiary,
        )
        self.redraw()

    def set_coloring(self, mode: str) -> None:
        """"By Stability" or "By Branch".

        By Stability is the physics-first reading: every branch is drawn in
        one colour, and only the stretches that go imaginary switch to red
        and dash. That answers "is this structure dynamically stable, and
        where does it fail" at a glance, which is the usual question.

        By Branch gives each branch its own colour instead, so a single
        branch can be followed through the crossings and near-degeneracies
        where an all-one-colour plot makes it ambiguous which line continues
        into which. The branch palette deliberately contains no red: red is
        reserved for imaginary in both modes, so an unstable stretch stays
        red and dashed here too and can never be confused with a stable
        branch that merely happened to be assigned a warm colour.
        """
        self._coloring = mode

    def set_branch_filter(self, branches: list[int] | None) -> None:
        self._branch_filter = branches
        if self._dataset is not None:
            self.set_dataset(self._dataset, show_imaginary=self._show_imaginary)

    def set_units(self, units: str) -> None:
        self._units = units

    def set_dataset(self, dataset: PhononDataset, show_imaginary: bool = True) -> None:
        self.axes.clear()
        self._style_axes()
        self._dataset = dataset
        self._show_imaginary = show_imaginary
        self._selection_line = None
        self._selection_point = None

        if not dataset.has_dispersion:
            self.show_message("No dispersion path in this calculation")
            return

        distances = dataset.distances
        self._distances = distances
        matrix = convert_frequency(dataset.frequency_matrix, self._units)
        tolerance = convert_frequency(IMAGINARY_TOLERANCE_THZ, self._units)

        branches = (
            range(matrix.shape[1])
            if self._branch_filter is None
            else [b for b in self._branch_filter if b < matrix.shape[1]]
        )

        by_branch = self._coloring == "By Branch"
        for branch in branches:
            values = matrix[:, branch]
            imaginary = values < -tolerance
            # By Branch: the colour identifies which branch this is. By
            # Stability: one colour for everything real.
            stable_colour = _branch_colour(branch) if by_branch else STABLE_COLOR
            # Red and dashed in *both* modes, without exception. Imaginary is
            # the one thing on this plot that must never be ambiguous, so it
            # keeps the legend's colour even when every other curve is being
            # coloured by something else entirely.
            unstable_colour = IMAGINARY_COLOR

            if imaginary.any() and not show_imaginary:
                # Hide only the imaginary part of the branch, not the branch:
                # dropping it entirely would leave a gap that reads as missing
                # data rather than as a hidden instability.
                values = np.where(imaginary, np.nan, values)
                self.axes.plot(distances, values, color=stable_colour, linewidth=0.9)
                continue

            if imaginary.any():
                # Split so the unstable stretch is drawn distinctly from the
                # rest, rather than colouring a whole branch by its worst point.
                self.axes.plot(
                    distances, np.where(imaginary, np.nan, values),
                    color=stable_colour, linewidth=0.9,
                )
                self.axes.plot(
                    distances, np.where(imaginary, values, np.nan),
                    color=unstable_colour, linewidth=0.9, linestyle=(0, (4, 2)),
                )
            else:
                self.axes.plot(distances, values, color=stable_colour, linewidth=0.9)

        # Zero line: the boundary between stable and unstable.
        self.axes.axhline(
            0.0, color=GUIDE_COLOR, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1
        )

        # High-symmetry guides and labels come from the file, never invented.
        ticks, labels = [], []
        for index, label in dataset.labelled_qpoints():
            position = float(distances[index])
            if ticks and abs(position - ticks[-1]) < 1e-9:
                # Two segments meeting at one point: merge into "X|M" — unless
                # it is a straight-through point like Γ-M-K's M, where both
                # segments name it the same thing and "M|M" would just be the
                # same label written twice.
                if label != labels[-1]:
                    labels[-1] = f"{labels[-1]}|{label}"
                continue
            ticks.append(position)
            labels.append(label)
            self.axes.axvline(position, color=GUIDE_COLOR, linewidth=0.7, zorder=1)
        if ticks:
            self.axes.set_xticks(ticks)
            self.axes.set_xticklabels(labels)
        else:
            self.axes.set_xlabel("Wave vector")

        self.axes.set_ylabel(f"Frequency ({self._units})")
        self.axes.set_xlim(float(distances.min()), float(distances.max()))
        self.axes.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        self.redraw()

    def frequency_limits(self) -> tuple[float, float]:
        return tuple(self.axes.get_ylim())  # type: ignore[return-value]

    def set_selection(self, qpoint_index: int, branch_index: int) -> None:
        """Mark the selected mode without redrawing every branch."""
        dataset = self._dataset
        if dataset is None or self._distances is None:
            return
        if not (0 <= qpoint_index < dataset.n_qpoints):
            return
        x = float(self._distances[qpoint_index])
        y = float(convert_frequency(
            dataset.bands[qpoint_index].frequencies[branch_index], self._units
        ))

        if self._selection_line is None:
            self._selection_line = self.axes.axvline(
                x, color=PALETTE.accent, linewidth=1.2, zorder=5
            )
        else:
            self._selection_line.set_xdata([x, x])

        if self._selection_point is None:
            (self._selection_point,) = self.axes.plot(
                [x], [y], marker="o", markersize=7.0,
                markerfacecolor=PALETTE.accent, markeredgecolor="#ffffff",
                markeredgewidth=1.4, linestyle="none", zorder=6,
            )
        else:
            self._selection_point.set_data([x], [y])
        self.draw_idle()

    def _on_click(self, event) -> None:
        dataset = self._dataset
        if (
            dataset is None
            or self._distances is None
            or event.inaxes is not self.axes
            or event.xdata is None
            or event.ydata is None
        ):
            return

        # Nearest q-point along x, then nearest branch in frequency there —
        # picking on both axes at once would let a click near a crossing land
        # on a branch the user was not pointing at.
        qpoint_index = int(np.argmin(np.abs(self._distances - float(event.xdata))))
        frequencies = convert_frequency(
            dataset.bands[qpoint_index].frequencies, self._units
        )
        branch_index = int(np.argmin(np.abs(frequencies - float(event.ydata))))
        self.modePicked.emit(qpoint_index, branch_index)


class PhononDosPlot(PlotCanvas):
    """Phonon DOS drawn with frequency on the vertical axis."""

    def __init__(self) -> None:
        super().__init__(DOS_MARGINS)
        self._cursor = None
        self._units = "THz"
        self.setMinimumHeight(320)

    def set_units(self, units: str) -> None:
        self._units = units

    def show_message(self, message: str) -> None:
        self.axes.clear()
        self._style_axes()
        self._cursor = None
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.axes.text(
            0.5, 0.5, message, transform=self.axes.transAxes,
            ha="center", va="center", fontsize=9, color=PALETTE.text_tertiary,
        )
        self.redraw()

    def set_dos(self, dos, show_projected: bool, limits: tuple[float, float]) -> None:
        self.axes.clear()
        self._style_axes()
        self._cursor = None

        frequencies = convert_frequency(dos.frequencies, self._units)
        self.axes.plot(
            dos.total, frequencies, color=TOTAL_DOS_COLOR, linewidth=1.1, label="Total"
        )
        self.axes.fill_betweenx(
            frequencies, 0.0, dos.total, color=TOTAL_DOS_COLOR, alpha=0.07
        )

        if show_projected and dos.has_projection:
            for element, values in dos.projected.items():
                self.axes.plot(
                    values, frequencies, linewidth=0.9,
                    color=effective_color(element), label=element,
                )

        self.axes.axhline(
            0.0, color=GUIDE_COLOR, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1
        )
        self.axes.set_xlabel(f"DOS (states/{self._units})")
        self.axes.set_ylim(*limits)
        self.axes.set_xlim(left=0.0)
        self.axes.xaxis.set_major_locator(MaxNLocator(nbins=3))
        # The frequency axis is the dispersion plot's; repeating its numbers
        # here would just be the same scale written twice.
        self.axes.set_yticklabels([])

        legend = self.axes.legend(
            loc="upper right", fontsize=7.5, frameon=False,
            handlelength=1.2, labelspacing=0.25, borderaxespad=0.2,
        )
        if legend is not None:
            for text in legend.get_texts():
                text.set_color(PALETTE.text_secondary)
        self.redraw()

    def set_frequency_cursor(self, frequency: float | None) -> None:
        """Mark the selected mode's frequency, tying the two plots together."""
        if self._cursor is not None:
            self._cursor.remove()
            self._cursor = None
        if frequency is not None:
            self._cursor = self.axes.axhline(
                convert_frequency(frequency, self._units),
                color=PALETTE.accent, linewidth=1.2, zorder=6,
            )
        self.draw_idle()
