"""The density-of-states canvas.

Energy runs vertically so it can share the band plot's axis. Its pixel margins
deliberately match the band plot's top and bottom, which is what keeps the two
Fermi lines on the same row.
"""

from __future__ import annotations

import numpy as np
from matplotlib.ticker import MaxNLocator

from ....models.dos import DosData
from ...plotting import PixelMargins, PlotCanvas
from ...theme import PALETTE

# Top and bottom must equal BAND_MARGINS for the shared energy axis to align.
DOS_MARGINS = PixelMargins(left=40, right=14, top=12, bottom=42)


class DosPlot(PlotCanvas):
    """Total DOS drawn vertically, sharing the band plot's energy limits."""

    def __init__(self) -> None:
        super().__init__(DOS_MARGINS)
        self._dos: DosData | None = None
        self._reference = 0.0
        self._ylimits = (-10.0, 10.0)
        self._xmax = 8.0
        self.axes.set_xlabel("States / eV")

    # -- data ---------------------------------------------------------------

    def set_dos(self, dos: DosData | None) -> None:
        self._dos = dos
        self._rebuild()

    def set_reference(self, reference: float) -> None:
        if abs(reference - self._reference) < 1e-12:
            return
        self._reference = float(reference)
        self._rebuild()

    def set_ylimits(self, low: float, high: float) -> None:
        if high <= low:
            return
        self._ylimits = (float(low), float(high))
        self.axes.set_ylim(low, high)
        self.redraw()

    def set_xmax(self, xmax: float) -> None:
        if xmax <= 0:
            return
        self._xmax = float(xmax)
        self._apply_xlimits()
        self.redraw()

    def suggest_xmax(self) -> float | None:
        """Peak DOS inside the current energy window, for auto-scaling."""
        if self._dos is None:
            return None
        low, high = self._ylimits
        peak = self._dos.max_in_window(low + self._reference, high + self._reference)
        return float(peak)

    # -- drawing ------------------------------------------------------------

    def _apply_xlimits(self) -> None:
        if self._dos is not None and self._dos.is_spin_polarized:
            self.axes.set_xlim(-self._xmax, self._xmax)
        else:
            self.axes.set_xlim(0.0, self._xmax)

    def _rebuild(self) -> None:
        axes = self.axes
        axes.clear()
        self._style_axes()
        axes.set_xlabel("States / eV")
        axes.tick_params(labelleft=True)

        dos = self._dos
        if dos is None:
            axes.set_xticks([])
            axes.set_ylim(*self._ylimits)
            self.redraw()
            return

        energies = dos.energies - self._reference

        if dos.is_spin_polarized:
            # Spin down is mirrored for display only; the data stays positive.
            axes.fill_betweenx(
                energies, 0.0, dos.total[0], color=PALETTE.accent, alpha=0.14, zorder=2
            )
            axes.plot(dos.total[0], energies, color=PALETTE.accent, linewidth=1.0, zorder=3)
            axes.fill_betweenx(
                energies, 0.0, -dos.total[1], color="#7c3aed", alpha=0.14, zorder=2
            )
            axes.plot(-dos.total[1], energies, color="#7c3aed", linewidth=1.0, zorder=3)
            axes.axvline(0.0, color=PALETTE.border, linewidth=0.8, zorder=1)
        else:
            axes.fill_betweenx(
                energies, 0.0, dos.total[0], color=PALETTE.accent, alpha=0.14, zorder=2
            )
            axes.plot(dos.total[0], energies, color=PALETTE.accent, linewidth=1.0, zorder=3)

        axes.axhline(
            0.0, color=PALETTE.fermi, linewidth=1.0, linestyle=(0, (5, 3)), zorder=4
        )

        axes.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        axes.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        self._apply_xlimits()
        axes.set_ylim(*self._ylimits)
        self.redraw()


def nice_upper_bound(value: float) -> float:
    """Round a peak value up to a readable slider maximum."""
    if value <= 0 or not np.isfinite(value):
        return 10.0
    magnitude = 10 ** np.floor(np.log10(value))
    for step in (1.0, 2.0, 2.5, 5.0, 10.0):
        candidate = step * magnitude
        if candidate >= value:
            return float(candidate)
    return float(10 * magnitude)
