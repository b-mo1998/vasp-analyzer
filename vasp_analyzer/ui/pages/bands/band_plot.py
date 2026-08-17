"""The band-structure canvas."""

from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator

from ....models.bands import BandData
from ...plotting import GUIDE_COLOR, PixelMargins, PlotCanvas
from ...theme import PALETTE

BAND_MARGINS = PixelMargins(left=62, right=14, top=12, bottom=42)


def ENERGY_LOCATOR() -> MaxNLocator:  # noqa: N802 - reads as a constant factory
    """Energy ticks on familiar steps (…, 5, 10, 20) rather than 4s and 8s."""
    return MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10], integer=False)


class BandPlot(PlotCanvas):
    """Draws eigenvalues against the k-axis, referenced to a chosen energy.

    Energies stay absolute in the model; the reference is subtracted here, so
    moving the Fermi control never re-reads or mutates the parsed data.
    """

    def __init__(self) -> None:
        super().__init__(BAND_MARGINS)
        self._bands: BandData | None = None
        self._wannier: tuple[np.ndarray, np.ndarray] | None = None
        self._reference = 0.0
        self._ylimits = (-10.0, 10.0)
        self._collections: list[LineCollection] = []
        self._fermi_line = None
        self._configure()

    # -- configuration ------------------------------------------------------

    def _configure(self) -> None:
        axes = self.axes
        axes.set_ylabel("Energy (eV)")
        axes.set_xlabel("k-point index")
        axes.set_xlim(0, 1)
        axes.set_ylim(*self._ylimits)

    # -- data ---------------------------------------------------------------

    def set_bands(self, bands: BandData | None) -> None:
        self._bands = bands
        self._rebuild()

    def set_wannier(self, x: np.ndarray | None, energies: np.ndarray | None) -> None:
        self._wannier = None if x is None or energies is None else (x, energies)
        self._rebuild()

    def set_reference(self, reference: float) -> None:
        """Set the energy plotted as zero (the Fermi level control)."""
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

    def reference(self) -> float:
        return self._reference

    # -- drawing ------------------------------------------------------------

    def _rebuild(self) -> None:
        axes = self.axes
        self._collections.clear()
        axes.clear()
        self._style_axes()
        axes.set_ylabel("Energy (eV)")

        bands = self._bands
        if bands is None:
            axes.set_xticks([])
            axes.set_xlabel("")
            axes.set_ylim(*self._ylimits)
            self.redraw()
            return

        x = bands.x()
        axes.set_xlim(float(x[0]), float(x[-1]) if x[-1] > x[0] else float(x[0]) + 1.0)

        # --- VASP bands ----------------------------------------------------
        # Connecting eigenvalues with a line asserts that consecutive k-points
        # are adjacent in reciprocal space. That is true along a band path and
        # false for a Monkhorst-Pack or Γ-centred mesh, where the list is an
        # unordered set of irreducible points — so a mesh is drawn as points.
        spin_colors = [PALETTE.accent, "#7c3aed"]
        for spin in range(bands.n_spin):
            energies = bands.energies[spin] - self._reference  # (nk, nbands)
            color = spin_colors[min(spin, len(spin_colors) - 1)]

            if bands.is_line_mode:
                segments = [
                    np.column_stack([x, energies[:, band]])
                    for band in range(bands.n_bands)
                ]
                collection = LineCollection(
                    segments,
                    colors=color,
                    linewidths=1.1,
                    zorder=3,
                    capstyle="round",
                )
                axes.add_collection(collection)
                self._collections.append(collection)
            else:
                axes.plot(
                    np.repeat(x[:, None], bands.n_bands, axis=1).ravel(),
                    energies.ravel(),
                    linestyle="none",
                    marker=".",
                    markersize=1.6,
                    color=color,
                    alpha=0.65,
                    zorder=3,
                    rasterized=True,
                )

        # --- Wannier90 interpolation ---------------------------------------
        if self._wannier is not None:
            wx, wenergies = self._wannier
            segments = [
                np.column_stack([wx, wenergies[band] - self._reference])
                for band in range(wenergies.shape[0])
            ]
            collection = LineCollection(
                segments,
                colors=PALETTE.wannier,
                linewidths=1.0,
                linestyles=(0, (4, 2.5)),
                zorder=4,
            )
            axes.add_collection(collection)
            self._collections.append(collection)

        # --- k-axis ---------------------------------------------------------
        if bands.labels:
            positions = [label.position for label in bands.labels]
            axes.set_xticks(positions)
            axes.set_xticklabels([label.label for label in bands.labels])
            for position in positions[1:-1]:
                axes.axvline(
                    position, color=GUIDE_COLOR, linewidth=0.8, zorder=1
                )
            axes.set_xlabel("")
        else:
            axes.set_xlabel(bands.x_axis_title)
            axes.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

        # --- Fermi reference -------------------------------------------------
        axes.axhline(
            0.0,
            color=PALETTE.fermi,
            linewidth=1.0,
            linestyle=(0, (5, 3)),
            zorder=2,
        )

        axes.yaxis.set_major_locator(ENERGY_LOCATOR())
        axes.set_ylim(*self._ylimits)
        self.redraw()
