"""The band-structure canvas."""

from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator
from PySide6.QtWidgets import QToolTip

from ....models.bands import BandData, BandGap
from ....parsers.procar import ProcarData
from ...plotting import GUIDE_COLOR, PixelMargins, PlotCanvas
from ...theme import PALETTE
from .projection import ProjectionOverlay

# Orbitals under this fraction of the band's total weight at a point are
# noise, not signal -- omitted from the hover breakdown rather than listed
# at "0.4%".
BREAKDOWN_MIN_FRACTION = 0.03
BREAKDOWN_MAX_ROWS = 4

# Distinct from every other colour already on this plot (accent blue for
# spin-up, violet for spin-down, orange/amber Fermi guide, grey Wannier
# dashes) so the markers read as their own thing at a glance.
VBM_COLOR = "#16a34a"   # green — occupied edge
CBM_COLOR = "#dc2626"   # red — unoccupied edge

# The bottom inset clears the k-point labels *and* the axis title beneath
# them; at 42px a title like "Wave vector (Å⁻¹)" was cut in half by the
# canvas edge. DOS_MARGINS must match, or the shared energy axis skews.
BAND_MARGINS = PixelMargins(left=62, right=14, top=12, bottom=54)

# Requested darker/larger than the app-wide plot defaults (9px, mid-grey —
# see plotting.py's TICK_COLOR): the energy-axis numbers and the k-point
# labels here are the primary reading surface of this page, not incidental
# axis furniture, so they get their own override rather than inheriting the
# global tick style shared by every other plot in the app. 10.5 is the
# global 9px tick size +15%; 11.5 is the k-labels' own 10px (set on an
# earlier pass) with another +15% on top of that.
TICK_LABEL_SIZE = 10.5
KLABEL_SIZE = 11.5


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
        self._projection: ProjectionOverlay | None = None
        self._band_gap: BandGap | None = None
        self._show_band_gap = True
        self._reference = 0.0
        self._ylimits = (-10.0, 10.0)
        self._collections: list[LineCollection] = []
        self._fermi_line = None
        self._procar: ProcarData | None = None
        self._species: list[str] | None = None
        self._hover_enabled = True
        self._configure()
        self.mpl_connect("motion_notify_event", self._on_hover)
        self.mpl_connect("figure_leave_event", self._on_leave)

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

    def set_projection(self, overlay: ProjectionOverlay | None) -> None:
        """Show (or clear) a fatbands overlay — coloured, sized markers layered
        over the base band lines, e.g. from PROCAR element projection."""
        self._projection = overlay
        self._rebuild()

    def set_band_gap(self, gap: BandGap | None) -> None:
        """The VBM/CBM markers — same gap the sidebar's own readout reports,
        drawn where it actually sits on the k-axis rather than only stated
        as a number, so a direct-vs-indirect gap is visible at a glance."""
        self._band_gap = gap
        self._rebuild()

    def set_band_gap_visible(self, visible: bool) -> None:
        self._show_band_gap = visible
        self._rebuild()

    def set_orbital_source(
        self, procar: ProcarData | None, species: list[str] | None
    ) -> None:
        """PROCAR weights backing the hover tooltip's orbital breakdown.

        Kept separate from ``set_projection`` -- the breakdown should be
        available under the cursor whether or not a fatbands overlay is
        currently switched on, so it is not tied to that state.
        """
        self._procar = procar
        self._species = species

    def set_hover_enabled(self, enabled: bool) -> None:
        self._hover_enabled = enabled
        if not enabled:
            QToolTip.hideText()

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
        axes.tick_params(axis="both", colors=PALETTE.text, labelsize=TICK_LABEL_SIZE)
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
        #
        # With a projection overlay active, the base bands step back to a thin
        # grey skeleton so the coloured, weight-sized dots read as the primary
        # signal — the standard "fatbands" look — rather than competing with a
        # bold accent-blue line underneath them.
        has_projection = self._projection is not None
        spin_colors = [PALETTE.accent, "#7c3aed"]
        base_alpha = 0.35 if has_projection else 1.0
        for spin in range(bands.n_spin):
            energies = bands.energies[spin] - self._reference  # (nk, nbands)
            color = GUIDE_COLOR if has_projection else spin_colors[min(spin, len(spin_colors) - 1)]

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
                    alpha=base_alpha,
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
                    alpha=0.65 * base_alpha,
                    zorder=3,
                    rasterized=True,
                )

        # --- Projection overlay (fatbands) ----------------------------------
        if has_projection:
            for spin in range(min(bands.n_spin, len(self._projection.colors))):
                energies = bands.energies[spin] - self._reference
                X = np.repeat(x[:, None], bands.n_bands, axis=1)
                axes.scatter(
                    X.ravel(),
                    energies.ravel(),
                    s=self._projection.sizes[spin].ravel(),
                    c=self._projection.colors[spin].reshape(-1, 4),
                    zorder=5,
                    linewidths=0,
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
            # High-symmetry point names (Γ, X, M, ...), not ordinary tick
            # numbers — sized and coloured to stand out from the rest of the
            # axis furniture, per KLABEL_SIZE above.
            axes.set_xticklabels(
                [label.label for label in bands.labels],
                fontsize=KLABEL_SIZE,
                color=PALETTE.text,
            )
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

        # --- VBM / CBM markers -----------------------------------------------
        # The same numbers the sidebar's band-gap readout already reports,
        # placed on the plot at the k-point they actually occur at — direct
        # vs. indirect is then something you *see* (the two markers share an
        # x position or don't) rather than only something you read.
        gap = self._band_gap
        if (
            self._show_band_gap
            and gap is not None
            and 0 <= gap.vbm_kpoint < len(x)
            and 0 <= gap.cbm_kpoint < len(x)
        ):
            vbm_xy = (float(x[gap.vbm_kpoint]), gap.vbm - self._reference)
            cbm_xy = (float(x[gap.cbm_kpoint]), gap.cbm - self._reference)
            if bands.is_line_mode and not gap.is_direct:
                # A path between the two only means something when the
                # k-axis itself is a path — a mesh's points are independent,
                # so no connector is drawn for one.
                axes.plot(
                    [vbm_xy[0], cbm_xy[0]],
                    [vbm_xy[1], cbm_xy[1]],
                    color=PALETTE.text_secondary,
                    linewidth=1.0,
                    linestyle=(0, (1, 1.5)),
                    zorder=6,
                )
            for xy, color, label in (
                (vbm_xy, VBM_COLOR, "VBM"),
                (cbm_xy, CBM_COLOR, "CBM"),
            ):
                axes.scatter(
                    [xy[0]],
                    [xy[1]],
                    marker="o",
                    s=48,
                    facecolor=color,
                    edgecolor="white",
                    linewidths=1.1,
                    zorder=7,
                )

        axes.yaxis.set_major_locator(ENERGY_LOCATOR())
        axes.set_ylim(*self._ylimits)
        self.redraw()

    # -- hover tooltip --------------------------------------------------------

    def _on_leave(self, _event) -> None:
        QToolTip.hideText()

    def _on_hover(self, event) -> None:
        bands = self._bands
        if (
            not self._hover_enabled
            or bands is None
            or event.inaxes is not self.axes
            or event.xdata is None
            or event.ydata is None
            or event.guiEvent is None
        ):
            QToolTip.hideText()
            return

        x = bands.x()
        k_index = int(np.argmin(np.abs(x - event.xdata)))

        # The nearest band in energy at that k-point, searched across every
        # spin channel -- a spin-down band can be the closer one even while
        # hovering the spin-up line, and the tooltip should report whichever
        # is actually under the cursor.
        best: tuple[float, int, int] | None = None
        for spin in range(bands.n_spin):
            energies = bands.energies[spin, k_index] - self._reference
            band_index = int(np.argmin(np.abs(energies - event.ydata)))
            distance = abs(float(energies[band_index] - event.ydata))
            if best is None or distance < best[0]:
                best = (distance, spin, band_index)
        distance, spin, band_index = best

        # Only fire within a tolerance scaled to the visible energy range, so
        # hovering empty space between bands stays quiet instead of always
        # snapping to the nearest line.
        y_range = max(self._ylimits[1] - self._ylimits[0], 1e-9)
        if distance > 0.03 * y_range:
            QToolTip.hideText()
            return

        energy_abs = float(bands.energies[spin, k_index, band_index])
        lines = [self._k_point_label(k_index, bands), f"Band {band_index + 1}"]
        if bands.n_spin > 1:
            lines.append("Spin up" if spin == 0 else "Spin down")
        lines.append(f"E = {energy_abs - self._reference:+.3f} eV")
        lines.append(f"({energy_abs:.3f} eV absolute)")

        breakdown = self._orbital_breakdown(spin, k_index, band_index)
        if breakdown:
            lines.append("")
            lines.extend(breakdown)

        QToolTip.showText(
            event.guiEvent.globalPosition().toPoint(), "\n".join(lines), self
        )

    def _k_point_label(self, k_index: int, bands: BandData) -> str:
        for point_label in bands.labels:
            if point_label.index == k_index:
                return f"k-point {k_index} ({point_label.label})"
        return f"k-point {k_index}"

    def _orbital_breakdown(self, spin: int, k_index: int, band_index: int) -> list[str]:
        """Per-orbital PROCAR weight at one (spin, k-point, band), summed over
        ions and grouped by element -- the top few contributors, as
        "Element orbital: NN%" lines, dominant contributor first."""
        procar = self._procar
        if procar is None or self._species is None or spin >= procar.n_spin:
            return []
        if procar.n_kpoints != self._bands.n_kpoints or procar.n_bands != self._bands.n_bands:
            return []
        if len(self._species) != procar.n_ions:
            return []

        # (n_ions, n_orbitals)
        weights = procar.ion_weights[spin, k_index, band_index]
        total = float(weights.sum())
        if total <= 1e-9:
            return ["No PROCAR weight at this point"]

        rows: list[tuple[float, str]] = []
        for element in sorted(set(self._species)):
            mask = [s == element for s in self._species]
            per_orbital = weights[mask].sum(axis=0)  # (n_orbitals,)
            for orbital, value in zip(procar.orbitals, per_orbital):
                fraction = float(value) / total
                if fraction >= BREAKDOWN_MIN_FRACTION:
                    rows.append((fraction, f"{element} {orbital}"))

        rows.sort(key=lambda row: row[0], reverse=True)
        return [f"{name}: {fraction * 100:.0f}%" for fraction, name in rows[:BREAKDOWN_MAX_ROWS]]
