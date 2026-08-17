"""Export of the Bands page.

Rather than screen-grabbing the widgets, a fresh matplotlib figure is composed
from the current view state. PNG, SVG and PDF then all come from real vector
artists, so an exported figure is publication-quality rather than a screenshot
of one.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from ....core.logging_setup import get_logger
from ....models.bands import BandData
from ....models.dos import DosData
from ...plotting import AXIS_COLOR, GUIDE_COLOR
from ...theme import PALETTE

log = get_logger("ui.bands.export")


@dataclass
class ExportRequest:
    """Everything needed to reproduce the current view off-screen."""

    bands: BandData | None
    dos: DosData | None
    reference: float
    y_min: float
    y_max: float
    dos_xmax: float
    include_dos: bool
    title: str = ""


def build_figure(request: ExportRequest, width: float = 8.0, height: float = 6.0) -> Figure:
    """Compose a figure matching what the page currently shows."""
    figure = Figure(figsize=(width, height), dpi=300, facecolor="white")

    if request.include_dos and request.dos is not None:
        axes_band, axes_dos = figure.subplots(
            1, 2, sharey=True, gridspec_kw={"width_ratios": [3, 1], "wspace": 0.04}
        )
    else:
        axes_band = figure.subplots(1, 1)
        axes_dos = None

    _draw_bands(axes_band, request)
    if axes_dos is not None:
        _draw_dos(axes_dos, request)

    if request.title:
        figure.suptitle(request.title, fontsize=11, y=0.98)

    figure.tight_layout()
    return figure


def _style(axes) -> None:
    for spine in axes.spines.values():
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(0.8)
    axes.tick_params(direction="out", length=3, width=0.8, labelsize=9)


def _draw_bands(axes, request: ExportRequest) -> None:
    _style(axes)
    axes.set_ylabel("Energy (eV)")
    axes.set_ylim(request.y_min, request.y_max)
    axes.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))

    bands = request.bands
    if bands is None:
        return

    x = bands.x()
    axes.set_xlim(float(x[0]), float(x[-1]) if x[-1] > x[0] else float(x[0]) + 1.0)

    colors = [PALETTE.accent, "#7c3aed"]
    for spin in range(bands.n_spin):
        energies = bands.energies[spin] - request.reference
        color = colors[min(spin, len(colors) - 1)]
        if bands.is_line_mode:
            segments = [
                np.column_stack([x, energies[:, band]])
                for band in range(bands.n_bands)
            ]
            axes.add_collection(
                LineCollection(segments, colors=color, linewidths=0.9, zorder=3)
            )
        else:
            axes.plot(
                np.repeat(x[:, None], bands.n_bands, axis=1).ravel(),
                energies.ravel(),
                linestyle="none",
                marker=".",
                markersize=1.0,
                color=color,
                alpha=0.7,
                zorder=3,
            )

    if bands.labels:
        positions = [label.position for label in bands.labels]
        axes.set_xticks(positions)
        axes.set_xticklabels([label.label for label in bands.labels])
        for position in positions[1:-1]:
            axes.axvline(position, color=GUIDE_COLOR, linewidth=0.7, zorder=1)
    else:
        axes.set_xlabel(bands.x_axis_title)

    axes.axhline(
        0.0, color=PALETTE.fermi, linewidth=0.9, linestyle=(0, (5, 3)), zorder=2
    )


def _draw_dos(axes, request: ExportRequest) -> None:
    _style(axes)
    axes.set_xlabel("States / eV")
    dos = request.dos
    if dos is None:
        return

    energies = dos.energies - request.reference
    if dos.is_spin_polarized:
        axes.fill_betweenx(energies, 0.0, dos.total[0], color=PALETTE.accent, alpha=0.14)
        axes.plot(dos.total[0], energies, color=PALETTE.accent, linewidth=0.9)
        axes.fill_betweenx(energies, 0.0, -dos.total[1], color="#7c3aed", alpha=0.14)
        axes.plot(-dos.total[1], energies, color="#7c3aed", linewidth=0.9)
        axes.set_xlim(-request.dos_xmax, request.dos_xmax)
    else:
        axes.fill_betweenx(energies, 0.0, dos.total[0], color=PALETTE.accent, alpha=0.14)
        axes.plot(dos.total[0], energies, color=PALETTE.accent, linewidth=0.9)
        axes.set_xlim(0.0, request.dos_xmax)

    axes.axhline(0.0, color=PALETTE.fermi, linewidth=0.9, linestyle=(0, (5, 3)))
    axes.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))


def save_figure(request: ExportRequest, path: Path) -> None:
    figure = build_figure(request)
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    log.info("Exported figure to %s", path)


def save_band_csv(request: ExportRequest, path: Path) -> None:
    """Write the plotted eigenvalues as CSV, referenced exactly as displayed."""
    bands = request.bands
    if bands is None:
        raise ValueError("no band data to export")

    x = bands.x()
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["k_index", "k_a", "k_b", "k_c", "x_axis"]
        for spin in range(bands.n_spin):
            suffix = "" if bands.n_spin == 1 else f"_spin{spin + 1}"
            header += [f"band{band + 1}{suffix}" for band in range(bands.n_bands)]
        writer.writerow(header)
        writer.writerow(
            [f"# energies in eV relative to {request.reference:.6f} eV"]
        )

        for k in range(bands.n_kpoints):
            row = [
                k,
                f"{bands.kpoints[k, 0]:.8f}",
                f"{bands.kpoints[k, 1]:.8f}",
                f"{bands.kpoints[k, 2]:.8f}",
                f"{x[k]:.8f}",
            ]
            for spin in range(bands.n_spin):
                row += [
                    f"{value:.6f}"
                    for value in (bands.energies[spin, k] - request.reference)
                ]
            writer.writerow(row)
    log.info("Exported band data to %s", path)


def save_dos_csv(request: ExportRequest, path: Path) -> None:
    dos = request.dos
    if dos is None:
        raise ValueError("no DOS data to export")

    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if dos.is_spin_polarized:
            writer.writerow(["energy_eV", "dos_up", "dos_down", "int_up", "int_down"])
            for i in range(dos.n_points):
                writer.writerow(
                    [
                        f"{dos.energies[i] - request.reference:.6f}",
                        f"{dos.total[0, i]:.6f}",
                        f"{dos.total[1, i]:.6f}",
                        f"{dos.integrated[0, i]:.6f}",
                        f"{dos.integrated[1, i]:.6f}",
                    ]
                )
        else:
            writer.writerow(["energy_eV", "dos", "integrated"])
            for i in range(dos.n_points):
                writer.writerow(
                    [
                        f"{dos.energies[i] - request.reference:.6f}",
                        f"{dos.total[0, i]:.6f}",
                        f"{dos.integrated[0, i]:.6f}",
                    ]
                )
    log.info("Exported DOS data to %s", path)
