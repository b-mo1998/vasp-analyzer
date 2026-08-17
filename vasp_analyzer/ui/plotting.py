"""Matplotlib canvas base class and shared plot styling.

Two things matter here beyond looks:

* **Axis alignment.** The band plot and the DOS plot live in separate cards, but
  their Fermi lines must sit on the same screen row. Fractional margins cannot
  guarantee that, so margins are specified in *pixels* and converted to
  fractions against the live canvas height on every resize. Give two canvases
  the same height and the same pixel margins and their axes occupy identical
  rows.
* **Cheap updates.** Changing an axis limit calls ``set_ylim`` and a deferred
  redraw rather than rebuilding the artists, which keeps slider dragging smooth.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtCore import QSize  # noqa: E402
from PySide6.QtGui import QColor, QPalette  # noqa: E402

from .theme import PALETTE, TYPE  # noqa: E402

AXIS_COLOR = "#c8c8cc"
TICK_COLOR = "#8a8a8f"
LABEL_COLOR = "#3a3a3c"
GUIDE_COLOR = "#d4d4d8"


@dataclass(frozen=True)
class PixelMargins:
    """Axes insets measured in device-independent pixels."""

    left: int = 62
    right: int = 14
    top: int = 12
    bottom: int = 42


def apply_plot_style() -> None:
    """Global matplotlib defaults matching the application's typography."""
    matplotlib.rcParams.update(
        {
            "font.family": TYPE.plot_family,
            "font.size": 9.5,
            "axes.linewidth": 0.8,
            "axes.edgecolor": AXIS_COLOR,
            "axes.labelcolor": LABEL_COLOR,
            "axes.labelsize": 10,
            "xtick.color": TICK_COLOR,
            "ytick.color": TICK_COLOR,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "figure.facecolor": PALETTE.surface,
            "axes.facecolor": PALETTE.surface,
            "savefig.facecolor": PALETTE.surface,
            "path.simplify": True,
            "path.simplify_threshold": 0.6,
            "agg.path.chunksize": 10000,
        }
    )


class PlotCanvas(FigureCanvasQTAgg):
    """A themed matplotlib canvas with pixel-accurate margins."""

    def __init__(self, margins: PixelMargins | None = None) -> None:
        figure = Figure(figsize=(4, 3), dpi=100, facecolor=PALETTE.surface)
        super().__init__(figure)
        self._margins = margins or PixelMargins()
        self.axes = figure.add_subplot(111)
        self._style_axes()
        self.setMinimumSize(QSize(120, 120))

        # The canvas paints its own white; stop Qt drawing a grey behind it.
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(PALETTE.surface))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

    # -- styling ------------------------------------------------------------

    def _style_axes(self) -> None:
        axes = self.axes
        for spine in axes.spines.values():
            spine.set_color(AXIS_COLOR)
            spine.set_linewidth(0.8)
        axes.tick_params(direction="out", length=3, width=0.8, pad=4)
        axes.set_axisbelow(True)

    # -- geometry -----------------------------------------------------------

    def set_margins(self, margins: PixelMargins) -> None:
        self._margins = margins
        self._apply_margins()

    def _apply_margins(self) -> None:
        width = max(1.0, float(self.figure.get_figwidth() * self.figure.dpi))
        height = max(1.0, float(self.figure.get_figheight() * self.figure.dpi))
        ratio = float(self.devicePixelRatioF() or 1.0)
        m = self._margins

        left = (m.left * ratio) / width
        right = 1.0 - (m.right * ratio) / width
        bottom = (m.bottom * ratio) / height
        top = 1.0 - (m.top * ratio) / height

        # Degenerate geometry during early layout would raise; skip instead.
        if right <= left or top <= bottom:
            return
        self.figure.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._apply_margins()

    # -- convenience --------------------------------------------------------

    def redraw(self) -> None:
        self._apply_margins()
        self.draw_idle()
