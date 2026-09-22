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


# Room reserved beside the y tick labels for the tick marks, the label pad and
# the axis title.
AXIS_LABEL_ALLOWANCE = 26.0


class PlotCanvas(FigureCanvasQTAgg):
    """A themed matplotlib canvas with pixel-accurate margins."""

    def __init__(self, margins: PixelMargins | None = None) -> None:
        figure = Figure(figsize=(4, 3), dpi=100, facecolor=PALETTE.surface)
        super().__init__(figure)
        self._margins = margins or PixelMargins()
        # Widened at draw time when the y tick labels turn out not to fit;
        # see _fit_left_margin.
        self._extra_left = 0.0
        # FigureCanvasQTAgg.sizeHint() reports the figure's *current* pixel
        # size, which the base class's resizeEvent keeps in sync with
        # whatever size the canvas is actually drawn at. That is fine while
        # it only ever grows to fill a layout slot — the bug is that popping
        # this canvas out into a much larger standalone window (see
        # widgets/popout.py) grows the figure to match, and the grown size
        # is never un-done on the way back: sizeHint keeps reporting the
        # popped-out size forever, so the card it lives in asks its layout
        # for more room than it used to on every trip through the pop-out,
        # compounding a little further each time. A fixed preferred size,
        # independent of whatever size the figure happens to be drawn at
        # right now, is what makes the widget size back down correctly once
        # it's back in its normal (stretch-driven) layout slot.
        self._preferred_size = QSize(int(figure.get_figwidth() * figure.dpi),
                                      int(figure.get_figheight() * figure.dpi))
        self.axes = figure.add_subplot(111)
        self._style_axes()
        self.setMinimumSize(QSize(120, 120))

        # The canvas paints its own white; stop Qt drawing a grey behind it.
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(PALETTE.surface))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        return self._preferred_size

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        return self.minimumSize()

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

        left = ((m.left + self._extra_left) * ratio) / width
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
        self._fit_left_margin()
        self.draw_idle()

    def _fit_left_margin(self) -> None:
        """Widen the left inset when the y tick labels do not fit inside it.

        A fixed inset is a guess about how many characters a tick label will
        need, and the guess fails exactly where it matters most: an MD total
        energy of -458.85 eV rendered inside a 52px inset loses its leading
        digits and reads as "58.85", which is not a smaller number but a
        different one. Measuring the labels that were actually drawn is the
        only way to be right for every dataset.
        """
        try:
            renderer = self.figure.canvas.get_renderer()
        except (AttributeError, RuntimeError):
            return

        widest = 0.0
        for label in self.axes.get_yticklabels():
            if not label.get_text():
                continue
            try:
                extent = label.get_window_extent(renderer=renderer)
            except (RuntimeError, ValueError):
                continue
            widest = max(widest, float(extent.width))
        if widest <= 0.0:
            return

        ratio = float(self.devicePixelRatioF() or 1.0)
        # Tick marks, the label-to-axis pad and the y-axis title all live in
        # the same inset, so the labels alone must not consume the whole of it.
        needed = widest / ratio + AXIS_LABEL_ALLOWANCE
        extra = max(0.0, needed - self._margins.left)
        # Only ever grow, and only in visible steps, so a redraw triggered by
        # the adjustment cannot oscillate.
        if extra > self._extra_left + 1.0:
            self._extra_left = extra
            self._apply_margins()
