"""The floating tool cluster that sits over a 3D viewport.

One widget shared by every page with a 3D view, rather than the near-identical
copy each page used to build for itself. Those copies had quietly drifted:
the Editor offered zoom-out and fit, Structure offered zoom-out but not fit,
and MD, NEB and Charge offered neither — so the same viewer gained and lost
controls depending on which tab you were looking at, for no reason a user
could infer. Worse, only the Editor had keyboard shortcuts for its tools at
all, so ``S``/``R``/``T`` silently did nothing on the other four pages.

Measure is the one genuinely page-specific tool. It answers "how far apart
are these two atoms", which is a question about *a* structure — on a
trajectory or a reaction path, where the geometry differs frame to frame, a
measurement is only true of whichever frame happened to be showing when it
was taken. So it is offered where it means something (Structure, Editor) and
withheld where it would mislead, via ``measure=True``.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QPushButton, QWidget

from ..style import icon_url
from ..theme import METRICS, SPACING

# (key, icon, tooltip-without-shortcut, view-mode attribute, shortcut)
#
# Shortcut letters match the Editor's, which is where they already existed:
# S select, R rotate, T translate/pan, M measure. T rather than P for pan
# because P is a poor fit next to "Play" elsewhere and T reads as
# "translate", which is what panning the camera is.
TOOL_SPECS = [
    ("select", "select.svg", "Select atoms", "MODE_SELECT", "S"),
    ("rotate", "rotate.svg", "Rotate the camera", "MODE_ROTATE", "R"),
    ("pan", "pan.svg", "Pan the camera", "MODE_PAN", "T"),
    ("measure", "measure.svg",
     "Measure — click two atoms for a distance, three for an angle",
     "MODE_MEASURE", "M"),
]

ZOOM_STEP = 1.2


class ViewerToolbar(QWidget):
    """Select / rotate / pan (/ measure), zoom in, zoom out and fit."""

    def __init__(
        self,
        view,
        parent: QWidget | None = None,
        *,
        measure: bool = False,
        on_fit: Callable[[], None] | None = None,
    ) -> None:
        """``on_fit`` lets a page supply its own framing.

        A trajectory or reaction-path viewer has to frame the *whole
        sequence*, not just the frame currently on screen — fitting to one
        frame would make the structure jump in and out of view as playback
        moved through it. Pages that need that pass their own callable;
        anything else falls back to the view's own ``fit_view``.
        """
        super().__init__(parent)
        self._view = view
        self._on_fit = on_fit
        self.setObjectName("ViewerToolbar")
        self.setAttribute(Qt.WA_StyledBackground, True)

        row = QHBoxLayout(self)
        # No margins: the strip's background is transparent, so a margin here
        # only pushed its buttons 4px below the orientation bar and pop-out
        # button, which every page places at the same offset. All three now
        # share one top edge and one 28px height.
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING.xs)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QPushButton] = {}

        specs = [s for s in TOOL_SPECS if measure or s[0] != "measure"]
        for index, (key, icon_name, tip, mode_attr, shortcut) in enumerate(specs):
            button = QPushButton()
            button.setIcon(QIcon(icon_url(icon_name)))
            button.setIconSize(QSize(15, 15))
            button.setObjectName("ToolButton")
            button.setCheckable(True)
            button.setFixedSize(32, METRICS.viewer_control_height)
            button.setToolTip(f"{tip}  ({shortcut})")
            button.setCursor(Qt.PointingHandCursor)
            button.setProperty("mode", getattr(view, mode_attr))
            self._group.addButton(button, index)
            row.addWidget(button)
            self._buttons[key] = button
        self._group.idClicked.connect(self._on_tool_clicked)

        # Icons rather than the "＋" "−" "⤢" glyphs these used to be: each of
        # those came from a different fallback font, so they sat at different
        # heights and weights inside identical boxes and read as misaligned.
        for icon_name, tip, slot in (
            ("zoom-in.svg", "Zoom in", lambda: self._view.zoom_by(ZOOM_STEP)),
            ("zoom-out.svg", "Zoom out", lambda: self._view.zoom_by(1 / ZOOM_STEP)),
            ("fit.svg", "Fit the structure in view", self.fit),
        ):
            button = QPushButton()
            button.setIcon(QIcon(icon_url(icon_name)))
            button.setIconSize(QSize(15, 15))
            button.setObjectName("ToolButton")
            button.setFixedSize(32, METRICS.viewer_control_height)
            button.setToolTip(tip)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(slot)
            row.addWidget(button)

        # Through set_tool, not a bare setChecked: setChecked paints a button
        # active without emitting `clicked`, so the view would keep whatever
        # mode it was constructed with while the strip claimed otherwise.
        # That exact mismatch is what made the Editor advertise "Select"
        # while every click actually orbited the camera.
        self.set_tool("rotate")

    # -- tools --------------------------------------------------------------

    def set_tool(self, key: str) -> None:
        """Switch tools, keeping the view and the highlighted button in step."""
        button = self._buttons.get(key)
        if button is None:
            return
        button.setChecked(True)
        self._view.set_mode(button.property("mode"))

    def current_tool(self) -> str:
        for key, button in self._buttons.items():
            if button.isChecked():
                return key
        return ""

    def _on_tool_clicked(self, index: int) -> None:
        button = self._group.button(index)
        if button is not None:
            self._view.set_mode(button.property("mode"))

    def fit(self) -> None:
        if self._on_fit is not None:
            self._on_fit()
        else:
            self._view.fit_view()

    # -- shortcuts ----------------------------------------------------------

    def install_shortcuts(self, owner: QWidget) -> None:
        """Register this toolbar's keys on ``owner`` (the page).

        Qt.WindowShortcut rather than WidgetWithChildrenShortcut: a page's
        sidebar lives in MainWindow's own sidebar stack rather than inside
        the page widget, so a widget-scoped shortcut goes dead the moment
        focus lands there. Window scope still refuses to fire while a
        different page is showing, because Qt requires the shortcut's own
        widget to be visible.
        """
        specs = [
            (shortcut, key)
            for key, _icon, _tip, _mode, shortcut in TOOL_SPECS
            if key in self._buttons
        ]
        for keys, key in specs:
            shortcut = QShortcut(QKeySequence(keys), owner)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(lambda k=key: self.set_tool(k))

        for keys, slot in (
            ("+", lambda: self._view.zoom_by(ZOOM_STEP)),
            ("=", lambda: self._view.zoom_by(ZOOM_STEP)),   # unshifted "+"
            ("-", lambda: self._view.zoom_by(1 / ZOOM_STEP)),
            ("F", self.fit),
        ):
            shortcut = QShortcut(QKeySequence(keys), owner)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(slot)


__all__ = ["ViewerToolbar", "TOOL_SPECS"]
