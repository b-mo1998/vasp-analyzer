"""Keep the mouse wheel scrolling the page instead of editing controls.

Qt's default is that a wheel event over a combo box, a spin box or a slider
changes its value. In a dialog with three widgets that is a convenience. In a
long sidebar of dropdowns and sliders it is a hazard: scrolling down the panel
silently changes whichever control the pointer happened to cross, and on a
page like Charge or Phonons that means a re-render from a value nobody chose.
The page also appears to stick, because the wheel event is consumed by the
control rather than reaching the scroll area behind it.

A matplotlib canvas has the same problem for a different reason — it accepts
every wheel event to forward to its own scroll callbacks, of which this
application registers none — so a plot inside a scrolling panel is a dead
patch the page will not scroll past.

Both are fixed the same way: intercept the wheel event before the control sees
it and hand it to the nearest scrolling ancestor. A control the user has
deliberately focused keeps its wheel behaviour, so keyboard-driven fine
adjustment still works.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QScrollBar,
    QWidget,
)

from ..plotting import PlotCanvas
from .painted_slider import PaintedSlider

# Value-editing controls whose wheel behaviour is a trap inside a scrolling
# panel. Deliberately not QAbstractScrollArea subclasses (tables, text views,
# the combo box's own popup): those scroll their own contents, which is what
# the wheel is for.
GUARDED = (QComboBox, QAbstractSpinBox, QAbstractSlider, PaintedSlider, PlotCanvas)

# A scroll bar is a QAbstractSlider, and a scroll area delivers wheel events to
# its own scroll bars — so guarding them would send the event straight back to
# the area that produced it, forever. They are also the one slider whose whole
# purpose is to respond to the wheel.
EXEMPT = (QScrollBar,)


def _scrolling_ancestor(widget: QWidget) -> QAbstractScrollArea | None:
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QAbstractScrollArea):
            return parent
        parent = parent.parentWidget()
    return None


class ScrollGuard(QObject):
    """Application-wide filter; install once on the QApplication."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._forwarding = False

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt naming
        if self._forwarding:
            return False
        if event.type() != QEvent.Wheel or not isinstance(obj, GUARDED):
            return False
        if isinstance(obj, EXEMPT) or obj.hasFocus():
            return False

        area = _scrolling_ancestor(obj)
        if area is not None:
            # Re-aimed rather than discarded, so the gesture still does the
            # obvious thing: the panel scrolls. The event carries positions
            # relative to the control, which QAbstractScrollArea ignores — it
            # reads only the wheel delta.
            self._forwarding = True
            try:
                QApplication.sendEvent(area.viewport(), event)
            finally:
                self._forwarding = False
            return True

        # Nothing to scroll behind it. Swallowing the event is still right:
        # the alternative is the control quietly changing value under a
        # gesture that meant "move the page", which is the complaint this
        # exists to answer.
        return True


__all__ = ["ScrollGuard"]
