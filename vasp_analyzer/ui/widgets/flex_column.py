"""A vertical layout that lets a scroll area treat it as freely resizable.

A ``QScrollArea`` with ``widgetResizable`` sizes its content one of two ways.
When the content's layout reports no height-for-width, the content is sized
to the viewport (never below its minimum), so a stretch item inside absorbs
any slack and a scrollbar appears only when the *minimum* no longer fits.
When the layout does report height-for-width -- which a single word-wrapped
``QLabel`` anywhere inside it is enough to cause -- the scroll area instead
grows the content to its *preferred* height-for-width, and a scrollbar
appears as soon as the preferred height exceeds the viewport, even though
the stretch item could have taken the difference.

That second behaviour is what made the NEB info column scroll by ~60px on
a window it plainly fitted into: the "Source Files" caption wraps, so the
whole column was laid out at the energy plot's preferred 300px rather than
letting the plot shrink to what was left.

This layout reports no height-for-width to its owner, so the scroll area
takes the first path. Its own items still wrap correctly -- QBoxLayout
computes per-item heights from their own height-for-width internally, and
that is unaffected by what the layout reports outward.
"""

from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget


class FlexColumn(QVBoxLayout):
    """A ``QVBoxLayout`` whose stretch items win over a wrapped label."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt naming
        return False


__all__ = ["FlexColumn"]
