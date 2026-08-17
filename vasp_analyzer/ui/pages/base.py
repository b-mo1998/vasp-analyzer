"""Common page contract.

Every page owns two widgets: the content shown in the main area and the
controls shown in the left sidebar. The shell places them; the page decides
what they contain and how they react to the current calculation.
"""

from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget

from ...models.calculation import Calculation
from ..theme import SPACING


class Page(QWidget):
    """Base class for a page in the application shell."""

    title = "Page"
    subtitle = ""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._calculation: Calculation | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING.lg)
        self._layout = layout

    def body(self) -> QVBoxLayout:
        return self._layout

    # -- shell interface ---------------------------------------------------

    def sidebar(self) -> QWidget:
        """The control panel for this page. Overridden by subclasses."""
        return QWidget()

    def set_calculation(self, calculation: Calculation | None) -> None:
        """Called whenever a directory is opened or closed."""
        self._calculation = calculation

    def calculation(self) -> Calculation | None:
        return self._calculation

    # -- toolbar actions ---------------------------------------------------

    def reset(self) -> None:
        """Restore the page's controls to their defaults."""

    def copy_to_clipboard(self) -> bool:
        """Copy the page's main visual. Returns True when something was copied."""
        return False

    def export(self) -> None:
        """Offer to save the page's content."""
