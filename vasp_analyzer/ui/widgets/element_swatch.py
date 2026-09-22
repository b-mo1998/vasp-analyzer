"""The clickable element-colour dot used in every 3D viewer's legend.

A colour chosen for an element is a display preference that belongs to the
element, not to one calculation — so the picker lives in a shared widget and
writes through :mod:`vasp_analyzer.services.appearance`, and every viewer that
draws atoms resolves its colours from that same place. Changing carbon on the
MD page changes carbon on the Structure page too, which is the behaviour a
user expects from something that reads as a global palette.

``dark`` selects the border treatment: the legends that float over the dark
3D canvases need a light hairline to separate a dark swatch from the panel
behind it, where the Structure page's light-backed legend needs the opposite.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QColorDialog, QLabel, QMenu, QWidget

from ...services import appearance


class ElementSwatch(QLabel):
    """A clickable colour dot for one element.

    Left click opens a colour picker; right click offers a reset to the
    default Jmol-style colour. Both apply immediately and persist across
    sessions and calculations.
    """

    changed = Signal()

    def __init__(
        self,
        element: str,
        size: int = 10,
        dark: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._element = element
        self._size = size
        self._dark = dark
        self.setFixedSize(size, size)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(f"{element}: click to change colour, right-click to reset")
        self._apply_style()

    @property
    def element(self) -> str:
        return self._element

    def _apply_style(self) -> None:
        colour = appearance.effective_color(self._element)
        border = "rgba(255,255,255,0.25)" if self._dark else "rgba(0,0,0,0.15)"
        self.setStyleSheet(
            f"background: {colour}; border-radius: {self._size / 2}px;"
            f" border: 1px solid {border};"
        )

    def refresh(self) -> None:
        """Repaint from the current effective colour — for when the override
        was changed somewhere other than this swatch."""
        self._apply_style()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.button() == Qt.LeftButton:
            self._pick_color()
        elif event.button() == Qt.RightButton:
            self._show_menu(event.globalPosition().toPoint())
        super().mousePressEvent(event)

    def _pick_color(self) -> None:
        current = QColor(appearance.effective_color(self._element))
        chosen = QColorDialog.getColor(current, self, f"Colour for {self._element}")
        if chosen.isValid():
            appearance.set_override(self._element, chosen.name())
            self._apply_style()
            self.changed.emit()

    def _show_menu(self, global_pos) -> None:
        menu = QMenu(self)
        reset = menu.addAction("Reset to Default Colour")
        reset.setEnabled(appearance.has_override(self._element))
        if menu.exec(global_pos) is reset:
            appearance.clear_override(self._element)
            self._apply_style()
            self.changed.emit()
