"""A small floating button that pops a widget out into its own window.

Reparents the real widget rather than duplicating it: a popped-out plot or 3D
viewer needs to stay the same live, interactive thing — the same matplotlib
canvas the user can still click into, the same viewer they can still orbit —
not a frozen picture of it. Qt reparents a widget cheaply and correctly (its
own children, event filters and internal state all move with it), so the
pop-out window *is* the original widget, just temporarily living in a
different top-level window; closing that window puts it right back where it
was, in the same layout slot.

Not a right-click menu: the 3D viewers already use a right-click for their
own thing (clearing an in-progress measurement), so a context menu here would
either shadow that or need to coexist awkwardly with it. A small persistent
button is unambiguous and, unlike a menu, visible before the user goes
looking for it.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QGuiApplication, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QBoxLayout,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLayout,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..style import icon_url
from ..theme import METRICS, PALETTE, RADIUS, SPACING


def _locate(widget: QWidget) -> tuple[QLayout, int] | None:
    """The QLayout that directly manages ``widget``, and its index in it.

    ``widget.parentWidget().layout()`` only finds this correctly when the
    widget was added straight to that layout. Most of the cards this button
    sits on build a *nested* layout instead — a QHBoxLayout created on its
    own and installed via ``outer.addLayout(inner)`` rather than owning a
    widget of its own — and a nested layout has no widget to be found
    through ``parentWidget()``; that call lands on the ancestor several
    levels up that owns the outermost layout instead. Reinserting into the
    wrong (outer) layout is exactly what silently broke the Charge page's
    Sectional Slices card after a pop-in: the plot landed appended at the
    end of the *card's* layout instead of back inside the row it shares
    with the plane controls, leaving a gap where it used to sit and the
    plot itself full-width beneath everything else. Searching the whole
    layout tree for whichever (possibly nested) layout actually contains
    the widget is the only way to get this right in general.
    """
    parent = widget.parentWidget()
    top = parent.layout() if parent is not None else None
    if top is None:
        return None
    return _search(top, widget)


def _search(layout: QLayout, widget: QWidget) -> tuple[QLayout, int] | None:
    index = layout.indexOf(widget)
    if index != -1:
        return layout, index
    for i in range(layout.count()):
        sub = layout.itemAt(i).layout()
        if sub is not None:
            found = _search(sub, widget)
            if found is not None:
                return found
    return None


def _grab_image(target: QWidget):
    """A matplotlib canvas renders through its own Agg backend, not Qt's
    paint system, so a plain ``grab()`` only sees whatever was last blitted
    to screen — usually fine, but ``figure.canvas.buffer_rgba()`` after a
    forced draw is what actually reflects the current data. Anything else
    (a 3D viewer) paints through QPainter, where ``grab()`` already is the
    real content."""
    from PySide6.QtGui import QImage

    figure = getattr(target, "figure", None)
    if figure is not None:
        target.draw()
        width, height = figure.canvas.get_width_height()
        buffer = figure.canvas.buffer_rgba()
        image = QImage(buffer, width, height, QImage.Format_RGBA8888).copy()
        return image
    return target.grab().toImage()


class _PlotPopoutToolbar(QWidget):
    """Copy / Export affordances riding on every pop-out window, not only
    the ones a page's own toolbar happens to cover — a plot popped out into
    its own window is otherwise a dead end for getting that image out of the
    app without closing the window and going back to hunt for the page-level
    Copy/Export buttons, which may not even target this specific plot."""

    def __init__(self, target: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._target = target
        # Fixed vertical policy, or this strip competes with the content for
        # the window's spare height. A matplotlib canvas reports
        # Preferred/Preferred rather than Expanding, so a plain QWidget
        # toolbar above it is just as entitled to grow — which is how a
        # 26px button row ended up 240px tall in a 720x540 pop-out, parking
        # the Copy/Export icons a visible inch above the plot they act on.
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACING.sm, SPACING.xs, SPACING.sm, SPACING.xs)
        layout.setSpacing(SPACING.xs)
        layout.addStretch(1)

        copy_button = self._button("copy.svg", "Copy image to clipboard")
        copy_button.clicked.connect(self._copy)
        layout.addWidget(copy_button)

        export_button = self._button("export.svg", "Export image to a file…")
        export_button.clicked.connect(self._export)
        layout.addWidget(export_button)

    def _button(self, icon: str, tooltip: str) -> QPushButton:
        button = QPushButton()
        button.setCursor(Qt.PointingHandCursor)
        button.setFixedSize(26, 26)
        button.setIcon(QIcon(icon_url(icon)))
        button.setToolTip(tooltip)
        button.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none;"
            f" border-radius: {RADIUS.sm}px; }}"
            f"QPushButton:hover {{ background: {PALETTE.border}; }}"
        )
        return button

    def _copy(self) -> None:
        QGuiApplication.clipboard().setImage(_grab_image(self._target))

    def _export(self) -> None:
        figure = getattr(self._target, "figure", None)
        filters = (
            "PNG image (*.png);;SVG vector (*.svg);;PDF document (*.pdf)"
            if figure is not None
            else "PNG image (*.png)"
        )
        path_text, _selected = QFileDialog.getSaveFileName(
            self, "Export Image", "", filters
        )
        if not path_text:
            return
        if figure is not None:
            figure.savefig(path_text, bbox_inches="tight", facecolor="white")
        else:
            _grab_image(self._target).save(path_text)


class _PopoutHost(QDialog):
    """The floating window a widget moves into while popped out."""

    def __init__(self, title: str, target: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        # Not modal: the rest of the application must stay usable, and more
        # than one plot or viewer can be popped out at the same time.
        self.setModal(False)
        self.setWindowFlag(Qt.Window, True)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowCloseButtonHint, True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(_PlotPopoutToolbar(target, self), 0)
        self._layout = layout

    def body(self) -> QVBoxLayout:
        return self._layout


class PopoutButton(QPushButton):
    """A small "open in its own window" affordance.

    Meant to float over the top-right-ish corner of whatever it is placed on
    (the caller positions it — this widget only handles the reparenting).
    Clicking it moves ``target`` into a standalone, resizable window; the
    same click, now on the button riding along inside that window's corner,
    moves it back.
    """

    def __init__(
        self,
        target: QWidget,
        title: str,
        parent: QWidget | None = None,
        flat: bool = False,
        shortcuts: list[tuple[QKeySequence | str, Callable[[], None]]] | None = None,
    ) -> None:
        """``flat=True`` for a button sitting inline in ordinary UI chrome (a
        plot card's header row); the default suits one floating over a 3D
        canvas instead, where a plain icon can vanish against the render
        beneath it and a small opaque pill keeps it legible regardless of
        what's drawn behind it.

        ``shortcuts`` re-installs the page's own keyboard shortcuts onto the
        pop-out window. A ``QShortcut`` parented to the page only fires while
        keyboard focus sits inside that page's widget tree — which the
        popped-out window is no longer part of once its content moves into a
        separate top-level window, so without this, the Editor's tool
        shortcuts (S/R/T/G/D/M, undo/redo/delete) would silently stop
        working the moment the viewer was popped out. Installed with the
        window as the QShortcut's own parent, so they are scoped to (and
        active only while) this specific pop-out is the focused window.
        """
        super().__init__(parent)
        self._target = target
        self._title = title
        self._shortcuts = list(shortcuts) if shortcuts else []
        self._host: _PopoutHost | None = None
        self._origin_parent: QWidget | None = None
        self._origin_layout: QLayout | None = None
        self._origin_index: int = -1
        self._origin_stretch: int = 0

        self.setObjectName("PopoutButton")
        self.setCursor(Qt.PointingHandCursor)
        # The same box as the viewer's tool buttons, so the row of controls
        # along the top of a viewer is one height end to end.
        self.setFixedSize(32, METRICS.viewer_control_height)
        self.setIconSize(QSize(15, 15))
        self.setIcon(QIcon(icon_url("popout.svg")))
        self.setToolTip("Open in its own window")
        if flat:
            self.setStyleSheet(
                f"#PopoutButton {{ background: transparent; border: none;"
                f" border-radius: {RADIUS.sm}px; }}"
                f"#PopoutButton:hover {{ background: {PALETTE.border}; }}"
            )
        else:
            self.setStyleSheet(
                f"#PopoutButton {{ background: rgba(255,255,255,0.90);"
                f" border: 1px solid {PALETTE.border}; border-radius: {RADIUS.sm}px; }}"
                f"#PopoutButton:hover {{ background: rgba(255,255,255,1.0); }}"
            )
        self.clicked.connect(self._toggle)

    def is_popped_out(self) -> bool:
        return self._host is not None

    def _toggle(self) -> None:
        if self._host is None:
            self._pop_out()
        else:
            self._host.close()  # triggers _pop_in via `finished`

    def _pop_out(self) -> None:
        self._origin_parent = self._target.parentWidget()
        located = _locate(self._target)
        if located is not None:
            self._origin_layout, self._origin_index = located
            self._origin_stretch = (
                self._origin_layout.stretch(self._origin_index)
                if isinstance(self._origin_layout, QBoxLayout)
                else 0
            )
        else:
            self._origin_layout, self._origin_index, self._origin_stretch = None, -1, 0

        host = _PopoutHost(self._title, self._target, self.window())
        # Comfortably larger than the embedded size by default — the whole
        # point is more room to work in, not a same-size window.
        host.resize(
            max(720, int(self._target.width() * 1.3) + 40),
            max(540, int(self._target.height() * 1.3) + 40),
        )
        # Stretch 1: the content is what the window's spare height is for,
        # so it starts directly beneath the toolbar and fills the rest.
        host.body().addWidget(self._target, 1)
        host.finished.connect(self._on_host_finished)

        for keys, slot in self._shortcuts:
            shortcut = QShortcut(QKeySequence(keys), host)
            # The default for a QShortcut parented to a top-level window:
            # active whenever that window is the focused one, regardless of
            # which of its child widgets has focus — exactly "these
            # shortcuts work while this pop-out is the window in front."
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(slot)

        self._host = host
        host.show()
        host.raise_()
        host.activateWindow()
        self.setToolTip("Return to the page")

    def _on_host_finished(self, _result: int) -> None:
        self._pop_in()

    def _pop_in(self) -> None:
        if self._host is None:
            return
        host, self._host = self._host, None

        if self._origin_layout is not None:
            index = self._origin_index
            count = self._origin_layout.count()
            if isinstance(self._origin_layout, QBoxLayout):
                # insertWidget's own stretch argument, rather than a
                # separate setStretch call after the fact — the widget has
                # no stretch entry at all until it is back in the layout.
                if 0 <= index <= count:
                    self._origin_layout.insertWidget(
                        index, self._target, self._origin_stretch
                    )
                else:
                    self._origin_layout.addWidget(self._target, self._origin_stretch)
            elif 0 <= index <= count:
                self._origin_layout.insertWidget(index, self._target)
            else:
                self._origin_layout.addWidget(self._target)

            # Reinserting into a nested layout does not always propagate a
            # size-hint change up through every ancestor on its own — the
            # container the layout itself is filed under can keep an old,
            # now-wrong size hint. Walking up and invalidating each one
            # forces a real recompute instead of a stale cached geometry.
            self._origin_layout.invalidate()
            ancestor = self._origin_parent
            while ancestor is not None:
                ancestor.updateGeometry()
                if ancestor.layout() is not None:
                    ancestor.layout().activate()
                ancestor = ancestor.parentWidget()
        elif self._origin_parent is not None:
            self._target.setParent(self._origin_parent)
        self._target.show()

        host.deleteLater()
        self.setToolTip("Open in its own window")


__all__ = ["PopoutButton"]
