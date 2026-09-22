"""The NEB image strip: a small rendering of every image, and navigation.

Thumbnails are rendered once, off-screen, into a cached QPixmap each. Painting
them live would mean running the full structure renderer once per image on every
repaint, which is wasteful for pictures that never change.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ....models.neb import NEBPath
from ....models.structure import StructureData
from ....services.appearance import effective_color
from ....services.bonds import build_sites_and_bonds, unit_cell_edges
from ...theme import PALETTE, RADIUS, SPACING, TYPE
from ..sequence_view import VIEWER_BACKGROUND, lift_for_dark
from ..structure.camera import Camera

THUMB_SIZE = QSize(64, 52)
CLIMB_MARK = "★"


def render_thumbnail(
    structure: StructureData, camera: Camera, size: QSize = THUMB_SIZE
) -> QPixmap:
    """A tiny wireframe-and-dots picture of one image.

    Deliberately simpler than the main viewer — no gradients, no ghosts, no
    depth sorting beyond the atoms themselves. At 64x52 those refinements cost
    time and show nothing.
    """
    pixmap = QPixmap(size)
    pixmap.fill(QColor(VIEWER_BACKGROUND))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)

    width, height = size.width(), size.height()
    bonds = build_sites_and_bonds(structure, include_ghosts=False)
    positions = np.array([site.position for site in bonds.sites])

    edges = unit_cell_edges(structure.lattice)
    edge_points = np.array([point for edge in edges for point in edge])
    screen_edges, _, _ = camera.project(edge_points, width, height)
    painter.setPen(QColor(255, 255, 255, 60))
    for index in range(len(edges)):
        start, end = screen_edges[2 * index], screen_edges[2 * index + 1]
        painter.drawLine(
            int(start[0]), int(start[1]), int(end[0]), int(end[1])
        )

    if positions.size:
        screen, depth, scale = camera.project(positions, width, height)
        order = np.argsort(depth)
        painter.setPen(Qt.NoPen)
        for index in order:
            colour = lift_for_dark(QColor(effective_color(bonds.sites[index].symbol)))
            painter.setBrush(colour)
            radius = max(1.2, float(scale[index]) * 0.10)
            painter.drawEllipse(
                int(screen[index][0] - radius),
                int(screen[index][1] - radius),
                int(radius * 2),
                int(radius * 2),
            )
    painter.end()
    return pixmap


class ImageThumbnail(QPushButton):
    """One clickable image in the strip."""

    def __init__(
        self, index: int, caption: str, climbing: bool, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._index = index
        self._selected = False
        self.setCursor(Qt.PointingHandCursor)
        self.setFlat(True)
        self.setCheckable(True)
        # A QPushButton hosting a layout still sizes itself from its (empty)
        # text unless told otherwise, which squashes the picture to nothing.
        self.setFixedSize(
            THUMB_SIZE.width() + 2 * SPACING.sm,
            THUMB_SIZE.height() + 2 * SPACING.xs + 18,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING.sm, SPACING.xs, SPACING.sm, SPACING.xs)
        layout.setSpacing(3)

        self._picture = QLabel()
        self._picture.setFixedSize(THUMB_SIZE)
        self._picture.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._picture, 0, Qt.AlignHCenter)

        text = f"{CLIMB_MARK} {caption}" if climbing else caption
        self._caption = QLabel(text)
        self._caption.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._caption)

        self._apply_style()

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self._picture.setPixmap(pixmap)

    def set_selected(self, selected: bool) -> None:
        if selected == self._selected:
            return
        self._selected = selected
        self.setChecked(selected)
        self._apply_style()

    def index(self) -> int:
        return self._index

    def _apply_style(self) -> None:
        border = PALETTE.accent if self._selected else PALETTE.border
        background = PALETTE.accent_soft if self._selected else PALETTE.surface
        self.setStyleSheet(
            f"QPushButton {{ border: {2 if self._selected else 1}px solid {border};"
            f" border-radius: {RADIUS.sm}px; background: {background}; }}"
            f"QPushButton:hover {{ border-color: {PALETTE.accent}; }}"
        )
        self._caption.setStyleSheet(
            f"font-size: {TYPE.tiny}px; background: transparent;"
            f" color: {PALETTE.accent if self._selected else PALETTE.text_secondary};"
        )


class ImageStrip(QWidget):
    """The horizontal row of NEB images with previous/next navigation."""

    imageSelected = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._thumbnails: list[ImageThumbnail] = []
        self._selected = 0
        self._count = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING.xs)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(SPACING.sm)
        title = QLabel("Images")
        title.setStyleSheet(f"font-size: {TYPE.body}px; font-weight: 600;")
        header.addWidget(title)

        self._previous = QPushButton("‹")
        self._next = QPushButton("›")
        self._counter = QLabel("— / —")
        self._counter.setStyleSheet(
            f"font-size: {TYPE.small}px; color: {PALETTE.text_secondary};"
        )
        for button in (self._previous, self._next):
            button.setFixedSize(24, 22)
            button.setCursor(Qt.PointingHandCursor)
            button.setStyleSheet(
                f"QPushButton {{ border: 1px solid {PALETTE.border};"
                f" border-radius: {RADIUS.sm}px; background: {PALETTE.surface};"
                f" color: {PALETTE.text}; font-size: 13px; padding-bottom: 2px; }}"
                f"QPushButton:hover:enabled {{ background: {PALETTE.surface_hover}; }}"
                f"QPushButton:disabled {{ color: {PALETTE.disabled_text}; }}"
            )
        self._previous.clicked.connect(lambda: self.select(self._selected - 1))
        self._next.clicked.connect(lambda: self.select(self._selected + 1))

        header.addWidget(self._previous)
        header.addWidget(self._counter)
        header.addWidget(self._next)
        header.addStretch(1)
        layout.addLayout(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFixedHeight(THUMB_SIZE.height() + 2 * SPACING.xs + 18 + 14)

        self._row_holder = QWidget()
        self._row = QHBoxLayout(self._row_holder)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(SPACING.xs)
        self._row.addStretch(1)
        scroll.setWidget(self._row_holder)
        layout.addWidget(scroll)

    # -- population -------------------------------------------------------------

    def clear(self) -> None:
        for thumbnail in self._thumbnails:
            self._row.removeWidget(thumbnail)
            thumbnail.setParent(None)
            thumbnail.deleteLater()
        self._thumbnails = []
        self._count = 0
        self._counter.setText("— / —")

    def set_path(self, path: NEBPath, camera: Camera) -> None:
        self.clear()
        self._count = path.n_images

        for position, image in enumerate(path.images):
            caption = image.folder
            if position == 0:
                caption = f"{caption} (Initial)"
            elif position == path.n_images - 1:
                caption = f"{caption} (Final)"

            thumbnail = ImageThumbnail(
                position, caption, climbing=position == path.climbing_image
            )
            if image.structure is not None:
                thumbnail.set_pixmap(render_thumbnail(image.structure, camera))
            thumbnail.setToolTip(
                f"Image {image.folder}"
                + (f" — {image.note}" if image.note else "")
            )
            thumbnail.clicked.connect(
                lambda _checked=False, i=position: self.imageSelected.emit(i)
            )
            self._row.insertWidget(self._row.count() - 1, thumbnail)
            self._thumbnails.append(thumbnail)

        self.select(0, notify=False)

    def select(self, index: int, notify: bool = True) -> None:
        if self._count == 0:
            return
        index = max(0, min(self._count - 1, index))
        self._selected = index
        for thumbnail in self._thumbnails:
            thumbnail.set_selected(thumbnail.index() == index)
        self._counter.setText(f"{index + 1} / {self._count}")
        self._previous.setEnabled(index > 0)
        self._next.setEnabled(index < self._count - 1)
        if notify:
            self.imageSelected.emit(index)

    def selected(self) -> int:
        return self._selected
