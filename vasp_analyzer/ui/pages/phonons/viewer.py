"""The eigenvector viewer: a structure that oscillates, with displacement arrows.

Built on :class:`SequenceView`, which already animates a precomputed sequence of
positions by index. That is exactly the shape a phonon animation has — one
oscillation sampled at N phases — so the frames are generated once when the
mode changes and playback is then just an index advancing on a timer. Nothing
is re-meshed or rebuilt per frame.

Arrows are drawn through the base class's ``_extra_primitives`` hook so they
take part in the same back-to-front depth sort as atoms and bonds; an arrow on
an atom at the back of the cell is then correctly hidden behind atoms in front
of it.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from ..sequence_view import SequenceView

ARROW_COLOR = "#4d8dff"
# Arrows shorter than this many pixels are noise rather than information.
MIN_ARROW_PIXELS = 4.0


class PhononView(SequenceView):
    """Animated structure with per-atom displacement vectors."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frames: np.ndarray | None = None       # (n_phases, n_atoms, 3)
        self._displacements: np.ndarray | None = None
        self._show_vectors = True
        self._vector_style = "Arrows"
        self._vector_scale = 1.0

    # -- mode data ---------------------------------------------------------

    def set_frames(self, frames: np.ndarray | None) -> None:
        """The full oscillation, as cartesian positions per phase."""
        self._frames = None if frames is None else np.asarray(frames, dtype=float)
        self._frame = 0
        if self._frames is not None:
            # SequenceView draws whichever frame index is current, straight
            # from this array — no scene rebuild between frames.
            self.set_path(self._frames)
        else:
            self.set_path(None)
        self.update()

    def set_displacements(self, displacements: np.ndarray | None) -> None:
        self._displacements = (
            None if displacements is None else np.asarray(displacements, dtype=float)
        )
        self.update()

    def set_vector_options(
        self, show: bool, style: str = "Arrows", scale: float = 1.0
    ) -> None:
        self._show_vectors = show
        self._vector_style = style
        self._vector_scale = float(scale)
        self.update()

    def advance(self) -> None:
        if self._frames is None or len(self._frames) == 0:
            return
        self._frame = (self._frame + 1) % len(self._frames)
        self.update()

    def set_phase_index(self, index: int) -> None:
        if self._frames is None or len(self._frames) == 0:
            return
        self._frame = int(index) % len(self._frames)
        self.update()

    @property
    def phase_index(self) -> int:
        return self._frame

    @property
    def n_phases(self) -> int:
        return 0 if self._frames is None else len(self._frames)

    # -- painting ----------------------------------------------------------

    def _extra_primitives(self) -> list[tuple[float, str, tuple]]:
        items = list(super()._extra_primitives())
        if not self._show_vectors or self._vector_style == "Hidden":
            return items
        if self._displacements is None or self._bonds is None:
            return items

        # One primitive per atom, depth-sorted with everything else so an
        # arrow behind the cell does not paint over the atoms in front of it.
        screen, depth, _scale = self._project_sites()
        for index, site in enumerate(self._bonds.sites):
            if site.is_ghost or site.index >= len(self._displacements):
                continue
            items.append((float(depth[index]) + 0.01, "vector", (index, site.index)))
        return items

    def _draw_extra(self, painter: QPainter, kind: str, payload: tuple) -> None:
        if kind != "vector":
            super()._draw_extra(painter, kind, payload)
            return
        if self._displacements is None or self._bonds is None:
            return

        site_index, atom_index = payload
        displacement = self._displacements[atom_index] * self._vector_scale
        site = self._bonds.sites[site_index]

        # Project the arrow's tail and head separately rather than scaling a
        # screen-space direction: under perspective the two ends sit at
        # different depths, and a single projected direction would point
        # slightly wrong for any atom away from the centre.
        points = np.array([site.position, site.position + displacement])
        projected, _depth, _scale = self._camera.project(
            points, self.width(), self.height()
        )
        tail = QPointF(float(projected[0][0]), float(projected[0][1]))
        head = QPointF(float(projected[1][0]), float(projected[1][1]))

        delta = np.array([head.x() - tail.x(), head.y() - tail.y()])
        length = float(np.linalg.norm(delta))
        if length < MIN_ARROW_PIXELS:
            return

        colour = QColor(ARROW_COLOR)
        pen = QPen(colour, 1.8)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawLine(tail, head)

        if self._vector_style != "Arrows":
            return

        unit = delta / length
        normal = np.array([-unit[1], unit[0]])
        size = min(7.0, length * 0.4)
        base = np.array([head.x(), head.y()]) - unit * size
        left = base + normal * size * 0.5
        right = base - normal * size * 0.5
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(colour))
        painter.drawPolygon(
            QPolygonF(
                [
                    head,
                    QPointF(float(left[0]), float(left[1])),
                    QPointF(float(right[0]), float(right[1])),
                ]
            )
        )
