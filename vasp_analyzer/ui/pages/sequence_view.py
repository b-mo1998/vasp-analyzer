"""A structure viewer for a *sequence* of frames — MD trajectories and NEB bands.

Both pages show the same thing: one structure out of an ordered series, with a
stable camera, optional motion trails and an optionally tracked atom. Rather
than grow a second renderer, this extends :class:`StructureView` with the few
behaviours a sequence needs:

* **Stable framing.** The camera is fitted once for the whole sequence and then
  left alone. Refitting per frame makes a trajectory appear to breathe.
* **Trails.** The recent path of each atom, drawn from *unwrapped* coordinates
  so a boundary crossing never draws a line straight across the cell, then
  translated so the trail meets the atom exactly where it is displayed.
* **Tracking.** Keeping a chosen atom centred while the sequence plays.

Bond topology is recomputed per frame for ordinary cell sizes — bonds really do
stretch and break during MD, and freezing them would misrepresent that — and
frozen above :data:`BOND_REBUILD_LIMIT` atoms, where the per-frame neighbour
search would cost more than the accuracy is worth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ...models.structure import StructureData
from ...services.appearance import effective_color
from ...services.bonds import unit_cell_edges
from ...services.slab import VacuumInfo, framing_corners
from ..theme import PALETTE
from .structure.viewer import StructureView

# Above this many atoms, bonds are computed once and only their endpoints move
# with the atoms; below it, they are rebuilt every frame.
BOND_REBUILD_LIMIT = 600

VIEWER_BACKGROUND = "#26272b"
SEQUENCE_CELL_COLOR = "#b9bec6"
TRAIL_WIDTH = 1.4
TRAIL_MIN_ALPHA = 12
TRAIL_MAX_ALPHA = 150
# A cyan distinct from the Jmol element palette (which is dominated by reds,
# greys, whites and browns) and from the R/G/B axis triad, so a trail never
# reads as either an atom or an axis on the dark canvas.
TRAIL_COLOR = "#4fd1e0"

# Elements darker than this (perceived luminance, 0-255) are lifted towards
# grey so they stay visible on the dark canvas. Carbon's standard Jmol colour
# is #313131, which is all but invisible against #26272b.
MIN_LUMINANCE = 96


def lift_for_dark(color: QColor) -> QColor:
    """Brighten a very dark element colour without changing its hue."""
    luminance = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
    if luminance >= MIN_LUMINANCE:
        return color
    # A pure grey has no hue to preserve, so it is simply raised; a dark
    # coloured element keeps its hue and gains value.
    hue, saturation, value, alpha = color.getHsv()
    target = int(MIN_LUMINANCE + (MIN_LUMINANCE - luminance) * 0.6)
    lifted = QColor()
    lifted.setHsv(max(hue, 0), saturation, min(255, max(value, target)), alpha)
    return lifted


@dataclass
class TrailOptions:
    """How motion trails are drawn."""

    enabled: bool = True
    length: int = 10                 # frames of history
    selected_only: bool = False      # only the tracked atom's trail


class SequenceView(StructureView):
    """A dark-canvas structure viewer for one frame of an ordered sequence."""

    # The full multi-selection after any change, as ordered structure
    # indices (click order, last clicked at the end).
    selectionChanged = Signal(object)   # list[int]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.set_background(VIEWER_BACKGROUND)
        # Ordered, because "the atom clicked last" is the one tracking
        # follows -- a set would forget which that was.
        self._multi: list[int] = []
        self._trail_options = TrailOptions()
        # (n_frames, n_atoms, 3) unwrapped cartesian coordinates for trails,
        # supplied by the page; None disables trails entirely.
        self._path: np.ndarray | None = None
        self._frame = 0
        self._tracked: int | None = None
        self._frozen_bonds = False

    # -- sequence data ------------------------------------------------------

    def element_color(self, symbol: str) -> QColor:
        """The dark-canvas version of whatever the base class would choose.

        Delegating rather than going straight to ``effective_color`` is what
        makes the Coloring control work at all here: the base implementation
        is the one that honours ``ViewOptions.coloring``, so hard-coding the
        per-element lookup meant "Uniform" was accepted by the combo box,
        stored on the options, and then silently ignored on the MD and NEB
        pages -- the only two that use this view.
        """
        return lift_for_dark(super().element_color(symbol))

    # -- multi-selection ----------------------------------------------------

    def set_structure(self, structure, reset_camera: bool = True) -> None:  # type: ignore[override]
        # A genuinely new structure means the stored indices name different
        # atoms; silently dropped rather than emitted, because the page
        # resets its own selection display alongside a new calculation.
        self._multi = []
        super().set_structure(structure, reset_camera)

    def selection(self) -> list[int]:
        return list(self._multi)

    def clear_selection(self) -> None:
        if not self._multi and self._selected is None:
            return
        self._multi = []
        self._selected = None
        self.selectionChanged.emit([])
        self.update()

    def _selection_set(self) -> set[int]:
        return set(self._multi)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Multi-selection on top of the base viewer's single-click select.

        A plain click replaces the selection with the clicked atom -- the
        behaviour every viewer here has always had. Holding Ctrl/Cmd or
        Shift toggles the clicked atom in and out of the selection instead,
        matching the Editor's conventions, so several atoms' trails can be
        followed at once. ``atomSelected`` keeps firing with the last
        clicked atom (or None), so the tracked-atom machinery and the
        sidebar chip behave exactly as before for single clicks.
        """
        if self._mode == self.MODE_SELECT and event.button() == Qt.LeftButton:
            self._last_mouse = QPointF(event.position())
            hit = self._pick(event.position())
            additive = bool(
                event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier)
            )
            if hit is None:
                if not additive:
                    self.clear_selection()
                self.update()
                self.atomSelected.emit(None if not additive else self._selected)
                return
            assert self._bonds is not None
            index = self._bonds.sites[hit].index
            if additive and index in self._multi:
                self._multi.remove(index)
                self._selected = self._multi[-1] if self._multi else None
            elif additive:
                self._multi.append(index)
                self._selected = index
            else:
                self._multi = [index]
                self._selected = index
            self.selectionChanged.emit(list(self._multi))
            self.atomSelected.emit(self._selected)
            self.update()
            return
        super().mousePressEvent(event)

    def set_path(self, path: np.ndarray | None) -> None:
        """Continuous (unwrapped) cartesian coordinates for the whole sequence."""
        self._path = None if path is None else np.asarray(path, dtype=float)
        self.update()

    def set_frame_index(self, frame: int) -> None:
        self._frame = int(frame)

    def set_trail_options(self, options: TrailOptions) -> None:
        self._trail_options = options
        self.update()

    def set_tracked_atom(self, index: int | None) -> None:
        """Keep this atom centred as the sequence plays (None to stop)."""
        self._tracked = index
        self.update()

    def tracked_atom(self) -> int | None:
        return self._tracked

    # -- frame updates ------------------------------------------------------

    def show_frame(self, structure: StructureData, frame: int) -> None:
        """Display one frame without disturbing the camera.

        The first call fits the camera to the frame; later calls keep whatever
        orientation and zoom the user has set, which is what makes stepping
        through a sequence readable.
        """
        first = self._structure is None
        self._frame = int(frame)
        self._frozen_bonds = structure.n_atoms > BOND_REBUILD_LIMIT

        keep_selection = self._selected
        if self._frozen_bonds and not first and self._bonds is not None:
            self._move_existing_sites(structure)
        else:
            super().set_structure(structure, reset_camera=first)
        self._selected = keep_selection

        if self._tracked is not None:
            self._centre_on_tracked()
        self.update()

    def _move_existing_sites(self, structure: StructureData) -> None:
        """Reuse the existing bond topology and only move the sites.

        Ghost sites carry the same lattice offset they were created with, so
        they can be recomputed from the new coordinates without another
        neighbour search.
        """
        assert self._bonds is not None
        wrapped = structure.frac_coords % 1.0
        cartesian = wrapped @ structure.lattice
        for site in self._bonds.sites:
            if site.index >= len(cartesian):
                continue
            base = cartesian[site.index]
            # A ghost is its parent displaced by a whole lattice translation,
            # recorded when the bond search created it.
            site.position = base if site.shift is None else base + site.shift
        self._structure = structure

    def _centre_on_tracked(self) -> None:
        """Pan so the tracked atom stays at the middle of the view.

        The camera centre is moved rather than the pan offset, so the atom
        stays put under rotation as well as translation.
        """
        if self._bonds is None or self._tracked is None:
            return
        for site in self._bonds.sites:
            if site.index == self._tracked and not site.is_ghost:
                self._camera.centre = np.asarray(site.position, dtype=float)
                return

    # -- painting ------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().paintEvent(event)
        if self._structure is None or self._bonds is None:
            return
        if not self._trail_options.enabled or self._path is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        self._draw_trails(painter)
        painter.end()

    def _draw_trails(self, painter: QPainter) -> None:
        """Fading polylines showing where each atom has recently been."""
        assert self._bonds is not None
        path = self._path
        if path is None or path.ndim != 3:
            return

        options = self._trail_options
        end = int(np.clip(self._frame, 0, path.shape[0] - 1))
        start = max(0, end - max(1, options.length))
        if end - start < 1:
            return

        history = path[start : end + 1]           # (n, n_atoms, 3)
        n_atoms = min(history.shape[1], self._structure.n_atoms)

        # Trails are drawn from continuous coordinates but must meet the atom
        # where it is actually painted, which — with wrapping on — is somewhere
        # else entirely. Translating each trail by that difference keeps it
        # attached without ever drawing a line across the whole cell.
        current_drawn = np.array(
            [
                self._bonds.sites[i].position
                for i in range(min(n_atoms, len(self._bonds.sites)))
            ]
        )
        if current_drawn.size == 0:
            return
        offsets = current_drawn - history[-1][: len(current_drawn)]

        if options.selected_only and (self._multi or self._tracked is not None):
            # Every selected atom's trail, not just the tracked one -- with
            # multi-selection, "show me these atoms' paths" is the point.
            chosen = list(dict.fromkeys(
                [*self._multi]
                + ([self._tracked] if self._tracked is not None else [])
            ))
            indices = chosen
        else:
            indices = range(len(current_drawn))

        for atom in indices:
            if atom is None or atom >= len(current_drawn):
                continue
            points = history[:, atom, :] + offsets[atom]
            screen, _, _ = self._camera.project(points, self.width(), self.height())
            # One distinct trail colour rather than each atom's own: drawn in
            # the element colour, a trail is the same hue as the atom sitting
            # on top of it and as every other atom of that element, so in a
            # cell of one species the paths were indistinguishable from the
            # structure they run through. A single contrasting colour reads
            # as "path" at a glance and stays legible against any element.
            colour = QColor(TRAIL_COLOR)

            for step in range(len(screen) - 1):
                # Older segments fade out, so the direction of travel reads
                # without needing an arrowhead.
                fraction = (step + 1) / max(1, len(screen) - 1)
                alpha = int(
                    TRAIL_MIN_ALPHA + (TRAIL_MAX_ALPHA - TRAIL_MIN_ALPHA) * fraction
                )
                colour.setAlpha(alpha)
                pen = QPen(colour, TRAIL_WIDTH)
                pen.setCapStyle(Qt.RoundCap)
                painter.setPen(pen)
                painter.drawLine(
                    QPointF(float(screen[step][0]), float(screen[step][1])),
                    QPointF(float(screen[step + 1][0]), float(screen[step + 1][1])),
                )

    def _draw_edge(self, painter: QPainter, start: np.ndarray, end: np.ndarray) -> None:
        # The light-theme cell colour disappears against the dark canvas.
        pen = QPen(QColor(SEQUENCE_CELL_COLOR), 1.0)
        painter.setPen(pen)
        painter.drawLine(
            QPointF(float(start[0]), float(start[1])),
            QPointF(float(end[0]), float(end[1])),
        )

    def fit_to_sequence(self, corners: np.ndarray, padding: float = 1.12) -> None:
        """Frame the whole sequence once, so no frame falls outside the view.

        ``padding`` enlarges the fitted radius a little. The camera's own
        framing margin is tuned for a static structure viewed head-on; under
        perspective the near corners of a cell project outside it, and without
        the extra room they are clipped by the canvas edge.
        """
        if corners.size:
            self._camera.fit(np.asarray(corners, dtype=float))
            self._camera.extent = max(self._camera.extent * float(padding), 1e-3)
        self._camera.reset()
        self.update()


def sequence_extent(
    lattice: np.ndarray,
    positions: np.ndarray,
    vacuum: VacuumInfo | None = None,
) -> np.ndarray:
    """Points the camera should frame: the cell box plus every sampled atom.

    Fitting to one frame alone lets a later frame drift out of view, so the
    caller passes coordinates spanning the sequence. When a vacuum gap is
    detected the box is clipped along that axis to the atoms themselves — a
    graphene sheet in a 14 Å cell is mostly empty space, and framing the whole
    cell shrinks the sheet to a sliver.
    """
    points = np.asarray(positions, dtype=float).reshape(-1, 3)
    if vacuum is not None and points.size:
        corners = framing_corners(lattice, vacuum, points)
    else:
        corners = np.array(
            [point for edge in unit_cell_edges(lattice) for point in edge]
        )
    if points.size == 0:
        return corners
    return np.vstack([corners, points])


__all__ = [
    "SequenceView",
    "TrailOptions",
    "sequence_extent",
    "VIEWER_BACKGROUND",
    "PALETTE",
]
