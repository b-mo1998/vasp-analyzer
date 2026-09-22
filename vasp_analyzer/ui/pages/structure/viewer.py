"""The interactive crystal-structure viewer.

Rendering is a depth-sorted QPainter pass rather than OpenGL: for the unit
cells this application shows (tens to a few hundred atoms) it is fast, renders
identically everywhere, antialiases beautifully and makes picking a simple
screen-space hit test. ``StructureView`` is deliberately the only place that
knows how atoms are drawn, so a GPU backend can replace it for large supercells
without touching the page.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QPainter,
    QPen,
    QPolygonF,
    QRadialGradient,
)
from PySide6.QtWidgets import QWidget

from ....models.structure import StructureData
from ....resources.elements import covalent_radius
from ....services.appearance import effective_color
from ....services.bonds import BondSet, build_sites_and_bonds, unit_cell_edges
from ....services.slab import VacuumInfo, detect_vacuum, framing_corners
from ...theme import PALETTE
from .camera import Camera

BALL_SCALE = 0.28          # fraction of the covalent radius drawn as a sphere
SPACE_FILLING_SCALE = 1.0
# A small vertex dot rather than 0: an atom with no bonds (an adsorbate,
# a vacancy edge, a lone dopant) would otherwise be completely invisible in
# Wireframe, since it has no stick to be represented by. Small enough to
# read as a joint rather than a ball — the whole distinction Wireframe is
# supposed to make from Ball-and-Stick.
WIREFRAME_SCALE = 0.05
BOND_WIDTH = 0.13          # in ångström, converted to pixels at draw time
# Thinner than Ball-and-Stick's — an actual wire, not a slightly smaller rod.
BOND_WIDTH_WIREFRAME = 0.045
CELL_COLOR = "#9aa0a6"
GHOST_ALPHA = 120

AXIS_COLORS = {"a": "#e5484d", "b": "#30a46c", "c": "#3b82f6"}
# Used by the "Uniform" coloring mode. A desaturated slate rather than one of
# the element colours, so it cannot be mistaken for a species, and mid-toned
# enough to stay legible on the light Structure canvas and the dark
# Editor/MD/NEB one alike.
UNIFORM_COLOR = "#8a94a6"
MEASURE_COLOR = "#f59e0b"
MAX_MEASURE_POINTS = 3
# The secondary ring drawn on atoms that are symmetry-equivalent to the
# selected one. Amber rather than the accent blue, so "this is the one you
# clicked" and "these are the ones it maps onto" stay distinguishable.
EQUIVALENT_COLOR = "#f59e0b"
# Arrows shorter than this on screen are noise, not information.
MIN_OVERLAY_ARROW_PIXELS = 3.0


@dataclass
class VectorOverlay:
    """A per-atom arrow field: magnetic moments, relaxation displacements…

    ``vectors`` is (n_atoms, 3) in Å of the *displayed* structure (a page that
    tiles a supercell tiles these to match). ``scale`` is a display
    exaggeration only -- a 0.02 Å relaxation is invisible at 1:1 -- and
    ``max_length`` clamps the longest arrow, so one atom that moved a lot
    cannot draw a line across the whole viewport.
    """

    vectors: np.ndarray
    color: str
    scale: float = 1.0
    max_length: float = 3.0        # Å, after scaling
    signed_color: str | None = None   # for vectors whose sign matters (spins)
    # Which component decides that sign. A collinear moment lives on z; an
    # isolated S_x view has to test x instead, or every arrow reads as "up".
    signed_axis: int = 2


@dataclass
class ViewOptions:
    style: str = "Ball-and-Stick"     # or "Space Filling" / "Wireframe"
    coloring: str = "By Element"
    show_cell: bool = True
    show_axes: bool = True
    perspective: bool = True


@dataclass(frozen=True)
class Measurement:
    """A distance (two atoms) or an angle (three atoms), in the crystal's own
    cartesian frame — measured directly from the same site positions the
    viewer draws, so it matches what's on screen exactly, ghosts included."""

    kind: str            # "distance" or "angle"
    value: float          # Å for a distance, degrees for an angle
    symbols: list[str]    # element symbols in click order


def _measure(positions: list[np.ndarray], symbols: list[str]) -> Measurement | None:
    if len(positions) == 2:
        distance = float(np.linalg.norm(positions[1] - positions[0]))
        return Measurement(kind="distance", value=distance, symbols=symbols)
    if len(positions) == 3:
        v1 = positions[0] - positions[1]
        v2 = positions[2] - positions[1]
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom < 1e-12:
            return None
        cosine = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
        angle = float(np.degrees(np.arccos(cosine)))
        return Measurement(kind="angle", value=angle, symbols=symbols)
    return None


class StructureView(QWidget):
    """Draws a structure and handles orbit / pan / zoom / pick."""

    atomSelected = Signal(object)        # site index in the source structure, or None
    measurementChanged = Signal(object)  # Measurement, or None when cleared
    viewChanged = Signal()

    MODE_SELECT = "select"
    MODE_ROTATE = "rotate"
    MODE_PAN = "pan"
    MODE_MEASURE = "measure"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 260)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)

        self._structure: StructureData | None = None
        self._bonds: BondSet | None = None
        self._vacuum: VacuumInfo | None = None
        self._camera = Camera()
        self._camera.reset()
        self._options = ViewOptions()
        self._mode = self.MODE_ROTATE
        self._selected: int | None = None
        self._hovered: int | None = None
        self._measure_points: list[int] = []  # site indices, in click order
        # Structure indices to leave undrawn and unpickable. A *view* concern
        # only: hiding an atom must never change the structure, or "hide" and
        # "delete" would differ merely in how recoverable they feel. Held as
        # structure indices rather than site indices so periodic ghosts hide
        # along with the atom they are an image of.
        self._hidden: set[int] = set()
        # Structure indices to ring in the secondary colour -- the atoms
        # symmetry-equivalent to the selection.
        self._highlighted: set[int] = set()
        # Named per-atom arrow fields, drawn depth-sorted with everything
        # else. Keyed so a page can show moments and displacements at once
        # and toggle each independently.
        self._overlays: dict[str, VectorOverlay] = {}
        self._last_mouse: QPointF | None = None
        self._screen_cache: np.ndarray | None = None
        self._radius_cache: np.ndarray | None = None
        self._depth_cache: np.ndarray | None = None
        # Overridden by the trajectory/NEB viewer, which sits on a dark canvas.
        self._background = PALETTE.surface

    # -- data ---------------------------------------------------------------

    def set_background(self, color: str) -> None:
        self._background = color
        self.update()

    def set_structure(
        self, structure: StructureData | None, reset_camera: bool = True
    ) -> None:
        """Show a structure, rebuilding bonds and (by default) reframing.

        ``reset_camera=False`` keeps the current orientation and zoom, which is
        what stepping through a trajectory or an NEB band needs: the point of
        those pages is to watch atoms move against a fixed viewpoint, and
        refitting the camera every frame would make the structure jitter.
        """
        self._structure = structure
        self._selected = None
        self._clear_measurement(notify=False)
        # Indexed per atom of the *previous* structure; a page that wants
        # them on the new one re-installs them after this returns.
        self._highlighted = set()
        self._overlays = {}
        self._bonds = None
        self._vacuum = None
        if structure is not None:
            self._bonds = build_sites_and_bonds(structure)
            self._vacuum = detect_vacuum(structure)
            if reset_camera:
                self._fit_camera()
        # The orientation presets are about the cell's own axes, so the camera
        # needs the lattice whether or not it is being reframed — an unchanged
        # camera still has to answer "look down b" correctly after the
        # structure under it changes.
        self._camera.set_lattice(None if structure is None else structure.lattice)
        if reset_camera:
            self._camera.reset()
        self.update()

    def _fit_camera(self) -> None:
        """Centre and scale the camera on the cell and its real contents.

        Periodic images are drawn where they fall, but letting them drive the
        framing would shrink the cell to a corner whenever bonds cross many
        faces. When a slab's vacuum gap is detected, the vacuum axis is framed
        to the atoms themselves instead of the full (mostly empty) cell — the
        true cell is still what gets drawn, only the framing changes.
        """
        if self._structure is None or self._bonds is None:
            return
        real = np.array(
            [site.position for site in self._bonds.sites if not site.is_ghost]
        )
        corners = framing_corners(self._structure.lattice, self._vacuum, real)
        self._camera.fit(np.vstack([real, corners]) if real.size else corners)

    def fit_view(self) -> None:
        """Reframe *and* re-orient — what a "fit" button should do.

        Distinct from :meth:`reset_view`, which only restores the orientation
        and zoom: a page that keeps its camera across edits (the Editor does,
        so a drag is not fought by a jumping view) still needs a way to say
        "I have a different structure now, frame it".
        """
        self._fit_camera()
        self._camera.reset()
        self.update()
        self.viewChanged.emit()

    def vacuum_info(self) -> VacuumInfo | None:
        return self._vacuum

    def set_options(self, options: ViewOptions) -> None:
        self._options = options
        self._camera.perspective = options.perspective
        self.update()

    def options(self) -> ViewOptions:
        return self._options

    def set_hidden_atoms(self, indices: set[int]) -> None:
        """Leave these structure indices undrawn and unclickable."""
        self._hidden = set(indices)
        self.update()

    def hidden_atoms(self) -> set[int]:
        return set(self._hidden)

    def _site_is_hidden(self, site_index: int) -> bool:
        if not self._hidden or self._bonds is None:
            return False
        if site_index >= len(self._bonds.sites):
            return False
        return self._bonds.sites[site_index].index in self._hidden

    def set_mode(self, mode: str) -> None:
        if self._mode == self.MODE_MEASURE and mode != self.MODE_MEASURE:
            self._clear_measurement()
        self._mode = mode
        cursors = {
            self.MODE_SELECT: Qt.ArrowCursor,
            self.MODE_ROTATE: Qt.OpenHandCursor,
            self.MODE_PAN: Qt.SizeAllCursor,
            self.MODE_MEASURE: Qt.CrossCursor,
        }
        self.setCursor(cursors.get(mode, Qt.ArrowCursor))

    def mode(self) -> str:
        return self._mode

    def camera(self) -> Camera:
        return self._camera

    def selected_index(self) -> int | None:
        return self._selected

    def set_highlighted_atoms(self, indices: set[int]) -> None:
        """Ring these structure indices in the secondary colour."""
        self._highlighted = set(indices)
        self.update()

    def highlighted_atoms(self) -> set[int]:
        return set(self._highlighted)

    def set_vector_overlay(self, name: str, overlay: VectorOverlay | None) -> None:
        """Install (or with None, remove) a per-atom arrow field."""
        if overlay is None:
            self._overlays.pop(name, None)
        else:
            self._overlays[name] = overlay
        self.update()

    def vector_overlay(self, name: str) -> VectorOverlay | None:
        return self._overlays.get(name)

    def _selection_set(self) -> set[int]:
        """Which structure indices get a selection ring.

        A hook so a subclass with a richer selection model (the sequence
        viewer's multi-selection, say) can highlight all of it without this
        class growing a second selection concept of its own.
        """
        return set() if self._selected is None else {self._selected}

    # -- measurement ----------------------------------------------------------

    def _clear_measurement(self, notify: bool = True) -> None:
        if not self._measure_points:
            return
        self._measure_points = []
        self.update()
        if notify:
            self.measurementChanged.emit(None)

    def _add_measure_point(self, site_index: int) -> None:
        assert self._bonds is not None
        if len(self._measure_points) >= MAX_MEASURE_POINTS:
            self._measure_points = []
        self._measure_points.append(site_index)

        result: Measurement | None = None
        if len(self._measure_points) >= 2:
            sites = [self._bonds.sites[i] for i in self._measure_points]
            result = _measure([s.position for s in sites], [s.symbol for s in sites])

        self.update()
        self.measurementChanged.emit(result)

    def reset_view(self) -> None:
        self._camera.reset()
        self.update()
        self.viewChanged.emit()

    def orient(self, key: str) -> None:
        """Swing the camera to one of the cell-axis presets ("a", "c*", ...)
        or back to the default view ("default")."""
        self._camera.orient(key)
        self.update()
        self.viewChanged.emit()

    def zoom_by(self, factor: float) -> None:
        self._camera.scale(factor)
        self.update()
        self.viewChanged.emit()

    # -- geometry helpers ------------------------------------------------------

    def _atom_radii(self) -> np.ndarray:
        assert self._bonds is not None
        style = self._options.style
        if style == "Space Filling":
            scale = SPACE_FILLING_SCALE
        elif style == "Wireframe":
            scale = WIREFRAME_SCALE
        else:
            scale = BALL_SCALE
        return np.array(
            [covalent_radius(site.symbol) * scale for site in self._bonds.sites]
        )

    def _project_sites(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self._bonds is not None
        points = np.array([site.position for site in self._bonds.sites])
        return self._camera.project(points, self.width(), self.height())

    # -- painting -----------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(self._background))

        if self._structure is None or self._bonds is None:
            painter.end()
            return

        screen, depth, scale = self._project_sites()
        self._screen_cache = screen
        self._depth_cache = depth

        radii_model = self._atom_radii()
        radii_px = radii_model * scale
        self._radius_cache = radii_px

        # Everything is collected into one list and painted back to front, so
        # bonds, atoms and cell edges interleave correctly by depth.
        primitives: list[tuple[float, str, tuple]] = []

        if self._options.show_cell:
            primitives.extend(self._cell_primitives())

        if self._options.style != "Space Filling":
            primitives.extend(self._bond_primitives(screen, depth, scale))

        for index in range(len(screen)):
            if self._site_is_hidden(index):
                continue
            primitives.append((float(depth[index]), "atom", (index,)))

        primitives.extend(self._extra_primitives())
        primitives.sort(key=lambda item: item[0])

        for _, kind, payload in primitives:
            if kind == "atom":
                self._draw_atom(painter, payload[0], screen, radii_px)
            elif kind == "bond":
                self._draw_bond(painter, *payload)
            elif kind == "edge":
                self._draw_edge(painter, *payload)
            else:
                self._draw_extra(painter, kind, payload)

        if self._options.show_axes:
            self._draw_axes(painter)

        if self._measure_points:
            self._draw_measurement(painter)

        painter.end()

    # -- extension hooks ------------------------------------------------------
    #
    # Subclasses add their own drawable items through these two methods so
    # they take part in the same back-to-front sort as atoms and bonds. That
    # is what lets a translucent isosurface interleave correctly with the
    # atoms it surrounds instead of being painted wholly over or under them.

    def _extra_primitives(self) -> list[tuple[float, str, tuple]]:
        """(depth, kind, payload) items to merge into the paint order."""
        if not self._overlays or self._bonds is None:
            return []
        # One primitive per (overlay, real atom), nudged just in front of its
        # atom's depth so the arrow draws over the sphere it belongs to
        # rather than being swallowed by it.
        _screen, depth, _scale = self._project_sites()
        items: list[tuple[float, str, tuple]] = []
        for name, overlay in self._overlays.items():
            for site_index, site in enumerate(self._bonds.sites):
                if site.is_ghost or site.index >= len(overlay.vectors):
                    continue
                if self._site_is_hidden(site_index):
                    continue
                items.append(
                    (float(depth[site_index]) + 0.01, "overlay_arrow", (name, site_index))
                )
        return items

    def _draw_extra(self, painter: QPainter, kind: str, payload: tuple) -> None:
        """Draw one item produced by :meth:`_extra_primitives`."""
        if kind == "overlay_arrow":
            self._draw_overlay_arrow(painter, *payload)

    def _draw_overlay_arrow(self, painter: QPainter, name: str, site_index: int) -> None:
        overlay = self._overlays.get(name)
        if overlay is None or self._bonds is None:
            return
        site = self._bonds.sites[site_index]
        vector = np.asarray(overlay.vectors[site.index], dtype=float) * overlay.scale
        length_a = float(np.linalg.norm(vector))
        if length_a <= 0.0:
            return
        if length_a > overlay.max_length:
            vector = vector * (overlay.max_length / length_a)

        # Tail and head projected separately: under perspective they sit at
        # different depths, and scaling one projected direction would point
        # slightly wrong for anything off-centre.
        points = np.array([site.position, site.position + vector])
        projected, _depth, _scale = self._camera.project(
            points, self.width(), self.height()
        )
        tail = QPointF(float(projected[0][0]), float(projected[0][1]))
        head = QPointF(float(projected[1][0]), float(projected[1][1]))
        delta = np.array([head.x() - tail.x(), head.y() - tail.y()])
        length = float(np.linalg.norm(delta))
        if length < MIN_OVERLAY_ARROW_PIXELS:
            return

        # A signed field (a collinear moment) colours "down" differently from
        # "up", so antiferromagnetic order reads as alternating colours and
        # not just alternating directions.
        colour_hex = overlay.color
        if overlay.signed_color is not None and vector[overlay.signed_axis] < 0:
            colour_hex = overlay.signed_color
        colour = QColor(colour_hex)

        pen = QPen(colour, 2.0)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawLine(tail, head)

        unit = delta / length
        normal = np.array([-unit[1], unit[0]])
        size = min(8.0, length * 0.45)
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

    def _cell_primitives(self) -> list[tuple[float, str, tuple]]:
        assert self._structure is not None
        edges = unit_cell_edges(self._structure.lattice)
        points = np.array([point for edge in edges for point in edge])
        screen, depth, _ = self._camera.project(points, self.width(), self.height())

        primitives = []
        for index in range(len(edges)):
            start, end = screen[2 * index], screen[2 * index + 1]
            mid_depth = float((depth[2 * index] + depth[2 * index + 1]) / 2)
            primitives.append((mid_depth, "edge", (start, end)))
        return primitives

    def _bond_primitives(
        self, screen: np.ndarray, depth: np.ndarray, scale: np.ndarray
    ) -> list[tuple[float, str, tuple]]:
        assert self._bonds is not None
        primitives = []
        base_width = (
            BOND_WIDTH_WIREFRAME if self._options.style == "Wireframe" else BOND_WIDTH
        )
        for i, j in self._bonds.bonds:
            # A bond to a hidden atom would otherwise dangle in mid-air,
            # pointing at nothing.
            if self._site_is_hidden(i) or self._site_is_hidden(j):
                continue
            start, end = screen[i], screen[j]
            middle = (start + end) / 2.0
            width = base_width * float((scale[i] + scale[j]) / 2)
            mid_depth = float((depth[i] + depth[j]) / 2)
            # Two half-bonds, each in its own atom's colour.
            primitives.append(
                (
                    mid_depth,
                    "bond",
                    (start, middle, self._site_color(i), width),
                )
            )
            primitives.append(
                (
                    mid_depth,
                    "bond",
                    (middle, end, self._site_color(j), width),
                )
            )
        return primitives

    def element_color(self, symbol: str) -> QColor:
        """The colour an element is drawn in.

        A hook rather than a direct call so a viewer on a different background
        can adapt it — the sequence viewer lifts very dark elements (carbon's
        Jmol colour is near-black) that would otherwise vanish on its dark
        canvas.
        """
        if self._options.coloring == "Uniform":
            return QColor(UNIFORM_COLOR)
        return QColor(effective_color(symbol))

    def _site_color(self, index: int) -> QColor:
        assert self._bonds is not None
        site = self._bonds.sites[index]
        colour = self.element_color(site.symbol)
        if site.is_ghost:
            colour.setAlpha(GHOST_ALPHA)
        return colour

    def _draw_atom(
        self, painter: QPainter, index: int, screen: np.ndarray, radii: np.ndarray
    ) -> None:
        assert self._bonds is not None
        site = self._bonds.sites[index]
        radius = max(1.5, float(radii[index]))
        centre = QPointF(float(screen[index][0]), float(screen[index][1]))
        base = self.element_color(site.symbol)

        # A soft radial gradient reads as a sphere without needing lighting.
        gradient = QRadialGradient(
            centre + QPointF(-radius * 0.32, -radius * 0.34), radius * 1.45
        )
        highlight = base.lighter(150)
        shadow = base.darker(135)
        if site.is_ghost:
            for colour in (highlight, base, shadow):
                colour.setAlpha(GHOST_ALPHA)
        gradient.setColorAt(0.0, highlight)
        gradient.setColorAt(0.55, base)
        gradient.setColorAt(1.0, shadow)

        painter.setBrush(QBrush(gradient))
        outline = QColor(0, 0, 0, 40 if not site.is_ghost else 20)
        painter.setPen(QPen(outline, 0.8))
        painter.drawEllipse(centre, radius, radius)

        # Equivalence ring first, so an atom that is both selected and in the
        # highlighted set shows the selection ring on top.
        if (
            site.index in self._highlighted
            and site.index not in self._selection_set()
            and not site.is_ghost
        ):
            painter.setBrush(Qt.NoBrush)
            pen = QPen(QColor(EQUIVALENT_COLOR), 1.8)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawEllipse(centre, radius + 3.5, radius + 3.5)

        if site.index in self._selection_set() and not site.is_ghost:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(PALETTE.accent), 2.0))
            painter.drawEllipse(centre, radius + 3.5, radius + 3.5)
        elif self._hovered == index:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(PALETTE.accent), 1.2))
            painter.drawEllipse(centre, radius + 2.5, radius + 2.5)

    def _draw_bond(
        self,
        painter: QPainter,
        start: np.ndarray,
        end: np.ndarray,
        colour: QColor,
        width: float,
    ) -> None:
        pen = QPen(colour, max(1.4, width))
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawLine(
            QPointF(float(start[0]), float(start[1])),
            QPointF(float(end[0]), float(end[1])),
        )

    def _draw_edge(self, painter: QPainter, start: np.ndarray, end: np.ndarray) -> None:
        pen = QPen(QColor(CELL_COLOR), 1.0)
        painter.setPen(pen)
        painter.drawLine(
            QPointF(float(start[0]), float(start[1])),
            QPointF(float(end[0]), float(end[1])),
        )

    def _draw_axes(self, painter: QPainter) -> None:
        if self._structure is None:
            return
        origin = QPointF(48.0, self.height() - 48.0)
        lattice = self._structure.lattice
        directions = self._camera.direction_screen(
            lattice / np.linalg.norm(lattice, axis=1, keepdims=True)
        )

        painter.save()
        for index, name in enumerate("abc"):
            vector = directions[index] * 30.0
            tip = origin + QPointF(float(vector[0]), float(vector[1]))
            pen = QPen(QColor(AXIS_COLORS[name]), 1.8)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(origin, tip)

            painter.setPen(QPen(QColor(AXIS_COLORS[name]), 1.0))
            label_offset = QPointF(float(vector[0]) * 0.28, float(vector[1]) * 0.28)
            painter.drawText(
                QRectF(tip + label_offset - QPointF(7, 8), tip + label_offset + QPointF(7, 8)),
                Qt.AlignCenter,
                name,
            )
        painter.restore()

    def _draw_measurement(self, painter: QPainter) -> None:
        """Numbered badges at each clicked site, a connecting line (or two,
        for an angle), and the resulting distance/angle drawn near it. Always
        painted last, on top of everything else — a measurement in progress
        should never get lost behind an atom."""
        assert self._bonds is not None
        if self._screen_cache is None:
            return

        points = [
            (i, self._screen_cache[i]) for i in self._measure_points if i < len(self._screen_cache)
        ]
        if not points:
            return

        pen = QPen(QColor(MEASURE_COLOR), 1.6)
        pen.setDashPattern([4.0, 3.0])
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        for (_, a), (_, b) in zip(points, points[1:]):
            painter.drawLine(QPointF(float(a[0]), float(a[1])), QPointF(float(b[0]), float(b[1])))

        painter.setPen(QPen(QColor("#ffffff"), 1.0))
        painter.setBrush(QColor(MEASURE_COLOR))
        for order, (_, screen_pos) in enumerate(points, start=1):
            centre = QPointF(float(screen_pos[0]), float(screen_pos[1]))
            painter.drawEllipse(centre, 8.0, 8.0)
            painter.drawText(QRectF(centre - QPointF(8, 8), centre + QPointF(8, 8)), Qt.AlignCenter, str(order))

        sites = [self._bonds.sites[i] for i in self._measure_points]
        result = _measure([s.position for s in sites], [s.symbol for s in sites])
        if result is None:
            return

        label = f"{result.value:.3f} Å" if result.kind == "distance" else f"{result.value:.1f}°"
        if result.kind == "distance":
            anchor = (points[0][1] + points[1][1]) / 2.0
        else:
            anchor = points[1][1]  # the vertex

        text_pos = QPointF(float(anchor[0]) + 12, float(anchor[1]) - 12)
        metrics = painter.fontMetrics()
        rect = metrics.boundingRect(label).adjusted(-5, -3, 5, 3)
        rect.moveTopLeft((text_pos - QPointF(0, rect.height())).toPoint())
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(26, 26, 26, 215))
        painter.drawRoundedRect(rect, 4, 4)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(rect, Qt.AlignCenter, label)

    # -- interaction ---------------------------------------------------------------

    def _pick(self, position: QPointF) -> int | None:
        """Nearest atom under the cursor, front-most first."""
        if self._screen_cache is None or self._radius_cache is None:
            return None
        if self._depth_cache is None:
            return None
        point = np.array([position.x(), position.y()])
        distances = np.linalg.norm(self._screen_cache - point, axis=1)
        candidates = np.nonzero(distances <= np.maximum(self._radius_cache, 4.0))[0]
        # An atom you cannot see must not be selectable through the empty
        # space where it used to be — hiding is often exactly a way to reach
        # what is behind it.
        if self._hidden:
            candidates = np.array(
                [c for c in candidates if not self._site_is_hidden(int(c))], dtype=int
            )
        if candidates.size == 0:
            return None
        return int(candidates[np.argmax(self._depth_cache[candidates])])

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._last_mouse = QPointF(event.position())
        if self._mode == self.MODE_SELECT and event.button() == Qt.LeftButton:
            hit = self._pick(event.position())
            if hit is None:
                self._selected = None
                self.atomSelected.emit(None)
            else:
                assert self._bonds is not None
                self._selected = self._bonds.sites[hit].index
                self.atomSelected.emit(self._selected)
            self.update()
        elif self._mode == self.MODE_MEASURE:
            if event.button() == Qt.RightButton:
                self._clear_measurement()
            elif event.button() == Qt.LeftButton:
                hit = self._pick(event.position())
                if hit is not None:
                    self._add_measure_point(hit)
        elif self._mode == self.MODE_ROTATE:
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        position = QPointF(event.position())

        if self._last_mouse is not None and event.buttons() & (
            Qt.LeftButton | Qt.MiddleButton | Qt.RightButton
        ):
            delta = position - self._last_mouse
            panning = (
                self._mode == self.MODE_PAN
                or event.buttons() & Qt.MiddleButton
                or event.modifiers() & Qt.ShiftModifier
            )
            if panning:
                self._camera.move(delta.x(), delta.y())
            elif self._mode != self.MODE_SELECT:
                self._camera.orbit(delta.x(), delta.y())
            self._last_mouse = position
            self.update()
            self.viewChanged.emit()
            return

        self._last_mouse = position
        if self._mode in (self.MODE_SELECT, self.MODE_MEASURE):
            hovered = self._pick(position)
            if hovered != self._hovered:
                self._hovered = hovered
                self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._last_mouse = None
        if self._mode == self.MODE_ROTATE:
            self.setCursor(Qt.OpenHandCursor)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt naming
        steps = event.angleDelta().y() / 120.0
        if steps:
            self._camera.scale(1.0 + 0.12 * steps)
            self.update()
            self.viewChanged.emit()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.reset_view()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.key() == Qt.Key_Escape and self._measure_points:
            self._clear_measurement()
            return
        super().keyPressEvent(event)
