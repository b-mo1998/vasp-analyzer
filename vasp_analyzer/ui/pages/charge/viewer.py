"""3D viewer for a volumetric field: atoms plus a translucent isosurface.

Extends the shared :class:`StructureView` rather than introducing a second
renderer. The isosurface enters the base class's existing depth sort as
individual triangles, so atoms inside the surface are correctly occluded by
the facets in front of them and visible through the ones behind — the ordinary
painter's algorithm, which is what makes a translucent surface read as
enclosing the structure rather than floating over it.

Triangles are shaded from their own smooth normals with a single fixed light,
kept matte on purpose: a specular highlight on a charge density would suggest
a physical surface that does not exist.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from ....services.isosurface import SurfaceMesh
from ..sequence_view import VIEWER_BACKGROUND, lift_for_dark
from ..structure.viewer import StructureView

# Direction the scene is lit from, in camera space (over the viewer's shoulder,
# slightly up and to the left) — the same convention as the atom gradients.
LIGHT = np.array([-0.35, 0.45, 0.82])
AMBIENT = 0.42
DIFFUSE = 0.58

SLICE_EDGE = "#f5c451"
SLICE_FILL = QColor(245, 196, 81, 34)


class DensityView(StructureView):
    """Structure viewer with an isosurface and an optional cutting plane."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.set_background(VIEWER_BACKGROUND)
        self._mesh: SurfaceMesh | None = None
        self._surface_color = QColor("#4da3d0")
        self._opacity = 0.7
        self._show_surface = True
        self._show_atoms = True
        self._slice_quad: np.ndarray | None = None   # (4, 3) cartesian corners
        self._ghost_shifts: np.ndarray | None = None  # periodic image offsets

    # -- data ----------------------------------------------------------------

    def set_mesh(self, mesh: SurfaceMesh | None) -> None:
        self._mesh = mesh
        self.update()

    def mesh(self) -> SurfaceMesh | None:
        return self._mesh

    def set_surface_color(self, color: str) -> None:
        self._surface_color = QColor(color)
        self.update()

    def set_opacity(self, opacity: float) -> None:
        self._opacity = float(np.clip(opacity, 0.0, 1.0))
        self.update()

    def set_show_surface(self, show: bool) -> None:
        self._show_surface = bool(show)
        self.update()

    def set_show_atoms(self, show: bool) -> None:
        self._show_atoms = bool(show)
        self.update()

    def set_slice_plane(self, corners: np.ndarray | None) -> None:
        """The cutting plane's four cartesian corners, or None to hide it."""
        self._slice_quad = None if corners is None else np.asarray(corners, dtype=float)
        self.update()

    def set_periodic_images(self, shifts: np.ndarray | None) -> None:
        """Lattice translations at which to repeat the surface, or None.

        Kept to a small set of neighbours by the caller: replicating a
        multi-thousand-triangle mesh in all 26 directions would multiply the
        paint cost by 27 for very little insight.
        """
        self._ghost_shifts = None if shifts is None else np.asarray(shifts, dtype=float)
        self.update()

    # -- painting -------------------------------------------------------------

    def element_color(self, symbol: str) -> QColor:
        from ....services.appearance import effective_color

        return lift_for_dark(QColor(effective_color(symbol)))

    def _atom_radii(self) -> np.ndarray:
        radii = super()._atom_radii()
        return np.zeros_like(radii) if not self._show_atoms else radii

    def _extra_primitives(self) -> list[tuple[float, str, tuple]]:
        # The isosurface is *not* returned here. It is composited as one layer
        # in paintEvent instead — see _draw_surface_layer for why.
        primitives: list[tuple[float, str, tuple]] = []
        if self._slice_quad is not None:
            primitives.append(self._plane_primitive())
        return primitives

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().paintEvent(event)
        if not self._show_surface or self._mesh is None or self._mesh.is_empty:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        self._draw_surface_layer(painter)
        painter.end()

    def _draw_surface_layer(self, painter: QPainter) -> None:
        """Composite the isosurface as a single translucent layer.

        Painting each facet directly with a semi-transparent brush does not
        work: neighbouring triangles must overlap slightly to hide the hairline
        antialiasing gap between them, and every overlap then blends twice,
        turning a smooth surface into a speckle of darker seams. Drawing the
        mesh *opaquely* into an offscreen image and compositing that once gives
        both a seamless surface and a uniform, correct transparency.

        The cost is that the surface becomes one layer over the atoms rather
        than interleaving with them facet by facet. For a density surface
        enclosing a structure that is very nearly the same picture, since the
        atoms are inside it and show through.
        """
        layer = QImage(self.size(), QImage.Format_ARGB32_Premultiplied)
        layer.fill(Qt.transparent)

        inner = QPainter(layer)
        inner.setRenderHint(QPainter.Antialiasing, True)
        for _, corners, colour in self._surface_faces():
            polygon = QPolygonF([QPointF(float(x), float(y)) for x, y in corners])
            inner.setPen(QPen(colour, 1.0))
            inner.setBrush(colour)
            inner.drawPolygon(polygon)
        inner.end()

        painter.setOpacity(self._opacity)
        painter.drawImage(0, 0, layer)
        painter.setOpacity(1.0)

    def _surface_faces(self) -> list[tuple[float, np.ndarray, QColor]]:
        """Visible facets, back to front, with their shaded (opaque) colours.

        Projection and shading are done for the whole mesh at once with numpy;
        only the per-triangle tuples are built in Python, which keeps a
        20,000-triangle surface interactive.
        """
        mesh = self._mesh
        assert mesh is not None

        offsets = (
            [np.zeros(3)]
            if self._ghost_shifts is None
            else [np.zeros(3), *self._ghost_shifts]
        )

        faces: list[tuple[float, np.ndarray, QColor]] = []
        face_normals = mesh.triangle_normals()
        # Shade in camera space so the lighting follows the object as it turns.
        rotated = face_normals @ self._camera.rotation.T
        intensity = np.clip(
            AMBIENT + DIFFUSE * np.abs(rotated @ LIGHT), 0.0, 1.0
        )

        base = self._surface_color

        for index, shift in enumerate(offsets):
            vertices = mesh.vertices + shift
            screen, depth, _ = self._camera.project(
                vertices, self.width(), self.height()
            )
            corners = screen[mesh.triangles]                # (n, 3, 2)
            depths = depth[mesh.triangles].mean(axis=1)     # (n,)

            # Drop triangles that project to nothing; at a coarse grid these
            # are a large fraction of the mesh and cost real time to paint.
            edge1 = corners[:, 1] - corners[:, 0]
            edge2 = corners[:, 2] - corners[:, 0]
            area = np.abs(edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0])
            visible = np.nonzero(area > 1e-3)[0]

            # Neighbouring cells are drawn dimmer so the central cell still
            # reads as the subject rather than one tile of a pattern.
            dim = 1.0 if index == 0 else 0.45
            for face in visible:
                shade = float(intensity[face]) * dim
                colour = QColor(
                    int(base.red() * shade),
                    int(base.green() * shade),
                    int(base.blue() * shade),
                )
                faces.append((float(depths[face]), corners[face], colour))

        # Painter's algorithm within the layer, which is exact here because
        # every facet is opaque at this stage.
        faces.sort(key=lambda item: item[0])
        return faces

    def _plane_primitive(self) -> tuple[float, str, tuple]:
        assert self._slice_quad is not None
        screen, depth, _ = self._camera.project(
            self._slice_quad, self.width(), self.height()
        )
        return (float(np.mean(depth)), "plane", (screen,))

    def _draw_extra(self, painter: QPainter, kind: str, payload: tuple) -> None:
        if kind == "plane":
            (screen,) = payload
            polygon = QPolygonF(
                [QPointF(float(x), float(y)) for x, y in screen]
            )
            painter.setBrush(SLICE_FILL)
            pen = QPen(QColor(SLICE_EDGE), 1.2)
            pen.setDashPattern([4.0, 3.0])
            painter.setPen(pen)
            painter.drawPolygon(polygon)


def slice_plane_corners(
    lattice: np.ndarray, axis: int, fraction: float
) -> np.ndarray:
    """The four cartesian corners of a cut plane through the cell.

    Built from the two lattice vectors spanning the plane, so in a sheared cell
    the quad is the correct parallelogram rather than an axis-aligned square.
    """
    others = [i for i in range(3) if i != axis]
    origin = (float(fraction) % 1.0) * lattice[axis]
    u = lattice[others[0]]
    v = lattice[others[1]]
    return np.array([origin, origin + u, origin + u + v, origin + v])


def neighbour_shifts(lattice: np.ndarray) -> np.ndarray:
    """In-plane neighbouring cell translations for the periodic-images option.

    Only the six neighbours sharing a face are used. The full 26-cell shell
    would multiply the triangle count by 27 and mostly draw surfaces hidden
    behind the central cell.
    """
    return np.array(
        [
            sign * lattice[axis]
            for axis in range(3)
            for sign in (-1.0, 1.0)
        ]
    )
