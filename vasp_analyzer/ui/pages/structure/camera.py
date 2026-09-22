"""Camera and projection for the structure viewer.

Kept free of Qt so the projection maths can be tested without a window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Right-handed rotation about an arbitrary axis (Rodrigues' formula)."""
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12 or abs(angle) < 1e-12:
        return np.eye(3)
    axis = axis / norm
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def orientation_matrix(toward_viewer: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    """A camera rotation whose rows are (screen-right, screen-up, toward-viewer).

    ``project`` computes ``local = p @ rotation.T``, so row 0 becomes the
    horizontal screen coordinate, row 1 the vertical one and row 2 the depth
    used for sorting and for the perspective divide — which is why the rows
    are exactly the camera's own basis vectors and why the third one points
    *at* the viewer rather than away.
    """
    forward = np.asarray(toward_viewer, dtype=float)
    norm = float(np.linalg.norm(forward))
    if norm < 1e-12:
        return np.eye(3)
    forward = forward / norm

    up = np.asarray(up_hint, dtype=float)
    up = up - float(np.dot(up, forward)) * forward
    if float(np.linalg.norm(up)) < 1e-9:
        # The hint was (nearly) the view direction itself and says nothing
        # about which way is up. Any perpendicular will do; pick whichever
        # cartesian axis the view direction leans on least.
        fallback = np.eye(3)[int(np.argmin(np.abs(forward)))]
        up = fallback - float(np.dot(fallback, forward)) * forward
    up = up / float(np.linalg.norm(up))

    right = np.cross(up, forward)
    right = right / float(np.linalg.norm(right))
    return np.array([right, up, forward])


# Which cell axis is drawn "up" for each orientation preset, and which
# direction the camera looks down. Reciprocal axes matter because in an
# oblique cell the normal to a plane is not the axis perpendicular to it on
# paper: a* is normal to the b-c plane, which is what "view down a*" means.
ORIENTATION_UP_HINT = {
    "a": 2, "b": 2, "c": 1,
    "a*": 2, "b*": 2, "c*": 1,
}

# Degrees of tilt applied to the default view, away from looking exactly down
# **a**. Exactly down an axis is a poor default: every atom behind another
# hides completely and the cell degenerates to a flat outline. A few degrees
# off-axis restores the depth cue while keeping the familiar
# b-across / c-up reading of the cell.
DEFAULT_TILT_DEGREES = (-10.0, 7.0)


@dataclass
class Camera:
    """Orbit camera producing screen coordinates plus a depth value.

    The model is centred on the structure, rotated, then projected either
    orthographically or in perspective. Depth is returned so the painter can
    sort primitives back-to-front.
    """

    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    zoom: float = 1.0
    pan: np.ndarray = field(default_factory=lambda: np.zeros(2))
    perspective: bool = True
    centre: np.ndarray = field(default_factory=lambda: np.zeros(3))
    extent: float = 10.0          # model radius, sets the default framing
    field_of_view: float = 32.0   # degrees
    # Rows a, b, c. Set from the structure so the orientation presets can be
    # about the *crystal* axes rather than about cartesian x/y/z, which are
    # only the same thing for a cell that happens to be axis-aligned.
    lattice: np.ndarray | None = None

    # -- interaction -------------------------------------------------------

    def orbit(self, dx: float, dy: float, speed: float = 0.01) -> None:
        """Rotate about screen axes, so dragging feels like turning the object."""
        yaw = rotation_matrix(np.array([0.0, 1.0, 0.0]), dx * speed)
        pitch = rotation_matrix(np.array([1.0, 0.0, 0.0]), dy * speed)
        self.rotation = pitch @ yaw @ self.rotation

    def roll(self, angle: float) -> None:
        self.rotation = rotation_matrix(np.array([0.0, 0.0, 1.0]), angle) @ self.rotation

    def move(self, dx: float, dy: float) -> None:
        self.pan = self.pan + np.array([dx, dy], dtype=float)

    def scale(self, factor: float) -> None:
        self.zoom = float(np.clip(self.zoom * factor, 0.15, 12.0))

    def set_lattice(self, lattice: np.ndarray | None) -> None:
        self.lattice = None if lattice is None else np.asarray(lattice, dtype=float)

    def _axes(self) -> np.ndarray:
        """Real-space cell axes, falling back to cartesian for no structure."""
        if self.lattice is None or np.asarray(self.lattice).shape != (3, 3):
            return np.eye(3)
        lattice = np.asarray(self.lattice, dtype=float)
        if abs(float(np.linalg.det(lattice))) < 1e-12:
            return np.eye(3)
        return lattice

    def view_direction(self, key: str) -> np.ndarray | None:
        """The axis a preset looks down, in cartesian coordinates.

        Reciprocal axes are built as the cross products of the other two real
        axes rather than through a full inverse, which is the same vector up
        to the 2pi/V factor an orientation does not care about.
        """
        axes = self._axes()
        a, b, c = axes[0], axes[1], axes[2]
        directions = {
            "a": a, "b": b, "c": c,
            "a*": np.cross(b, c),
            "b*": np.cross(c, a),
            "c*": np.cross(a, b),
        }
        direction = directions.get(key)
        if direction is None or float(np.linalg.norm(direction)) < 1e-12:
            return None
        return np.asarray(direction, dtype=float)

    def orient(self, key: str) -> None:
        """Point the camera down one of the cell's real or reciprocal axes.

        The axis ends up pointing *at* the viewer, so "c" gives the familiar
        plan view onto the a-b plane rather than a view of its underside.
        Zoom is deliberately left alone: switching orientation is a question
        about direction, and having it silently undo the zoom the user set
        would make the buttons unusable for looking closely at anything.
        """
        if key == "default":
            self.reset_orientation()
            return
        direction = self.view_direction(key)
        if direction is None:
            return
        axes = self._axes()
        self.rotation = orientation_matrix(
            direction, axes[ORIENTATION_UP_HINT.get(key, 2)]
        )
        self.pan = np.zeros(2)

    def reset_orientation(self) -> None:
        """The default view: b across, c up, a toward the viewer, tilted.

        This is the orientation the structure pages open in and the one the
        default-view button returns to, so a cell always presents itself the
        same way however it was reached.
        """
        axes = self._axes()
        self.rotation = orientation_matrix(axes[0], axes[2])
        yaw, pitch = DEFAULT_TILT_DEGREES
        self.rotation = (
            rotation_matrix(np.array([1.0, 0.0, 0.0]), np.radians(pitch))
            @ rotation_matrix(np.array([0.0, 1.0, 0.0]), np.radians(yaw))
            @ self.rotation
        )
        self.pan = np.zeros(2)

    def reset(self) -> None:
        self.reset_orientation()
        self.zoom = 1.0
        self.pan = np.zeros(2)

    def look_along(self, axis: str) -> None:
        """Backwards-compatible alias for :meth:`orient`."""
        self.orient(axis)

    # -- projection ---------------------------------------------------------

    def fit(self, points: np.ndarray) -> None:
        """Centre on a point cloud and record its radius."""
        if points.size == 0:
            self.centre = np.zeros(3)
            self.extent = 10.0
            return
        self.centre = points.mean(axis=0)
        radius = float(np.linalg.norm(points - self.centre, axis=1).max())
        self.extent = max(radius, 1e-3)

    def project(
        self, points: np.ndarray, width: float, height: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project cartesian points.

        Returns ``(screen_xy, depth, scale)`` where ``scale`` is the per-point
        size multiplier — constant under orthographic projection and distance
        dependent under perspective.
        """
        if points.size == 0:
            return np.zeros((0, 2)), np.zeros(0), np.zeros(0)

        local = (points - self.centre) @ self.rotation.T
        margin = 0.62
        base = min(width, height) * margin / self.extent * self.zoom

        if self.perspective:
            # Place the eye far enough back that the cell fits the frame.
            distance = self.extent / np.tan(np.radians(self.field_of_view) / 2.0)
            eye_z = distance + self.extent
            denominator = np.clip(eye_z - local[:, 2], 1e-3, None)
            factor = distance / denominator
        else:
            factor = np.ones(len(local))

        screen = np.empty((len(local), 2))
        screen[:, 0] = width / 2 + (local[:, 0] * base * factor) + self.pan[0]
        screen[:, 1] = height / 2 - (local[:, 1] * base * factor) + self.pan[1]
        return screen, local[:, 2], base * factor

    def direction_screen(self, vectors: np.ndarray) -> np.ndarray:
        """Rotate direction vectors into screen space (for the axis triad)."""
        local = vectors @ self.rotation.T
        out = np.empty_like(local[:, :2])
        out[:, 0] = local[:, 0]
        out[:, 1] = -local[:, 1]
        return out
