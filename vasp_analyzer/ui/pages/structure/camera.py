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

    def reset(self) -> None:
        # A gentle three-quarter view reads better than looking straight down
        # an axis, where atoms hide behind one another.
        self.rotation = rotation_matrix(
            np.array([1.0, 0.0, 0.0]), np.radians(-18.0)
        ) @ rotation_matrix(np.array([0.0, 1.0, 0.0]), np.radians(28.0))
        self.zoom = 1.0
        self.pan = np.zeros(2)

    def look_along(self, axis: str) -> None:
        """Align the view with a cell axis."""
        presets = {
            "a": rotation_matrix(np.array([0.0, 1.0, 0.0]), np.radians(-90.0)),
            "b": rotation_matrix(np.array([1.0, 0.0, 0.0]), np.radians(90.0)),
            "c": np.eye(3),
        }
        self.rotation = presets.get(axis, np.eye(3))
        self.pan = np.zeros(2)

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
