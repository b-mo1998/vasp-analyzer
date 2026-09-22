"""Isosurface extraction from a volumetric field.

Uses **marching tetrahedra** rather than marching cubes. Each grid cell is cut
into six tetrahedra, and a tetrahedron has only 2⁴ = 16 corner configurations
against 2⁸ = 256 for a cube. That difference matters for a hand-written
implementation: the cube algorithm needs a 256-entry triangle table that is
tedious to transcribe and easy to get subtly wrong, while the tetrahedral
cases are small enough to enumerate in a few lines and are inherently free of
the ambiguous-face problem that makes naive marching cubes produce holes.
The cost is roughly twice as many triangles for the same surface, which is a
good trade when the alternative is a table that renders plausibly and is
wrong.

Vertices come out in **fractional** coordinates and are converted with the
lattice matrix, so a hexagonal or triclinic cell is handled correctly rather
than being approximated by a stretched cube.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.logging_setup import get_logger
from ..models.volumetric import VolumetricField

log = get_logger("services.isosurface")

# Cube corners in (i, j, k) offsets, in the order the tetrahedra below index.
_CORNERS = np.array(
    [
        [0, 0, 0],  # 0
        [1, 0, 0],  # 1
        [1, 1, 0],  # 2
        [0, 1, 0],  # 3
        [0, 0, 1],  # 4
        [1, 0, 1],  # 5
        [1, 1, 1],  # 6
        [0, 1, 1],  # 7
    ],
    dtype=np.int64,
)

# Six tetrahedra tiling the cube, all sharing the 0-6 body diagonal.
_TETRAHEDRA = (
    (0, 1, 2, 6),
    (0, 2, 3, 6),
    (0, 3, 7, 6),
    (0, 7, 4, 6),
    (0, 4, 5, 6),
    (0, 5, 1, 6),
)

# The six edges of a tetrahedron, as (local vertex, local vertex).
_TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))

# For each of the 16 inside/outside patterns, which edges the surface crosses,
# as triangles. Bit n of the case index is set when local vertex n is inside.
_TET_CASES: dict[int, tuple[tuple[int, int, int], ...]] = {
    0b0000: (),
    0b1111: (),
    0b0001: ((0, 1, 2),),
    0b1110: ((0, 2, 1),),
    0b0010: ((0, 4, 3),),
    0b1101: ((0, 3, 4),),
    0b0100: ((1, 3, 5),),
    0b1011: ((1, 5, 3),),
    0b1000: ((2, 5, 4),),
    0b0111: ((2, 4, 5),),
    0b0011: ((1, 4, 2), (1, 3, 4)),
    0b1100: ((1, 2, 4), (1, 4, 3)),
    0b0101: ((0, 3, 2), (2, 3, 5)),
    0b1010: ((0, 2, 3), (2, 5, 3)),
    0b1001: ((0, 1, 4), (4, 1, 5)),
    0b0110: ((0, 4, 1), (4, 5, 1)),
}

MAX_RENDER_POINTS = 3_000_000     # above this, auto-subsample before meshing

# Facet size to aim for. Much finer than this and the triangles fall below a
# pixel on screen: they cost time to paint and, being antialiased at partial
# coverage, render as a speckle rather than a solid surface.
TARGET_SPACING_ANGSTROM = 0.30


def suggested_subsample(field: VolumetricField) -> tuple[int, int, int]:
    """Per-axis rendering strides giving facets of a usable size.

    Per-axis rather than a single factor because VASP grids are routinely
    anisotropic: the graphene slab here is 36×36×200, sampling c twice as
    finely as a and b. One shared stride would either leave c over-tessellated
    or coarsen a and b past the point of usefulness.
    """
    strides = []
    for spacing in field.grid_spacing:
        if spacing <= 0:
            strides.append(1)
        else:
            strides.append(int(max(1, round(TARGET_SPACING_ANGSTROM / spacing))))
    return tuple(strides)  # type: ignore[return-value]


@dataclass
class SurfaceMesh:
    """A triangulated isosurface in cartesian coordinates."""

    vertices: np.ndarray      # (n_vertices, 3) Å
    triangles: np.ndarray     # (n_triangles, 3) indices into vertices
    normals: np.ndarray       # (n_vertices, 3) unit vectors
    isovalue: float = 0.0

    @property
    def n_triangles(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def is_empty(self) -> bool:
        return self.triangles.size == 0

    def triangle_centroids(self) -> np.ndarray:
        return self.vertices[self.triangles].mean(axis=1)

    def triangle_normals(self) -> np.ndarray:
        """Per-face normals, averaged from the smooth vertex normals.

        Using the vertex normals rather than the geometric face normal keeps
        shading continuous across the facets, so a coarse mesh still reads as
        a curved surface instead of a faceted crystal.
        """
        averaged = self.normals[self.triangles].mean(axis=1)
        lengths = np.linalg.norm(averaged, axis=1, keepdims=True)
        return averaged / np.where(lengths > 1e-12, lengths, 1.0)


def _padded(values: np.ndarray) -> np.ndarray:
    """Wrap one extra layer on each far face.

    The field is periodic, so without this the surface would be cut off at the
    cell boundary and appear to have flat lids wherever it crosses a face.
    """
    return np.pad(values, ((0, 1), (0, 1), (0, 1)), mode="wrap")


def _gradient_normals(values: np.ndarray) -> np.ndarray:
    """∇f at every grid point, evaluated periodically.

    The gradient of the scalar field is normal to its level sets, so
    interpolating it at each vertex gives smooth normals directly — no
    face-normal averaging pass, and no seams where facets meet.
    """
    gradients = np.gradient(values, edge_order=1)
    return np.stack(gradients, axis=-1)


def _weld(
    vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray, decimals: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge coincident vertices into a connected mesh.

    Rounding before comparison is safe here because the coordinates are grid
    indices interpolated along an edge: two triangles meeting on a shared edge
    compute the identical crossing point from the identical pair of corner
    values, so their vertices agree to floating-point round-off, far below the
    tolerance. Normals are averaged over the merged copies, which is what makes
    shading continuous across facets.
    """
    keys = np.round(vertices, decimals)
    _, first_index, inverse = np.unique(
        keys, axis=0, return_index=True, return_inverse=True
    )
    inverse = inverse.reshape(-1)

    merged = vertices[first_index]
    accumulated = np.zeros_like(merged)
    np.add.at(accumulated, inverse, normals)
    counts = np.zeros(len(merged))
    np.add.at(counts, inverse, 1)
    averaged = accumulated / np.where(counts > 0, counts, 1)[:, None]

    return merged, inverse[faces], averaged


def extract_isosurface(
    field: VolumetricField,
    isovalue: float,
    subsample: int | tuple[int, int, int] = 1,
    max_points: int = MAX_RENDER_POINTS,
) -> SurfaceMesh:
    """Triangulate ``field`` at ``isovalue``.

    ``subsample`` coarsens the grid before meshing — one factor for all three
    axes, or one per axis. It affects only what is drawn, never any reported
    quantity. When the grid is still very large a further automatic step is
    applied so a 300³ CHGCAR cannot lock up the UI.
    """
    values = field.values
    if isinstance(subsample, (int, np.integer)):
        steps = [max(1, int(subsample))] * 3
    else:
        steps = [max(1, int(s)) for s in subsample]

    # An automatic floor on top of the caller's choice: meshing tens of
    # millions of points interactively is not viable whatever the control says.
    while values[:: steps[0], :: steps[1], :: steps[2]].size > max_points:
        steps[int(np.argmin(steps))] += 1

    grid = np.ascontiguousarray(
        values[:: steps[0], :: steps[1], :: steps[2]], dtype=np.float64
    )

    # Nudge the level off any exact tie with a sample value. When the isovalue
    # coincides with a grid value the surface passes exactly through that
    # corner: several tetrahedra then emit the identical vertex, welding fuses
    # them into a non-manifold junction, and the degenerate zero-area facets
    # left behind confuse anything that later relies on adjacency. The shift is
    # far below the resolution the surface is meaningful at.
    isovalue = float(isovalue)
    if np.any(grid == isovalue):
        span = float(np.ptp(grid)) or 1.0
        isovalue += span * 1e-9

    shape = np.array(grid.shape)
    if np.any(shape < 2):
        return SurfaceMesh(
            vertices=np.zeros((0, 3)),
            triangles=np.zeros((0, 3), dtype=int),
            normals=np.zeros((0, 3)),
            isovalue=isovalue,
        )

    padded = _padded(grid)
    gradient = _gradient_normals(padded)

    nx, ny, nz = shape
    # Corner values for every cell, as eight (nx, ny, nz) arrays.
    corner_values = np.empty((8, nx, ny, nz))
    corner_index = np.empty((8, nx, ny, nz), dtype=np.int64)
    flat_shape = padded.shape
    for position, (di, dj, dk) in enumerate(_CORNERS):
        corner_values[position] = padded[
            di : di + nx, dj : dj + ny, dk : dk + nz
        ]
        ii, jj, kk = np.meshgrid(
            np.arange(nx) + di, np.arange(ny) + dj, np.arange(nz) + dk, indexing="ij"
        )
        corner_index[position] = np.ravel_multi_index((ii, jj, kk), flat_shape)

    inside = corner_values >= isovalue
    corner_values = corner_values.reshape(8, -1)
    corner_index = corner_index.reshape(8, -1)
    inside = inside.reshape(8, -1)

    flat_gradient = gradient.reshape(-1, 3)
    flat_positions = np.stack(
        np.unravel_index(np.arange(padded.size), flat_shape), axis=-1
    ).astype(np.float64)

    vertex_chunks: list[np.ndarray] = []
    normal_chunks: list[np.ndarray] = []
    total = 0
    triangles: list[np.ndarray] = []

    for tetra in _TETRAHEDRA:
        local_values = corner_values[list(tetra)]      # (4, n_cells)
        local_inside = inside[list(tetra)]
        local_index = corner_index[list(tetra)]

        case = (
            local_inside[0].astype(np.int64)
            | (local_inside[1].astype(np.int64) << 1)
            | (local_inside[2].astype(np.int64) << 2)
            | (local_inside[3].astype(np.int64) << 3)
        )

        for pattern, faces in _TET_CASES.items():
            if not faces:
                continue
            selected = np.nonzero(case == pattern)[0]
            if selected.size == 0:
                continue

            # Interpolate once per crossed edge for the whole batch.
            edge_points: dict[int, np.ndarray] = {}
            edge_normals: dict[int, np.ndarray] = {}
            needed = {edge for face in faces for edge in face}
            for edge in needed:
                a, b = _TET_EDGES[edge]
                va = local_values[a][selected]
                vb = local_values[b][selected]
                denominator = vb - va
                t = np.where(
                    np.abs(denominator) > 1e-12, (isovalue - va) / denominator, 0.5
                )
                t = np.clip(t, 0.0, 1.0)[:, None]

                ia = local_index[a][selected]
                ib = local_index[b][selected]
                edge_points[edge] = (
                    flat_positions[ia] * (1.0 - t) + flat_positions[ib] * t
                )
                edge_normals[edge] = (
                    flat_gradient[ia] * (1.0 - t) + flat_gradient[ib] * t
                )

            for face in faces:
                indices = []
                for edge in face:
                    vertex_chunks.append(edge_points[edge])
                    normal_chunks.append(edge_normals[edge])
                    indices.append(np.arange(total, total + selected.size))
                    total += selected.size
                triangles.append(np.stack(indices, axis=-1))

    if not vertex_chunks:
        return SurfaceMesh(
            vertices=np.zeros((0, 3)),
            triangles=np.zeros((0, 3), dtype=int),
            normals=np.zeros((0, 3)),
            isovalue=isovalue,
        )

    grid_vertices = np.concatenate(vertex_chunks, axis=0)
    raw_normals = np.concatenate(normal_chunks, axis=0)
    faces_array = np.concatenate(triangles, axis=0)

    # Weld coincident vertices. Each triangle is emitted with its own copy of
    # every corner, so without this the mesh is a pile of disconnected facets
    # that merely *look* joined: neighbouring triangles share no index, so any
    # operation needing adjacency is wrong. Smoothing is the clearest victim —
    # a vertex's only "neighbours" would be the two other corners of its own
    # triangle, so every facet would collapse toward its own centroid.
    grid_vertices, faces_array, raw_normals = _weld(
        grid_vertices, faces_array, raw_normals
    )

    # Grid index -> fractional -> cartesian. Doing the lattice multiplication
    # here is what makes non-orthogonal cells correct.
    # Grid index -> fractional, using each axis's own coarsened point count.
    fractional = grid_vertices / np.array(
        [grid.shape[i] for i in range(3)], dtype=float
    )
    cartesian = fractional @ field.lattice

    # The gradient is with respect to grid indices; converting it to a real
    # direction needs the inverse-transpose of the lattice, exactly as for any
    # covariant vector. Skipping this tilts the lighting on a sheared cell.
    inverse = np.linalg.inv(field.lattice)
    normals = raw_normals @ inverse
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = -normals / np.where(lengths > 1e-12, lengths, 1.0)

    log.info(
        "Isosurface at %.4g: %d triangles from a %d×%d×%d grid (strides %s)",
        isovalue,
        faces_array.shape[0],
        *grid.shape,
        "×".join(str(s) for s in steps),
    )
    return SurfaceMesh(
        vertices=cartesian,
        triangles=faces_array,
        normals=normals,
        isovalue=float(isovalue),
    )


def smooth_mesh(mesh: SurfaceMesh, strength: float, iterations: int = 2) -> SurfaceMesh:
    """Taubin (λ/μ) smoothing of vertex positions.

    Deliberately a *mesh* operation, not a blur of the scalar field: it changes
    how the surface looks without touching the data any statistic, integral or
    slice is computed from. Smoothing the field instead would silently alter
    the numbers reported elsewhere on the page.

    Taubin rather than plain Laplacian because Laplacian smoothing shrinks.
    That is harmless on a fat convex blob and destructive on a thin one: a
    density isosurface around a 2D sheet has two faces a fraction of an
    ångström apart, and successive shrinking pulls them into each other,
    collapsing facets into slivers. Following each inward pass (λ > 0) with a
    slightly larger outward pass (μ < 0) removes high-frequency noise while
    leaving the surface's scale essentially unchanged.
    """
    if mesh.is_empty or strength <= 0.0:
        return mesh

    vertices = mesh.vertices.copy()
    faces = mesh.triangles
    lam = float(np.clip(strength, 0.0, 1.0)) * 0.5
    mu = -lam * 1.02          # slightly larger, which is what cancels shrinkage

    # Neighbour sums are the same every pass, so the index arrays are built once.
    first_indices = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    second_indices = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    counts = np.zeros(len(vertices))
    np.add.at(counts, first_indices, 1)
    np.add.at(counts, second_indices, 1)
    safe = np.where(counts > 0, counts, 1)[:, None]

    def pass_once(points: np.ndarray, factor: float) -> np.ndarray:
        accumulated = np.zeros_like(points)
        np.add.at(accumulated, first_indices, points[second_indices])
        np.add.at(accumulated, second_indices, points[first_indices])
        return points + factor * (accumulated / safe - points)

    for _ in range(max(1, iterations)):
        vertices = pass_once(vertices, lam)
        vertices = pass_once(vertices, mu)

    return SurfaceMesh(
        vertices=vertices,
        triangles=faces,
        normals=mesh.normals,
        isovalue=mesh.isovalue,
    )
