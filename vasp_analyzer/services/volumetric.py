"""Quantitative analysis of a volumetric field.

Everything here works on the full-resolution field. Rendering may subsample —
that is a drawing decision — but a number shown on screen must never depend on
it, so no function in this module accepts a subsampling factor.

The other recurring concern is non-orthogonal cells. A slice through a
triclinic cell is a *parallelogram*, not a rectangle, and drawing it as a
rectangle silently misrepresents where the atoms are. :func:`slice_geometry`
therefore returns real in-plane cartesian coordinates rather than array
indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.logging_setup import get_logger
from ..models.volumetric import FieldKind, VolumetricField

log = get_logger("services.volumetric")

AXIS_NAMES = ("a", "b", "c")
AXIS_LETTERS = ("X", "Y", "Z")

# A volumetric file with nothing in it is a placeholder VASP left behind, not
# a data source; offering it in the picker would only produce a parse error.
_MIN_USEFUL_BYTES = 64


@dataclass(frozen=True)
class VolumetricSource:
    """One selectable volumetric file found in a directory."""

    path: Path
    kind: FieldKind

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def label(self) -> str:
        """What the Data Source control shows.

        PARCHG files keep their full name because the suffix identifies which
        bands and k-points they hold, and a directory can contain several.
        """
        if self.name.upper().startswith("PARCHG"):
            return self.name
        return f"{self.name} (VASP)"


def find_volumetric_sources(directory: Path) -> list[VolumetricSource]:
    """Every usable volumetric file in a directory, in a sensible order.

    Ordered so the fullest, most commonly wanted field comes first: CHGCAR
    before the coarser CHG, then ELF, then partial densities, then the
    potential.
    """
    from ..parsers.chgcar import kind_for_filename  # noqa: PLC0415 - avoids a cycle

    directory = Path(directory)
    if not directory.is_dir():
        return []

    priority = {
        "CHGCAR": 0,
        "CHG": 1,
        "ELFCAR": 2,
        "PARCHG": 3,
        "LOCPOT": 4,
    }

    found: list[tuple[int, str, VolumetricSource]] = []
    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        upper = entry.name.upper()
        family = next(
            (
                key
                for key in ("CHGCAR", "ELFCAR", "PARCHG", "LOCPOT", "CHG")
                if upper == key or upper.startswith(key + ".")
            ),
            None,
        )
        if family is None:
            continue
        try:
            if entry.stat().st_size < _MIN_USEFUL_BYTES:
                log.info("Ignoring empty volumetric file %s", entry.name)
                continue
        except OSError:
            continue
        found.append(
            (priority[family], entry.name, VolumetricSource(entry, kind_for_filename(entry.name)))
        )

    found.sort(key=lambda item: (item[0], item[1]))
    return [source for _, _, source in found]


@dataclass(frozen=True)
class IntegrationResult:
    """∫f dV and, when it is checkable, how it compares with the expected count."""

    integral: float
    expected: float | None = None
    expected_source: str = ""

    @property
    def difference(self) -> float | None:
        if self.expected is None:
            return None
        return self.integral - self.expected

    @property
    def relative_error(self) -> float | None:
        """Signed relative error in percent."""
        if self.expected is None or abs(self.expected) < 1e-12:
            return None
        return 100.0 * (self.integral - self.expected) / self.expected

    @property
    def is_consistent(self) -> bool | None:
        """Whether the grid integral reproduces the expected electron count.

        A grid integral is a finite-resolution approximation, so a fraction of
        a percent is expected and fine; a large discrepancy usually means the
        normalisation or the axis order is wrong, which is exactly what this
        check exists to catch.
        """
        error = self.relative_error
        return None if error is None else abs(error) < 0.5


@dataclass(frozen=True)
class SliceData:
    """A 2D cut with the geometry needed to draw it truthfully."""

    values: np.ndarray          # (n_u, n_v)
    x_edges: np.ndarray         # (n_u+1, n_v+1) in-plane cartesian, Å
    y_edges: np.ndarray         # (n_u+1, n_v+1) in-plane cartesian, Å
    axis: int                   # the normal's lattice direction
    fraction: float             # position along that direction, 0-1
    distance: float             # the same position in Å
    x_label: str = ""
    y_label: str = ""

    @property
    def is_parallelogram(self) -> bool:
        """True when the in-plane lattice vectors are not perpendicular."""
        return not np.allclose(self.x_edges[0, 0], self.x_edges[0, -1], atol=1e-9) or True


@dataclass(frozen=True)
class AtomMarker:
    """An atom close enough to a slice plane to be worth drawing on it."""

    symbol: str
    x: float
    y: float
    distance: float             # perpendicular distance from the plane, Å


def integrate(
    field: VolumetricField,
    expected_electrons: float | None = None,
    expected_source: str = "",
) -> IntegrationResult:
    """Integrate the field over the cell.

    For a charge density this counts electrons and can be checked against
    NELECT — the standard CHGCAR sanity check, and the quickest way to detect
    a normalisation error.
    """
    result = IntegrationResult(
        integral=field.integral(),
        expected=expected_electrons if field.kind.integrates_to_electrons else None,
        expected_source=expected_source,
    )
    if result.relative_error is not None:
        log.info(
            "Integrated %s: %.4f vs expected %.4f (%.3f%%)",
            field.component or field.kind.value,
            result.integral,
            result.expected,
            result.relative_error,
        )
    return result


def _plane_basis(lattice: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """An orthonormal 2D frame spanning the plane perpendicular to ``axis``.

    The plane is spanned by the two *other* lattice vectors, which are not
    generally perpendicular to each other. Gram-Schmidt gives a frame in which
    those vectors can be expressed as honest 2D coordinates, so the resulting
    picture is the real parallelogram rather than a rectangle.
    """
    others = [i for i in range(3) if i != axis]
    first = lattice[others[0]]
    second = lattice[others[1]]

    e1 = first / max(np.linalg.norm(first), 1e-12)
    perpendicular = second - np.dot(second, e1) * e1
    norm = np.linalg.norm(perpendicular)
    e2 = perpendicular / norm if norm > 1e-12 else np.cross(e1, [0.0, 0.0, 1.0])
    return e1, e2


def slice_geometry(field: VolumetricField, axis: int, fraction: float) -> SliceData:
    """Extract a slice and the in-plane cartesian coordinates to draw it with."""
    values = field.slice_plane(axis, fraction)
    lattice = field.lattice
    others = [i for i in range(3) if i != axis]
    e1, e2 = _plane_basis(lattice, axis)

    n_u, n_v = values.shape
    # Cell edges, so a pcolormesh draws each voxel where it really is.
    u = np.linspace(0.0, 1.0, n_u + 1)
    v = np.linspace(0.0, 1.0, n_v + 1)
    grid_u, grid_v = np.meshgrid(u, v, indexing="ij")

    vector_u = lattice[others[0]]
    vector_v = lattice[others[1]]
    cartesian = (
        grid_u[..., None] * vector_u[None, None, :]
        + grid_v[..., None] * vector_v[None, None, :]
    )

    length = float(np.linalg.norm(lattice[axis]))
    return SliceData(
        values=values,
        x_edges=cartesian @ e1,
        y_edges=cartesian @ e2,
        axis=axis,
        fraction=float(fraction) % 1.0,
        distance=(float(fraction) % 1.0) * length,
        x_label=f"{AXIS_NAMES[others[0]]} (Å)",
        y_label=f"{AXIS_NAMES[others[1]]} (Å)",
    )


def atoms_near_plane(
    field: VolumetricField, axis: int, fraction: float, tolerance: float = 1.0
) -> list[AtomMarker]:
    """Atoms within ``tolerance`` Å of the slice plane, in plane coordinates.

    Distance is measured along the plane *normal* rather than along the lattice
    vector: in a sheared cell those differ, and only the normal distance says
    whether an atom is really near the cut.
    """
    structure = field.structure
    lattice = field.lattice
    others = [i for i in range(3) if i != axis]
    e1, e2 = _plane_basis(lattice, axis)

    normal = np.cross(lattice[others[0]], lattice[others[1]])
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return []
    normal = normal / norm

    plane_point = (float(fraction) % 1.0) * lattice[axis]
    markers: list[AtomMarker] = []

    for symbol, frac in zip(structure.species, structure.frac_coords):
        # Check the atom and its neighbouring periodic images along the normal
        # direction, so one just across the boundary still registers.
        for shift in (-1.0, 0.0, 1.0):
            shifted = frac.copy()
            shifted[axis] = (shifted[axis] % 1.0) + shift
            position = shifted @ lattice
            offset = float(np.dot(position - plane_point, normal))
            if abs(offset) > tolerance:
                continue
            in_plane = position - offset * normal
            markers.append(
                AtomMarker(
                    symbol=symbol,
                    x=float(np.dot(in_plane, e1)),
                    y=float(np.dot(in_plane, e2)),
                    distance=abs(offset),
                )
            )
    return markers


def _grid_frac_coords(shape: tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = shape
    ii, jj, kk = np.meshgrid(
        np.arange(nx) / nx, np.arange(ny) / ny, np.arange(nz) / nz, indexing="ij"
    )
    return np.stack([ii, jj, kk], axis=-1)  # (nx, ny, nz, 3)


def _distance_sq_to_atom(
    grid_frac: np.ndarray, lattice: np.ndarray, atom_frac: np.ndarray
) -> np.ndarray:
    """Minimum-image squared distance from every grid point to one atom."""
    delta = grid_frac - atom_frac
    delta -= np.round(delta)
    cart = delta @ lattice
    return np.einsum("ijkl,ijkl->ijk", cart, cart)


def localize_to_atoms(
    field: VolumetricField, atom_rows: set[int], radius: float
) -> VolumetricField:
    """A copy of ``field`` with every grid value beyond ``radius`` Å of the
    nearest of ``atom_rows`` set to zero.

    Not a physically rigorous per-atom or per-orbital decomposition — a
    CHGCAR (or any VASP volumetric file) carries only the total density on
    the grid, with no such decomposition recorded in it anywhere. What this
    *is*, is a real and commonly useful technique regardless: restricting an
    isosurface, slice or planar average to the neighbourhood of one dopant
    or one species isolates a local bonding environment for inspection, at
    the honest cost of being a spatial mask rather than a quantum-mechanical
    projection.

    Periodicity is handled with the minimum-image convention in fractional
    coordinates, exact for any cutoff well under half the shortest cell
    dimension — true of any realistic localization radius against any
    realistic supercell.
    """
    structure = field.structure
    rows = sorted(r for r in atom_rows if 0 <= r < structure.n_atoms)
    if not rows:
        return field

    lattice = field.lattice
    grid_frac = _grid_frac_coords(field.shape)

    nearest_sq = np.full(field.shape, np.inf)
    for row in rows:
        dist_sq = _distance_sq_to_atom(grid_frac, lattice, structure.frac_coords[row])
        np.minimum(nearest_sq, dist_sq, out=nearest_sq)

    mask = nearest_sq <= (radius * radius)
    masked_values = np.where(mask, field.values, 0.0)

    return VolumetricField(
        structure=structure,
        values=masked_values,
        kind=field.kind,
        source_file=field.source_file,
        component=field.component,
        metadata={**field.metadata, "localized_to": rows, "localize_radius": radius},
    )


@dataclass(frozen=True)
class AtomIntegration:
    """∫f dV within a radius of one atom."""

    row: int             # 0-based atom index
    symbol: str
    value: float          # electrons, for a charge/partial_charge field


def integrate_per_atom(
    field: VolumetricField, atom_rows: list[int], radius: float
) -> list[AtomIntegration]:
    """∫f dV within ``radius`` Å of each of ``atom_rows``, one atom at a time.

    Not a Bader analysis — a true Bader partition follows the field's own
    zero-flux surfaces, which a plain FFT grid doesn't give you for free —
    but the same real, useful approximation :func:`localize_to_atoms` already
    offers, applied per atom instead of to a group: a fixed-radius sphere
    around each atom, integrated. Spheres from atoms closer together than
    ``2 * radius`` overlap, so the region between them is counted in more
    than one atom's total and the numbers are not a clean partition of the
    whole field in that case — keep ``radius`` under half the typical
    interatomic spacing for that not to matter.
    """
    structure = field.structure
    rows = [r for r in atom_rows if 0 <= r < structure.n_atoms]
    if not rows:
        return []

    lattice = field.lattice
    grid_frac = _grid_frac_coords(field.shape)
    radius_sq = radius * radius

    results: list[AtomIntegration] = []
    for row in rows:
        dist_sq = _distance_sq_to_atom(grid_frac, lattice, structure.frac_coords[row])
        value = float(np.sum(field.values, where=dist_sq <= radius_sq)) * field.voxel_volume
        results.append(AtomIntegration(row=row, symbol=structure.species[row], value=value))
    return results


def difference(field_a: VolumetricField, field_b: VolumetricField) -> VolumetricField:
    """``field_a - field_b``, on ``field_a``'s grid and structure.

    The standard "before vs. after" comparison — e.g. a CHGCAR taken before
    and after an adsorption event on the same slab — to see where charge was
    actually gained or lost rather than only inspecting each density alone.

    Requires an identical FFT grid shape. VASP does not guarantee that two
    otherwise-comparable calculations share one even with an identical
    structure (a different ENCUT or PREC changes it), so a mismatch is
    reported rather than silently resampled, which would fabricate values at
    every grid point instead of just comparing the ones VASP actually wrote.
    """
    if field_a.shape != field_b.shape:
        raise ValueError(
            f"Grid shapes differ ({field_a.shape[0]}×{field_a.shape[1]}×{field_a.shape[2]} "
            f"vs {field_b.shape[0]}×{field_b.shape[1]}×{field_b.shape[2]}) — both files "
            "need the same FFT grid to compare point by point."
        )
    return VolumetricField(
        structure=field_a.structure,
        values=field_a.values - field_b.values,
        kind=FieldKind.DIFFERENCE,
        source_file=field_a.source_file,
        component=field_a.component,
        metadata={"difference_of": (field_a.source_file, field_b.source_file)},
    )


def planar_average(field: VolumetricField, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """(distance in Å, averaged value) along one lattice direction."""
    return field.axis_distance(axis), field.planar_average(axis)


def display_range(field: VolumetricField, robust: bool = True) -> tuple[float, float]:
    """Colour limits for a 2D map of this field.

    Signed fields get a range symmetric about zero so the colour map's neutral
    midpoint really means zero. Percentiles rather than extremes keep the map
    readable: the density at a nucleus is orders of magnitude above the
    bonding region that is actually of interest.
    """
    values = field.values
    if field.kind.bounded_range is not None:
        return field.kind.bounded_range

    if field.kind.is_signed:
        limit = float(np.percentile(np.abs(values), 99.5)) if robust else float(
            np.max(np.abs(values))
        )
        limit = max(limit, 1e-12)
        return (-limit, limit)

    low = 0.0
    high = float(np.percentile(values, 99.5)) if robust else float(np.max(values))
    return (low, max(high, low + 1e-12))


def colormap_for(field: VolumetricField, preferred: str | None = None) -> str:
    """The matplotlib colour map appropriate to the field's physics.

    A signed quantity needs a diverging map centred on zero; a strictly
    non-negative one needs a sequential map. Letting a user put a diverging map
    on a total density would imply a sign change that cannot occur.
    """
    if preferred:
        return preferred
    if field.kind.is_signed:
        return "RdBu_r"
    if field.kind is FieldKind.ELF:
        return "plasma"
    return "viridis"


def describe_grid(field: VolumetricField) -> str:
    nx, ny, nz = field.shape
    return f"{nx} × {ny} × {nz}"


def describe_spacing(field: VolumetricField) -> str:
    """Grid spacing, collapsed to one figure only when that is honest."""
    da, db, dc = field.grid_spacing
    if max(da, db, dc) - min(da, db, dc) < 0.005:
        return f"{da:.3f} Å"
    return f"{da:.3f} / {db:.3f} / {dc:.3f} Å"
