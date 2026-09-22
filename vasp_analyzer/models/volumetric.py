"""Real-space scalar fields on a VASP FFT grid.

CHGCAR, CHG, PARCHG, ELFCAR and LOCPOT all share one layout: a POSCAR header
followed by a uniform grid sampled in *fractional* coordinates. Modelling them
as one :class:`VolumetricField` rather than as separate per-file types is what
lets the page's rendering and analysis widgets work on "a scalar field" and
gain new sources (a density difference, an electrostatic potential) without
being rewritten.

The distinction the UI actually depends on is not which file a field came from
but what kind of quantity it is — whether it is signed, whether it integrates
to something meaningful, what units it carries. That is :class:`FieldKind`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .structure import StructureData


class FieldKind(str, Enum):
    """What a scalar field physically represents.

    Drives units, colour-map family, whether an integral is meaningful, and
    whether the display scale should be centred on zero.
    """

    CHARGE = "charge"                  # electron density, e/Å³, non-negative
    MAGNETIZATION = "magnetization"    # spin density, e/Å³, signed
    PARTIAL_CHARGE = "partial_charge"  # band/k-resolved density, e/Å³
    ELF = "elf"                        # electron localisation function, 0-1
    POTENTIAL = "potential"            # electrostatic potential, eV
    DIFFERENCE = "difference"          # a difference of two densities, signed

    @property
    def label(self) -> str:
        return {
            FieldKind.CHARGE: "Total Charge Density",
            FieldKind.MAGNETIZATION: "Magnetization Density",
            FieldKind.PARTIAL_CHARGE: "Partial Charge Density",
            FieldKind.ELF: "Electron Localization Function",
            FieldKind.POTENTIAL: "Electrostatic Potential",
            FieldKind.DIFFERENCE: "Charge Density Difference",
        }[self]

    @property
    def units(self) -> str:
        """Unit string, empty for dimensionless fields.

        ELF is a dimensionless ratio; labelling it e/Å³ — as a naive reuse of
        the density UI would — states something false about the quantity.
        """
        return {
            FieldKind.CHARGE: "e/Å³",
            FieldKind.MAGNETIZATION: "e/Å³",
            FieldKind.PARTIAL_CHARGE: "e/Å³",
            FieldKind.ELF: "",
            FieldKind.POTENTIAL: "eV",
            FieldKind.DIFFERENCE: "e/Å³",
        }[self]

    @property
    def is_signed(self) -> bool:
        """True when negative values are physically expected.

        A total electron density is non-negative by construction, so it gets a
        sequential colour map; a magnetization or a difference is meaningfully
        two-sided and gets a diverging one centred on zero.
        """
        return self in (FieldKind.MAGNETIZATION, FieldKind.DIFFERENCE, FieldKind.POTENTIAL)

    @property
    def integrates_to_electrons(self) -> bool:
        """True when ∫f dV counts electrons and can be checked against NELECT."""
        return self in (FieldKind.CHARGE, FieldKind.PARTIAL_CHARGE)

    @property
    def bounded_range(self) -> tuple[float, float] | None:
        """The field's own natural range, when it has one."""
        return (0.0, 1.0) if self is FieldKind.ELF else None


@dataclass
class VolumetricField:
    """One scalar field sampled on the FFT grid of one structure.

    ``values`` is indexed ``[i, j, k]`` along the a, b and c lattice
    directions, already normalised into :attr:`kind`'s units. Grid point
    ``(i, j, k)`` sits at fractional coordinate ``(i/nx, j/ny, k/nz)``, so
    converting to cartesian is a multiplication by the lattice — which is what
    keeps non-orthogonal cells correct.
    """

    structure: StructureData
    values: np.ndarray                 # (nx, ny, nz)
    kind: FieldKind = FieldKind.CHARGE
    source_file: str = ""
    component: str = ""                # e.g. "total", "magnetization", "m_x"
    metadata: dict = field(default_factory=dict)

    # -- grid ---------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.values.shape)  # type: ignore[return-value]

    @property
    def n_points(self) -> int:
        return int(self.values.size)

    @property
    def lattice(self) -> np.ndarray:
        return self.structure.lattice

    @property
    def volume(self) -> float:
        return self.structure.volume

    @property
    def voxel_volume(self) -> float:
        """Volume each grid point represents, in Å³."""
        return self.volume / max(self.n_points, 1)

    @property
    def grid_spacing(self) -> tuple[float, float, float]:
        """Spacing along each lattice direction in Å.

        Reported per axis rather than as one number: the grid is uniform in
        *fractional* coordinates, so a cell with unequal lattice vectors — or a
        slab with a long vacuum axis — has genuinely different spacings, and a
        single averaged figure would be misleading.
        """
        lengths = self.structure.lattice_parameters
        nx, ny, nz = self.shape
        return (lengths[0] / nx, lengths[1] / ny, lengths[2] / nz)

    @property
    def is_orthogonal(self) -> bool:
        angles = self.structure.lattice_angles
        return all(abs(angle - 90.0) < 1e-6 for angle in angles)

    # -- statistics -----------------------------------------------------------

    @property
    def minimum(self) -> float:
        return float(np.min(self.values))

    @property
    def maximum(self) -> float:
        return float(np.max(self.values))

    @property
    def mean(self) -> float:
        return float(np.mean(self.values))

    def integral(self) -> float:
        """∫ f dV over the cell.

        For a charge density this is the number of electrons the grid
        represents. The grid is uniform in fractional space, so every point
        carries the same weight and the sum times the voxel volume is exact to
        the grid's own resolution.
        """
        return float(np.sum(self.values)) * self.voxel_volume

    def percentile_range(self, low: float = 50.0, high: float = 99.9) -> tuple[float, float]:
        """A practical display range from the data's own distribution.

        A fixed 0.001–0.1 slider is meaningless across systems: densities in a
        molecular crystal and a dense metal differ by orders of magnitude. The
        upper percentile rather than the maximum avoids a range dominated by
        the few points nearest a nucleus, where the PAW density spikes.
        """
        finite = self.values[np.isfinite(self.values)]
        if finite.size == 0:
            return (0.0, 1.0)
        lo = float(np.percentile(finite, low))
        hi = float(np.percentile(finite, high))
        if hi <= lo:
            hi = lo + max(abs(lo) * 0.1, 1e-6)
        return (lo, hi)

    def suggested_isovalue(self) -> float:
        """A starting isovalue that shows structure rather than nothing.

        For a density, a high percentile sits inside the bonding regions; the
        median is nearly vacuum in most cells and would draw a surface hugging
        the cell walls.
        """
        bounded = self.kind.bounded_range
        if bounded is not None:
            return 0.5 * (bounded[0] + bounded[1])
        if self.kind.is_signed:
            return float(np.percentile(np.abs(self.values), 99.0)) * 0.5
        return float(np.percentile(self.values, 92.0))

    # -- sampling ---------------------------------------------------------------

    def slice_plane(self, axis: int, fraction: float) -> np.ndarray:
        """A 2D cut at fractional position ``fraction`` along ``axis``.

        Linearly interpolated between the two bracketing grid planes and
        wrapped periodically, so the plane can be moved continuously rather
        than snapping to grid points.
        """
        n = self.shape[axis]
        position = (float(fraction) % 1.0) * n
        lower = int(np.floor(position)) % n
        upper = (lower + 1) % n
        weight = position - np.floor(position)

        first = np.take(self.values, lower, axis=axis)
        second = np.take(self.values, upper, axis=axis)
        return (1.0 - weight) * first + weight * second

    def planar_average(self, axis: int) -> np.ndarray:
        """Average over the two directions perpendicular to ``axis``.

        Because the grid is uniform in fractional coordinates, every point on a
        given plane represents an equal area of that plane whatever the cell
        shape, so a plain mean is already the correct planar average — no
        Jacobian is needed even for a triclinic cell.
        """
        others = tuple(i for i in range(3) if i != axis)
        return np.mean(self.values, axis=others)

    def axis_distance(self, axis: int) -> np.ndarray:
        """Distance along ``axis`` for each grid plane, in Å.

        Measured along the lattice vector itself. For a non-orthogonal cell
        that is the physically meaningful coordinate for a planar average: the
        planes are separated along the vector, not along a cartesian axis.
        """
        length = self.structure.lattice_parameters[axis]
        n = self.shape[axis]
        return np.arange(n, dtype=float) * (length / n)

    def subsample(self, factor: int) -> "VolumetricField":
        """A coarser copy for rendering only.

        Quantitative results are always computed from the full-resolution
        field; this exists so a 300³ grid can be meshed interactively without
        changing any reported number.
        """
        if factor <= 1:
            return self
        coarse = self.values[::factor, ::factor, ::factor]
        return VolumetricField(
            structure=self.structure,
            values=np.ascontiguousarray(coarse),
            kind=self.kind,
            source_file=self.source_file,
            component=self.component,
            metadata={**self.metadata, "subsampled_by": factor},
        )


@dataclass
class VolumetricFile:
    """Every field one file contained.

    A spin-polarised CHGCAR holds the total density *and* the magnetization
    density; a non-collinear one holds three magnetization components. Keeping
    them together — rather than returning only the first — is what lets the
    Data Type control offer exactly what the file actually has.
    """

    path_name: str
    fields: list[VolumetricField] = field(default_factory=list)
    structure: StructureData | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_spin_polarized(self) -> bool:
        return any(f.kind is FieldKind.MAGNETIZATION for f in self.fields)

    @property
    def n_components(self) -> int:
        return len(self.fields)

    def field_for(self, component: str) -> VolumetricField | None:
        for candidate in self.fields:
            if candidate.component == component:
                return candidate
        return self.fields[0] if self.fields else None

    @property
    def spin_mode(self) -> str:
        """"Non-polarised" / "Collinear" / "Non-collinear", from block count."""
        magnetic = sum(1 for f in self.fields if f.kind is FieldKind.MAGNETIZATION)
        if magnetic >= 3:
            return "Non-collinear"
        if magnetic == 1:
            return "Collinear"
        return "Non-polarised"
