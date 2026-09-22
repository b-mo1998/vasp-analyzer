"""Phonon data model.

Frequencies are stored in **THz**, with imaginary modes carried as *negative*
numbers. Phonopy reports an imaginary frequency this way already (it takes the
signed square root of a negative eigenvalue of the dynamical matrix), and
keeping that convention means a dispersion plot is a plain line plot and an
instability is visible as a branch dipping below zero rather than as a
separately-encoded special case.

Eigenvectors are stored as **displacement** vectors, not the mass-weighted
eigenvectors of the dynamical matrix. What the dynamical matrix diagonalises
is mass-weighted, so its components must be divided by sqrt(mass) before they
describe how far each atom actually moves; drawing the raw components would
show a heavy atom moving as far as a light one for the same mode, which is
wrong in a way that looks plausible. Conversion happens once, at parse time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .structure import StructureData

# Below this magnitude a negative frequency is numerical noise in the acoustic
# modes at Γ rather than a real instability. Phonopy's own acoustic sum rule
# leaves residuals of this order, and calling them "imaginary" would report
# every well-converged calculation as unstable.
IMAGINARY_TOLERANCE_THZ = 0.1

# THz -> other units, for display only.
THZ_TO_CM1 = 33.35641
THZ_TO_MEV = 4.135667

# Display units. THz stays the internal representation; these only convert on
# the way to a label or an axis, so switching never touches the data.
UNIT_FACTORS = {"THz": 1.0, "cm⁻¹": THZ_TO_CM1, "meV": THZ_TO_MEV}


def convert_frequency(value_thz, units: str):
    """Convert a THz frequency (or array) into the chosen display unit."""
    return value_thz * UNIT_FACTORS.get(units, 1.0)


@dataclass
class PhononBand:
    """One q-point along a path, with every branch's frequency there."""

    qpoint: np.ndarray                     # (3,) fractional reciprocal coords
    distance: float                        # cumulative path length, arbitrary units
    frequencies: np.ndarray                # (n_branches,) THz, negative = imaginary
    # (n_branches, n_atoms, 3) complex displacement eigenvectors, or None when
    # the source carried frequencies only.
    eigenvectors: np.ndarray | None = None
    label: str = ""                        # high-symmetry label, when this q-point has one


@dataclass
class PhononDos:
    """Total and per-element phonon density of states."""

    frequencies: np.ndarray                # (n_points,) THz
    total: np.ndarray                      # (n_points,)
    projected: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def has_projection(self) -> bool:
        return bool(self.projected)


@dataclass
class PhononDataset:
    """Everything parsed from a phonon calculation."""

    structure: StructureData | None = None
    bands: list[PhononBand] = field(default_factory=list)
    dos: PhononDos | None = None
    # Frequencies sampled on a mesh, used to recompute a DOS at a chosen
    # smearing. (n_qpoints, n_branches) THz.
    mesh_frequencies: np.ndarray | None = None
    mesh_weights: np.ndarray | None = None
    # Per-atom element symbols for mesh projection, matching `structure`.
    mesh_eigenvectors: np.ndarray | None = None

    supercell: tuple[int, int, int] | None = None
    mesh: tuple[int, int, int] | None = None
    calculation: str = ""                  # e.g. "DFPT (Phonopy)"
    source_files: list[str] = field(default_factory=list)
    directory_name: str = ""

    # -- dimensions --------------------------------------------------------

    @property
    def n_qpoints(self) -> int:
        return len(self.bands)

    @property
    def n_branches(self) -> int:
        return int(self.bands[0].frequencies.size) if self.bands else 0

    @property
    def n_atoms(self) -> int:
        if self.structure is not None:
            return self.structure.n_atoms
        # 3N branches for N atoms — the only handle on atom count when the
        # structure itself was not readable.
        return self.n_branches // 3

    @property
    def has_eigenvectors(self) -> bool:
        return bool(self.bands) and self.bands[0].eigenvectors is not None

    @property
    def has_dispersion(self) -> bool:
        return self.n_qpoints > 1

    # -- derived quantities ------------------------------------------------

    @property
    def frequency_matrix(self) -> np.ndarray:
        """(n_qpoints, n_branches) THz."""
        if not self.bands:
            return np.zeros((0, 0))
        return np.array([band.frequencies for band in self.bands])

    @property
    def distances(self) -> np.ndarray:
        return np.array([band.distance for band in self.bands])

    @property
    def max_frequency(self) -> float:
        matrix = self.frequency_matrix
        return float(matrix.max()) if matrix.size else 0.0

    def imaginary_count(self, tolerance: float = IMAGINARY_TOLERANCE_THZ) -> int:
        """Modes genuinely below zero, ignoring near-zero acoustic noise."""
        matrix = self.frequency_matrix
        if not matrix.size:
            return 0
        return int((matrix < -abs(tolerance)).sum())

    @property
    def is_stable(self) -> bool:
        return self.imaginary_count() == 0

    def labelled_qpoints(self) -> list[tuple[int, str]]:
        """(index, label) for every q-point carrying a high-symmetry label."""
        return [(i, b.label) for i, b in enumerate(self.bands) if b.label]

    def group_velocity(self, qpoint_index: int, branch_index: int) -> float | None:
        """|dω/dq| along the path at this point, in km/s.

        A central difference along the plotted path — which is a *directional*
        derivative along that path, not the full gradient. It is the only
        velocity derivable from band data alone, so it is reported as such and
        withheld at path discontinuities where the difference is meaningless.
        """
        if not (0 <= qpoint_index < self.n_qpoints):
            return None
        lower = max(0, qpoint_index - 1)
        upper = min(self.n_qpoints - 1, qpoint_index + 1)
        if upper == lower:
            return None

        distance = self.bands[upper].distance - self.bands[lower].distance
        if distance <= 1e-12:
            # A discontinuity in the path (two segments meeting at one point)
            # has zero separation and no defined slope.
            return None
        delta_f = (
            self.bands[upper].frequencies[branch_index]
            - self.bands[lower].frequencies[branch_index]
        )
        # Path distances from phonopy are in 1/Å (2π excluded); THz·Å = 100 m/s.
        return abs(float(delta_f) / distance) * 0.1

    def degeneracy(self, qpoint_index: int, branch_index: int, tolerance: float = 0.05) -> int:
        """How many branches share this frequency at this q-point."""
        if not (0 <= qpoint_index < self.n_qpoints):
            return 1
        frequencies = self.bands[qpoint_index].frequencies
        target = frequencies[branch_index]
        return int((np.abs(frequencies - target) <= tolerance).sum())

    def mode_type(self, qpoint_index: int, branch_index: int) -> str:
        """Acoustic or optical, claimed only where it is actually decidable.

        The three lowest branches are acoustic *at Γ*, where they go to zero.
        Away from Γ branches cross and mix, and an index is no longer a
        reliable label — so anywhere else this reports "—" rather than
        guessing from the branch number.
        """
        if not (0 <= qpoint_index < self.n_qpoints):
            return "—"
        qpoint = self.bands[qpoint_index].qpoint
        at_gamma = bool(np.allclose(qpoint, 0.0, atol=1e-6))
        if not at_gamma:
            return "—"
        return "Acoustic" if branch_index < 3 else "Optical"


@dataclass(frozen=True)
class SelectedMode:
    """The one piece of selection state the whole page reads from."""

    qpoint_index: int = 0
    branch_index: int = 0
