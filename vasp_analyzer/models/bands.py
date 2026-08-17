"""Band-structure data model."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class KPathLabel:
    """A high-symmetry point positioned along the plotted k-axis."""

    index: int          # index into the k-point list
    position: float     # x coordinate on the plotted axis
    label: str          # display text, e.g. "Γ"


@dataclass(frozen=True)
class BandGap:
    """A resolved band gap, with enough to say whether it's direct."""

    gap: float          # eV
    vbm: float          # eV, absolute
    cbm: float          # eV, absolute
    vbm_kpoint: int
    cbm_kpoint: int

    @property
    def is_direct(self) -> bool:
        return self.vbm_kpoint == self.cbm_kpoint


@dataclass
class BandData:
    """Eigenvalues on a k-point list, as read from EIGENVAL.

    ``energies`` is ``(n_spin, n_kpoints, n_bands)`` in absolute eV — exactly as
    VASP wrote them. Referencing to the Fermi level happens at display time so
    the user can change the reference without re-reading the file.
    """

    kpoints: np.ndarray            # (nk, 3) fractional
    weights: np.ndarray            # (nk,)
    energies: np.ndarray           # (nspin, nk, nbands)
    occupations: np.ndarray        # (nspin, nk, nbands)
    n_electrons: float
    n_ions: int = 0
    source: str = "EIGENVAL"

    # Axis information, filled in by :mod:`vasp_analyzer.services.kpath`
    x_values: np.ndarray | None = None
    labels: list[KPathLabel] = field(default_factory=list)
    segment_breaks: list[int] = field(default_factory=list)
    is_line_mode: bool = False
    x_axis_title: str = "k-point index"

    @property
    def n_spin(self) -> int:
        return int(self.energies.shape[0])

    @property
    def n_kpoints(self) -> int:
        return int(self.energies.shape[1])

    @property
    def n_bands(self) -> int:
        return int(self.energies.shape[2])

    @property
    def is_spin_polarized(self) -> bool:
        return self.n_spin > 1

    @property
    def energy_range(self) -> tuple[float, float]:
        return float(self.energies.min()), float(self.energies.max())

    def x(self) -> np.ndarray:
        """Plot coordinate for each k-point."""
        if self.x_values is not None:
            return self.x_values
        return np.arange(self.n_kpoints, dtype=float)

    def estimate_fermi(self) -> float | None:
        """Fermi level implied by the occupations, or None if undeterminable.

        This is only used as a last-resort *fallback* and callers must label it
        as estimated: it is the midpoint of the highest occupied and lowest
        unoccupied eigenvalue, which is meaningless for a metal.
        """
        occupied = self.occupations > 0.5
        if not occupied.any() or occupied.all():
            return None
        homo = float(self.energies[occupied].max())
        lumo_mask = ~occupied
        if not lumo_mask.any():
            return None
        lumo = float(self.energies[lumo_mask].min())
        if lumo < homo:
            return None  # bands overlap: metallic, midpoint would be misleading
        return 0.5 * (homo + lumo)

    def band_gap(self, e_fermi: float) -> BandGap | None:
        """The band gap relative to a given Fermi level, classified as direct
        or indirect by whether the valence-band maximum and conduction-band
        minimum occur at the same k-point.

        Returns None when the system is metallic (VBM/CBM overlap or every
        state is occupied/unoccupied) or the gap can't be resolved.
        """
        occupied = self.energies <= e_fermi + 1e-6  # (nspin, nk, nbands)
        if not occupied.any() or occupied.all():
            return None

        # Per k-point, the highest occupied and lowest unoccupied energy
        # (across spin and band) — collapsing straight to a global max/min
        # would give the right *value* but lose which k-point it came from,
        # which is exactly what direct-vs-indirect depends on.
        occupied_energies = np.where(occupied, self.energies, -np.inf)
        unoccupied_energies = np.where(~occupied, self.energies, np.inf)
        per_k_vbm = occupied_energies.max(axis=(0, 2))
        per_k_cbm = unoccupied_energies.min(axis=(0, 2))

        k_vbm = int(np.argmax(per_k_vbm))
        k_cbm = int(np.argmin(per_k_cbm))
        vbm = float(per_k_vbm[k_vbm])
        cbm = float(per_k_cbm[k_cbm])
        gap = cbm - vbm
        if gap <= 1e-4:
            return None
        return BandGap(gap=gap, vbm=vbm, cbm=cbm, vbm_kpoint=k_vbm, cbm_kpoint=k_cbm)
