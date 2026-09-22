"""Molecular-dynamics trajectory model.

A trajectory is a sequence of structures sharing one species list. Coordinates
are kept as a single ``(n_frames, n_atoms, 3)`` fractional array rather than a
list of :class:`StructureData` objects: playback needs to hand one frame's
coordinates to the viewer thousands of times, and slicing a numpy array is free
where rebuilding a dataclass per frame is not.

The lattice is stored per frame so a variable-cell (NPT) run is representable,
even though a fixed-cell (NVT/NVE) run repeats the same matrix.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .structure import StructureData


@dataclass
class MDSeries:
    """One per-step scalar quantity, with the unit it is measured in.

    Series are optional and independently sized: an OSZICAR may carry
    temperature for every step while the pressure is never written at all, and
    a page must be able to show one without inventing the other.
    """

    name: str
    values: np.ndarray            # (n_steps,)
    unit: str = ""
    source: str = ""

    def __len__(self) -> int:
        return int(self.values.shape[0])


@dataclass
class MDTrajectory:
    """A parsed MD run: geometry over time plus whatever telemetry was found."""

    species: list[str]
    frac_coords: np.ndarray            # (n_frames, n_atoms, 3), as written
    lattices: np.ndarray               # (n_frames, 3, 3) row vectors in Å
    comment: str = ""
    source: str = "XDATCAR"

    # Timing. ``timestep`` is in femtoseconds and may be None when neither the
    # INCAR nor any output stated POTIM — the page then falls back to showing
    # frame indices rather than inventing a physical time axis.
    timestep: float | None = None
    # XDATCAR records every NBLOCK-th step, so frame n is ionic step
    # n * frame_stride. Without this a 1 fs run sampled every 10 steps would
    # report a tenth of its real duration.
    frame_stride: int = 1

    # Per-step telemetry, keyed by a short name ("temperature", "total_energy",
    # "potential_energy", "kinetic_energy", "pressure").
    series: dict[str, MDSeries] = field(default_factory=dict)

    ensemble: str | None = None
    target_temperature: tuple[float, float] | None = None  # (TEBEG, TEEND)
    source_files: list[str] = field(default_factory=list)

    # -- dimensions --------------------------------------------------------

    @property
    def n_frames(self) -> int:
        return int(self.frac_coords.shape[0])

    @property
    def n_atoms(self) -> int:
        return int(self.frac_coords.shape[1])

    @property
    def is_variable_cell(self) -> bool:
        """True when the cell actually changes, not merely when it could."""
        if self.n_frames < 2:
            return False
        return not bool(np.allclose(self.lattices, self.lattices[0], atol=1e-8))

    @property
    def elements(self) -> list[str]:
        """Distinct elements, in first-appearance order (POSCAR's own order)."""
        seen: list[str] = []
        for symbol in self.species:
            if symbol not in seen:
                seen.append(symbol)
        return seen

    @property
    def composition(self) -> dict[str, int]:
        return dict(Counter(self.species))

    @property
    def formula(self) -> str:
        return "".join(f"{el}{n}" for el, n in self.composition.items())

    # -- time axis ----------------------------------------------------------

    @property
    def has_time_axis(self) -> bool:
        return self.timestep is not None

    def times(self) -> np.ndarray:
        """Frame times in picoseconds, or frame indices when POTIM is unknown.

        Callers must consult :attr:`has_time_axis` before labelling the result
        as a time — this deliberately returns *something* plottable either way
        so a trajectory without POTIM still animates against frame number.
        """
        if self.timestep is None:
            return np.arange(self.n_frames, dtype=float)
        step_fs = self.timestep * self.frame_stride
        return np.arange(self.n_frames, dtype=float) * step_fs / 1000.0

    @property
    def duration_ps(self) -> float | None:
        if self.timestep is None or self.n_frames == 0:
            return None
        return float(self.n_frames * self.frame_stride * self.timestep / 1000.0)

    # -- geometry access ------------------------------------------------------

    def cart_coords(self, frame: int) -> np.ndarray:
        """(n_atoms, 3) cartesian coordinates of one frame, as recorded."""
        index = int(np.clip(frame, 0, self.n_frames - 1))
        return self.frac_coords[index] @ self.lattices[index]

    def structure(self, frame: int) -> StructureData:
        """One frame as a :class:`StructureData`, for code that wants the
        ordinary structure model (symmetry analysis, export, bond building)."""
        index = int(np.clip(frame, 0, self.n_frames - 1))
        return StructureData(
            lattice=self.lattices[index],
            species=list(self.species),
            frac_coords=self.frac_coords[index].copy(),
            comment=self.comment,
            source=f"{self.source} frame {index + 1}",
        )

    def series_for(self, name: str) -> MDSeries | None:
        series = self.series.get(name)
        if series is None or len(series) == 0:
            return None
        return series

    def has(self, name: str) -> bool:
        return self.series_for(name) is not None
