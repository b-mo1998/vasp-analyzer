"""Nudged-elastic-band path model.

An NEB calculation is a directory of numbered image folders rather than a
single file, so the "path" here is assembled from many partial sources. Every
per-image quantity is optional: an interrupted run leaves images whose energy
was never written, and the page must still be able to show their geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .structure import StructureData


@dataclass
class NEBImage:
    """One image along the band."""

    index: int
    folder: str
    structure: StructureData | None = None
    energy: float | None = None            # absolute, eV
    energy_source: str = ""                # the file the energy came from
    max_force: float | None = None         # eV/Å
    force_kind: str = ""                   # "NEB projected" or "atomic"
    is_endpoint: bool = False
    source_files: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def label(self) -> str:
        return self.folder


@dataclass
class NEBPath:
    """A discovered NEB calculation: its images and everything derived."""

    images: list[NEBImage] = field(default_factory=list)
    reaction_coordinate: np.ndarray | None = None   # (n_images,) cumulative Å
    method: str = "NEB"                             # "NEB" / "NEB (Climbing Image)"
    climbing_image: int | None = None               # index into images
    climbing_image_origin: str = ""                 # "explicit" / "inferred"
    force_threshold: float | None = None            # |EDIFFG|, eV/Å
    spring_constant: float | None = None
    n_movable: int | None = None                    # INCAR IMAGES
    source_files: list[str] = field(default_factory=list)
    directory_name: str = ""
    # VTST's own dense interpolation (spline.dat), when the run produced
    # one. Preferred over interpolating here — see parsers.vtst.
    spline: "SplineCurve | None" = None
    # Pseudopotential titles echoed by an image's OUTCAR ("PAW_PBE C 08Apr2002").
    # An NEB directory has no top-level OUTCAR, so this is the only place the
    # run's POTCAR provenance is recorded.
    potcar_titles: list[str] = field(default_factory=list)
    functional: str | None = None

    # -- dimensions --------------------------------------------------------

    @property
    def n_images(self) -> int:
        return len(self.images)

    @property
    def has_energies(self) -> bool:
        return any(image.energy is not None for image in self.images)

    @property
    def complete_energies(self) -> bool:
        return bool(self.images) and all(
            image.energy is not None for image in self.images
        )

    @property
    def is_climbing(self) -> bool:
        return "Climbing" in self.method

    # -- energies ------------------------------------------------------------

    @property
    def reference_energy(self) -> float | None:
        """The initial image's energy — the zero of the relative profile."""
        if not self.images:
            return None
        return self.images[0].energy

    def relative_energies(self) -> list[float | None]:
        """Energies measured from the initial image, in eV."""
        reference = self.reference_energy
        if reference is None:
            return [None] * self.n_images
        return [
            None if image.energy is None else image.energy - reference
            for image in self.images
        ]

    @property
    def initial_energy(self) -> float | None:
        return self.images[0].energy if self.images else None

    @property
    def final_energy(self) -> float | None:
        return self.images[-1].energy if self.images else None

    @property
    def peak_index(self) -> int | None:
        """Index of the highest-energy image, or None when no energy is known.

        This is the highest energy *that was found*, which is only the true
        maximum of the band when every image reported one — see
        :attr:`forward_barrier`, which refuses to answer otherwise.
        """
        candidates = [
            (image.energy, i) for i, image in enumerate(self.images) if image.energy is not None
        ]
        if not candidates:
            return None
        return max(candidates)[1]

    @property
    def forward_barrier(self) -> float | None:
        """E_max - E_initial, in eV.

        None unless every image has an energy: with an unrun intermediate image
        the true maximum may be the missing one, and reporting the highest
        *known* value as "the barrier" would understate it — exactly the
        mistake worth avoiding on a half-finished band.
        """
        if not self.complete_energies:
            return None
        peak = self.peak_index
        initial = self.initial_energy
        if peak is None or initial is None:
            return None
        energy = self.images[peak].energy
        return None if energy is None else energy - initial

    @property
    def reverse_barrier(self) -> float | None:
        """E_max - E_final, in eV. None unless every image has an energy."""
        if not self.complete_energies:
            return None
        peak = self.peak_index
        final = self.final_energy
        if peak is None or final is None:
            return None
        energy = self.images[peak].energy
        return None if energy is None else energy - final

    @property
    def reaction_energy(self) -> float | None:
        """E_final - E_initial, in eV."""
        if self.initial_energy is None or self.final_energy is None:
            return None
        return self.final_energy - self.initial_energy

    # -- convergence ---------------------------------------------------------

    @property
    def max_force(self) -> float | None:
        """The largest force magnitude across every image that reported one."""
        forces = [abs(i.max_force) for i in self.images if i.max_force is not None]
        return max(forces) if forces else None

    @property
    def complete_forces(self) -> bool:
        """True when every movable image reported a force.

        Endpoints are held fixed by the NEB and are not required to: a band
        whose interior images all converged is converged, whether or not the
        two frozen endpoints wrote a force at all.
        """
        movable = [
            image for image in self.images if not image.is_endpoint
        ] or self.images
        return bool(movable) and all(image.max_force is not None for image in movable)

    @property
    def converged(self) -> bool | None:
        """True/False against the calculation's own criterion, or None when
        the threshold is unknown or some movable image never reported a force.

        An unanswerable question must not be answered "converged" — a band with
        three unrun images would otherwise pass on the strength of its two
        fixed endpoints alone.
        """
        if self.force_threshold is None or not self.complete_forces:
            return None
        force = self.max_force
        return None if force is None else force <= self.force_threshold

    # -- geometry -------------------------------------------------------------

    @property
    def structures(self) -> list[StructureData | None]:
        return [image.structure for image in self.images]

    @property
    def species(self) -> list[str] | None:
        for image in self.images:
            if image.structure is not None:
                return list(image.structure.species)
        return None

    def coordinate_of(self, index: int) -> float | None:
        if self.reaction_coordinate is None:
            return None
        if not 0 <= index < len(self.reaction_coordinate):
            return None
        return float(self.reaction_coordinate[index])
