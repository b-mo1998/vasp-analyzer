"""Turns PROCAR projected character into a fatbands overlay for BandPlot.

Kept as a pure function, independent of Qt/matplotlib widgets, so the colour
and size mapping can be tested directly against known PROCAR weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.colors import to_rgb

from ....models.bands import BandData
from ....parsers.procar import ProcarData
from ....services.appearance import effective_color

MIN_MARKER_SIZE = 3.0
MAX_MARKER_SIZE = 90.0
NEUTRAL_RGB = (0.6, 0.6, 0.6)


@dataclass
class ProjectionOverlay:
    """Per-(k-point, band) colour and marker size, one entry per spin channel."""

    colors: list[np.ndarray]   # (nk, nbands, 4) RGBA in [0, 1]
    sizes: list[np.ndarray]    # (nk, nbands) matplotlib scatter marker area


def build_element_overlay(
    bands: BandData, procar: ProcarData, species: list[str], orbital: str | None = None
) -> ProjectionOverlay | None:
    """Colour every (k-point, band) point by a weighted blend of the elements
    it's built from, and size it by how much of the wavefunction PROCAR
    actually accounts for there — a band the local basis barely describes
    shows as a small, faint dot rather than a confident one.

    ``orbital`` restricts everything (colour blend and size alike) to one
    orbital's character — e.g. "pz" — rather than the total across all of
    them. For a single-element system like graphene, "by element" alone is
    visually flat (everything is carbon); filtered to pz it immediately
    picks out the pi bands.

    Returns ``None`` if PROCAR's dimensions don't match the bands being
    plotted. A stale PROCAR can pass the coarse discovery-time consistency
    check (matching ion count) while still describing a different band or
    k-point count, and drawing a misaligned overlay would be worse than
    drawing none.
    """
    if procar.n_kpoints != bands.n_kpoints or procar.n_bands != bands.n_bands:
        return None
    if len(species) != procar.n_ions:
        return None

    elements = sorted(set(species))
    element_rgb = {element: np.array(to_rgb(effective_color(element))) for element in elements}

    n_spin = min(bands.n_spin, procar.n_spin)
    colors: list[np.ndarray] = []
    sizes: list[np.ndarray] = []

    for spin in range(n_spin):
        total = procar.total_weight(spin, orbital=orbital)  # (nk, nbands)
        blended = np.zeros((*total.shape, 3))
        safe_total = np.where(total > 1e-9, total, 1.0)
        for element in elements:
            weight = procar.element_weight(species, element, spin, orbital=orbital)
            blended += (weight / safe_total)[..., None] * element_rgb[element]

        # Where PROCAR carries essentially no weight, a blended colour has no
        # real basis — show neutral grey rather than an arbitrary element hue.
        no_weight = total <= 1e-9
        blended[no_weight] = NEUTRAL_RGB

        alpha = np.clip(0.35 + 0.65 * total, 0.0, 1.0)
        alpha[no_weight] = 0.15
        rgba = np.concatenate([blended, alpha[..., None]], axis=-1)
        colors.append(rgba)

        size = MIN_MARKER_SIZE + (MAX_MARKER_SIZE - MIN_MARKER_SIZE) * np.clip(total, 0.0, 1.0)
        sizes.append(size)

    # A calculation whose PROCAR doesn't cover every spin channel (unusual,
    # but not impossible) simply gets no overlay on the channel it lacks,
    # rather than an index error.
    for _ in range(bands.n_spin - n_spin):
        colors.append(np.zeros((bands.n_kpoints, bands.n_bands, 4)))
        sizes.append(np.zeros((bands.n_kpoints, bands.n_bands)))

    return ProjectionOverlay(colors=colors, sizes=sizes)


def legend_elements(species: list[str]) -> list[tuple[str, str]]:
    """(element, colour) pairs for a projection legend, in a stable order."""
    return [(element, effective_color(element)) for element in sorted(set(species))]
