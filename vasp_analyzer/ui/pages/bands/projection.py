"""Turns PROCAR projected character into a fatbands overlay for BandPlot.

Kept as a pure function, independent of Qt/matplotlib widgets, so the colour
and size mapping can be tested directly against known PROCAR weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_rgb

from ....models.bands import BandData
from ....parsers.procar import ProcarData
from ....services.appearance import effective_color

MIN_MARKER_SIZE = 3.0
MAX_MARKER_SIZE = 90.0
# Marker sizes are tuned for a few hundred plotted points. A large cell can
# put tens of thousands on the same axes — a 118-atom nanotube with 800 bands
# over 50 k-points is 40,000 — and at the unscaled size every marker overlaps
# its neighbours until the plot is a solid block. The area budget below keeps
# the total ink roughly constant, so a dense band structure stays readable
# instead of going black.
MARKER_AREA_BUDGET = 90.0 * 600.0
MIN_MAX_MARKER_SIZE = 6.0


def _max_marker_size(n_points: int) -> float:
    """The largest marker to use for a plot of ``n_points`` (k-point, band)
    pairs, so a dense structure does not paint itself solid."""
    if n_points <= 0:
        return MAX_MARKER_SIZE
    scaled = MARKER_AREA_BUDGET / float(n_points)
    return float(min(MAX_MARKER_SIZE, max(MIN_MAX_MARKER_SIZE, scaled)))
NEUTRAL_RGB = (0.6, 0.6, 0.6)
SPIN_MARKER_SIZE = 12.0
# A violet distinct from the reds/blues/greens the element and orbital
# overlays already use, so an atom-projected plot is never confusable with
# either of them at a glance.
ATOM_PROJECTION_COLOR = "#7c3aed"

# low value (spin down) -> red, high value (spin up) -> blue, matching
# widgets.colorbar.DivergingColorbar's gradient.
SPIN_CMAP = colormaps["bwr_r"]

# Colour assigned to each orbital by the order it was checked in: 1st black,
# 2nd red, 3rd blue (as requested), then further distinguishable colours for
# whatever else gets added to the selection.
ORBITAL_PALETTE = [
    "#000000",
    "#e53e3e",
    "#3b82f6",
    "#f59e0b",
    "#10b981",
    "#8b5cf6",
    "#14b8a6",
    "#ec4899",
    "#78350f",
    "#06b6d4",
]


def orbital_color(index: int) -> str:
    return ORBITAL_PALETTE[index % len(ORBITAL_PALETTE)]


@dataclass
class ProjectionOverlay:
    """Per-(k-point, band) colour and marker size, one entry per spin channel."""

    colors: list[np.ndarray]   # (nk, nbands, 4) RGBA in [0, 1]
    sizes: list[np.ndarray]    # (nk, nbands) matplotlib scatter marker area


def build_element_overlay(
    bands: BandData,
    procar: ProcarData,
    species: list[str],
    orbitals: list[str] | None = None,
    orbitals_by_element: dict[str, list[str]] | None = None,
) -> ProjectionOverlay | None:
    """Colour every (k-point, band) point by a weighted blend of the elements
    it's built from, and size it by how much of the wavefunction PROCAR
    actually accounts for there — a band the local basis barely describes
    shows as a small, faint dot rather than a confident one.

    ``orbitals`` restricts everything (colour blend and size alike) to a
    chosen combination of orbitals — e.g. ``["pz"]`` or ``["px", "py"]`` —
    rather than the total across all of them. For a single-element system
    like graphene, "by element" alone is visually flat (everything is
    carbon); filtered to pz it immediately picks out the pi bands.

    ``orbitals_by_element`` gives each element its own combination, which is
    what a real multi-element question usually needs: a nanotube doped with
    tellurium is read by asking for carbon *p* against tellurium *d*, not by
    applying one orbital filter to both. An element absent from the mapping
    contributes nothing, so the selection also acts as an element filter.
    It takes precedence over ``orbitals`` where both are given.

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
    if orbitals_by_element is not None:
        # Only elements the caller actually asked for take part.
        elements = [e for e in elements if orbitals_by_element.get(e)]
        if not elements:
            return None
    element_rgb = {element: np.array(to_rgb(effective_color(element))) for element in elements}

    def _orbitals_for(element: str) -> list[str] | None:
        if orbitals_by_element is None:
            return orbitals
        return orbitals_by_element.get(element)

    n_spin = min(bands.n_spin, procar.n_spin)
    colors: list[np.ndarray] = []
    sizes: list[np.ndarray] = []

    for spin in range(n_spin):
        # Each element contributes only the orbitals chosen for it, so the
        # total has to be accumulated the same way rather than read from
        # total_weight() — which knows nothing about the per-element split.
        per_element = {
            element: procar.element_weight(
                species, element, spin, orbitals=_orbitals_for(element)
            )
            for element in elements
        }
        if orbitals_by_element is None:
            total = procar.total_weight(spin, orbitals=orbitals)  # (nk, nbands)
        else:
            total = sum(per_element.values())

        blended = np.zeros((*total.shape, 3))
        safe_total = np.where(total > 1e-9, total, 1.0)
        for element in elements:
            blended += (per_element[element] / safe_total)[..., None] * element_rgb[element]

        # Where PROCAR carries essentially no weight, a blended colour has no
        # real basis — show neutral grey rather than an arbitrary element hue.
        no_weight = total <= 1e-9
        blended[no_weight] = NEUTRAL_RGB

        alpha = np.clip(0.35 + 0.65 * total, 0.0, 1.0)
        alpha[no_weight] = 0.15
        # PROCAR carries three decimals, so a per-element (or per-orbital)
        # sum can exceed the "tot" column it is divided by -- 0.002 / 0.001
        # -- and push a channel past 1. matplotlib refuses such a colour
        # outright, and the whole overlay failed to draw for graphene.
        rgba = np.clip(np.concatenate([blended, alpha[..., None]], axis=-1), 0.0, 1.0)
        colors.append(rgba)

        largest = _max_marker_size(total.size)
        size = MIN_MARKER_SIZE + max(0.0, largest - MIN_MARKER_SIZE) * np.clip(
            total, 0.0, 1.0
        )
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


def build_orbital_overlay(
    bands: BandData, procar: ProcarData, orbitals: list[str]
) -> ProjectionOverlay | None:
    """Colour every (k-point, band) point by a weighted blend of the selected
    orbitals' own colours (see :data:`ORBITAL_PALETTE`), sized by how much of
    the two combined they actually account for.

    Coloring by element alone is flat for a single-element system like
    graphene — every point is the same colour, and only marker size hints at
    which orbital dominates. Colouring by orbital instead makes a px+py+pz
    selection visually separate the sigma and pi character immediately.

    ``orbitals`` must be non-empty — this is only meant to run once at least
    one orbital is actually selected; an unfiltered "all orbitals" selection
    reads better under :func:`build_element_overlay` instead.
    """
    if procar.n_kpoints != bands.n_kpoints or procar.n_bands != bands.n_bands:
        return None
    if not orbitals:
        return None

    orbital_rgb = {
        orbital: np.array(to_rgb(orbital_color(i))) for i, orbital in enumerate(orbitals)
    }

    n_spin = min(bands.n_spin, procar.n_spin)
    colors: list[np.ndarray] = []
    sizes: list[np.ndarray] = []

    for spin in range(n_spin):
        total = procar.total_weight(spin, orbitals=orbitals)  # (nk, nbands)
        blended = np.zeros((*total.shape, 3))
        safe_total = np.where(total > 1e-9, total, 1.0)
        for orbital in orbitals:
            weight = procar.total_weight(spin, orbitals=[orbital])
            blended += (weight / safe_total)[..., None] * orbital_rgb[orbital]

        no_weight = total <= 1e-9
        blended[no_weight] = NEUTRAL_RGB

        alpha = np.clip(0.35 + 0.65 * total, 0.0, 1.0)
        alpha[no_weight] = 0.15
        # PROCAR carries three decimals, so a per-element (or per-orbital)
        # sum can exceed the "tot" column it is divided by -- 0.002 / 0.001
        # -- and push a channel past 1. matplotlib refuses such a colour
        # outright, and the whole overlay failed to draw for graphene.
        rgba = np.clip(np.concatenate([blended, alpha[..., None]], axis=-1), 0.0, 1.0)
        colors.append(rgba)

        largest = _max_marker_size(total.size)
        size = MIN_MARKER_SIZE + max(0.0, largest - MIN_MARKER_SIZE) * np.clip(
            total, 0.0, 1.0
        )
        sizes.append(size)

    for _ in range(bands.n_spin - n_spin):
        colors.append(np.zeros((bands.n_kpoints, bands.n_bands, 4)))
        sizes.append(np.zeros((bands.n_kpoints, bands.n_bands)))

    return ProjectionOverlay(colors=colors, sizes=sizes)


def legend_orbitals(orbitals: list[str]) -> list[tuple[str, str]]:
    """(orbital, colour) pairs in selection order — the same order and
    colours :func:`build_orbital_overlay` used to paint them."""
    return [(orbital, orbital_color(i)) for i, orbital in enumerate(orbitals)]


def build_atom_overlay(
    bands: BandData,
    procar: ProcarData,
    rows: list[int],
    orbitals: list[str] | None,
) -> ProjectionOverlay | None:
    """Colour every (k-point, band) point by how much character sits on
    ``rows`` (0-based ion indices), restricted to ``orbitals`` if given.

    One fixed colour whose alpha and marker size track the fraction — the
    same visual language :func:`build_orbital_overlay` uses for a single
    quantity, chosen over a per-atom blend because the point of this overlay
    is "how much of this band lives on *this specific handful of atoms*",
    a single number, not a comparison between them.
    """
    if procar.n_kpoints != bands.n_kpoints or procar.n_bands != bands.n_bands:
        return None
    if not rows:
        return None

    base_rgb = np.array(to_rgb(ATOM_PROJECTION_COLOR))
    n_spin = min(bands.n_spin, procar.n_spin)
    colors: list[np.ndarray] = []
    sizes: list[np.ndarray] = []

    for spin in range(n_spin):
        weight = procar.atom_weight(rows, spin, orbitals=orbitals)  # (nk, nbands)
        no_weight = weight <= 1e-9

        rgb = np.broadcast_to(base_rgb, (*weight.shape, 3)).copy()
        rgb[no_weight] = NEUTRAL_RGB
        alpha = np.clip(0.35 + 0.65 * weight, 0.0, 1.0)
        alpha[no_weight] = 0.12
        colors.append(np.concatenate([rgb, alpha[..., None]], axis=-1))

        largest = _max_marker_size(weight.size)
        size = MIN_MARKER_SIZE + max(0.0, largest - MIN_MARKER_SIZE) * np.clip(
            weight, 0.0, 1.0
        )
        sizes.append(size)

    for _ in range(bands.n_spin - n_spin):
        colors.append(np.zeros((bands.n_kpoints, bands.n_bands, 4)))
        sizes.append(np.zeros((bands.n_kpoints, bands.n_bands)))

    return ProjectionOverlay(colors=colors, sizes=sizes)


AXIS_LABELS = ("x", "y", "z")


def build_spin_overlay(
    bands: BandData,
    procar: ProcarData,
    axis: int = 2,
    orbitals: list[str] | None = None,
    vmax: float | None = None,
) -> tuple[ProjectionOverlay, float] | None:
    """Colour every (k-point, band) point by its net spin along ``axis``
    (0=x, 1=y, 2=z) — blue for positive ("spin up"), red for negative ("spin
    down"), white near zero, using a colormap symmetric about zero so the two
    directions read as opposite rather than as different intensities of one
    hue.

    Works for either kind of spin-resolved PROCAR: a non-collinear run's
    ``spin_texture`` (mx/my/mz; ``axis`` picks the component), or a
    collinear ISPIN=2 run, where the "spin" plotted is simply the up-channel
    weight minus the down-channel one and ``axis`` is ignored — there's only
    one axis in that case.

    Returns ``None`` when ``procar`` carries no spin information at all
    (ISPIN=1, no SOC) or its dimensions don't match ``bands``. On success,
    also returns the scale's ``vmax`` — the largest |value| present, unless
    ``vmax`` was given explicitly — so a colourbar can be drawn to match.
    """
    if procar.n_kpoints != bands.n_kpoints or procar.n_bands != bands.n_bands:
        return None

    if procar.spin_texture is not None:
        values = procar.spin_projection(axis=axis, orbitals=orbitals)
    elif procar.n_spin >= 2:
        values = procar.total_weight(spin=0, orbitals=orbitals) - procar.total_weight(
            spin=1, orbitals=orbitals
        )
    else:
        return None
    if values is None:
        return None

    scale = vmax if vmax is not None else max(float(np.max(np.abs(values))), 1e-9)
    normalised = np.clip((values + scale) / (2 * scale), 0.0, 1.0)
    rgba = SPIN_CMAP(normalised)  # (nk, nbands, 4)
    size = np.full(values.shape, SPIN_MARKER_SIZE)

    colors = [np.asarray(rgba)]
    sizes = [size]
    for _ in range(bands.n_spin - 1):
        colors.append(np.zeros((bands.n_kpoints, bands.n_bands, 4)))
        sizes.append(np.zeros((bands.n_kpoints, bands.n_bands)))

    return ProjectionOverlay(colors=colors, sizes=sizes), scale
