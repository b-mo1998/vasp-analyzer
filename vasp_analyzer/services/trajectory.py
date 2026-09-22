"""Trajectory geometry: periodic unwrapping and mean-square displacement.

Kept free of Qt and of any plotting so the maths can be tested directly, and so
future analyses (diffusion coefficients, RDF, VACF) can reuse the unwrapped
coordinates rather than re-deriving them.

The central operation is unwrapping. XDATCAR stores fractional coordinates
wrapped into the cell, so an atom leaving through one face reappears at the
other and its recorded position jumps by nearly a full lattice vector. Taking
those coordinates at face value would make a diffusing atom look stationary and
would put enormous spurious spikes in any displacement measure. The fix is the
minimum-image convention applied to *consecutive differences*::

    ds        = s(t+1) - s(t)
    ds_true   = ds - round(ds)          # nothing moves more than half a cell
    s_cont(t) = s(0) + cumsum(ds_true)

This assumes no atom travels more than half a lattice vector between two
recorded frames, which is the same assumption every MD code makes and holds
comfortably at ordinary timesteps.
"""

from __future__ import annotations

import numpy as np

from ..core.logging_setup import get_logger
from ..models.trajectory import MDTrajectory

log = get_logger("services.trajectory")


def wrap_fractional(frac: np.ndarray) -> np.ndarray:
    """Fold fractional coordinates into [0, 1)."""
    return np.asarray(frac) % 1.0


def unwrap_fractional(frac: np.ndarray) -> np.ndarray:
    """Continuous fractional coordinates from wrapped ones.

    ``frac`` is ``(n_frames, n_atoms, 3)``. The first frame is left where it
    is and every later frame accumulates minimum-image corrected steps, so the
    result is continuous in time and may run outside [0, 1) — which is exactly
    what a diffusing atom should do.
    """
    frac = np.asarray(frac, dtype=float)
    if frac.ndim != 3:
        raise ValueError("expected a (n_frames, n_atoms, 3) array")
    if frac.shape[0] < 2:
        return frac.copy()

    deltas = np.diff(frac, axis=0)
    deltas -= np.round(deltas)
    unwrapped = np.empty_like(frac)
    unwrapped[0] = frac[0]
    np.cumsum(deltas, axis=0, out=unwrapped[1:])
    unwrapped[1:] += frac[0]
    return unwrapped


def unwrapped_cartesian(trajectory: MDTrajectory) -> np.ndarray:
    """(n_frames, n_atoms, 3) continuous cartesian coordinates in Å.

    Each minimum-image corrected fractional step is converted with the lattice
    of the frame it lands on, so a variable-cell (NPT) run accumulates real
    displacement rather than displacement measured in a cell that was changing
    underneath it.
    """
    frac = np.asarray(trajectory.frac_coords, dtype=float)
    lattices = np.asarray(trajectory.lattices, dtype=float)
    n_frames = frac.shape[0]

    if n_frames < 2:
        return frac @ lattices[0]

    if not trajectory.is_variable_cell:
        # One matrix for the whole run: unwrap in fractional space and convert
        # once, which is both faster and exactly equivalent.
        return unwrap_fractional(frac) @ lattices[0]

    deltas = np.diff(frac, axis=0)
    deltas -= np.round(deltas)
    steps = np.einsum("fai,fij->faj", deltas, lattices[1:])
    cartesian = np.empty_like(frac)
    cartesian[0] = frac[0] @ lattices[0]
    np.cumsum(steps, axis=0, out=cartesian[1:])
    cartesian[1:] += cartesian[0]
    return cartesian


def wrapped_cartesian(trajectory: MDTrajectory, frame: int) -> np.ndarray:
    """(n_atoms, 3) cartesian coordinates of one frame, folded into the cell."""
    index = int(np.clip(frame, 0, trajectory.n_frames - 1))
    return wrap_fractional(trajectory.frac_coords[index]) @ trajectory.lattices[index]


def mean_square_displacement(
    trajectory: MDTrajectory, positions: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Per-element MSD in Å² against the *same* frame axis the page plays.

    For element α::

        MSD_a(t) = (1/N_a) * sum_{i in a} |r_i(t) - r_i(0)|^2

    measured on unwrapped coordinates. Referencing every frame to t = 0 (rather
    than averaging over many time origins) is what makes this directly
    comparable with the playback cursor: the value under the cursor is the
    displacement the structure on screen has actually accumulated.
    :func:`mean_square_displacement_lagged` provides the time-origin-averaged
    form for quantitative diffusion work.

    Returns ``{element: (n_frames,) array}``, plus a ``"total"`` entry over all
    atoms. An empty dict is returned for a single-frame trajectory.
    """
    if trajectory.n_frames < 2:
        return {}

    cartesian = unwrapped_cartesian(trajectory) if positions is None else positions
    displacement = cartesian - cartesian[0]
    squared = np.einsum("fai,fai->fa", displacement, displacement)  # (n_frames, n_atoms)

    species = np.array(trajectory.species)
    result: dict[str, np.ndarray] = {}
    for element in trajectory.elements:
        mask = species == element
        if mask.any():
            result[element] = squared[:, mask].mean(axis=1)
    result["total"] = squared.mean(axis=1)

    log.info(
        "Calculated MSD for %s over %d frames",
        ", ".join(trajectory.elements),
        trajectory.n_frames,
    )
    return result


def mean_square_displacement_lagged(
    trajectory: MDTrajectory,
    positions: np.ndarray | None = None,
    max_lag_fraction: float = 0.5,
) -> dict[str, np.ndarray]:
    """Time-origin-averaged MSD against lag time τ rather than absolute time.

    ``MSD_a(tau) = < |r_i(t0+tau) - r_i(t0)|^2 >`` averaged over every origin
    ``t0`` and every atom of element α. This is the statistically robust form
    a diffusion coefficient should be fitted to; it is *not* what the page
    plots, because its x-axis is a lag and cannot carry a playback cursor.

    Lags beyond ``max_lag_fraction`` of the run are omitted: those are averaged
    over so few origins that they are dominated by noise.
    """
    if trajectory.n_frames < 2:
        return {}

    cartesian = unwrapped_cartesian(trajectory) if positions is None else positions
    n_frames = cartesian.shape[0]
    max_lag = max(1, int(n_frames * max_lag_fraction))

    species = np.array(trajectory.species)
    result: dict[str, np.ndarray] = {}
    masks = {element: species == element for element in trajectory.elements}
    masks["total"] = np.ones(len(species), dtype=bool)
    curves = {name: np.zeros(max_lag) for name in masks}

    for lag in range(1, max_lag):
        difference = cartesian[lag:] - cartesian[:-lag]
        squared = np.einsum("fai,fai->fa", difference, difference)
        for name, mask in masks.items():
            if mask.any():
                curves[name][lag] = float(squared[:, mask].mean())

    for name, mask in masks.items():
        if mask.any():
            result[name] = curves[name]
    return result


def centre_of_mass_drift(trajectory: MDTrajectory) -> np.ndarray:
    """(n_frames, 3) drift of the unweighted centroid, in Å.

    A thermostat should not translate the whole cell; a visible drift here is
    the usual explanation for an MSD that rises linearly for every element at
    once, so it is worth being able to check.
    """
    cartesian = unwrapped_cartesian(trajectory)
    centre = cartesian.mean(axis=1)
    return centre - centre[0]


# --- geometry of a selection over time ---------------------------------------


def _minimum_image_vectors(
    trajectory: MDTrajectory, a: int, b: int
) -> np.ndarray:
    """(n_frames, 3) cartesian vector from atom ``a`` to atom ``b`` per frame,
    under the minimum-image convention in each frame's own cell.

    Per-frame lattices, because a variable-cell run changes them; minimum
    image, because XDATCAR coordinates are wrapped and a bonded pair that
    straddles a face would otherwise look a whole cell apart.
    """
    frac = trajectory.frac_coords
    delta = frac[:, b, :] - frac[:, a, :]
    delta -= np.round(delta)
    # einsum over frames: each frame's fractional delta times that frame's
    # lattice rows.
    return np.einsum("fi,fij->fj", delta, trajectory.lattices)


def pair_distance_series(trajectory: MDTrajectory, a: int, b: int) -> np.ndarray:
    """Distance between two atoms in every frame, Å."""
    return np.linalg.norm(_minimum_image_vectors(trajectory, a, b), axis=1)


def angle_series(trajectory: MDTrajectory, a: int, vertex: int, c: int) -> np.ndarray:
    """The a–vertex–c angle in every frame, degrees."""
    u = _minimum_image_vectors(trajectory, vertex, a)
    v = _minimum_image_vectors(trajectory, vertex, c)
    norms = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.einsum("fi,fi->f", u, v) / norms
    cosine = np.clip(np.nan_to_num(cosine, nan=1.0), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))
