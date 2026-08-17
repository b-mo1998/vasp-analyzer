"""k-axis construction for band plots.

EIGENVAL stores fractional k-points and nothing else — no path, no labels. What
can honestly be drawn depends on what else is present:

* KPOINTS in line mode + a structure  → true reciprocal-space distance in Å⁻¹
  with high-symmetry labels at the segment joins.
* KPOINTS in line mode, no structure  → fractional path length, labels kept.
* anything else (a Monkhorst-Pack or Γ-centred mesh)  → k-point index, and the
  axis says so. No labels are invented.
"""

from __future__ import annotations

import numpy as np

from ..core.logging_setup import get_logger
from ..models.bands import BandData, KPathLabel
from ..models.structure import StructureData
from ..parsers.kpoints import KpointsSpec

log = get_logger("services.kpath")

# A jump this many times the median step is treated as a path discontinuity.
BREAK_FACTOR = 3.0


def _cartesian(kpoints: np.ndarray, structure: StructureData | None) -> np.ndarray:
    if structure is None:
        return kpoints
    return kpoints @ structure.reciprocal_lattice


def _cumulative_distance(points: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Cumulative path length, holding x fixed across discontinuities."""
    if len(points) < 2:
        return np.zeros(len(points)), []

    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    finite = steps[steps > 1e-12]
    median = float(np.median(finite)) if finite.size else 0.0
    breaks: list[int] = []

    advance = steps.copy()
    if median > 0:
        jumps = steps > BREAK_FACTOR * median
        advance[jumps] = 0.0
        breaks = [int(i + 1) for i in np.nonzero(jumps)[0]]

    x = np.concatenate([[0.0], np.cumsum(advance)])
    return x, breaks


def build_kpath(
    bands: BandData,
    spec: KpointsSpec | None,
    structure: StructureData | None,
) -> None:
    """Populate ``bands.x_values``, ``labels`` and ``segment_breaks`` in place."""
    n_k = bands.n_kpoints

    if spec is None or not spec.is_line_mode:
        bands.x_values = np.arange(n_k, dtype=float)
        bands.is_line_mode = False
        bands.labels = []
        bands.segment_breaks = []
        bands.x_axis_title = "k-point index"
        reason = (
            f"{spec.mode} mesh" if spec is not None else "no KPOINTS file"
        )
        log.info("k-axis: index (%s), %d points", reason, n_k)
        return

    points = _cartesian(bands.kpoints, structure)
    x, breaks = _cumulative_distance(points)
    bands.x_values = x
    bands.segment_breaks = breaks
    bands.is_line_mode = True
    bands.x_axis_title = (
        "Wave vector (Å⁻¹)" if structure is not None else "Wave vector (fractional)"
    )

    bands.labels = _place_labels(spec, x, n_k)
    log.info(
        "k-axis: line mode, %d points, %d labels, %d break(s)",
        n_k,
        len(bands.labels),
        len(breaks),
    )


def _place_labels(spec: KpointsSpec, x: np.ndarray, n_k: int) -> list[KPathLabel]:
    """Map the labels declared in KPOINTS onto plotted positions.

    In line mode VASP writes ``n_divisions`` k-points per segment, and KPOINTS
    lists a start and end point per segment. Consecutive segments that meet at
    the same point collapse into one label such as ``X|M``.
    """
    divisions = spec.n_divisions
    if divisions <= 0 or not spec.path_labels:
        return []

    n_segments = len(spec.path_labels) // 2
    if n_segments == 0:
        return []

    entries: list[tuple[int, str]] = []
    for segment in range(n_segments):
        start_index = segment * divisions
        end_index = start_index + divisions - 1
        if start_index >= n_k:
            break
        entries.append((start_index, spec.path_labels[2 * segment]))
        entries.append((min(end_index, n_k - 1), spec.path_labels[2 * segment + 1]))

    # Segment i ends and segment i+1 begins at the same k-point, which VASP
    # writes twice. Labels closer together than a small fraction of the axis
    # are therefore the same point: collapse them into "X", or into "X|M" where
    # the path jumps between two different points.
    span = float(x[-1] - x[0]) if len(x) > 1 else 0.0
    tolerance = max(span * 0.01, 1e-9)

    merged: list[KPathLabel] = []
    for index, text in entries:
        if not text:
            continue
        if merged and abs(x[index] - merged[-1].position) <= tolerance:
            previous = merged[-1]
            if previous.label != text:
                merged[-1] = KPathLabel(
                    index=previous.index,
                    position=previous.position,
                    label=f"{previous.label}|{text}",
                )
            continue
        merged.append(KPathLabel(index=index, position=float(x[index]), label=text))
    return merged
