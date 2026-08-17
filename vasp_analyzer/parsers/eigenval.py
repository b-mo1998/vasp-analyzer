"""EIGENVAL parser.

Layout::

    line 1   NIONS  NIONS  NBLOCK  ISPIN
    line 2   volume and lattice constants
    line 3   POTIM / temperature
    line 4   coordinate flag ("CAR")
    line 5   system name
    line 6   NELECT  NKPTS  NBANDS
    then, per k-point:  blank line, "kx ky kz weight", NBANDS eigenvalue rows

Eigenvalue rows carry 3 columns for a single spin channel (band, energy,
occupation) and 5 for ISPIN=2 (band, E_up, E_dn, occ_up, occ_dn). Non-collinear
runs write a single channel, so the column count — not the INCAR — decides.

Truncated files (an interrupted job, a partial copy) are read up to the last
complete k-point and reported as a warning rather than an error.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.errors import Issue, ParseResult, Severity, safe_parser
from ..core.logging_setup import get_logger
from ..models.bands import BandData

log = get_logger("parsers.eigenval")


@dataclass(frozen=True)
class EigenvalHeader:
    """The first six lines of EIGENVAL — enough to identify the calculation."""

    n_ions: int
    n_electrons: float
    n_kpoints: int
    n_bands: int


def peek_eigenval_header(path: Path) -> EigenvalHeader | None:
    """Read only the header. Used by discovery to stay fast and by validation
    to check that other files describe the same calculation."""
    try:
        with Path(path).open("r", errors="replace") as handle:
            lines = [handle.readline() for _ in range(6)]
        if not lines[5]:
            return None
        first = lines[0].split()
        dims = lines[5].split()
        if len(dims) < 3:
            return None
        return EigenvalHeader(
            n_ions=int(float(first[0])) if first else 0,
            n_electrons=float(dims[0]),
            n_kpoints=int(dims[1]),
            n_bands=int(dims[2]),
        )
    except (OSError, ValueError, IndexError) as exc:
        log.warning("Could not peek EIGENVAL header at %s: %s", path, exc)
        return None


@safe_parser("EIGENVAL")
def parse_eigenval(path: Path) -> ParseResult[BandData]:
    text = Path(path).read_text(errors="replace")
    lines = text.splitlines()
    if len(lines) < 7:
        return ParseResult.failure(
            "eigenval_truncated",
            "EIGENVAL is too short to contain any eigenvalues.",
            detail=f"only {len(lines)} lines",
            source="EIGENVAL",
        )

    issues: list[Issue] = []

    header_counts = lines[0].split()
    n_ions = int(float(header_counts[0])) if header_counts else 0

    dims = lines[5].split()
    if len(dims) < 3:
        return ParseResult.failure(
            "eigenval_bad_header",
            "EIGENVAL header does not declare k-point and band counts.",
            detail=f"line 6: {lines[5]!r}",
            source="EIGENVAL",
        )
    n_electrons = float(dims[0])
    n_kpoints = int(dims[1])
    n_bands = int(dims[2])

    if n_kpoints <= 0 or n_bands <= 0:
        return ParseResult.failure(
            "eigenval_empty",
            "EIGENVAL declares no k-points or no bands.",
            detail=f"NKPTS={n_kpoints} NBANDS={n_bands}",
            source="EIGENVAL",
        )

    body = [ln for ln in lines[6:] if ln.strip()]
    if not body:
        return ParseResult.failure(
            "eigenval_no_data",
            "EIGENVAL contains a header but no eigenvalues.",
            source="EIGENVAL",
        )

    # Column count of the first eigenvalue row decides the spin layout.
    n_columns = len(body[1].split()) if len(body) > 1 else 3
    if n_columns >= 5:
        n_spin = 2
    elif n_columns >= 3:
        n_spin = 1
    else:
        return ParseResult.failure(
            "eigenval_bad_columns",
            "EIGENVAL eigenvalue rows have an unrecognised column layout.",
            detail=f"first data row: {body[1]!r}",
            source="EIGENVAL",
        )

    values = np.fromstring(" ".join(body), sep=" ")
    block_floats = 4 + n_bands * n_columns
    complete = int(values.size // block_floats)

    if complete == 0:
        return ParseResult.failure(
            "eigenval_truncated",
            "EIGENVAL ends before the first complete k-point.",
            detail=f"{values.size} numbers, need {block_floats} per k-point",
            source="EIGENVAL",
        )

    if complete < n_kpoints:
        issues.append(
            Issue(
                code="eigenval_truncated",
                message=(
                    f"EIGENVAL is incomplete: {complete} of {n_kpoints} k-points "
                    "were read."
                ),
                detail=f"expected {n_kpoints * block_floats} numbers, found {values.size}",
                severity=Severity.WARNING,
                source="EIGENVAL",
            )
        )
        n_kpoints = complete

    block = values[: n_kpoints * block_floats].reshape(n_kpoints, block_floats)
    kpoints = block[:, :3].copy()
    weights = block[:, 3].copy()
    rows = block[:, 4:].reshape(n_kpoints, n_bands, n_columns)

    energies = np.empty((n_spin, n_kpoints, n_bands), dtype=float)
    occupations = np.zeros((n_spin, n_kpoints, n_bands), dtype=float)

    if n_spin == 1:
        energies[0] = rows[:, :, 1]
        if n_columns >= 3:
            occupations[0] = rows[:, :, 2]
    else:
        energies[0] = rows[:, :, 1]
        energies[1] = rows[:, :, 2]
        occupations[0] = rows[:, :, 3]
        occupations[1] = rows[:, :, 4]

    log.info(
        "Parsed EIGENVAL: %d k-points, %d bands, %d spin channel(s)",
        n_kpoints,
        n_bands,
        n_spin,
    )

    data = BandData(
        kpoints=kpoints,
        weights=weights,
        energies=energies,
        occupations=occupations,
        n_electrons=n_electrons,
        n_ions=n_ions,
    )
    return ParseResult.success(data, issues)
