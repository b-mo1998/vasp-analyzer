"""DOSCAR parser.

Layout::

    line 1   NIONS  NIONS_PDOS  PARTIAL_FLAG  NCDIJ
    lines 2-5  volume/lattice, POTIM, coordinate flag, system name
    line 6   EMAX  EMIN  NEDOS  EFERMI  weight
    then NEDOS rows of the total DOS, then — if PARTIAL_FLAG is set — a repeat
    of line 6 followed by NEDOS rows for each ion.

Total-DOS rows have 3 columns for one spin channel (E, DOS, integrated) and 5
for ISPIN=2. Only the total DOS is read eagerly; the projected blocks are large
and are left for the element-projection feature to read on demand.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.errors import Issue, ParseResult, Severity, safe_parser
from ..core.logging_setup import get_logger
from ..models.dos import DosData

log = get_logger("parsers.doscar")

HEADER_LINES = 6


@dataclass(frozen=True)
class DoscarHeader:
    """DOSCAR's first six lines: identity plus the Fermi level."""

    n_ions: int
    n_points: int
    e_fermi: float | None
    has_projected: bool


def peek_doscar_header(path: Path) -> DoscarHeader | None:
    """Read only the header — this is where the Fermi level lives, and reading
    it costs nothing compared with the full density of states."""
    try:
        with Path(path).open("r", errors="replace") as handle:
            lines = [handle.readline() for _ in range(HEADER_LINES)]
        if not lines[5]:
            return None
        first = lines[0].split()
        control = lines[5].split()
        if len(control) < 4:
            return None
        return DoscarHeader(
            n_ions=int(float(first[0])) if first else 0,
            n_points=int(float(control[2])),
            e_fermi=float(control[3]),
            has_projected=len(first) >= 3 and int(float(first[2])) != 0,
        )
    except (OSError, ValueError, IndexError) as exc:
        log.warning("Could not peek DOSCAR header at %s: %s", path, exc)
        return None


@safe_parser("DOSCAR")
def parse_doscar(path: Path) -> ParseResult[DosData]:
    path = Path(path)
    issues: list[Issue] = []

    with path.open("r", errors="replace") as handle:
        header = [handle.readline() for _ in range(HEADER_LINES)]
        if not header[-1]:
            return ParseResult.failure(
                "doscar_truncated",
                "DOSCAR is too short to contain a density of states.",
                source="DOSCAR",
            )

        first = header[0].split()
        n_ions = int(float(first[0])) if first else 0
        has_projected = len(first) >= 3 and int(float(first[2])) != 0

        control = header[5].split()
        if len(control) < 4:
            return ParseResult.failure(
                "doscar_bad_header",
                "DOSCAR header does not declare the energy grid.",
                detail=f"line 6: {header[5]!r}",
                source="DOSCAR",
            )
        n_points = int(float(control[2]))
        e_fermi = float(control[3])

        if n_points <= 0:
            return ParseResult.failure(
                "doscar_empty",
                "DOSCAR declares an empty energy grid.",
                detail=f"NEDOS={n_points}",
                source="DOSCAR",
            )

        rows: list[str] = []
        for _ in range(n_points):
            line = handle.readline()
            if not line:
                break
            rows.append(line)

    if not rows:
        return ParseResult.failure(
            "doscar_no_data",
            "DOSCAR contains a header but no density of states.",
            source="DOSCAR",
        )

    n_columns = len(rows[0].split())
    if n_columns >= 5:
        n_spin = 2
    elif n_columns >= 3:
        n_spin = 1
    else:
        return ParseResult.failure(
            "doscar_bad_columns",
            "DOSCAR rows have an unrecognised column layout.",
            detail=f"first data row: {rows[0]!r}",
            source="DOSCAR",
        )

    values = np.fromstring(" ".join(rows), sep=" ")
    complete = int(values.size // n_columns)
    if complete == 0:
        return ParseResult.failure(
            "doscar_truncated",
            "DOSCAR ends before any usable data.",
            source="DOSCAR",
        )
    if complete < n_points:
        issues.append(
            Issue(
                code="doscar_truncated",
                message=(
                    f"DOSCAR is incomplete: {complete} of {n_points} energy points "
                    "were read."
                ),
                severity=Severity.WARNING,
                source="DOSCAR",
            )
        )

    table = values[: complete * n_columns].reshape(complete, n_columns)
    energies = table[:, 0].copy()

    if n_spin == 1:
        total = table[:, 1][None, :].copy()
        integrated = table[:, 2][None, :].copy()
    else:
        total = np.stack([table[:, 1], table[:, 2]])
        integrated = np.stack([table[:, 3], table[:, 4]])

    log.info(
        "Parsed DOSCAR: %d points, %d spin channel(s), E_fermi=%.4f eV",
        complete,
        n_spin,
        e_fermi,
    )

    data = DosData(
        energies=energies,
        total=total,
        integrated=integrated,
        e_fermi=e_fermi,
        n_ions=n_ions,
        has_projected=has_projected,
    )
    return ParseResult.success(data, issues)
