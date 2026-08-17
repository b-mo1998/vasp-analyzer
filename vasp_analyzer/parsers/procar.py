"""PROCAR parser: per-ion, per-orbital projected band character.

Layout, for the standard collinear (ISPIN=1) case::

    PROCAR lm decomposed
    # of k-points:   90         # of bands:    8         # of ions:    2

     k-point    1 :    0.00 0.00 0.00     weight = 0.0111

    band     1 # energy  -21.670 # occ.  2.000

    ion      s     py     pz     px    dxy    dyz    dz2    dxz  x2-y2    tot
      1  0.322  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.322
      2  0.322  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.322
    tot  0.643  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.643

For ISPIN=2, the whole k-point block repeats twice (spin up, then spin down).
For a non-collinear run (LNONCOLLINEAR=T), each band carries four consecutive
ion tables — total character, then the x/y/z magnetisation components — rather
than one. That variant is detected and reported rather than silently
mis-parsed: only the total-character table is unambiguous across VASP
versions, and getting the other three wrong would misrepresent the spin
texture, not just omit it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.errors import Issue, ParseResult, Severity, safe_parser
from ..core.logging_setup import get_logger

log = get_logger("parsers.procar")


@dataclass(frozen=True)
class ProcarHeader:
    """The declared dimensions, read without touching the projection data."""

    n_kpoints: int
    n_bands: int
    n_ions: int


def peek_procar_header(path: Path) -> ProcarHeader | None:
    """Read just the second line — cheap enough for discovery-time indexing
    and for cross-checking against EIGENVAL, the same way DOSCAR is peeked."""
    try:
        with Path(path).open("r", errors="replace") as handle:
            handle.readline()
            second = handle.readline()
        parts = second.split()
        # "# of k-points:   90   # of bands:   8   # of ions:   2"
        numbers = [int(p) for p in parts if p.isdigit()]
        if len(numbers) < 3:
            return None
        return ProcarHeader(n_kpoints=numbers[0], n_bands=numbers[1], n_ions=numbers[2])
    except (OSError, ValueError, IndexError) as exc:
        log.warning("Could not peek PROCAR header at %s: %s", path, exc)
        return None


@dataclass
class ProcarData:
    """Projected character, per spin channel, k-point, band, ion and orbital."""

    orbitals: list[str]
    ion_weights: np.ndarray   # (n_spin, n_kpoints, n_bands, n_ions, n_orbitals)
    source: str = "PROCAR"

    @property
    def n_spin(self) -> int:
        return int(self.ion_weights.shape[0])

    @property
    def n_kpoints(self) -> int:
        return int(self.ion_weights.shape[1])

    @property
    def n_bands(self) -> int:
        return int(self.ion_weights.shape[2])

    @property
    def n_ions(self) -> int:
        return int(self.ion_weights.shape[3])

    def element_weight(self, species: list[str], element: str, spin: int = 0) -> np.ndarray:
        """(n_kpoints, n_bands) total character on one element, summed over its
        ions and every orbital."""
        mask = np.array([s == element for s in species], dtype=bool)
        if mask.size != self.n_ions or not mask.any():
            return np.zeros((self.n_kpoints, self.n_bands))
        spin = min(spin, self.n_spin - 1)
        return self.ion_weights[spin][:, :, mask, :].sum(axis=(2, 3))

    def total_weight(self, spin: int = 0) -> np.ndarray:
        """(n_kpoints, n_bands) character summed over every ion and orbital —
        used to spot bands the basis barely describes (weight well below 1)."""
        spin = min(spin, self.n_spin - 1)
        return self.ion_weights[spin].sum(axis=(2, 3))


def _split_ion_row(line: str) -> list[float] | None:
    parts = line.split()
    if len(parts) < 2:
        return None
    try:
        return [float(p) for p in parts[1:]]  # drop the ion index / "tot" label
    except ValueError:
        return None


def _read_one_table(
    lines: list[str], index: int, n_ions: int, n_orbitals: int | None
) -> tuple[list[list[float]] | None, list[str] | None, int]:
    """Read one ``ion ... tot`` table starting at ``index``.

    Returns ``(rows, orbital_names, next_index)``. ``rows`` is ``None`` when
    the table is missing or incomplete (EOF, or a "tot"/next-block line
    appearing before ``n_ions`` rows were read).
    """
    total = len(lines)
    if index >= total or not lines[index].strip().startswith("ion"):
        return None, None, index

    header_line = lines[index]
    index += 1
    orbital_names = header_line.split()[1:-1] if n_orbitals is None else None
    # Each data row also carries a trailing per-ion "tot" value after the
    # orbital columns, so rows must always be truncated to the orbital count
    # — including on this first table, before the caller has learned it from
    # orbital_names. Skipping that on the first table left it one column
    # longer than every later one, which numpy then refused to stack.
    effective_orbitals = n_orbitals if n_orbitals is not None else len(orbital_names)

    rows: list[list[float]] = []
    for _ in range(n_ions):
        if index >= total:
            return None, orbital_names, index
        values = _split_ion_row(lines[index])
        if values is None:
            return None, orbital_names, index
        rows.append(values[:effective_orbitals])
        index += 1

    # The "tot" summary row, if present — advance past it either way.
    if index < total and lines[index].strip().startswith("tot"):
        index += 1

    return rows, orbital_names, index


def _skip_blank(lines: list[str], index: int) -> int:
    total = len(lines)
    while index < total and not lines[index].strip():
        index += 1
    return index


@safe_parser("PROCAR")
def parse_procar(path: Path) -> ParseResult[ProcarData]:
    path = Path(path)
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) < 8:
        return ParseResult.failure(
            "procar_truncated", "PROCAR is too short to contain any data.", source="PROCAR"
        )

    header = peek_procar_header(path)
    if header is None:
        return ParseResult.failure(
            "procar_bad_header",
            "PROCAR's declared dimensions could not be read.",
            source="PROCAR",
        )

    issues: list[Issue] = []
    n_ions = header.n_ions

    orbitals: list[str] | None = None
    blocks: list[list[list[float]]] = []  # one entry per (kpoint, band): n_ions x n_orbitals
    kpoint_blocks_seen = 0

    index = 0
    total = len(lines)
    while index < total:
        stripped = lines[index].strip()

        if stripped.startswith("k-point"):
            kpoint_blocks_seen += 1
            index += 1
            continue

        if not stripped.startswith("band"):
            index += 1
            continue

        index = _skip_blank(lines, index + 1)
        n_orbitals = len(orbitals) if orbitals is not None else None
        rows, names, index = _read_one_table(lines, index, n_ions, n_orbitals)
        if orbitals is None and names is not None:
            orbitals = names

        if rows is None:
            break
        blocks.append(rows)

        # A non-collinear PROCAR carries three more tables per band (the x/y/z
        # magnetisation components); only the total-character table just read
        # is used, so skip past any further "ion ... tot" tables that belong
        # to this same band before looking for the next "band"/"k-point" line.
        while True:
            index = _skip_blank(lines, index)
            if index >= total or not lines[index].strip().startswith("ion"):
                break
            _, _, index = _read_one_table(lines, index, n_ions, len(orbitals) if orbitals else None)

    if orbitals is None or not blocks:
        return ParseResult.failure(
            "procar_no_data",
            "PROCAR contains a header but no projection data.",
            source="PROCAR",
        )

    # Reduce to whatever's actually complete, the same way EIGENVAL handles a
    # truncated file: a "k-point" here means a full set of n_bands band tables,
    # so an interrupted job that stopped mid-band, or mid-k-point, loses only
    # that partial tail rather than failing the whole file.
    complete_kpoint_units = len(blocks) // header.n_bands
    if complete_kpoint_units == 0:
        return ParseResult.failure(
            "procar_truncated",
            "PROCAR ends before a single complete k-point was read.",
            source="PROCAR",
        )

    is_multi_spin_file = kpoint_blocks_seen >= 2 * header.n_kpoints
    if is_multi_spin_file and complete_kpoint_units >= header.n_kpoints:
        n_spin = max(1, complete_kpoint_units // header.n_kpoints)
        n_kpoints = header.n_kpoints
    else:
        n_spin = 1
        n_kpoints = min(complete_kpoint_units, header.n_kpoints)

    used_blocks = n_spin * n_kpoints * header.n_bands
    if used_blocks < len(blocks) or n_kpoints < header.n_kpoints:
        issues.append(
            Issue(
                code="procar_truncated",
                message=(
                    f"PROCAR is incomplete: {n_kpoints} of {header.n_kpoints} "
                    "k-points were read."
                ),
                severity=Severity.WARNING,
                source="PROCAR",
            )
        )
    blocks = blocks[:used_blocks]

    array = np.array(blocks, dtype=float)  # (n_spin*n_kpts*n_bands, n_ions, n_orbitals)
    array = array.reshape(n_spin, n_kpoints, header.n_bands, n_ions, len(orbitals))

    log.info(
        "Parsed PROCAR: %d k-points, %d bands, %d ions, %d orbitals, %d spin channel(s)",
        n_kpoints,
        header.n_bands,
        n_ions,
        len(orbitals),
        n_spin,
    )

    return ParseResult.success(ProcarData(orbitals=orbitals, ion_weights=array), issues)
