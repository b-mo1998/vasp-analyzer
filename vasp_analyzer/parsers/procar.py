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
For a non-collinear run (LSORBIT/LNONCOLLINEAR=T), each band carries four
consecutive ion tables — total character, then the x/y/z magnetisation
components (mx, my, mz) — rather than one. All four are captured: the total
table feeds the ordinary element/orbital weights exactly as before, and the
three magnetisation tables become ``ProcarData.spin_texture``, only when
every band in the file carries exactly three of them consistently — a file
that doesn't is left without spin texture rather than risk misaligning it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.errors import Issue, ParseResult, Severity, safe_parser
from ..core.logging_setup import get_logger

log = get_logger("parsers.procar")

# Largest PROCAR to attempt, in MB. Parsing costs roughly twice the file size
# in RAM for the line list, plus the numeric arrays built from it — so ~2 GB
# of input is already several GB resident. Past this the honest answer is to
# decline rather than to risk an out-of-memory kill mid-parse.
MAX_PROCAR_MB = 2048.0


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
    # (n_kpoints, n_bands, n_ions, n_orbitals, 3) mx/my/mz — only set for a
    # non-collinear PROCAR whose spin tables were consistently readable.
    spin_texture: np.ndarray | None = None
    source: str = "PROCAR"

    @property
    def n_spin(self) -> int:
        return int(self.ion_weights.shape[0])

    @property
    def is_noncollinear(self) -> bool:
        return self.spin_texture is not None

    @property
    def n_kpoints(self) -> int:
        return int(self.ion_weights.shape[1])

    @property
    def n_bands(self) -> int:
        return int(self.ion_weights.shape[2])

    @property
    def n_ions(self) -> int:
        return int(self.ion_weights.shape[3])

    def element_weight(
        self,
        species: list[str],
        element: str,
        spin: int = 0,
        orbitals: list[str] | None = None,
    ) -> np.ndarray:
        """(n_kpoints, n_bands) character on one element, summed over its ions.

        Summed over every orbital by default; pass ``orbitals`` (a subset of
        :attr:`orbitals`, e.g. ``["pz"]`` or ``["px", "py"]``) to restrict to
        just that combination — graphene's pz-dominated pi bands are
        invisible under "by element" alone when there's only one element,
        but stand out immediately filtered to pz.
        """
        mask = np.array([s == element for s in species], dtype=bool)
        if mask.size != self.n_ions or not mask.any():
            return np.zeros((self.n_kpoints, self.n_bands))
        spin = min(spin, self.n_spin - 1)
        weights = self.ion_weights[spin][:, :, mask, :]
        weights = self._restrict_to_orbitals(weights, orbitals)
        return weights.sum(axis=(2, 3))

    def atom_weight(
        self,
        rows: list[int] | set[int],
        spin: int = 0,
        orbitals: list[str] | None = None,
    ) -> np.ndarray:
        """(n_kpoints, n_bands) character on exactly these ion rows, summed.

        The atom-ID counterpart to :meth:`element_weight`: that one groups by
        species, this one takes explicit 0-based rows — a Te dopant among 112
        carbons is a `[112]`-style set here, not something a species filter
        can pick out from the other carbons.
        """
        valid = sorted(r for r in rows if 0 <= r < self.n_ions)
        if not valid:
            return np.zeros((self.n_kpoints, self.n_bands))
        mask = np.zeros(self.n_ions, dtype=bool)
        mask[valid] = True
        spin = min(spin, self.n_spin - 1)
        weights = self.ion_weights[spin][:, :, mask, :]
        weights = self._restrict_to_orbitals(weights, orbitals)
        return weights.sum(axis=(2, 3))

    def total_weight(self, spin: int = 0, orbitals: list[str] | None = None) -> np.ndarray:
        """(n_kpoints, n_bands) character summed over every ion, and every
        orbital unless ``orbitals`` restricts it to a subset — used to spot
        bands the basis (or the selected orbitals) barely describes."""
        spin = min(spin, self.n_spin - 1)
        weights = self._restrict_to_orbitals(self.ion_weights[spin], orbitals)
        return weights.sum(axis=(2, 3))

    def spin_projection(
        self, axis: int = 2, orbitals: list[str] | None = None
    ) -> np.ndarray | None:
        """(n_kpoints, n_bands) net spin moment along ``axis`` (0=x, 1=y,
        2=z), summed over every ion and every orbital unless ``orbitals``
        restricts it to a subset.

        ``None`` for a collinear PROCAR (:attr:`spin_texture` unset) — there
        is no per-band spin *moment* to report there, only separate spin-up
        and spin-down channels via :attr:`ion_weights`.
        """
        if self.spin_texture is None:
            return None
        weights = self._restrict_to_orbitals(self.spin_texture[..., axis], orbitals)
        return weights.sum(axis=(2, 3))

    def _restrict_to_orbitals(
        self, weights: np.ndarray, orbitals: list[str] | None
    ) -> np.ndarray:
        if orbitals is None:
            return weights
        indices = [self.orbitals.index(o) for o in orbitals if o in self.orbitals]
        if not indices:
            return np.zeros_like(weights[..., :1])
        return weights[..., indices]


def _split_ion_row(line: str) -> list[str] | None:
    """The value tokens of one ion's data row (index/"tot" label dropped),
    left as strings rather than floats.

    Converting each value through Python's ``float()`` here, one at a time,
    used to account for essentially all of this parser's runtime — profiled
    at ~4.7 million calls and 8 of 13 seconds on a real 216-ion, 48-kpoint
    PROCAR. That conversion is deferred instead to a single bulk
    ``numpy.array(..., dtype=float)`` call over every row in the file at
    once (see ``parse_procar``): numpy parsing the same strings in C does
    the identical conversion roughly twice as fast, with no per-call Python
    overhead repeated millions of times.

    Validity here is checked on the index column alone — always present,
    always a plain integer — rather than every value column. That is enough
    to catch a genuinely truncated or non-numeric line (an unexpected
    header, a cut-off file); a value column that is present but not
    numeric (a VASP field-overflow artifact, not something any of these
    real PROCAR files exhibit) surfaces instead when the bulk conversion
    runs, as a single clear failure rather than millions of redundant
    per-value checks.
    """
    parts = line.split()
    if len(parts) < 2:
        return None
    try:
        float(parts[0])
    except ValueError:
        return None
    return parts[1:]


def _read_one_table(
    lines: list[str], index: int, n_ions: int, n_orbitals: int | None
) -> tuple[list[list[str]] | None, list[str] | None, int]:
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

    rows: list[list[str]] = []
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


def _read_data_rows_only(
    lines: list[str], index: int, n_ions: int, n_orbitals: int
) -> tuple[list[list[str]] | None, int]:
    """Read one ion-data table that has no ``ion ...`` header line of its own
    — the shape a non-collinear PROCAR's mx/my/mz sub-tables take, stacked
    directly after the (header-bearing) total-character table with no blank
    line or repeated header between them."""
    total = len(lines)
    rows: list[list[str]] = []
    for _ in range(n_ions):
        if index >= total:
            return None, index
        values = _split_ion_row(lines[index])
        if values is None:
            return None, index
        rows.append(values[:n_orbitals])
        index += 1
    if index < total and lines[index].strip().startswith("tot"):
        index += 1
    return rows, index


def _skip_blank(lines: list[str], index: int) -> int:
    total = len(lines)
    while index < total and not lines[index].strip():
        index += 1
    return index


@safe_parser("PROCAR")
def parse_procar(path: Path) -> ParseResult[ProcarData]:
    path = Path(path)

    size_mb = path.stat().st_size / 1_048_576 if path.is_file() else 0.0
    if size_mb > MAX_PROCAR_MB:
        # Refused rather than attempted: the line list alone costs roughly
        # twice the file size in RAM, so a file this large would push most
        # machines into swap or an out-of-memory kill. Losing the projection
        # is a far better outcome than taking the whole application (or the
        # user's session) down with it.
        return ParseResult.failure(
            "procar_too_large",
            (
                f"PROCAR is {size_mb / 1024:.1f} GB, beyond the "
                f"{MAX_PROCAR_MB / 1024:.1f} GB this can load without "
                "exhausting memory. Re-run with LORBIT=10 (no per-ion "
                "decomposition) or a coarser k-mesh for a smaller file."
            ),
            source="PROCAR",
        )

    # Read line by line rather than read_text().splitlines(): the latter holds
    # the entire file as one string *and* the split list at the same moment,
    # peaking near four times the file size. Streaming the handle never
    # materialises the whole-file string, which halves peak memory for no
    # change in the parsing below.
    with path.open("r", errors="replace") as handle:
        lines = [line.rstrip("\n") for line in handle]

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
    # One entry per (kpoint, band): n_ions rows, each still value *strings* —
    # see _split_ion_row for why conversion is deferred to one bulk call
    # below rather than done per row here.
    blocks: list[list[list[str]]] = []
    kpoint_blocks_seen = 0

    # A non-collinear PROCAR carries three more tables per band, right after
    # the total-character one: the x/y/z magnetisation components, always in
    # that order. Captured only while every band seen so far has carried
    # exactly the same count (0, or consistently 3) — a file that doesn't is
    # left without spin texture rather than risk misaligning mx/my/mz against
    # the wrong band.
    mx_blocks: list[list[list[str]]] = []
    my_blocks: list[list[list[str]]] = []
    mz_blocks: list[list[list[str]]] = []
    spin_expected: int | None = None
    spin_capture_ok = True

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

        # The three magnetisation sub-tables (mx, my, mz), when present, are
        # stacked immediately after the total-character table just read —
        # no blank line, no repeated "ion ..." header — so they're only
        # distinguishable from "nothing more here" by whether the very next
        # line still parses as an ion-data row at all.
        extras: list[list[list[float]]] = []
        for _ in range(3):
            if index >= total or _split_ion_row(lines[index]) is None:
                break
            extra_rows, index = _read_data_rows_only(lines, index, n_ions, len(orbitals))
            if extra_rows is None:
                break
            extras.append(extra_rows)

        if spin_expected is None:
            spin_expected = len(extras) if len(extras) in (0, 3) else -1
        if len(extras) != spin_expected:
            spin_capture_ok = False
        elif spin_expected == 3:
            mx_blocks.append(extras[0])
            my_blocks.append(extras[1])
            mz_blocks.append(extras[2])

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

    # Every row was validated only on its index column as it was read (see
    # _split_ion_row); the actual value strings are converted to float here,
    # in one bulk call, rather than the ~5 million individual float() calls
    # that used to happen one row at a time. A value that is present but not
    # numeric — a VASP field-overflow artifact, not something any of these
    # real PROCAR files exhibit — surfaces here as a single clear failure
    # instead of being caught (and silently truncated) row by row.
    try:
        array = np.array(blocks, dtype=float)  # (n_spin*n_kpts*n_bands, n_ions, n_orbitals)
    except ValueError as exc:
        return ParseResult.failure(
            "procar_bad_value",
            "PROCAR contains a non-numeric projection value.",
            detail=str(exc),
            source="PROCAR",
        )
    array = array.reshape(n_spin, n_kpoints, header.n_bands, n_ions, len(orbitals))

    spin_texture = None
    # Non-collinear PROCAR is inherently single-channel in the ion_weights
    # sense (the k-point loop isn't doubled the way ISPIN=2 doubles it), so
    # spin texture is only meaningful — and only ever populated above — when
    # n_spin came out as 1.
    if (
        spin_capture_ok
        and spin_expected == 3
        and n_spin == 1
        and len(mx_blocks) >= used_blocks
    ):
        shape = (n_kpoints, header.n_bands, n_ions, len(orbitals))
        try:
            mx = np.array(mx_blocks[:used_blocks], dtype=float).reshape(shape)
            my = np.array(my_blocks[:used_blocks], dtype=float).reshape(shape)
            mz = np.array(mz_blocks[:used_blocks], dtype=float).reshape(shape)
        except ValueError:
            # The main character table above already parsed cleanly — a bad
            # spin-texture value alone is a reason to drop just the spin
            # texture, not the whole (otherwise good) PROCAR.
            mx = my = mz = None
        if mx is not None:
            spin_texture = np.stack([mx, my, mz], axis=-1)

    log.info(
        "Parsed PROCAR: %d k-points, %d bands, %d ions, %d orbitals, %d spin channel(s)%s",
        n_kpoints,
        header.n_bands,
        n_ions,
        len(orbitals),
        n_spin,
        ", non-collinear spin texture" if spin_texture is not None else "",
    )

    return ParseResult.success(
        ProcarData(orbitals=orbitals, ion_weights=array, spin_texture=spin_texture), issues
    )
