"""OUTCAR metadata parser.

OUTCAR can be hundreds of megabytes, and everything the app needs sits either
in the header block (echoed INCAR, dimensions, POTCAR titles) or in the last
electronic step (Fermi level, timing). Rather than stream the whole file, small
windows are read from the head and the tail; a file below the window size is
read whole. This keeps a 1 GB OUTCAR as fast to index as a 1 MB one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.errors import ParseResult, safe_parser
from ..core.logging_setup import get_logger

log = get_logger("parsers.outcar")

WINDOW = 512 * 1024

_RE_FERMI = re.compile(r"E-fermi\s*:\s*(-?\d+\.?\d*)")
_RE_ENCUT = re.compile(r"ENCUT\s*=\s*(\d+\.?\d*)")
_RE_NBANDS = re.compile(r"NBANDS\s*=\s*(\d+)")
_RE_NKPTS = re.compile(r"NKPTS\s*=\s*(\d+)")
_RE_NELECT = re.compile(r"NELECT\s*=\s*(-?\d+\.?\d*)")
_RE_NIONS = re.compile(r"NIONS\s*=\s*(\d+)")
_RE_TITEL = re.compile(r"TITEL\s*=\s*(.+)")
_RE_KMESH = re.compile(r"generate k-points for:\s*(\d+)\s+(\d+)\s+(\d+)")
_RE_ELAPSED = re.compile(r"Elapsed time \(sec\):\s*(\d+\.?\d*)")
_RE_VERSION = re.compile(r"vasp\.(\S+)")
_RE_ISPIN = re.compile(r"ISPIN\s*=\s*(\d+)")
_RE_LNONCOL = re.compile(r"LNONCOLLINEAR\s*=\s*([TF])")
_RE_LSORBIT = re.compile(r"LSORBIT\s*=\s*([TF])")


@dataclass
class OutcarInfo:
    """Facts extracted from OUTCAR. Every field may legitimately be None."""

    e_fermi: float | None = None
    encut: float | None = None
    n_bands: int | None = None
    n_kpoints: int | None = None
    n_electrons: float | None = None
    n_ions: int | None = None
    kmesh: tuple[int, int, int] | None = None
    vasp_version: str | None = None
    elapsed_seconds: float | None = None
    ispin: int | None = None
    noncollinear: bool | None = None
    soc: bool | None = None
    potcar_species: list[str] = field(default_factory=list)
    potcar_titles: list[str] = field(default_factory=list)
    finished: bool = False

    @property
    def kmesh_label(self) -> str | None:
        if self.kmesh is None:
            return None
        a, b, c = self.kmesh
        return f"{a}×{b}×{c}"

    @property
    def spin_mode(self) -> str | None:
        if self.soc or self.noncollinear:
            return "Non-collinear"
        if self.ispin == 2:
            return "Spin-polarised"
        if self.ispin == 1:
            return "Non-spin-polarised"
        return None


def _read_windows(path: Path) -> tuple[str, str]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size <= 2 * WINDOW:
            head = handle.read().decode("utf-8", errors="replace")
            return head, head
        head = handle.read(WINDOW).decode("utf-8", errors="replace")
        handle.seek(-WINDOW, 2)
        tail = handle.read().decode("utf-8", errors="replace")
    return head, tail


@safe_parser("OUTCAR")
def parse_outcar(path: Path) -> ParseResult[OutcarInfo]:
    path = Path(path)
    head, tail = _read_windows(path)
    info = OutcarInfo()

    version = _RE_VERSION.search(head)
    if version:
        info.vasp_version = version.group(1)

    for regex, attr, cast in (
        (_RE_ENCUT, "encut", float),
        (_RE_NBANDS, "n_bands", int),
        (_RE_NKPTS, "n_kpoints", int),
        (_RE_NELECT, "n_electrons", float),
        (_RE_NIONS, "n_ions", int),
    ):
        match = regex.search(head)
        if match:
            setattr(info, attr, cast(match.group(1)))

    mesh = _RE_KMESH.search(head)
    if mesh:
        info.kmesh = (int(mesh.group(1)), int(mesh.group(2)), int(mesh.group(3)))

    # The runtime flags appear twice: the echoed INCAR and VASP's own summary.
    # The summary (single letter T/F) is authoritative.
    noncol = _RE_LNONCOL.search(head)
    if noncol:
        info.noncollinear = noncol.group(1) == "T"
    soc = _RE_LSORBIT.search(head)
    if soc:
        info.soc = soc.group(1) == "T"
    for match in _RE_ISPIN.finditer(head):
        info.ispin = int(match.group(1))

    for match in _RE_TITEL.finditer(head):
        title = match.group(1).strip()
        info.potcar_titles.append(title)
        parts = title.split()
        if len(parts) >= 2:
            info.potcar_species.append(parts[1].split("_")[0])

    # Fermi level and timing: last occurrence, from the tail window.
    fermi_matches = _RE_FERMI.findall(tail)
    if fermi_matches:
        info.e_fermi = float(fermi_matches[-1])
    elapsed = _RE_ELAPSED.findall(tail)
    if elapsed:
        info.elapsed_seconds = float(elapsed[-1])
        info.finished = True

    log.info(
        "Parsed OUTCAR: version=%s E_fermi=%s NBANDS=%s NKPTS=%s",
        info.vasp_version,
        info.e_fermi,
        info.n_bands,
        info.n_kpoints,
    )
    return ParseResult.success(info)


# --- final ionic state (used by the NEB page) ------------------------------------

# The number groups are written ``\d+(?:\.\d*)?`` rather than ``\d+\.?\d*``:
# both match the same strings, but the latter can split one run of digits
# between ``\d+`` and ``\d*`` in O(n) ways, so a crafted digit run that is not
# followed by whitespace makes the match backtrack O(n^2) times.
_RE_SIGMA0 = re.compile(r"energy\s+without\s+entropy\s*=\s*(-?\d+(?:\.\d*)?)\s+"
                        r"energy\(sigma->0\)\s*=\s*(-?\d+(?:\.\d*)?)")
_RE_TOTEN = re.compile(r"free\s+energy\s+TOTEN\s*=\s*(-?\d+\.?\d*)")
_FORCE_HEADER = "TOTAL-FORCE"


@dataclass
class IonicState:
    """The converged result of an OUTCAR's last ionic step."""

    energy: float | None = None         # energy(sigma->0), eV
    free_energy: float | None = None    # TOTEN, eV
    max_force: float | None = None      # largest |F| over atoms, eV/Å
    forces: np.ndarray | None = None    # (n_atoms, 3), eV/Å
    n_ionic_steps: int = 0

    @property
    def n_atoms(self) -> int:
        return 0 if self.forces is None else int(self.forces.shape[0])


def _read_tail(path: Path, size: int) -> str:
    total = path.stat().st_size
    with path.open("rb") as handle:
        if total > size:
            handle.seek(-size, 2)
        return handle.read().decode("utf-8", errors="replace")


def _last_force_block(lines: list[str]) -> np.ndarray | None:
    """Atomic forces from the final ``POSITION / TOTAL-FORCE`` table."""
    starts = [i for i, line in enumerate(lines) if _FORCE_HEADER in line]
    if not starts:
        return None
    # The header is followed by a dashed rule, then one row per atom holding
    # x y z fx fy fz, then another dashed rule.
    cursor = starts[-1] + 1
    while cursor < len(lines) and set(lines[cursor].strip()) <= {"-"}:
        cursor += 1

    rows: list[list[float]] = []
    for line in lines[cursor:]:
        tokens = line.split()
        if len(tokens) < 6:
            break
        try:
            rows.append([float(t) for t in tokens[3:6]])
        except ValueError:
            break
    return np.array(rows, dtype=float) if rows else None


@safe_parser("OUTCAR")
def parse_ionic_state(path: Path) -> ParseResult[IonicState]:
    """Read the last ionic step's energy and forces.

    ``energy(sigma->0)`` is used rather than TOTEN because it is the smearing-
    extrapolated energy, which is what NEB barriers should be compared with —
    and it is what VTST's own ``nebef.dat`` reports, so the two agree exactly.

    Only the tail of the file is read. The window is grown once if the final
    force table did not fit, which keeps a hundred-megabyte relaxation OUTCAR
    as cheap to summarise as a small one.
    """
    path = Path(path)
    state = IonicState()

    for window in (WINDOW, 8 * WINDOW):
        text = _read_tail(path, window)
        lines = text.splitlines()

        sigma = _RE_SIGMA0.findall(text)
        if sigma:
            state.energy = float(sigma[-1][1])
        toten = _RE_TOTEN.findall(text)
        if toten:
            state.free_energy = float(toten[-1])
        state.n_ionic_steps = len(toten)

        forces = _last_force_block(lines)
        if forces is not None:
            state.forces = forces
            state.max_force = float(np.linalg.norm(forces, axis=1).max())

        # A complete answer, or the whole file already read: stop.
        if (state.energy is not None and state.forces is not None) or window >= path.stat().st_size:
            break

    if state.energy is None and state.free_energy is None and state.forces is None:
        return ParseResult.failure(
            "outcar_no_ionic_state",
            "OUTCAR contains no completed ionic step.",
            source=path.name,
        )
    return ParseResult.success(state)


# --- magnetic moments ---------------------------------------------------------------

_MAG_HEADER = re.compile(r"^\s*magnetization \(([xyz])\)\s*$")


@dataclass
class MagneticMoments:
    """Per-atom magnetic moments from the last ionic step, in μB.

    ``vectors`` is always (n_atoms, 3) so a viewer can draw it without caring
    which kind of run produced it. A collinear (ISPIN=2) calculation reports
    one number per atom; VASP quantises that along z by convention, so it is
    stored as (0, 0, m) and ``collinear`` records that the x/y components are
    definitional zeros rather than measured ones.
    """

    vectors: np.ndarray
    collinear: bool

    @property
    def n_atoms(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def magnitudes(self) -> np.ndarray:
        return np.linalg.norm(self.vectors, axis=1)

    @property
    def scalar(self) -> np.ndarray:
        """Signed moment per atom — the z component for collinear runs, the
        magnitude otherwise (a non-collinear moment has no natural sign)."""
        return self.vectors[:, 2] if self.collinear else self.magnitudes

    @property
    def total(self) -> float:
        return float(self.scalar.sum())

    # -- spin components ----------------------------------------------------
    #
    # Only meaningful for a non-collinear run, where VASP writes three
    # separate magnetization tables and the moment is a genuine vector in
    # spin space. For a collinear run x and y are definitional zeros, so a
    # per-component view of one would be reporting a convention rather than
    # a result.

    def component(self, axis: str) -> np.ndarray:
        """Per-atom S_x, S_y or S_z, signed, in μB."""
        return self.vectors[:, "xyz".index(axis)]

    @property
    def total_vector(self) -> np.ndarray:
        """The cell's net moment as (S_x, S_y, S_z)."""
        return self.vectors.sum(axis=0)

    @property
    def net_magnitude(self) -> float:
        """|Σ m_i| — the size of the *net* moment.

        Distinct from ``total``: an antiferromagnet has a large summed
        magnitude per atom and a net vector of nearly zero, and conflating
        the two would describe it as non-magnetic.
        """
        return float(np.linalg.norm(self.total_vector))

    def component_overlay_vectors(self, axis: str) -> np.ndarray:
        """The chosen component alone, as (n_atoms, 3) for drawing.

        Isolating one component answers "how is S_z arranged?" without the
        other two rotating every arrow away from the axis being read.
        """
        isolated = np.zeros_like(self.vectors)
        index = "xyz".index(axis)
        isolated[:, index] = self.vectors[:, index]
        return isolated

    def dominant_axis(self) -> str:
        """Which spin axis carries the most weight across the cell.

        Used to open a non-collinear run on the component that actually has
        structure in it, rather than always on x.
        """
        weight = np.abs(self.vectors).sum(axis=0)
        return "xyz"[int(np.argmax(weight))]


def _parse_magnetization_block(lines: list[str], start: int) -> list[float] | None:
    """The ``tot`` column of one ``magnetization (x)`` table.

    Layout (LORBIT >= 10)::

         magnetization (x)

        # of ion       s       p       d       tot
        ------------------------------------------
            1        0.012   0.034   2.101   2.147
        ...
        ------------------------------------------
        tot          0.024   0.068   4.202   4.294
    """
    # Advance past the blank line and column header to the dashed rule that
    # opens the table; the rows start on the line after it. Bounded, so a
    # header with no table beneath it (a truncated file) cannot run away.
    cursor = start + 1
    found_rule = False
    while cursor < len(lines) and cursor - start <= 6:
        stripped = lines[cursor].strip()
        cursor += 1
        if stripped and set(stripped) <= {"-"}:
            found_rule = True
            break
    if not found_rule:
        return None
    values: list[float] = []
    for line in lines[cursor:]:
        stripped = line.strip()
        if not stripped or set(stripped) <= {"-"}:
            break
        tokens = stripped.split()
        if tokens[0] == "tot":
            break
        try:
            int(tokens[0])
            values.append(float(tokens[-1]))
        except (ValueError, IndexError):
            break
    return values or None


@safe_parser("OUTCAR")
def parse_magnetic_moments(path: Path) -> ParseResult[MagneticMoments]:
    """Per-atom moments from the *last* magnetization table(s) in OUTCAR.

    Only written for spin-polarised runs with LORBIT set, and only the final
    block matters — earlier ones describe unconverged ionic steps. Read from
    the tail, growing the window once, for the same reason as the ionic
    state: a relaxation OUTCAR can be enormous and the answer is at the end.
    """
    path = Path(path)
    for window in (WINDOW, 8 * WINDOW):
        lines = _read_tail(path, window).splitlines()
        blocks: dict[str, tuple[int, list[float]]] = {}
        for index, line in enumerate(lines):
            match = _MAG_HEADER.match(line)
            if not match:
                continue
            values = _parse_magnetization_block(lines, index)
            if values is not None:
                # Later blocks overwrite earlier ones: last ionic step wins.
                blocks[match.group(1)] = (index, values)

        if "x" in blocks:
            break
        if window >= path.stat().st_size:
            break

    if "x" not in blocks:
        return ParseResult.failure(
            "outcar_no_magnetization",
            "OUTCAR has no magnetization table (not spin-polarised, or LORBIT unset).",
            source=path.name,
        )

    x = np.asarray(blocks["x"][1], dtype=float)
    non_collinear = "y" in blocks and "z" in blocks
    if non_collinear:
        y = np.asarray(blocks["y"][1], dtype=float)
        z = np.asarray(blocks["z"][1], dtype=float)
        n = min(len(x), len(y), len(z))
        vectors = np.column_stack([x[:n], y[:n], z[:n]])
    else:
        # Collinear: VASP's single "(x)" table is the moment along the
        # quantisation axis, conventionally drawn along z.
        vectors = np.column_stack([np.zeros_like(x), np.zeros_like(x), x])

    moments = MagneticMoments(vectors=vectors, collinear=not non_collinear)
    log.info(
        "Parsed magnetic moments: %d atoms, %s, total %.3f μB",
        moments.n_atoms,
        "collinear" if moments.collinear else "non-collinear",
        moments.total,
    )
    return ParseResult.success(moments)
