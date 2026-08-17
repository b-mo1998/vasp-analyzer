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
