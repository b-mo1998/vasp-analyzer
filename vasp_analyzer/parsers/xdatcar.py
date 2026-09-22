"""XDATCAR parser: the ionic trajectory of an MD or relaxation run.

Two layouts exist and both appear in the wild:

*Fixed cell* (ISIF <= 2, the usual NVT/NVE case) writes the seven-line POSCAR
header exactly once, then one ``Direct configuration=`` block per step::

    Graphene 5x5
       1
       12.35  0.00  0.00
       ...
       C
       50
    Direct configuration=     1
      0.000  0.000  0.250
      ...
    Direct configuration=     2
      ...

*Variable cell* (NPT) repeats the whole header before every configuration, so
the lattice has to be tracked per frame rather than read once.

The file is read in one pass and converted with a single ``np.array`` over the
joined coordinate text: a 5,000-frame trajectory is tens of millions of tokens,
and calling ``float()`` on each one is what makes naive parsers slow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.errors import Issue, ParseResult, Severity, safe_parser
from ..core.logging_setup import get_logger
from ..models.trajectory import MDTrajectory

log = get_logger("parsers.xdatcar")

_CONFIG_MARKER = "configuration="


def _is_int_list(tokens: list[str]) -> bool:
    if not tokens:
        return False
    for token in tokens:
        try:
            int(token)
        except ValueError:
            return False
    return True


def _parse_header(lines: list[str], start: int) -> tuple[dict, int] | None:
    """Read a POSCAR-style header beginning at ``start``.

    Returns ``({lattice, species, n_atoms, comment}, next_index)``, or None when
    the lines there are not a header — which is how a repeated (NPT) header is
    told apart from the next configuration block.
    """
    if start + 6 >= len(lines):
        return None
    try:
        scale = float(lines[start + 1].split()[0])
        vectors = np.array(
            [[float(x) for x in lines[start + 2 + i].split()[:3]] for i in range(3)],
            dtype=float,
        )
    except (ValueError, IndexError):
        return None

    if scale < 0:
        current = abs(np.linalg.det(vectors))
        scale = (abs(scale) / current) ** (1.0 / 3.0) if current > 0 else 1.0
    lattice = vectors * scale

    name_tokens = lines[start + 5].split()
    if not name_tokens:
        return None

    if _is_int_list(name_tokens):
        # VASP 4 style: counts with no element names.
        counts = [int(t) for t in name_tokens]
        names = [f"El{i + 1}" for i in range(len(counts))]
        cursor = start + 6
    else:
        names = name_tokens
        try:
            counts = [int(float(t)) for t in lines[start + 6].split()]
        except ValueError:
            return None
        cursor = start + 7

    if not counts or len(names) != len(counts):
        return None

    species: list[str] = []
    for name, count in zip(names, counts):
        species.extend([name] * count)
    if not species:
        return None

    return (
        {
            "lattice": lattice,
            "species": species,
            "n_atoms": len(species),
            "comment": lines[start].strip(),
        },
        cursor,
    )


@safe_parser("XDATCAR")
def parse_xdatcar(path: Path) -> ParseResult[MDTrajectory]:
    path = Path(path)
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) < 8:
        return ParseResult.failure(
            "xdatcar_truncated",
            "XDATCAR is too short to contain a trajectory.",
            source="XDATCAR",
        )

    header = _parse_header(lines, 0)
    if header is None:
        return ParseResult.failure(
            "xdatcar_bad_header",
            "XDATCAR does not begin with a readable structure header.",
            source="XDATCAR",
        )
    info, cursor = header
    n_atoms = info["n_atoms"]
    species = info["species"]
    lattice = info["lattice"]

    issues: list[Issue] = []
    coord_blocks: list[str] = []
    frame_lattices: list[np.ndarray] = []
    total = len(lines)
    truncated = False

    while cursor < total:
        line = lines[cursor]
        stripped = line.strip()

        if not stripped:
            cursor += 1
            continue

        if _CONFIG_MARKER in stripped.replace(" ", "").lower() or stripped.lower().startswith(
            ("direct", "cartesian")
        ):
            block = lines[cursor + 1 : cursor + 1 + n_atoms]
            if len(block) < n_atoms:
                # An interrupted job leaves a partial final block; keep every
                # complete frame before it rather than rejecting the file.
                truncated = True
                break
            coord_blocks.append("\n".join(block))
            frame_lattices.append(lattice)
            cursor += 1 + n_atoms
            continue

        # Not a configuration line: an NPT run repeats the header here.
        repeated = _parse_header(lines, cursor)
        if repeated is not None and repeated[0]["n_atoms"] == n_atoms:
            lattice = repeated[0]["lattice"]
            cursor = repeated[1]
            continue

        cursor += 1

    if not coord_blocks:
        return ParseResult.failure(
            "xdatcar_no_frames",
            "XDATCAR contains a header but no atomic configurations.",
            source="XDATCAR",
        )

    try:
        flat = np.array("\n".join(coord_blocks).split(), dtype=float)
    except ValueError as exc:
        return ParseResult.failure(
            "xdatcar_bad_coordinates",
            "XDATCAR contains coordinates that could not be read as numbers.",
            detail=str(exc),
            source="XDATCAR",
        )

    n_frames = len(coord_blocks)
    expected = n_frames * n_atoms * 3
    if flat.size < expected:
        # Ragged rows (a partially written line) — drop back to whole frames.
        usable = (flat.size // (n_atoms * 3)) * n_atoms * 3
        if usable == 0:
            return ParseResult.failure(
                "xdatcar_truncated_frame",
                "XDATCAR ends before a single complete configuration was written.",
                source="XDATCAR",
            )
        flat = flat[:usable]
        n_frames = usable // (n_atoms * 3)
        frame_lattices = frame_lattices[:n_frames]
        truncated = True
    elif flat.size > expected:
        flat = flat[:expected]

    frac = flat.reshape(n_frames, n_atoms, 3)

    if truncated:
        issues.append(
            Issue(
                code="xdatcar_truncated",
                message=(
                    f"XDATCAR ends mid-configuration; {n_frames} complete frames "
                    "were read."
                ),
                severity=Severity.WARNING,
                source="XDATCAR",
            )
        )

    trajectory = MDTrajectory(
        species=species,
        frac_coords=frac,
        lattices=np.array(frame_lattices, dtype=float),
        comment=info["comment"],
        source="XDATCAR",
        source_files=["XDATCAR"],
    )
    log.info(
        "Parsed XDATCAR: %d frames, %d atoms, %s%s",
        n_frames,
        n_atoms,
        trajectory.formula,
        ", variable cell" if trajectory.is_variable_cell else "",
    )
    return ParseResult.success(trajectory, issues)


def peek_xdatcar_frames(path: Path) -> int | None:
    """Count configurations without reading coordinates.

    Used at discovery time, where the page only needs to know whether a usable
    trajectory exists — reading a multi-gigabyte XDATCAR to answer that would
    defeat the point of a cheap scan.
    """
    try:
        count = 0
        with Path(path).open("r", errors="replace") as handle:
            for line in handle:
                lowered = line.strip().lower()
                if _CONFIG_MARKER in lowered.replace(" ", ""):
                    count += 1
        return count or None
    except OSError as exc:
        log.warning("Could not peek XDATCAR at %s: %s", path, exc)
        return None
