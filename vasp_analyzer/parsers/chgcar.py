"""Volumetric VASP files: CHGCAR, CHG, PARCHG, ELFCAR and LOCPOT.

All of them share one layout::

    <POSCAR header, including the coordinates>
    <blank line>
     NGXF NGYF NGZF
    <NGXF*NGYF*NGZF values, whitespace separated>
    [augmentation occupancies ...]
    [<blank line>
     NGXF NGYF NGZF
     <a second block: the magnetization density>]

Two details decide whether the result is science or a plausible-looking
picture:

**Grid order.** VASP writes the *first* index fastest and the third slowest,
so the flat array must be read in Fortran order. Reshaping in C order instead
produces an array of exactly the right shape holding a transposed field — it
renders without complaint and is wrong. :func:`parse_volumetric` therefore
reshapes with ``order="F"`` and the test suite pins this with an asymmetric
grid where a transposition is unambiguous.

**Normalisation.** CHGCAR stores the density *multiplied by the cell volume*,
so the density in e/Å³ is ``data / V``. The electron count then follows as
``sum(data) / (nx*ny*nz)``, which is the standard sanity check: it must come
back as the calculation's valence-electron count. Plotting raw CHGCAR numbers
and labelling them e/Å³ would be wrong by a factor of the cell volume — for
the graphene cell here, a factor of about 1500.

LOCPOT and ELFCAR are *not* volume-scaled: they hold the quantity itself, so
they are read without that division.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..core.errors import Issue, ParseResult, Severity, safe_parser
from ..core.logging_setup import get_logger
from ..models.structure import StructureData
from ..models.volumetric import FieldKind, VolumetricField, VolumetricFile
from .poscar import parse_poscar

log = get_logger("parsers.chgcar")

# Files whose values are the quantity itself rather than quantity × volume.
_UNSCALED = (FieldKind.ELF, FieldKind.POTENTIAL)

_RE_GRID = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s*$")


def kind_for_filename(name: str) -> FieldKind:
    """Which physical quantity a filename implies."""
    upper = name.upper()
    if upper.startswith("ELFCAR"):
        return FieldKind.ELF
    if upper.startswith("LOCPOT"):
        return FieldKind.POTENTIAL
    if upper.startswith("PARCHG"):
        return FieldKind.PARTIAL_CHARGE
    return FieldKind.CHARGE


def _find_structure_end(lines: list[str]) -> int:
    """Index of the first line after the POSCAR header block.

    The header's length depends on whether the species line, selective
    dynamics and a velocity block are present, so it is found by counting
    atoms rather than assumed.
    """
    try:
        counts_line = 6
        tokens = lines[5].split()
        if all(_is_int(t) for t in tokens):
            counts_line = 5           # VASP 4: no species names
        counts = [int(float(t)) for t in lines[counts_line].split()]
    except (IndexError, ValueError):
        return -1

    n_atoms = sum(counts)
    cursor = counts_line + 1
    if cursor < len(lines) and lines[cursor].strip()[:1].upper() == "S":
        cursor += 1                    # selective dynamics
    cursor += 1                        # Direct / Cartesian
    return cursor + n_atoms


def _is_int(token: str) -> bool:
    try:
        int(token)
    except ValueError:
        return False
    return True


def _read_block(
    lines: list[str], start: int, expected: int
) -> tuple[np.ndarray | None, int]:
    """Read ``expected`` whitespace-separated floats beginning at ``start``.

    Returns the values and the index of the first unconsumed line. Parsing
    stops early at a line that is not numeric — that is how the PAW
    augmentation section, which follows the grid, is recognised without
    needing to know its length.
    """
    collected: list[str] = []
    count = 0
    cursor = start
    total = len(lines)

    while cursor < total and count < expected:
        stripped = lines[cursor].strip()
        if not stripped:
            cursor += 1
            continue
        first = stripped.split(maxsplit=1)[0]
        try:
            float(first)
        except ValueError:
            break                      # "augmentation occupancies ..." etc.
        collected.append(stripped)
        count += len(stripped.split())
        cursor += 1

    if count < expected:
        return None, cursor

    values = np.array(" ".join(collected).split()[:expected], dtype=float)
    return values, cursor


def _skip_augmentation(lines: list[str], start: int) -> int:
    """Advance past the PAW augmentation occupancies to the next grid header."""
    cursor = start
    total = len(lines)
    while cursor < total:
        if _RE_GRID.match(lines[cursor]):
            return cursor
        cursor += 1
    return total


@safe_parser("CHGCAR")
def parse_volumetric(
    path: Path, kind: FieldKind | None = None
) -> ParseResult[VolumetricFile]:
    """Read any VASP volumetric file into normalised scalar fields."""
    path = Path(path)
    name = path.name
    kind = kind or kind_for_filename(name)

    text = path.read_text(errors="replace")
    lines = text.splitlines()
    if len(lines) < 10:
        return ParseResult.failure(
            "volumetric_truncated",
            f"{name} is too short to contain a volumetric grid.",
            source=name,
        )

    issues: list[Issue] = []

    # --- structure ---------------------------------------------------------
    end = _find_structure_end(lines)
    if end < 0 or end >= len(lines):
        return ParseResult.failure(
            "volumetric_bad_header",
            f"{name} does not begin with a readable structure header.",
            source=name,
        )

    structure_result = parse_poscar_from_lines(lines[:end], source=name)
    if not structure_result.ok or structure_result.value is None:
        return ParseResult.failure(
            "volumetric_bad_structure",
            f"{name}'s structure header could not be read.",
            source=name,
        )
    structure = structure_result.value
    issues.extend(structure_result.issues)

    # --- grid dimensions ------------------------------------------------------
    cursor = end
    while cursor < len(lines) and not _RE_GRID.match(lines[cursor]):
        cursor += 1
    if cursor >= len(lines):
        return ParseResult.failure(
            "volumetric_no_grid",
            f"{name} contains a structure but no grid dimensions.",
            source=name,
        )

    match = _RE_GRID.match(lines[cursor])
    assert match is not None
    shape = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    expected = shape[0] * shape[1] * shape[2]
    if expected <= 0:
        return ParseResult.failure(
            "volumetric_empty_grid",
            f"{name} declares an empty grid ({shape[0]}×{shape[1]}×{shape[2]}).",
            source=name,
        )
    cursor += 1

    # --- one or more data blocks ------------------------------------------------
    volume = structure.volume
    scale = 1.0 if kind in _UNSCALED else (1.0 / volume if volume > 0 else 1.0)

    raw_blocks: list[np.ndarray] = []
    while True:
        values, cursor = _read_block(lines, cursor, expected)
        if values is None:
            break
        raw_blocks.append(values)

        cursor = _skip_augmentation(lines, cursor)
        if cursor >= len(lines):
            break
        following = _RE_GRID.match(lines[cursor])
        if not following:
            break
        if (
            int(following.group(1)),
            int(following.group(2)),
            int(following.group(3)),
        ) != shape:
            break                      # a differently shaped block is not ours
        cursor += 1

    if not raw_blocks:
        return ParseResult.failure(
            "volumetric_truncated_grid",
            (
                f"{name} declares a {shape[0]}×{shape[1]}×{shape[2]} grid but "
                "does not contain that many values."
            ),
            source=name,
        )

    # --- assemble the fields ------------------------------------------------------
    result = VolumetricFile(path_name=name, structure=structure)
    components = _component_names(kind, len(raw_blocks))

    for index, (block, (component, block_kind)) in enumerate(
        zip(raw_blocks, components)
    ):
        # order="F": VASP varies the first index fastest.
        grid = np.asfortranarray(block.reshape(shape, order="F")) * scale
        result.fields.append(
            VolumetricField(
                structure=structure,
                values=np.ascontiguousarray(grid),
                kind=block_kind,
                source_file=name,
                component=component,
                metadata={"block": index, "grid": shape},
            )
        )

    if len(raw_blocks) > 1:
        log.info("%s carries %d volumetric blocks", name, len(raw_blocks))
    if len(raw_blocks) == 3 or len(raw_blocks) > 4:
        issues.append(
            Issue(
                code="volumetric_unexpected_blocks",
                message=(
                    f"{name} contains {len(raw_blocks)} grid blocks; the extra "
                    "ones were read but their meaning is not certain."
                ),
                severity=Severity.INFO,
                source=name,
            )
        )

    log.info(
        "Parsed %s: %d×%d×%d grid, %d block(s), %d atoms",
        name,
        shape[0],
        shape[1],
        shape[2],
        len(raw_blocks),
        structure.n_atoms,
    )
    return ParseResult.success(result, issues)


def _component_names(kind: FieldKind, count: int) -> list[tuple[str, FieldKind]]:
    """Name each grid block and say what it represents.

    A second block in a CHGCAR is the magnetization density, not another
    charge density; blocks three and four of a non-collinear run are the
    remaining magnetization components.
    """
    if kind in _UNSCALED:
        return [(kind.value, kind)] + [
            (f"block{i + 2}", kind) for i in range(count - 1)
        ]

    magnetic = FieldKind.MAGNETIZATION
    names: list[tuple[str, FieldKind]] = [("total", kind)]
    if count == 2:
        names.append(("magnetization", magnetic))
    elif count == 4:
        names.extend([("m_x", magnetic), ("m_y", magnetic), ("m_z", magnetic)])
    else:
        names.extend([(f"block{i + 2}", magnetic) for i in range(count - 1)])
    return names[:count]


def parse_poscar_from_lines(lines: list[str], source: str) -> ParseResult[StructureData]:
    """Parse a POSCAR header held in memory.

    A volumetric file's structure block must be read from the same text as its
    grid — attaching the grid to an external POSCAR risks pairing it with
    coordinates from a different ionic step.
    """
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".POSCAR", delete=False) as handle:
        handle.write("\n".join(lines) + "\n")
        temporary = Path(handle.name)
    try:
        return parse_poscar(temporary, source=source)
    finally:
        temporary.unlink(missing_ok=True)


def peek_volumetric_structure(path: Path) -> StructureData | None:
    """Read just the POSCAR header off the front of a volumetric file.

    Every CHGCAR/CHG/ELFCAR/LOCPOT/PARCHG opens with a complete POSCAR block,
    terminated by a blank line before the grid dimensions — so a directory
    (or a single dropped file) that has a CHGCAR but no POSCAR still knows
    exactly what structure the field belongs to.

    Bounded like :func:`peek_volumetric_grid` and for the same reason: this
    runs during the cheap discovery scan, and reading a multi-gigabyte
    density in full to answer "which atoms?" would make opening a folder feel
    broken. Stopping at the blank line reads a few kilobytes instead.
    """
    header: list[str] = []
    try:
        with Path(path).open("r", errors="replace") as handle:
            # Far more lines than any real structure block needs, but finite:
            # a truncated or non-volumetric file must not read forever.
            for _ in range(1_000_000):
                line = handle.readline()
                if not line or not line.strip():
                    break
                header.append(line.rstrip("\n"))
    except OSError as exc:
        log.warning("Could not peek structure at %s: %s", path, exc)
        return None

    if len(header) < 7:
        return None
    result = parse_poscar_from_lines(header, source=Path(path).name)
    return result.value if result.ok else None


def peek_volumetric_grid(path: Path) -> tuple[int, int, int] | None:
    """Read only the grid dimensions, for discovery-time capability checks.

    A CHGCAR can be hundreds of megabytes; reading one to decide whether a
    page should be enabled would defeat the point of a cheap directory scan.
    """
    try:
        with Path(path).open("r", errors="replace") as handle:
            seen_blank = False
            for _ in range(4096):
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    seen_blank = True
                    continue
                if seen_blank:
                    match = _RE_GRID.match(line)
                    if match:
                        return (
                            int(match.group(1)),
                            int(match.group(2)),
                            int(match.group(3)),
                        )
    except OSError as exc:
        log.warning("Could not peek volumetric grid at %s: %s", path, exc)
    return None
