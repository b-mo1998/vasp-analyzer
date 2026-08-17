"""POSCAR / CONTCAR parser.

Written by hand rather than delegating to pymatgen: the format is small and
regular, this keeps startup fast, and it lets us degrade gracefully on VASP 4
files that omit the species line (pymatgen raises there). Symmetry analysis —
the part that genuinely needs spglib — is delegated in
:mod:`vasp_analyzer.services.symmetry`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.errors import Issue, ParseResult, Severity, safe_parser
from ..core.logging_setup import get_logger
from ..models.structure import StructureData

log = get_logger("parsers.poscar")


def _is_int_list(tokens: list[str]) -> bool:
    if not tokens:
        return False
    for token in tokens:
        try:
            int(token)
        except ValueError:
            return False
    return True


@safe_parser("POSCAR")
def parse_poscar(path: Path, source: str | None = None) -> ParseResult[StructureData]:
    path = Path(path)
    label = source or path.name
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) < 7:
        return ParseResult.failure(
            "poscar_truncated",
            f"{label} is too short to describe a structure.",
            source=label,
        )

    issues: list[Issue] = []
    comment = lines[0].strip()

    scale = float(lines[1].split()[0])
    vectors = np.array(
        [[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)], dtype=float
    )
    if scale < 0:
        # A negative scale is a target volume in Å³.
        current = abs(np.linalg.det(vectors))
        scale = (abs(scale) / current) ** (1.0 / 3.0) if current > 0 else 1.0
    lattice = vectors * scale

    species_tokens = lines[5].split()
    if _is_int_list(species_tokens):
        # VASP 4 style: no species line, counts come first.
        counts = [int(t) for t in species_tokens]
        names = [f"El{i + 1}" for i in range(len(counts))]
        cursor = 6
        issues.append(
            Issue(
                code="poscar_no_species",
                message=(
                    f"{label} uses the VASP 4 format and does not name its elements."
                ),
                detail="Element labels are placeholders; species come from POTCAR.",
                severity=Severity.WARNING,
                source=label,
            )
        )
    else:
        names = species_tokens
        counts = [int(float(t)) for t in lines[6].split()]
        cursor = 7

    if len(names) != len(counts):
        return ParseResult.failure(
            "poscar_species_mismatch",
            f"{label} lists {len(names)} elements but {len(counts)} atom counts.",
            source=label,
        )

    species: list[str] = []
    for name, count in zip(names, counts):
        species.extend([name] * count)
    n_atoms = len(species)
    if n_atoms == 0:
        return ParseResult.failure(
            "poscar_no_atoms", f"{label} contains no atoms.", source=label
        )

    selective = False
    if cursor < len(lines) and lines[cursor].strip()[:1].upper() == "S":
        selective = True
        cursor += 1

    if cursor >= len(lines):
        return ParseResult.failure(
            "poscar_no_coordinates",
            f"{label} declares atoms but contains no coordinates.",
            source=label,
        )

    mode = lines[cursor].strip()[:1].upper()
    cartesian = mode in ("C", "K")
    cursor += 1

    coords = np.zeros((n_atoms, 3), dtype=float)
    flags = np.ones((n_atoms, 3), dtype=bool) if selective else None
    read = 0
    for raw in lines[cursor : cursor + n_atoms]:
        tokens = raw.split()
        if len(tokens) < 3:
            break
        coords[read] = [float(t) for t in tokens[:3]]
        if flags is not None and len(tokens) >= 6:
            flags[read] = [t.upper().startswith("T") for t in tokens[3:6]]
        read += 1

    if read < n_atoms:
        return ParseResult.failure(
            "poscar_truncated_coordinates",
            f"{label} lists {n_atoms} atoms but only {read} coordinate rows.",
            source=label,
        )

    if cartesian:
        coords = coords * scale @ np.linalg.inv(lattice)

    log.info("Parsed %s: %d atoms, %s", label, n_atoms, "".join(names))

    structure = StructureData(
        lattice=lattice,
        species=species,
        frac_coords=coords,
        comment=comment,
        source=label,
        selective_dynamics=flags,
    )
    return ParseResult.success(structure, issues)
