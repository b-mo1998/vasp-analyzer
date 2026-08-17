"""Symmetry analysis via spglib, reached through pymatgen.

Imported lazily: pymatgen costs seconds to import and the answer is only needed
when the Structure page is opened. Failure is never fatal — an unanalysable
structure simply has no space group, and the UI shows an em dash.
"""

from __future__ import annotations

from ..core.logging_setup import get_logger
from ..models.structure import StructureData

log = get_logger("services.symmetry")

SYMPREC = 1e-3


def analyse(structure: StructureData) -> bool:
    """Fill in space group and crystal system. Returns True on success."""
    if structure.space_group_number is not None:
        return True
    try:
        from pymatgen.core import Lattice, Structure  # noqa: PLC0415
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # noqa: PLC0415

        pmg = Structure(
            Lattice(structure.lattice),
            structure.species,
            structure.frac_coords,
            coords_are_cartesian=False,
        )
        analyzer = SpacegroupAnalyzer(pmg, symprec=SYMPREC)
        structure.space_group_symbol = analyzer.get_space_group_symbol()
        structure.space_group_number = analyzer.get_space_group_number()
        structure.crystal_system = analyzer.get_crystal_system().capitalize()
        log.info(
            "Symmetry: %s (%s), %s",
            structure.space_group_symbol,
            structure.space_group_number,
            structure.crystal_system,
        )
        return True
    except Exception as exc:  # noqa: BLE001 - optional analysis, never fatal
        log.warning("Symmetry analysis failed: %s: %s", type(exc).__name__, exc)
        return False


def space_group_label(structure: StructureData) -> str:
    if structure.space_group_symbol is None or structure.space_group_number is None:
        return "—"
    return f"{structure.space_group_symbol} ({structure.space_group_number})"
