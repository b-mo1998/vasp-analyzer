"""Symmetry analysis via spglib, reached through pymatgen.

Imported lazily: pymatgen costs seconds to import and the answer is only needed
when the Structure page is opened. Failure is never fatal — an unanalysable
structure simply has no space group, and the UI shows an em dash.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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


@dataclass(frozen=True)
class SymmetryEquivalence:
    """Which atoms a space group maps onto one another.

    ``representative[i]`` is the index of the atom chosen to stand for the
    orbit atom ``i`` belongs to; two atoms are symmetry-equivalent exactly
    when their representatives match. ``wyckoff[i]`` is the Wyckoff letter
    and ``site_symmetry[i]`` the point-group symbol of the site.
    """

    representative: np.ndarray       # (n_atoms,) int
    wyckoff: list[str]
    site_symmetry: list[str]

    def equivalents_of(self, index: int) -> list[int]:
        """Every atom in the same orbit as ``index``, including itself."""
        if index < 0 or index >= len(self.representative):
            return []
        target = self.representative[index]
        return [int(i) for i in np.nonzero(self.representative == target)[0]]

    def multiplicity(self, index: int) -> int:
        return len(self.equivalents_of(index))

    def wyckoff_label(self, index: int) -> str:
        """The Wyckoff letter alone -- not the familiar "8a" form.

        The multiplicity in "8a" is defined for the *conventional* cell, and
        spglib reports the letter relative to that cell. The orbit size we
        can count here is the one in whatever cell was opened, which for a
        3x3x3 supercell of diamond silicon is 216, not 8: prefixing it would
        produce a "216a" that looks like a Wyckoff label and is not one. The
        orbit size is reported on its own, as what it is.
        """
        if index < 0 or index >= len(self.wyckoff):
            return "—"
        return self.wyckoff[index]


def equivalence(structure: StructureData) -> SymmetryEquivalence | None:
    """The orbit structure spglib already computes and the space-group lookup
    throws away. Never fatal: an unanalysable structure just has no
    equivalence information."""
    try:
        from pymatgen.core import Lattice, Structure  # noqa: PLC0415
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # noqa: PLC0415

        pmg = Structure(
            Lattice(structure.lattice),
            structure.species,
            structure.frac_coords,
            coords_are_cartesian=False,
        )
        dataset = SpacegroupAnalyzer(pmg, symprec=SYMPREC).get_symmetry_dataset()
        if dataset is None:
            return None

        # spglib returns a dataclass in recent versions and a dict in older
        # ones; read both the same way.
        def field(name):
            return dataset[name] if isinstance(dataset, dict) else getattr(dataset, name)

        representative = np.asarray(field("equivalent_atoms"), dtype=int)
        wyckoff = [str(w) for w in field("wyckoffs")]
        site_symmetry = [str(s) for s in field("site_symmetry_symbols")]
        if len(representative) != structure.n_atoms:
            return None
        return SymmetryEquivalence(representative, wyckoff, site_symmetry)
    except Exception as exc:  # noqa: BLE001 - optional analysis, never fatal
        log.warning("Equivalence analysis failed: %s: %s", type(exc).__name__, exc)
        return None


def space_group_label(structure: StructureData) -> str:
    if structure.space_group_symbol is None or structure.space_group_number is None:
        return "—"
    return f"{structure.space_group_symbol} ({structure.space_group_number})"
