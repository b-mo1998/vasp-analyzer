"""Writing a StructureData back out as a real structure file.

Kept as pure functions on StructureData — no Qt — mirroring how the Bands
page's export.py separates file writing from the QFileDialog plumbing in
page.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ....models.structure import StructureData

CIF_TEMPLATE = """data_{formula}
_cell_length_a    {a:.6f}
_cell_length_b    {b:.6f}
_cell_length_c    {c:.6f}
_cell_angle_alpha {alpha:.4f}
_cell_angle_beta  {beta:.4f}
_cell_angle_gamma {gamma:.4f}
_symmetry_space_group_name_H-M   '{space_group}'
_symmetry_Int_Tables_number      {space_group_number}
_chemical_formula_sum '{full_formula}'

loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
{atom_rows}
"""


def save_cif(structure: StructureData, path: Path) -> None:
    """Write a minimal, always-valid P1-style CIF: every atom listed
    explicitly rather than symmetry-reduced, so it opens correctly in any CIF
    reader whether or not this structure's symmetry was ever determined."""
    counts: dict[str, int] = {}
    rows = []
    for species, frac in zip(structure.species, structure.frac_coords):
        counts[species] = counts.get(species, 0) + 1
        label = f"{species}{counts[species]}"
        rows.append(f" {label:<6s}{species:<4s}{frac[0]:11.6f}{frac[1]:11.6f}{frac[2]:11.6f}")

    alpha, beta, gamma = structure.lattice_angles
    a, b, c = structure.lattice_parameters
    text = CIF_TEMPLATE.format(
        formula=structure.formula or "structure",
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        space_group=structure.space_group_symbol or "P 1",
        space_group_number=structure.space_group_number or 1,
        full_formula=structure.full_formula,
        atom_rows="\n".join(rows),
    )
    Path(path).write_text(text)


def save_poscar(structure: StructureData, path: Path) -> None:
    """Write VASP POSCAR/CONTCAR format — the same layout parsers/poscar.py
    reads, so a round trip through this exporter and back is lossless.

    Atoms are written grouped by element. POSCAR's counts line describes
    *runs* of one element, so interleaved species have no representation in
    the format: writing "Si O / 2 1" while the coordinates run Si, O, Si
    silently relabels the oxygen as silicon and vice versa. Reordering the
    rows to match the counts is the only way to keep the file honest, and it
    is what VASP itself expects.
    """
    # First-appearance element order, then every atom of each element in its
    # original relative order.
    elements: list[str] = []
    for symbol in structure.species:
        if symbol not in elements:
            elements.append(symbol)
    order: list[int] = []
    for element in elements:
        order.extend(i for i, s in enumerate(structure.species) if s == element)
    counts = {e: sum(1 for s in structure.species if s == e) for e in elements}

    flags = structure.selective_dynamics
    if flags is not None:
        flags = np.asarray(flags, dtype=bool)
        if flags.ndim == 1:
            flags = np.repeat(flags[:, None], 3, axis=1)

    # Two explicit spaces between every field, never relying on fixed-width
    # padding alone to separate columns: a value like -0.0000000000000000 is
    # exactly as wide as a 19-character field, leaving no padding at all, so
    # two such values back to back glue together into one unparsable token.
    lines = [
        structure.comment or structure.formula or "structure",
        "   1.0000000000000000",
    ]
    for vector in structure.lattice:
        lines.append(
            "  " + "  ".join(f"{component:20.16f}" for component in vector)
        )
    lines.append("   " + "   ".join(elements))
    lines.append("   " + "   ".join(str(counts[e]) for e in elements))
    if flags is not None:
        lines.append("Selective dynamics")
    lines.append("Direct")
    for row in order:
        frac = structure.frac_coords[row]
        line = "  " + "  ".join(f"{component:19.16f}" for component in frac)
        if flags is not None:
            line += "  " + "  ".join("T" if free else "F" for free in flags[row])
        lines.append(line)

    Path(path).write_text("\n".join(lines) + "\n")
