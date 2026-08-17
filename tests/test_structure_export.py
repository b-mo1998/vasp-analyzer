"""Structure export: CIF/POSCAR writers, and the round trip back through
parse_poscar that should always reproduce the original structure exactly.

The POSCAR writer had a real spacing bug caught here: a coordinate of exactly
-0.0000000000000000 is precisely as wide as its fixed-format field, leaving no
padding at all, so two such values written back to back glued into one token
that parse_poscar's split()-based row reader couldn't parse — silently
truncating the file it had just written.
"""

from __future__ import annotations

import numpy as np
import pytest

from vasp_analyzer.models.structure import StructureData
from vasp_analyzer.parsers.poscar import parse_poscar
from vasp_analyzer.ui.pages.structure.export import save_cif, save_poscar


def graphene() -> StructureData:
    lattice = np.array(
        [
            [2.4694910408650514, -0.0, 0.0],
            [-1.2347455204325257, 2.1386419486320141, -0.0],
            [0.0, 0.0, 14.4682012468615859],
        ]
    )
    frac = np.array([[-0.0, -0.0, 0.25], [0.6666666699999979, 0.3333333300000021, 0.25]])
    return StructureData(lattice=lattice, species=["C", "C"], frac_coords=frac, comment="graphene")


def rutile() -> StructureData:
    lattice = np.diag([4.5941, 4.5941, 2.9589])
    frac = np.array(
        [
            [0.0, 0.0, 0.0], [0.5, 0.5, 0.5],
            [0.3053, 0.3053, 0.0], [0.6947, 0.6947, 0.0],
            [0.8053, 0.1947, 0.5], [0.1947, 0.8053, 0.5],
        ]
    )
    return StructureData(
        lattice=lattice, species=["Ti", "Ti", "O", "O", "O", "O"], frac_coords=frac
    )


# --- POSCAR ------------------------------------------------------------------


def test_poscar_round_trip_is_exact_for_negative_zero_coordinates(tmp_path):
    """The specific case that broke: fractional coordinates of -0.0 sit right
    at the fixed-width field boundary."""
    structure = graphene()
    path = tmp_path / "POSCAR"
    save_poscar(structure, path)

    result = parse_poscar(path)
    assert result.ok, result.issues
    back = result.value
    assert back.n_atoms == structure.n_atoms
    assert np.allclose(back.lattice, structure.lattice)
    assert np.allclose(back.frac_coords, structure.frac_coords, atol=1e-10)
    assert back.species == structure.species


def test_poscar_fields_are_always_whitespace_separated(tmp_path):
    """Direct regression for the glued-token bug: every row, split on
    whitespace, must yield exactly as many tokens as expected — never fewer
    because two numbers ran together."""
    structure = graphene()
    path = tmp_path / "POSCAR"
    save_poscar(structure, path)

    lines = path.read_text().splitlines()
    for vector_line in lines[2:5]:
        assert len(vector_line.split()) == 3
    assert lines[7] == "Direct"
    for coord_line in lines[8:10]:
        assert len(coord_line.split()) == 3


def test_poscar_round_trip_multi_element(tmp_path):
    structure = rutile()
    path = tmp_path / "POSCAR"
    save_poscar(structure, path)

    back = parse_poscar(path).value
    assert back.species == structure.species
    assert np.allclose(back.frac_coords, structure.frac_coords)
    assert np.allclose(back.lattice, structure.lattice)


def test_poscar_groups_species_into_element_blocks(tmp_path):
    structure = rutile()
    path = tmp_path / "POSCAR"
    save_poscar(structure, path)

    lines = path.read_text().splitlines()
    assert lines[5].split() == ["Ti", "O"]
    assert lines[6].split() == ["2", "4"]


# --- CIF -----------------------------------------------------------------------


def test_cif_contains_every_atom_explicitly(tmp_path):
    structure = rutile()
    path = tmp_path / "structure.cif"
    save_cif(structure, path)

    text = path.read_text()
    assert text.count("\n Ti") + text.count("\n O") >= structure.n_atoms
    assert "_chemical_formula_sum 'Ti2O4'" in text


def test_cif_falls_back_to_p1_without_symmetry_analysis(tmp_path):
    structure = rutile()  # space_group_* left as None: analyse() was never run
    path = tmp_path / "structure.cif"
    save_cif(structure, path)

    text = path.read_text()
    assert "'P 1'" in text
    assert "_symmetry_Int_Tables_number      1" in text


def test_cif_uses_determined_space_group_when_available(tmp_path):
    structure = graphene()
    structure.space_group_symbol = "P6/mmm"
    structure.space_group_number = 191
    path = tmp_path / "structure.cif"
    save_cif(structure, path)

    text = path.read_text()
    assert "P6/mmm" in text
    assert "191" in text


def test_cif_cell_parameters_match_the_structure(tmp_path):
    structure = rutile()
    path = tmp_path / "structure.cif"
    save_cif(structure, path)

    text = path.read_text()
    assert "_cell_length_a    4.594100" in text
    assert "_cell_length_c    2.958900" in text
    assert "_cell_angle_gamma 90.0000" in text
