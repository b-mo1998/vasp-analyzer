"""The editable structure model and its undo stack.

These are the invariants the Editor page depends on to stay trustworthy:
identity survives reordering, a coordinate-system switch is not a move, and a
round trip through POSCAR loses nothing — including the per-axis selective
dynamics flags, which are the part most easily dropped silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from vasp_analyzer.models.editable import EditableStructure
from vasp_analyzer.models.structure import StructureData
from vasp_analyzer.parsers.poscar import parse_poscar
from vasp_analyzer.services.editing import CommandStack, EditCommand
from vasp_analyzer.services.supercell import build_supercell
from vasp_analyzer.ui.pages.structure.export import save_poscar


def _structure(species=("Si", "Si", "O"), lattice=None) -> StructureData:
    n = len(species)
    coords = np.array([[0.1 * i, 0.2 * i, 0.3 * i] for i in range(n)])
    return StructureData(
        lattice=np.eye(3) * 5.0 if lattice is None else lattice,
        species=list(species),
        frac_coords=coords,
    )


def _editable(**kwargs) -> EditableStructure:
    return EditableStructure.from_structure(_structure(**kwargs))


# --- identity -----------------------------------------------------------------


def test_ids_are_unique_and_survive_deletion():
    structure = _editable(species=("Si", "O", "Si", "O"))
    original = list(structure.ids)
    assert len(set(original)) == 4

    after = structure.with_atoms_removed([1])
    assert after.ids == [original[0], original[2], original[3]]
    # The atom that was at index 2 is now at index 1, but it is the same atom.
    assert after.index_of(original[2]) == 1


def test_sorting_moves_rows_but_not_identity():
    structure = _editable(species=("Si", "O", "Si"))
    sorted_structure = structure.sorted_by_element()

    assert sorted_structure.species == ["Si", "Si", "O"]
    assert sorted_structure.ids == [structure.ids[0], structure.ids[2], structure.ids[1]]
    # Each id still points at the coordinates it always had.
    for atom_id in structure.ids:
        before = structure.frac_coords[structure.index_of(atom_id)]
        after = sorted_structure.frac_coords[sorted_structure.index_of(atom_id)]
        np.testing.assert_allclose(before, after)


def test_duplicated_atoms_get_fresh_ids():
    structure = _editable()
    duplicated, new_ids = structure.with_atoms_duplicated([0, 1])

    assert len(new_ids) == 2
    assert not set(new_ids) & set(structure.ids)
    assert duplicated.n_atoms == structure.n_atoms + 2


def test_added_atom_never_reuses_a_deleted_id():
    structure = _editable()
    removed = structure.with_atoms_removed([0, 1, 2])
    grown, atom_id = removed.with_atom_added("H", [0.5, 0.5, 0.5])

    assert grown.n_atoms == 1
    assert atom_id not in structure.ids


def test_mutations_do_not_touch_the_original():
    """Undo restores a previous object, so mutating helpers must not alias."""
    structure = _editable()
    before = structure.frac_coords.copy()
    structure.translated([0], np.array([0.5, 0.5, 0.5]))
    structure.with_atoms_removed([0])
    structure.with_selective([0], np.array([[False, False, False]]))

    np.testing.assert_allclose(structure.frac_coords, before)
    assert structure.n_atoms == 3


# --- coordinates ---------------------------------------------------------------


def test_cartesian_round_trip_is_exact_for_a_triclinic_cell():
    lattice = np.array([[4.0, 0.0, 0.0], [1.3, 3.7, 0.0], [0.9, 1.1, 5.2]])
    structure = _editable(lattice=lattice)

    cart = structure.cart_coords
    back = structure.to_fractional(cart)
    np.testing.assert_allclose(back, structure.frac_coords, atol=1e-12)


def test_cartesian_uses_the_full_lattice_not_just_its_diagonal():
    # A non-orthogonal cell: assuming a diagonal lattice would drop the
    # off-diagonal contribution and put the atom in the wrong place.
    lattice = np.array([[4.0, 0.0, 0.0], [2.0, 3.0, 0.0], [0.0, 0.0, 5.0]])
    structure = EditableStructure.from_structure(
        StructureData(
            lattice=lattice, species=["Si"], frac_coords=np.array([[0.5, 0.5, 0.0]])
        )
    )
    np.testing.assert_allclose(structure.cart_coords[0], [3.0, 1.5, 0.0])


def test_minimum_image_distance_wraps_across_the_boundary():
    structure = EditableStructure.from_structure(
        StructureData(
            lattice=np.eye(3) * 10.0,
            species=["Si", "Si"],
            frac_coords=np.array([[0.05, 0.0, 0.0], [0.95, 0.0, 0.0]]),
        )
    )
    # Naively 9 Å apart; across the periodic boundary they are 1 Å apart.
    assert structure.minimum_image_distance(0, 1) == pytest.approx(1.0)


def test_close_pairs_finds_overlaps_across_the_boundary():
    structure = EditableStructure.from_structure(
        StructureData(
            lattice=np.eye(3) * 10.0,
            species=["Si", "Si"],
            frac_coords=np.array([[0.02, 0.0, 0.0], [0.98, 0.0, 0.0]]),
        )
    )
    pairs = structure.close_pairs(1.0)
    assert len(pairs) == 1
    assert pairs[0][2] == pytest.approx(0.4)


# --- operations ------------------------------------------------------------------


def test_wrapping_brings_every_atom_into_the_cell():
    structure = _editable()
    strayed = structure.translated([0], np.array([1.7, -2.3, 0.0]))
    wrapped = strayed.wrapped()

    assert ((wrapped.frac_coords >= 0) & (wrapped.frac_coords < 1)).all()


def test_centring_preserves_every_interatomic_distance():
    structure = _editable()
    before = [structure.minimum_image_distance(0, j) for j in (1, 2)]
    centred = structure.centred()
    after = [centred.minimum_image_distance(0, j) for j in (1, 2)]

    np.testing.assert_allclose(before, after, atol=1e-12)
    np.testing.assert_allclose(centred.frac_coords.mean(axis=0), [0.5, 0.5, 0.5])


def test_is_element_grouped_detects_interleaving():
    assert _editable(species=("Si", "Si", "O")).is_element_grouped()
    assert not _editable(species=("Si", "O", "Si")).is_element_grouped()


def test_supercell_preserves_selective_dynamics():
    flags = np.array([[True, True, True], [False, False, False], [True, False, True]])
    structure = _structure()
    structure.selective_dynamics = flags

    tiled = build_supercell(structure, 2, 1, 1)

    assert tiled.selective_dynamics is not None
    assert len(tiled.selective_dynamics) == 6
    # Each copy of an atom inherits that atom's constraints.
    np.testing.assert_array_equal(tiled.selective_dynamics[:3], flags)
    np.testing.assert_array_equal(tiled.selective_dynamics[3:], flags)


# --- POSCAR round trip -----------------------------------------------------------


def test_interleaved_species_survive_a_poscar_round_trip(tmp_path):
    """Regression: POSCAR counts describe runs of one element, so writing
    interleaved species without reordering relabels atoms — an O silently
    became an Si and vice versa."""
    structure = _structure(species=("Si", "O", "Si"))
    save_poscar(structure, tmp_path / "POSCAR")
    reloaded = parse_poscar(tmp_path / "POSCAR").value

    def pairs(s):
        return sorted(zip(s.species, (tuple(np.round(c, 10)) for c in s.frac_coords)))

    assert pairs(reloaded) == pairs(structure)


def test_selective_dynamics_survive_a_poscar_round_trip(tmp_path):
    flags = np.array([[True, True, True], [False, False, False], [True, False, True]])
    structure = _structure()
    structure.selective_dynamics = flags

    save_poscar(structure, tmp_path / "POSCAR")
    reloaded = parse_poscar(tmp_path / "POSCAR").value

    assert reloaded.selective_dynamics is not None
    np.testing.assert_array_equal(
        np.asarray(reloaded.selective_dynamics, dtype=bool), flags
    )


def test_no_selective_dynamics_block_when_none_was_set(tmp_path):
    """Writing the block for a structure that never had one would impose
    constraints the user did not ask for."""
    save_poscar(_structure(), tmp_path / "POSCAR")
    assert "elective" not in (tmp_path / "POSCAR").read_text()


def test_editable_round_trip_through_structure_data():
    structure = _editable()
    structure = structure.with_selective([0], np.array([[False, True, False]]))

    back = EditableStructure.from_structure(structure.to_structure())

    assert back.species == structure.species
    np.testing.assert_allclose(back.frac_coords, structure.frac_coords)
    np.testing.assert_array_equal(back.selective, structure.selective)


# --- undo stack --------------------------------------------------------------------


def _command(label, before, after, selection=frozenset()):
    return EditCommand(
        label=label,
        before=before,
        after=after,
        selection_before=set(),
        selection_after=set(selection),
    )


def test_undo_and_redo_walk_the_history():
    a, b, c = _editable(), _editable(species=("Si",)), _editable(species=("O",))
    stack = CommandStack()
    stack.push(_command("one", a, b))
    stack.push(_command("two", b, c))

    assert stack.can_undo and not stack.can_redo
    assert stack.undo().before is b
    assert stack.undo().before is a
    assert not stack.can_undo
    assert stack.redo().after is b


def test_a_new_change_discards_the_redo_branch():
    a, b, c = _editable(), _editable(species=("Si",)), _editable(species=("O",))
    stack = CommandStack()
    stack.push(_command("one", a, b))
    stack.undo()
    assert stack.can_redo

    stack.push(_command("other", a, c))
    assert not stack.can_redo


def test_history_is_depth_limited():
    stack = CommandStack(limit=3)
    structure = _editable()
    for index in range(6):
        stack.push(_command(f"edit {index}", structure, structure))

    assert len(stack._done) == 3
    assert stack.undo_label == "edit 5"


# --- nearest-neighbour geometry (the Editor's move-gizmo bond handles) ------------


def _graphene() -> EditableStructure:
    """A real, physically exact graphene primitive cell: 2 atoms, a-b = 2.46 Å
    hexagonal, bond length exactly 2.46/sqrt(3) Å. Each atom has three real
    C-C bonds, all to periodic copies of the *other* basis atom."""
    a = 2.46
    lattice = np.array(
        [[a, 0.0, 0.0], [-a / 2, a * np.sqrt(3) / 2, 0.0], [0.0, 0.0, 20.0]]
    )
    coords = np.array([[0.0, 0.0, 0.5], [1 / 3, 2 / 3, 0.5]])
    return EditableStructure.from_structure(
        StructureData(lattice=lattice, species=["C", "C"], frac_coords=coords)
    )


def test_nearest_neighbors_finds_all_three_bonds_in_a_two_atom_cell():
    """Regression: a plain minimum-image search can only ever return the one
    closest periodic copy of each *atom*, so it collapses graphene's three
    physical C-C bonds (all to the same other basis atom, via three different
    periodic images) down to one. Every one of the 27 neighbouring images has
    to be a candidate, not just the nearest per atom row."""
    structure = _graphene()
    neighbors = structure.nearest_neighbors(0, k=3)

    assert len(neighbors) == 3
    assert all(row == 1 for row, _direction, _distance in neighbors)
    bond_length = 2.46 / np.sqrt(3)
    for _row, _direction, distance in neighbors:
        assert distance == pytest.approx(bond_length, rel=1e-6)


def test_nearest_neighbor_directions_are_120_degrees_apart():
    structure = _graphene()
    directions = np.array([d for _r, d, _dist in structure.nearest_neighbors(0, k=3)])

    for i in range(3):
        assert np.linalg.norm(directions[i]) == pytest.approx(1.0)
    for i in range(3):
        for j in range(i + 1, 3):
            cosine = float(np.dot(directions[i], directions[j]))
            angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
            assert angle == pytest.approx(120.0, abs=0.5)


def test_nearest_neighbors_excludes_the_atom_itself():
    structure = _editable(species=("Si", "Si"))
    for row, _direction, distance in structure.nearest_neighbors(0, k=5):
        assert row != 0
        assert distance > 1e-6


def test_nearest_neighbors_is_empty_for_a_single_atom():
    structure = _editable(species=("Si",))
    assert structure.nearest_neighbors(0, k=3) == []


def test_direction_to_matches_nearest_neighbor_for_the_closest_pair():
    structure = _graphene()
    direction, distance = structure.direction_to(0, 1)
    nearest = structure.nearest_neighbors(0, k=1)[0]

    np.testing.assert_allclose(direction, nearest[1], atol=1e-9)
    assert distance == pytest.approx(nearest[2])


def test_direction_to_uses_the_shortest_periodic_path():
    """Two atoms placed near opposite faces of the cell are, physically,
    close neighbours across the boundary — the naive (non-periodic) cartesian
    difference would report them as being almost a full lattice vector apart."""
    structure = EditableStructure.from_structure(
        StructureData(
            lattice=np.eye(3) * 10.0,
            species=["Si", "Si"],
            frac_coords=np.array([[0.02, 0.0, 0.0], [0.98, 0.0, 0.0]]),
        )
    )
    direction, distance = structure.direction_to(0, 1)
    assert distance == pytest.approx(0.4, abs=1e-6)
    # Atom 1's nearest image sits in the -x direction from atom 0.
    assert direction[0] < 0


def test_direction_to_atom_itself_is_degenerate_not_a_crash():
    structure = _editable()
    direction, distance = structure.direction_to(0, 0)
    assert distance == 0.0
    np.testing.assert_allclose(direction, np.zeros(3))


def test_tetrahedral_silicon_has_four_equal_bonds():
    """Diamond-cubic silicon: a real, non-planar coordination check — each
    atom has four neighbours at the canonical (±1,±1,±1)/√3 tetrahedral
    angle, not three coplanar ones."""
    a = 5.43
    lattice = np.eye(3) * a
    coords = np.array(
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
        + [[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]
        + [[0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]]
    )
    structure = EditableStructure.from_structure(
        StructureData(lattice=lattice, species=["Si"] * 8, frac_coords=coords)
    )
    neighbors = structure.nearest_neighbors(0, k=4)
    distances = [d for _r, _dir, d in neighbors]
    assert max(distances) - min(distances) < 1e-6

    directions = np.array([d for _r, d, _dist in neighbors])
    # Tetrahedral angle is arccos(-1/3) ≈ 109.47°.
    for i in range(4):
        for j in range(i + 1, 4):
            cosine = float(np.dot(directions[i], directions[j]))
            angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
            assert angle == pytest.approx(109.47, abs=0.5)
