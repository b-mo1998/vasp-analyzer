"""Charge-density localization: restricting a field to the neighbourhood of
chosen atoms via a periodic minimum-image distance mask."""

from __future__ import annotations

import numpy as np

from vasp_analyzer.models.structure import StructureData
from vasp_analyzer.models.volumetric import FieldKind, VolumetricField
from vasp_analyzer.services.volumetric import localize_to_atoms


def _cubic_field(n: int = 20, cell: float = 10.0) -> VolumetricField:
    """A 10 Å cubic cell, uniform density 1.0 everywhere, two atoms at
    opposite corners of the cell (0,0,0) and the cell centre (0.5,0.5,0.5) —
    exactly 5 Å apart in a straight line, and each exactly 5*sqrt(3)/... Å
    from cell corners via periodic images, chosen so the geometry is easy to
    reason about by hand."""
    structure = StructureData(
        lattice=np.eye(3) * cell,
        species=["Fe", "O"],
        frac_coords=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
    )
    values = np.ones((n, n, n))
    return VolumetricField(structure=structure, values=values, kind=FieldKind.CHARGE)


def test_no_atoms_selected_returns_the_field_unchanged():
    field = _cubic_field()
    out = localize_to_atoms(field, set(), radius=2.0)
    assert out is field


def test_values_far_from_every_selected_atom_are_zeroed():
    field = _cubic_field(n=20, cell=10.0)
    out = localize_to_atoms(field, {0}, radius=2.0)  # around Fe at the origin

    # Grid point at the cell centre (fractional 0.5,0.5,0.5) is 5*sqrt(3) Å
    # from the origin atom under minimum image — well outside a 2 Å radius.
    centre_index = 10  # n=20 -> index 10 sits at fractional 0.5
    assert out.values[centre_index, centre_index, centre_index] == 0.0

    # The grid point coinciding with the atom itself must survive.
    assert out.values[0, 0, 0] == 1.0


def test_periodic_image_across_the_boundary_still_counts_as_near():
    """An atom at fractional 0 is also right next to fractional ~0.98 via the
    periodic image on the other side — minimum-image distance, not naive
    displacement, must be what decides the mask."""
    field = _cubic_field(n=20, cell=10.0)
    out = localize_to_atoms(field, {0}, radius=1.5)

    near_boundary_index = 19  # fractional 19/20 = 0.95 -> 0.5 A from the origin
    assert out.values[near_boundary_index, 0, 0] == 1.0


def test_two_selected_atoms_each_contribute_their_own_neighbourhood():
    field = _cubic_field(n=20, cell=10.0)
    out_one = localize_to_atoms(field, {0}, radius=1.0)
    out_both = localize_to_atoms(field, {0, 1}, radius=1.0)

    centre_index = 10
    # Within 1 A of the second atom (fractional 0.5,0.5,0.5) but nowhere near
    # the first: zeroed when only atom 0 is selected, kept when both are.
    assert out_one.values[centre_index, centre_index, centre_index] == 0.0
    assert out_both.values[centre_index, centre_index, centre_index] == 1.0


def test_out_of_range_atom_indices_are_ignored_not_an_error():
    field = _cubic_field()
    out = localize_to_atoms(field, {0, 99}, radius=2.0)
    assert out.values[0, 0, 0] == 1.0


def test_localized_field_keeps_the_same_structure_and_kind():
    field = _cubic_field()
    out = localize_to_atoms(field, {0}, radius=2.0)
    assert out.structure is field.structure
    assert out.kind == field.kind
    assert out.shape == field.shape
