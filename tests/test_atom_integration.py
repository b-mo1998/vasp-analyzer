"""Per-atom charge integration: a Bader-style approximation built from the
same per-atom sphere mask ``localize_to_atoms`` already draws, applied one
atom at a time and integrated rather than shown as a single combined mask.
"""

from __future__ import annotations

import numpy as np
import pytest

from vasp_analyzer.models.structure import StructureData
from vasp_analyzer.models.volumetric import FieldKind, VolumetricField
from vasp_analyzer.services.volumetric import integrate_per_atom


def _cubic_field(n: int = 20, cell: float = 10.0) -> VolumetricField:
    """Two atoms 5 A apart in a 10 A cubic cell, uniform density 1.0
    everywhere -- same fixture shape as test_localize_to_atoms.py, so the
    geometry is easy to reason about by hand."""
    structure = StructureData(
        lattice=np.eye(3) * cell,
        species=["Fe", "O"],
        frac_coords=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
    )
    values = np.ones((n, n, n))
    return VolumetricField(structure=structure, values=values, kind=FieldKind.CHARGE)


def test_no_atoms_selected_returns_nothing():
    field = _cubic_field()
    assert integrate_per_atom(field, [], radius=2.0) == []


def test_one_result_per_selected_atom_with_symbol_and_row():
    field = _cubic_field()
    results = integrate_per_atom(field, [0, 1], radius=2.0)

    assert [r.row for r in results] == [0, 1]
    assert [r.symbol for r in results] == ["Fe", "O"]


def test_integral_matches_uniform_density_times_sphere_volume():
    """A uniform density of 1.0 e/A^3 integrated over a sphere of radius r
    should equal (4/3)*pi*r^3, up to the grid's own discretisation error."""
    field = _cubic_field(n=60, cell=10.0)  # fine grid for a tight tolerance
    radius = 1.5
    [result] = integrate_per_atom(field, [0], radius=radius)

    expected = (4.0 / 3.0) * np.pi * radius**3
    assert result.value == pytest.approx(expected, rel=0.05)


def test_out_of_range_indices_are_ignored_not_an_error():
    field = _cubic_field()
    results = integrate_per_atom(field, [0, 99], radius=2.0)
    assert [r.row for r in results] == [0]


def test_overlapping_spheres_each_report_the_shared_region():
    """Two atoms closer together than 2*radius double-count the overlap --
    the docstring's own caveat, verified rather than only stated: pulling
    the atoms close together (radius > half their separation) makes each
    atom's own integral approach the full-sphere value despite the region
    between them being shared."""
    structure = StructureData(
        lattice=np.eye(3) * 10.0,
        species=["Fe", "Fe"],
        frac_coords=np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]]),  # 0.5 A apart
    )
    field = VolumetricField(
        structure=structure, values=np.ones((60, 60, 60)), kind=FieldKind.CHARGE
    )
    radius = 2.0  # >> half the 0.5 A separation
    results = integrate_per_atom(field, [0, 1], radius)

    full_sphere = (4.0 / 3.0) * np.pi * radius**3
    for result in results:
        assert result.value == pytest.approx(full_sphere, rel=0.05)


def test_difference_field_can_report_a_negative_integral():
    structure = StructureData(
        lattice=np.eye(3) * 10.0,
        species=["Fe"],
        frac_coords=np.array([[0.0, 0.0, 0.0]]),
    )
    field = VolumetricField(
        structure=structure, values=np.full((20, 20, 20), -1.0), kind=FieldKind.DIFFERENCE
    )
    [result] = integrate_per_atom(field, [0], radius=2.0)
    assert result.value < 0.0
