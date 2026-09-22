"""Charge-difference maps: subtracting one volumetric field from another to
see where density was gained or lost -- e.g. before vs. after an adsorption
event on the same slab."""

from __future__ import annotations

import numpy as np
import pytest

from vasp_analyzer.models.structure import StructureData
from vasp_analyzer.models.volumetric import FieldKind, VolumetricField
from vasp_analyzer.services.volumetric import colormap_for, difference


def _field(values: np.ndarray, kind: FieldKind = FieldKind.CHARGE) -> VolumetricField:
    structure = StructureData(
        lattice=np.eye(3) * 10.0,
        species=["Fe"],
        frac_coords=np.array([[0.0, 0.0, 0.0]]),
    )
    return VolumetricField(structure=structure, values=values, kind=kind, source_file="CHGCAR")


def test_difference_is_pointwise_subtraction():
    a = _field(np.full((4, 4, 4), 3.0))
    b = _field(np.full((4, 4, 4), 1.0))

    result = difference(a, b)

    assert np.allclose(result.values, 2.0)


def test_difference_field_is_kind_difference_and_signed():
    a = _field(np.ones((4, 4, 4)))
    b = _field(np.ones((4, 4, 4)))

    result = difference(a, b)

    assert result.kind is FieldKind.DIFFERENCE
    assert result.kind.is_signed
    assert colormap_for(result) == "RdBu_r"


def test_difference_does_not_integrate_to_electrons():
    a = _field(np.ones((4, 4, 4)))
    b = _field(np.ones((4, 4, 4)))

    result = difference(a, b)

    assert not result.kind.integrates_to_electrons


def test_difference_keeps_field_as_structure():
    a = _field(np.ones((4, 4, 4)))
    b = _field(np.full((4, 4, 4), 2.0))

    result = difference(a, b)

    assert result.structure is a.structure


def test_mismatched_grid_shapes_raise_a_clear_error():
    a = _field(np.ones((4, 4, 4)))
    b = _field(np.ones((5, 5, 5)))

    with pytest.raises(ValueError, match=r"4×4×4.*5×5×5"):
        difference(a, b)


def test_subtracting_a_field_from_itself_is_exactly_zero():
    a = _field(np.random.default_rng(0).random((6, 6, 6)))

    result = difference(a, a)

    assert np.all(result.values == 0.0)
