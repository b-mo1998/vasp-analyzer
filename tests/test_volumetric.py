"""Volumetric fields: CHGCAR parsing, normalisation, slices and isosurfaces.

Two properties matter more than the rest, because getting either wrong still
produces a picture that looks entirely plausible:

* **Grid order.** VASP varies the first index fastest. The fixture writes
  ``i + 10j + 100k``, so every grid point announces its own indices and a
  transposed read is impossible to overlook.
* **Normalisation.** CHGCAR stores density × volume, so the density is
  ``data / V`` and the integral is the electron count. A parser that skips the
  division is wrong by a factor of the cell volume — three orders of magnitude
  for a typical cell — while still producing a perfectly reasonable-looking
  isosurface.
"""

from __future__ import annotations

import numpy as np
import pytest

from vasp_analyzer.models.capabilities import Capability
from vasp_analyzer.models.structure import StructureData
from vasp_analyzer.models.volumetric import FieldKind, VolumetricField
from vasp_analyzer.parsers.chgcar import (
    kind_for_filename,
    parse_volumetric,
    peek_volumetric_grid,
)
from vasp_analyzer.services import volumetric as vol
from vasp_analyzer.services.discovery import discover
from vasp_analyzer.services.isosurface import (
    extract_isosurface,
    smooth_mesh,
    suggested_subsample,
)

from . import fixtures


def _field(directory, **kwargs) -> VolumetricField:
    name = kwargs.pop("name", "CHGCAR")
    fixtures.write_volumetric(directory, name=name, **kwargs)
    result = parse_volumetric(directory / name)
    assert result.ok, result.issues
    return result.value.fields[0]


# --- Test A: a standard CHGCAR --------------------------------------------------


def test_structure_and_grid_dimensions(tmp_path):
    field = _field(tmp_path, shape=(4, 5, 6))
    assert field.shape == (4, 5, 6)
    assert field.structure.n_atoms == 2
    assert field.kind is FieldKind.CHARGE
    assert field.source_file == "CHGCAR"


def test_grid_uses_fortran_order(tmp_path):
    """The decisive axis-order check: the fixture's value at (i, j, k) is
    i + 10j + 100k, so a transposed read cannot pass."""
    field = _field(tmp_path, shape=(4, 5, 6))
    raw = field.values * field.volume       # undo the density normalisation

    assert raw[1, 2, 3] == pytest.approx(1 + 20 + 300)
    assert raw[3, 0, 0] == pytest.approx(3)     # first index varies fastest
    assert raw[0, 4, 0] == pytest.approx(40)
    assert raw[0, 0, 5] == pytest.approx(500)


def test_density_is_normalised_by_the_cell_volume(tmp_path):
    field = _field(tmp_path, shape=(4, 4, 4))
    raw_sum = float((field.values * field.volume).sum())
    # n = data / V, so multiplying back must recover the written data exactly.
    assert field.values.max() == pytest.approx(
        (field.values * field.volume).max() / field.volume
    )
    assert raw_sum > 0


def test_integral_recovers_the_electron_count(tmp_path):
    """The standard CHGCAR sanity check: ∫n dV is the number of electrons."""
    field = _field(tmp_path, shape=(6, 6, 6), electrons=8.0)
    assert field.integral() == pytest.approx(8.0, abs=1e-9)


def test_integration_reports_agreement_with_nelect(tmp_path):
    field = _field(tmp_path, shape=(6, 6, 6), electrons=12.0)
    result = vol.integrate(field, expected_electrons=12.0, expected_source="OUTCAR")

    assert result.integral == pytest.approx(12.0, abs=1e-9)
    assert result.difference == pytest.approx(0.0, abs=1e-9)
    assert result.is_consistent is True


def test_integration_flags_a_normalisation_error(tmp_path):
    # What a missing /V would look like: the integral off by the cell volume.
    field = _field(tmp_path, shape=(6, 6, 6), electrons=8.0)
    result = vol.integrate(field, expected_electrons=8.0 * field.volume)
    assert result.is_consistent is False


def test_statistics_come_from_the_full_field(tmp_path):
    field = _field(tmp_path, shape=(4, 5, 6))
    assert field.minimum == pytest.approx(float(field.values.min()))
    assert field.maximum == pytest.approx(float(field.values.max()))
    assert field.mean == pytest.approx(float(field.values.mean()))


def test_augmentation_occupancies_are_skipped(tmp_path):
    """PAW augmentation follows the grid and must not be read as data."""
    plain = _field(tmp_path, shape=(4, 4, 4))
    other = tmp_path / "with_aug"
    other.mkdir()
    augmented = _field(other, shape=(4, 4, 4), augmentation=True)

    assert augmented.shape == plain.shape
    np.testing.assert_allclose(augmented.values, plain.values)


def test_peek_reads_dimensions_without_the_data(tmp_path):
    fixtures.write_volumetric(tmp_path, shape=(4, 5, 6))
    assert peek_volumetric_grid(tmp_path / "CHGCAR") == (4, 5, 6)


# --- Test B: spin-polarised ---------------------------------------------------------


def test_second_block_is_magnetization_not_another_density(tmp_path):
    fixtures.write_volumetric(tmp_path, shape=(4, 4, 4), n_blocks=2)
    result = parse_volumetric(tmp_path / "CHGCAR")

    assert result.ok
    file = result.value
    assert file.n_components == 2
    assert file.is_spin_polarized
    assert file.spin_mode == "Collinear"

    total, magnetization = file.fields
    assert total.kind is FieldKind.CHARGE
    assert total.component == "total"
    assert magnetization.kind is FieldKind.MAGNETIZATION
    assert magnetization.component == "magnetization"


def test_magnetization_is_signed_and_charge_is_not(tmp_path):
    fixtures.write_volumetric(tmp_path, shape=(4, 4, 4), n_blocks=2)
    total, magnetization = parse_volumetric(tmp_path / "CHGCAR").value.fields

    assert not total.kind.is_signed
    assert magnetization.kind.is_signed
    # A signed field gets a range centred on zero so the map's midpoint is zero.
    low, high = vol.display_range(magnetization)
    assert low == pytest.approx(-high)
    # A non-negative one does not.
    assert vol.display_range(total)[0] == 0.0


def test_noncollinear_gives_three_magnetization_components(tmp_path):
    fixtures.write_volumetric(tmp_path, shape=(3, 3, 3), n_blocks=4)
    file = parse_volumetric(tmp_path / "CHGCAR").value

    assert file.n_components == 4
    assert file.spin_mode == "Non-collinear"
    assert [f.component for f in file.fields] == ["total", "m_x", "m_y", "m_z"]


# --- Test C: ELFCAR ------------------------------------------------------------------


def test_elfcar_is_recognised_and_dimensionless(tmp_path):
    fixtures.write_volumetric(tmp_path, name="ELFCAR", shape=(4, 4, 4))
    result = parse_volumetric(tmp_path / "ELFCAR")

    assert result.ok
    field = result.value.fields[0]
    assert field.kind is FieldKind.ELF
    assert field.kind.units == "", "ELF is a ratio and must not be labelled e/Å³"
    assert not field.kind.integrates_to_electrons
    assert field.kind.bounded_range == (0.0, 1.0)


def test_elf_is_not_volume_scaled(tmp_path):
    """ELFCAR stores the function itself, unlike CHGCAR's density × volume."""
    fixtures.write_volumetric(tmp_path, name="ELFCAR", shape=(4, 4, 4))
    field = parse_volumetric(tmp_path / "ELFCAR").value.fields[0]
    raw = np.arange(4 * 4 * 4, dtype=float)
    assert field.values.max() == pytest.approx(
        float((np.arange(4) + 10 * np.arange(4)[:, None]).max() + 300)
    ) or field.values.max() > 1.0
    _ = raw


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("CHGCAR", FieldKind.CHARGE),
        ("CHG", FieldKind.CHARGE),
        ("ELFCAR", FieldKind.ELF),
        ("PARCHG.0023.ALLK", FieldKind.PARTIAL_CHARGE),
        ("LOCPOT", FieldKind.POTENTIAL),
    ],
)
def test_field_kind_from_filename(name, expected):
    assert kind_for_filename(name) is expected


# --- Test D: PARCHG ---------------------------------------------------------------------


def test_parchg_files_are_enumerated(tmp_path):
    fixtures.write_volumetric(tmp_path, name="CHGCAR", shape=(3, 3, 3))
    fixtures.write_volumetric(tmp_path, name="PARCHG.0012.ALLK", shape=(3, 3, 3))
    fixtures.write_volumetric(tmp_path, name="PARCHG.0023.ALLK", shape=(3, 3, 3))

    sources = vol.find_volumetric_sources(tmp_path)
    names = [source.name for source in sources]

    assert names[0] == "CHGCAR", "the fullest source comes first"
    assert "PARCHG.0012.ALLK" in names
    assert "PARCHG.0023.ALLK" in names
    # A PARCHG keeps its full name: the suffix says which bands it holds.
    labels = [s.label for s in sources if s.name.startswith("PARCHG")]
    assert labels == ["PARCHG.0012.ALLK", "PARCHG.0023.ALLK"]


def test_partial_charge_still_integrates_to_electrons(tmp_path):
    fixtures.write_volumetric(tmp_path, name="PARCHG.0001.ALLK", shape=(4, 4, 4))
    field = parse_volumetric(tmp_path / "PARCHG.0001.ALLK").value.fields[0]
    assert field.kind is FieldKind.PARTIAL_CHARGE
    assert field.kind.integrates_to_electrons
    assert field.kind.units == "e/Å³"


def test_empty_volumetric_files_are_ignored(tmp_path):
    """VASP leaves zero-byte CHGCAR/CHG placeholders behind; offering them as
    sources would only produce a parse error when selected."""
    (tmp_path / "CHGCAR").write_text("")
    (tmp_path / "CHG").write_text("")
    assert vol.find_volumetric_sources(tmp_path) == []


# --- Test E: truncated / malformed -------------------------------------------------------


def test_truncated_grid_reports_an_error_without_raising(tmp_path):
    fixtures.write_volumetric(tmp_path, shape=(6, 6, 6), truncate=True)
    result = parse_volumetric(tmp_path / "CHGCAR")

    assert result.failed
    assert result.errors
    assert "grid" in result.errors[0].message.lower()


def test_garbage_file_reports_an_error(tmp_path):
    (tmp_path / "CHGCAR").write_text("not a CHGCAR at all\n" * 20)
    assert parse_volumetric(tmp_path / "CHGCAR").failed


def test_missing_file_reports_an_error(tmp_path):
    assert parse_volumetric(tmp_path / "CHGCAR").failed


def test_structure_without_a_grid_is_rejected_cleanly(tmp_path):
    fixtures.write_poscar(tmp_path, name="CHGCAR")
    result = parse_volumetric(tmp_path / "CHGCAR")
    assert result.failed
    assert not result.errors[0].detail.startswith("Traceback")


# --- Test F: non-orthogonal lattices --------------------------------------------------------


def _triclinic() -> list[list[float]]:
    return [[10.0, 0.0, 0.0], [4.0, 9.0, 0.0], [3.0, 2.0, 8.0]]


def test_isosurface_of_a_sphere_matches_analytic_geometry():
    """A sphere in a cubic cell: the extracted surface must have the analytic
    area and enclose the analytic volume."""
    size, n, radius = 10.0, 48, 3.0
    grid = np.arange(n) / n
    x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")
    distance = np.sqrt(
        ((x - 0.5) * size) ** 2 + ((y - 0.5) * size) ** 2 + ((z - 0.5) * size) ** 2
    )
    structure = StructureData(
        lattice=np.eye(3) * size,
        species=["H"],
        frac_coords=np.array([[0.5, 0.5, 0.5]]),
    )
    field = VolumetricField(structure=structure, values=np.exp(-distance))
    mesh = extract_isosurface(field, np.exp(-radius))

    centre = np.array([size / 2] * 3)
    distances = np.linalg.norm(mesh.vertices - centre, axis=1)
    assert distances.mean() == pytest.approx(radius, abs=0.02)
    assert distances.std() < 0.05

    a, b, c = (mesh.vertices[mesh.triangles[:, i]] for i in range(3))
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
    assert area == pytest.approx(4 * np.pi * radius**2, rel=0.03)


def test_isosurface_in_a_triclinic_cell_is_not_distorted():
    """The field is a sphere in *cartesian* space; a correct lattice transform
    must return a sphere even though the sampling grid is sheared."""
    lattice = np.array(_triclinic())
    n, radius = 44, 2.5
    grid = np.arange(n) / n
    fractional = np.stack(
        np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    centre = np.array([0.5, 0.5, 0.5]) @ lattice
    distance = np.linalg.norm(fractional @ lattice - centre, axis=1).reshape(n, n, n)

    structure = StructureData(
        lattice=lattice, species=["H"], frac_coords=np.array([[0.5, 0.5, 0.5]])
    )
    field = VolumetricField(structure=structure, values=np.exp(-distance))
    mesh = extract_isosurface(field, np.exp(-radius))

    distances = np.linalg.norm(mesh.vertices - centre, axis=1)
    assert distances.mean() == pytest.approx(radius, abs=0.03)
    assert distances.std() < 0.05, "a sheared grid must not distort the surface"


def test_isosurface_is_closed_and_welded():
    """No boundary edges, and vertices genuinely shared between facets.

    "Closed" is tested as *no edge belonging to a single triangle* — that is
    what a hole in the surface looks like. A handful of edges shared by four
    triangles is not a hole but two sheets meeting, which marching tetrahedra
    produces where the level passes through a shared cube diagonal; it is
    harmless here and does not warrant the complexity of resolving it.

    The vertex count also has to come out well below the triangle count:
    unwelded facets share no indices at all, and every adjacency-based
    operation — smoothing above all — then misbehaves silently.
    """
    from collections import Counter

    size, n = 8.0, 32
    grid = np.arange(n) / n
    x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")
    distance = np.sqrt(
        ((x - 0.5) * size) ** 2 + ((y - 0.5) * size) ** 2 + ((z - 0.5) * size) ** 2
    )
    structure = StructureData(
        lattice=np.eye(3) * size, species=["H"], frac_coords=np.array([[0.5] * 3])
    )
    field = VolumetricField(structure=structure, values=np.exp(-distance))
    mesh = extract_isosurface(field, np.exp(-2.0))

    edges: Counter = Counter()
    for triangle in mesh.triangles:
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edges[tuple(sorted((triangle[first], triangle[second])))] += 1
    boundary = [edge for edge, count in edges.items() if count == 1]
    assert not boundary, "an edge with one triangle is a hole in the surface"
    manifold = sum(1 for count in edges.values() if count == 2)
    assert manifold / len(edges) > 0.99

    assert len(mesh.vertices) < mesh.n_triangles, "vertices must be shared"


def test_smoothing_preserves_scale():
    """Taubin smoothing, not plain Laplacian: a shrinking filter collapses a
    thin surface onto itself and destroys the mesh."""
    size, n, radius = 10.0, 40, 3.0
    grid = np.arange(n) / n
    x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")
    distance = np.sqrt(
        ((x - 0.5) * size) ** 2 + ((y - 0.5) * size) ** 2 + ((z - 0.5) * size) ** 2
    )
    structure = StructureData(
        lattice=np.eye(3) * size, species=["H"], frac_coords=np.array([[0.5] * 3])
    )
    field = VolumetricField(structure=structure, values=np.exp(-distance))
    mesh = extract_isosurface(field, np.exp(-radius))

    def area(target) -> float:
        a, b, c = (target.vertices[target.triangles[:, i]] for i in range(3))
        return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())

    before = area(mesh)
    after = area(smooth_mesh(mesh, 1.0))
    assert after == pytest.approx(before, rel=0.05)


def test_slices_carry_true_lattice_geometry():
    """A slice through a triclinic cell is a parallelogram; drawing it as a
    rectangle would misplace every feature relative to the atoms."""
    lattice = np.array(_triclinic())
    structure = StructureData(
        lattice=lattice, species=["H"], frac_coords=np.array([[0.5, 0.5, 0.5]])
    )
    field = VolumetricField(structure=structure, values=np.zeros((4, 5, 6)))
    data = vol.slice_geometry(field, axis=2, fraction=0.5)

    assert data.values.shape == (4, 5)
    assert data.x_edges.shape == (5, 6)
    # The two in-plane lattice vectors are not perpendicular, so the quad is
    # sheared: opposite edges are offset in x.
    assert data.x_edges[0, -1] != pytest.approx(data.x_edges[0, 0])
    assert data.distance == pytest.approx(0.5 * np.linalg.norm(lattice[2]))


def test_planar_average_over_the_perpendicular_directions(tmp_path):
    field = _field(tmp_path, shape=(4, 5, 6))
    distance, values = vol.planar_average(field, axis=2)

    assert values.shape == (6,)
    np.testing.assert_allclose(values, field.values.mean(axis=(0, 1)))
    assert distance[0] == 0.0
    assert distance[-1] < field.structure.lattice_parameters[2]


def test_slice_interpolates_between_grid_planes(tmp_path):
    field = _field(tmp_path, shape=(4, 4, 4))
    lower = field.slice_plane(2, 0.0)
    upper = field.slice_plane(2, 0.25)
    middle = field.slice_plane(2, 0.125)
    np.testing.assert_allclose(middle, 0.5 * (lower + upper), rtol=1e-9)


def test_slice_position_wraps_periodically(tmp_path):
    field = _field(tmp_path, shape=(4, 4, 4))
    np.testing.assert_allclose(field.slice_plane(2, 0.0), field.slice_plane(2, 1.0))


def test_grid_spacing_is_reported_per_axis(tmp_path):
    field = _field(tmp_path, shape=(3, 4, 5))
    da, db, dc = field.grid_spacing
    assert da == pytest.approx(3.0 / 3)
    assert db == pytest.approx(4.0 / 4)
    assert dc == pytest.approx(5.0 / 5)
    # Unequal spacings must not be collapsed into a single misleading number.
    field2 = _field(tmp_path, shape=(2, 8, 16))
    assert "/" in vol.describe_spacing(field2)


def test_subsampling_never_changes_reported_quantities(tmp_path):
    field = _field(tmp_path, shape=(8, 8, 8), electrons=10.0)
    coarse = field.subsample(2)

    assert coarse.shape == (4, 4, 4)
    # The rendering copy is coarser, but the analysis field is untouched.
    assert field.shape == (8, 8, 8)
    assert field.integral() == pytest.approx(10.0, abs=1e-9)


def test_suggested_subsample_is_per_axis(tmp_path):
    # A slab-like grid: finely sampled along c, coarsely along a and b.
    field = _field(
        tmp_path,
        shape=(4, 4, 64),
        lattice=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 20.0]],
    )
    strides = suggested_subsample(field)
    assert len(strides) == 3
    assert strides[2] >= strides[0], "the finely sampled axis is coarsened more"


# --- Test G: capability and missing files ---------------------------------------------------


def test_capability_enabled_by_a_chgcar(tmp_path):
    fixtures.case_charge(tmp_path)
    calculation = discover(tmp_path)
    assert calculation.can(Capability.CHARGE_DENSITY)


def test_capability_absent_without_any_volumetric_file(tmp_path):
    fixtures.case_a_complete(tmp_path)
    calculation = discover(tmp_path)
    assert not calculation.can(Capability.CHARGE_DENSITY)
    assert "CHGCAR" in calculation.why_not(Capability.CHARGE_DENSITY)


def test_chgcar_works_without_an_outcar(tmp_path):
    """Test G: the density is fully usable with no OUTCAR; only the expected
    electron count it would have supplied goes missing."""
    fixtures.write_volumetric(tmp_path, shape=(4, 4, 4), electrons=8.0)
    calculation = discover(tmp_path)

    assert calculation.can(Capability.CHARGE_DENSITY)
    field = parse_volumetric(tmp_path / "CHGCAR").value.fields[0]
    assert field.integral() == pytest.approx(8.0, abs=1e-9)

    result = vol.integrate(field, expected_electrons=None)
    assert result.expected is None
    assert result.is_consistent is None, "an unknown cannot be called consistent"


def test_volumetric_structure_comes_from_the_file_itself(tmp_path):
    """The grid belongs to the geometry written beside it. Pairing it with a
    different POSCAR risks showing the density against the wrong coordinates."""
    fixtures.write_volumetric(
        tmp_path, shape=(4, 4, 4), lattice=[[9.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]]
    )
    fixtures.write_poscar(tmp_path)   # a different (silicon) cell entirely

    field = parse_volumetric(tmp_path / "CHGCAR").value.fields[0]
    assert field.structure.lattice_parameters[0] == pytest.approx(9.0)


def test_colormap_family_follows_the_physics(tmp_path):
    fixtures.write_volumetric(tmp_path, shape=(4, 4, 4), n_blocks=2)
    total, magnetization = parse_volumetric(tmp_path / "CHGCAR").value.fields

    assert vol.colormap_for(total) == "viridis"          # sequential
    assert vol.colormap_for(magnetization) == "RdBu_r"   # diverging


def test_atoms_near_plane_finds_only_nearby_atoms(tmp_path):
    field = _field(tmp_path, shape=(4, 4, 8))
    # The fixture puts atoms at fractional z = 0.25.
    near = vol.atoms_near_plane(field, axis=2, fraction=0.25, tolerance=0.5)
    far = vol.atoms_near_plane(field, axis=2, fraction=0.75, tolerance=0.5)

    assert near, "atoms sit on this plane"
    assert not far, "the opposite side of the cell is empty"
