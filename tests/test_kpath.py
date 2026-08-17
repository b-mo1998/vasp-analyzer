"""k-axis construction, including the refusal to invent a path."""

from __future__ import annotations

import numpy as np

from vasp_analyzer.parsers.eigenval import parse_eigenval
from vasp_analyzer.parsers.kpoints import parse_kpoints
from vasp_analyzer.parsers.poscar import parse_poscar
from vasp_analyzer.services.kpath import build_kpath

from . import fixtures


def _bands(tmp_path, **kwargs):
    fixtures.write_eigenval(tmp_path, **kwargs)
    return parse_eigenval(tmp_path / "EIGENVAL").value


def test_mesh_falls_back_to_index_and_invents_no_labels(tmp_path):
    bands = _bands(tmp_path, n_kpoints=12, n_bands=4)
    fixtures.write_kpoints_mesh(tmp_path)
    spec = parse_kpoints(tmp_path / "KPOINTS").value

    build_kpath(bands, spec, None)

    assert not bands.is_line_mode
    assert bands.labels == []
    assert bands.x_axis_title == "k-point index"
    assert np.allclose(bands.x(), np.arange(12))


def test_no_kpoints_file_still_produces_a_usable_axis(tmp_path):
    bands = _bands(tmp_path, n_kpoints=8, n_bands=3)
    build_kpath(bands, None, None)

    assert bands.x().shape == (8,)
    assert bands.labels == []


def test_line_mode_produces_labels_and_monotonic_distance(tmp_path):
    bands = _bands(tmp_path, n_kpoints=12, n_bands=4)
    fixtures.write_kpoints_line(tmp_path, divisions=4)
    fixtures.write_poscar(tmp_path, atoms=2)
    spec = parse_kpoints(tmp_path / "KPOINTS").value
    structure = parse_poscar(tmp_path / "POSCAR").value

    build_kpath(bands, spec, structure)

    assert bands.is_line_mode
    assert bands.labels, "line mode must produce high-symmetry labels"
    assert bands.labels[0].label == "Γ"
    assert bands.x_axis_title == "Wave vector (Å⁻¹)"
    assert np.all(np.diff(bands.x()) >= -1e-12), "path length must not decrease"


def test_line_mode_without_structure_uses_fractional_distance(tmp_path):
    bands = _bands(tmp_path, n_kpoints=12, n_bands=4)
    fixtures.write_kpoints_line(tmp_path, divisions=4)
    spec = parse_kpoints(tmp_path / "KPOINTS").value

    build_kpath(bands, spec, None)

    assert bands.is_line_mode
    assert "fractional" in bands.x_axis_title
