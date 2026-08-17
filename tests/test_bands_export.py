"""DOS export as its own figure, and the bands-less-DOS case (DOSCAR without
EIGENVAL) that export/copy previously couldn't handle at all — the whole
Bands page's export used to bail out whenever bands were absent, even when
DOS alone was perfectly exportable.
"""

from __future__ import annotations

import pytest

from vasp_analyzer.parsers.doscar import parse_doscar
from vasp_analyzer.ui.pages.bands.export import ExportRequest, build_dos_figure, save_dos_figure

from . import fixtures


def _dos_request(dos):
    return ExportRequest(
        bands=None,
        dos=dos,
        reference=dos.e_fermi if dos else 0.0,
        y_min=-10.0,
        y_max=10.0,
        dos_xmax=20.0,
        include_dos=True,
        title="test",
    )


# --- pure functions --------------------------------------------------------------


def test_build_dos_figure_has_its_own_axis_labels(tmp_path):
    fixtures.write_doscar(tmp_path, n_points=60)
    dos = parse_doscar(tmp_path / "DOSCAR").value
    figure = build_dos_figure(_dos_request(dos))

    axes = figure.axes[0]
    assert axes.get_xlabel() == "States / eV"
    assert axes.get_ylabel() == "Energy (eV)"
    assert axes.get_ylim() == (-10.0, 10.0)


def test_save_dos_figure_writes_a_file(tmp_path):
    fixtures.write_doscar(tmp_path, n_points=60)
    dos = parse_doscar(tmp_path / "DOSCAR").value
    out = tmp_path / "dos.png"
    save_dos_figure(_dos_request(dos), out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_dos_figure_without_dos_raises():
    with pytest.raises(ValueError):
        save_dos_figure(_dos_request(None), "unused.png")


def test_dos_figure_title_is_set(tmp_path):
    fixtures.write_doscar(tmp_path, n_points=40)
    dos = parse_doscar(tmp_path / "DOSCAR").value
    figure = build_dos_figure(_dos_request(dos))
    assert figure._suptitle is not None
    assert figure._suptitle.get_text() == "test"
