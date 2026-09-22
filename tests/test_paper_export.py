"""The single-column "paper" export preset: a combined band+DOS figure sized
and fonted for direct use in a journal figure, rather than the on-screen
figure's larger canvas shrunk down after the fact.
"""

from __future__ import annotations

import pytest

from vasp_analyzer.parsers.doscar import parse_doscar
from vasp_analyzer.parsers.eigenval import parse_eigenval
from vasp_analyzer.ui.pages.bands.export import (
    PAPER_AXIS_LABELSIZE,
    PAPER_DPI,
    PAPER_HEIGHT_IN,
    PAPER_TICK_LABELSIZE,
    PAPER_WIDTH_IN,
    ExportRequest,
    build_figure,
    build_paper_figure,
    save_paper_figure,
)

from . import fixtures


def _request(tmp_path, include_dos=True) -> ExportRequest:
    fixtures.write_eigenval(tmp_path)
    bands = parse_eigenval(tmp_path / "EIGENVAL").value
    dos = None
    if include_dos:
        fixtures.write_doscar(tmp_path, n_points=40)
        dos = parse_doscar(tmp_path / "DOSCAR").value
    return ExportRequest(
        bands=bands,
        dos=dos,
        reference=0.0,
        y_min=-10.0,
        y_max=10.0,
        dos_xmax=20.0,
        include_dos=include_dos,
        title="test",
    )


def test_paper_figure_is_single_column_sized(tmp_path):
    figure = build_paper_figure(_request(tmp_path))

    width, height = figure.get_size_inches()
    assert width == pytest.approx(PAPER_WIDTH_IN)
    assert height == pytest.approx(PAPER_HEIGHT_IN)
    assert figure.dpi == PAPER_DPI


def test_paper_figure_uses_smaller_fonts_than_the_on_screen_figure(tmp_path):
    request = _request(tmp_path)
    paper = build_paper_figure(request)
    screen = build_figure(request)

    paper_axes = paper.axes[0]
    screen_axes = screen.axes[0]
    assert paper_axes.yaxis.label.get_fontsize() == PAPER_AXIS_LABELSIZE
    assert paper_axes.yaxis.label.get_fontsize() < screen_axes.yaxis.label.get_fontsize()
    assert paper_axes.get_xticklabels()[0].get_fontsize() == PAPER_TICK_LABELSIZE


def test_paper_figure_without_dos_is_a_single_axes(tmp_path):
    figure = build_paper_figure(_request(tmp_path, include_dos=False))
    assert len(figure.axes) == 1


def test_save_paper_figure_writes_a_file(tmp_path):
    out = tmp_path / "paper.pdf"
    save_paper_figure(_request(tmp_path), out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_paper_figure_without_bands_raises(tmp_path):
    request = _request(tmp_path)
    request.bands = None
    with pytest.raises(ValueError):
        save_paper_figure(request, tmp_path / "unused.pdf")
