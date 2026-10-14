"""The Phonons page's "DOS data (*.csv)" export.

Its column headers are the species names, which come from the calculation's
own band.yaml. A name that a spreadsheet reads as a formula must not reach the
file as one -- whichever separator the spreadsheet splits the line on.
"""

from __future__ import annotations

import csv

import pytest

from vasp_analyzer.services import phonon as phonon_service
from vasp_analyzer.services.loader import DataLoader
from vasp_analyzer.ui.pages.phonons.page import PhononsPage

from . import phonon_fixtures as pf

FORMULA_STARTS = ("=", "+", "-", "@")
SEPARATORS = (",", ";", "\t", " ", "|")


@pytest.fixture()
def page(qtbot):
    widget = PhononsPage(DataLoader())
    qtbot.addWidget(widget)
    return widget


def _export_header(page, directory, symbol: str | None = None) -> str:
    """Export the DOS for a Si/O cell, optionally renaming Si, and return the
    header line. No POSCAR is written, so the band file supplies the cell."""
    band = pf.write_band_yaml(directory, species=["Si", "O"])
    if symbol is not None:
        text = band.read_text()
        assert "- symbol: Si\n" in text
        band.write_text(text.replace("- symbol: Si\n", f"- symbol: {symbol}\n"))
    result = phonon_service.load_phonons(directory)
    assert result.ok and result.value is not None
    page._dataset = result.value
    target = directory / "dos.csv"
    page._write_dos(target)
    return target.read_text().split("\n", 1)[0]


def _cells(header: str):
    for separator in SEPARATORS:
        yield from header.split(separator)
        yield from next(csv.reader([header], delimiter=separator))


def test_ordinary_element_symbols_are_written_unchanged(page, tmp_path):
    header = _export_header(page, tmp_path)
    assert header == "frequency_THz,total,Si,O"


@pytest.mark.parametrize(
    "symbol",
    [
        '=HYPERLINK("http://attacker/x";"Si")',
        "Si;=cmd|' /C calc'!A0",
        "Si\t=1+1",
        "Si =1+1",
        "Si;+1+1",
        "Si;-1+1",
        "Si;@SUM(1)",
        "Si,=1+1",
        'Si"=1+1',
        "@SUM(1)",
    ],
)
def test_no_header_cell_starts_a_formula(page, tmp_path, symbol):
    header = _export_header(page, tmp_path, symbol)
    for cell in _cells(header):
        text = cell.strip().lstrip('"').strip()
        assert not text.startswith(FORMULA_STARTS), (header, cell)
    # The name is still there to read, only defused.
    row = next(csv.reader([header]))
    assert row[:2] == ["frequency_THz", "total"]
    assert row[2].replace("'", "") == symbol.replace("'", "")
    assert row[3] == "O"
