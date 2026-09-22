"""Colour conventions on the phonon dispersion plot.

The one invariant that matters here: red means imaginary. The legend says so,
and dynamic instability is the single most consequential thing this plot
communicates -- so no stable branch may ever be drawn red, in any colouring
mode, and imaginary stretches stay red and dashed regardless of what the rest
of the plot is coloured by.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import pytest

from vasp_analyzer.services import phonon as phonon_service
from vasp_analyzer.ui.pages.phonons.plots import (
    BRANCH_COLORS,
    IMAGINARY_COLOR,
    DispersionPlot,
)

from . import phonon_fixtures as pf

RED = mcolors.to_rgba(IMAGINARY_COLOR)


@pytest.fixture()
def unstable_dataset(tmp_path):
    """A real negative force constant, so imaginary branches genuinely exist."""
    pf.unstable_phonon_calculation(tmp_path)
    result = phonon_service.load_phonons(tmp_path)
    assert result.ok
    return result.value


def _lines(qtbot, dataset, mode):
    plot = DispersionPlot()
    qtbot.addWidget(plot)
    plot.set_coloring(mode)
    plot.set_dataset(dataset, show_imaginary=True)
    return plot.axes.get_lines()


@pytest.mark.parametrize("mode", ["By Stability", "By Branch"])
def test_no_stable_branch_is_ever_red(qtbot, unstable_dataset, mode):
    """A cyclic colormap spanning the full hue wheel put a *stable* branch on
    red, contradicting the plot's own legend."""
    for line in _lines(qtbot, unstable_dataset, mode):
        solid = line.get_linestyle() in ("-", "solid")
        if solid:
            assert mcolors.to_rgba(line.get_color()) != RED


@pytest.mark.parametrize("mode", ["By Stability", "By Branch"])
def test_imaginary_stretches_are_red_and_dashed(qtbot, unstable_dataset, mode):
    dashed_red = [
        line
        for line in _lines(qtbot, unstable_dataset, mode)
        if line.get_linestyle() not in ("-", "solid")
        and mcolors.to_rgba(line.get_color()) == RED
    ]
    assert dashed_red, f"{mode} must still mark imaginary branches red and dashed"


def test_the_branch_palette_contains_no_red():
    assert all(mcolors.to_rgba(c) != RED for c in BRANCH_COLORS)


def test_branch_colours_stay_clear_of_red():
    """Not merely 'not exactly red' -- far enough away to read as a different
    colour at a glance on a thin 0.9px line."""
    import math

    for colour in BRANCH_COLORS:
        distance = math.dist(mcolors.to_rgba(colour)[:3], RED[:3])
        assert distance > 0.3, f"{colour} is too close to the imaginary red"


def test_by_branch_gives_neighbouring_branches_different_colours(
    qtbot, unstable_dataset
):
    solid = [
        line.get_color()
        for line in _lines(qtbot, unstable_dataset, "By Branch")
        if line.get_linestyle() in ("-", "solid")
    ]
    assert len(set(solid[:3])) > 1, "branches must be distinguishable"
