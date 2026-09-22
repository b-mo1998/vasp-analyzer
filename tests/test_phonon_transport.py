"""The Phonons page's animation transport: one play/pause button and a loop.

This replaced a "▶" / "❚❚" text pair beside a "■" button. Two problems, both
fixed here: the glyphs rendered as hairlines or tofu in several of the fonts
this app can fall back to, and the "■" read as a pause but rewound to frame 0
-- so the only thing that looked like pause actually threw away where you
were in the cycle.
"""

from __future__ import annotations

import pytest

from vasp_analyzer.services.loader import DataLoader
from vasp_analyzer.ui.pages.phonons.page import PhononsPage

from . import phonon_fixtures as pf


@pytest.fixture()
def page(qtbot, tmp_path):
    pf.complete_phonon_calculation(tmp_path)
    widget = PhononsPage(DataLoader())
    qtbot.addWidget(widget)
    return widget


def test_the_play_button_carries_an_icon_not_a_glyph(page):
    assert page._play.text() == ""
    assert not page._play.icon().isNull()


def test_the_second_button_is_a_loop_toggle_not_a_stop(page):
    assert page._loop.isCheckable()
    assert page._loop.isChecked(), "looping is the sensible default for an animation"
    assert not page._loop.icon().isNull()


def test_the_loop_button_carries_its_own_style_hook(page):
    """A dedicated object name, not the shared #ToolButton one: giving every
    checkable ToolButton a blue checked background would also hit the
    Select/Rotate/Pan tool groups, whose dark-stroke icons would lose all
    contrast against it."""
    assert page._loop.objectName() == "LoopButton"


def _load_frames(page, n_phases: int = 16) -> None:
    """Give the viewer a real animation to play.

    Play refuses to start with no frames, which is correct -- there is
    nothing to animate -- so a transport test has to supply some.
    """
    import numpy as np

    page._view.set_frames(np.zeros((n_phases, 2, 3)))


def test_play_toggles_to_pause_and_back(page, qtbot):
    _load_frames(page)

    playing = page._play.icon().cacheKey()
    page._toggle_play()
    assert page._playing
    paused = page._play.icon().cacheKey()
    assert paused != playing, "the icon must actually change on play"

    page._toggle_play()
    assert not page._playing
    assert page._play.icon().cacheKey() != paused


def test_pause_does_not_rewind_to_the_first_frame(page):
    """The old stop button's behaviour, and the specific complaint: pausing
    mid-cycle to look at a displacement pattern and having the view snap back
    to the undisplaced frame is the opposite of what pausing is for."""
    _load_frames(page)
    page._toggle_play()
    page._view.set_phase_index(7)

    page._toggle_play()

    assert page._view.phase_index == 7


def test_the_loop_icon_swaps_between_on_and_off_states(page):
    """A single fixed-colour icon would lose all contrast in one of the two
    states: the checked background is accent blue, so the icon has to turn
    white to stay visible against it, and dark again once unchecked."""
    on_pixmap = page._loop.icon().pixmap(14, 14).toImage()

    page._loop.setChecked(False)
    off_pixmap = page._loop.icon().pixmap(14, 14).toImage()
    assert off_pixmap != on_pixmap

    page._loop.setChecked(True)
    assert page._loop.icon().pixmap(14, 14).toImage() == on_pixmap


def test_the_loop_button_and_the_sidebar_repeat_stay_in_step(page):
    """One setting seen from two places, not two settings that can disagree."""
    page._loop.setChecked(False)
    assert not page._sidebar.repeat.isChecked()

    page._sidebar.repeat.setChecked(True)
    assert page._loop.isChecked()

    page._sidebar.repeat.setChecked(False)
    assert not page._loop.isChecked()
