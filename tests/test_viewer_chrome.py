"""The chrome that floats over a 3D viewport.

Two things this pins. First, the Structure page uses the same arrangement as
every other 3D page -- it was the odd one out, with an in-flow toolbar strip
above the canvas carrying a redundant "Structure Viewer" title. Second, the
floating toolbar has no panel of its own: each button already carries a white
fill and a border, so a second translucent slab behind them read as a grey box
swallowing a corner of the render.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QLabel

from vasp_analyzer.services.loader import DataLoader
from vasp_analyzer.ui.pages.charge.page import ChargePage
from vasp_analyzer.ui.pages.editor.page import StructureEditorPage
from vasp_analyzer.ui.pages.md.page import MDPage
from vasp_analyzer.ui.pages.neb.page import NEBPage
from vasp_analyzer.ui.pages.structure.page import StructurePage


@pytest.fixture()
def structure_page(qtbot):
    page = StructurePage(DataLoader())
    qtbot.addWidget(page)
    page.resize(1000, 700)
    page.show()
    return page


def test_the_structure_toolbar_floats_over_the_viewer(structure_page):
    """A child of the holder, positioned over the canvas -- not a sibling
    strip laid out above it."""
    page = structure_page
    assert page._toolbar.parentWidget() is page._holder
    assert page._axis_bar.parentWidget() is page._holder
    assert page._popout.parentWidget() is page._holder


def test_the_structure_toolbar_has_no_title_label(structure_page):
    """The card it sits in is already unambiguous, and the label pushed the
    tools away from the thing they act on."""
    labels = [
        label.text() for label in structure_page._toolbar.findChildren(QLabel)
    ]
    assert "Structure Viewer" not in labels


def test_the_structure_overlays_are_positioned_in_the_corners(structure_page, qtbot):
    page = structure_page
    holder = page._holder
    # Sized here rather than left to the page's own layout: with no
    # calculation open the card shows an empty state and the holder stays
    # narrower than the orientation bar, which makes "is it on the right?"
    # meaningless.
    holder.resize(900, 600)
    page._position_overlays()

    assert page._toolbar.x() < holder.width() / 2, "tools sit top-left"
    assert page._axis_bar.x() > holder.width() / 2, "orientation sits top-right"
    # The pop-out button tucks in beside the orientation bar, not over it.
    assert page._popout.x() + page._popout.width() <= page._axis_bar.x()


def test_popping_out_the_structure_viewer_takes_its_toolbar_along(structure_page):
    page = structure_page
    page._popout._toggle()
    try:
        assert page._holder.window() is page._popout._host
        # The floating children moved with their parent.
        assert page._toolbar.window() is page._popout._host
        assert page._axis_bar.window() is page._popout._host
    finally:
        page._popout._toggle()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: StructurePage(DataLoader()),
        lambda: StructureEditorPage(),
        lambda: MDPage(DataLoader()),
        lambda: NEBPage(DataLoader()),
        lambda: ChargePage(DataLoader()),
    ],
    ids=["structure", "editor", "md", "neb", "charge"],
)
def test_no_page_paints_its_own_panel_behind_the_floating_toolbar(qtbot, factory):
    """Styled centrally and transparently. A page setting its own opaque
    stylesheet here would put the grey box back on that page alone, which is
    exactly how the five drifted apart before."""
    page = factory()
    qtbot.addWidget(page)

    bar = page._toolbar
    assert bar.objectName() == "ViewerToolbar"
    assert "rgba(255,255,255" not in bar.styleSheet()
