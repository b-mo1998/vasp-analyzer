"""The shared floating viewer toolbar, and the page behaviours around it.

One ViewerToolbar class serves every 3D page, so the tool set can no longer
drift between tabs: all pages get select/rotate/pan + zoom in/out + fit,
Structure and the Editor additionally get measure, and the keyboard
shortcuts (S/R/T/M, +/-/F) exist everywhere instead of only on the Editor.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QPushButton

from vasp_analyzer.services.loader import DataLoader
from vasp_analyzer.ui.pages.charge.page import ChargePage
from vasp_analyzer.ui.pages.md.page import MDPage
from vasp_analyzer.ui.pages.neb.page import NEBPage
from vasp_analyzer.ui.pages.structure.page import StructurePage
from vasp_analyzer.ui.widgets.viewer_toolbar import ViewerToolbar

from . import fixtures


PAGES = [
    ("structure", lambda: StructurePage(DataLoader()), True),
    ("md", lambda: MDPage(DataLoader()), False),
    ("neb", lambda: NEBPage(DataLoader()), False),
    ("charge", lambda: ChargePage(DataLoader()), False),
]


@pytest.mark.parametrize("name,factory,expects_measure", PAGES,
                         ids=[p[0] for p in PAGES])
def test_every_page_uses_the_shared_toolbar(qtbot, name, factory, expects_measure):
    page = factory()
    qtbot.addWidget(page)

    assert isinstance(page._toolbar, ViewerToolbar)
    tips = " ".join(b.toolTip() for b in page._toolbar.findChildren(QPushButton))
    assert "Zoom in" in tips and "Zoom out" in tips and "Fit" in tips
    assert ("Measure" in tips) == expects_measure


@pytest.mark.parametrize("name,factory,_m", PAGES, ids=[p[0] for p in PAGES])
def test_every_viewer_starts_in_rotate(qtbot, name, factory, _m):
    page = factory()
    qtbot.addWidget(page)

    assert page._view.mode() == page._view.MODE_ROTATE
    assert page._toolbar.current_tool() == "rotate"


def test_toolbar_set_tool_drives_the_view_and_the_highlight(qtbot):
    page = MDPage(DataLoader())
    qtbot.addWidget(page)

    page._toolbar.set_tool("select")
    assert page._view.mode() == page._view.MODE_SELECT
    assert page._toolbar.current_tool() == "select"


def test_md_defaults_to_unwrapped_boundaries(qtbot):
    page = MDPage(DataLoader())
    qtbot.addWidget(page)
    side = page._sidebar

    assert side.boundary_unwrapped.isChecked()
    assert not side.wrap_atoms.isChecked()

    # And reset keeps that default rather than reverting to wrapped.
    side.reset()
    assert side.boundary_unwrapped.isChecked()
    assert not side.wrap_atoms.isChecked()


def test_neb_sidebar_has_no_trail_length_control(qtbot):
    page = NEBPage(DataLoader())
    qtbot.addWidget(page)

    assert not hasattr(page._sidebar, "trail_length")
    # Trails still cover the whole band via the constant stand-in.
    assert page._sidebar.trail_images() >= 2


def test_charge_sidebar_reset_is_signal_quiet(qtbot):
    """One Reset click must trigger at most one rebuild pipeline. Before the
    guard, the sidebar's own reset emitted localizeChanged twice and
    compareChanged once, each running a full isosurface extraction before
    the page's reset ran a fourth."""
    from vasp_analyzer.ui.pages.charge.sidebar import ChargeSidebar

    side = ChargeSidebar()
    qtbot.addWidget(side)
    side.localize_enabled.setChecked(True)
    side.localize_field.setText("1-4")

    fired = []
    side.localizeChanged.connect(lambda: fired.append("localize"))
    side.compareChanged.connect(lambda: fired.append("compare"))
    side.reset()

    assert fired == []
    assert not side.localize_enabled.isChecked()
    assert side.localize_field.text() == ""


def test_charge_reset_recomputes_the_field_from_base(qtbot, tmp_path):
    """After a reset turns localization off silently, self._field must be
    recomputed from the raw base -- not left pointing at the stale localized
    field."""
    import time

    fixtures.write_volumetric(tmp_path, species=("Si",), counts=(3,),
                              electrons=8.0)
    from vasp_analyzer.services.discovery import discover

    loader = DataLoader()
    page = ChargePage(loader)
    qtbot.addWidget(page)
    calc = discover(tmp_path)
    loader.set_calculation(calc)
    page.set_calculation(calc)
    deadline = time.time() + 20
    while page._field is None and time.time() < deadline:
        qtbot.wait(50)
    assert page._field is not None

    page._sidebar.localize_enabled.setChecked(True)
    page._sidebar.localize_field.setText("1")
    qtbot.wait(100)
    assert page._field is not page._field_raw, "localization should mask"

    page.reset()
    qtbot.wait(100)
    assert page._field is page._field_raw
