"""Regression test for the dark-mode sidebar bug.

On macOS in Dark Mode, Fusion resolves any palette role the stylesheet doesn't
touch from the system appearance. QScrollArea's viewport paints itself from
the Base role and isn't reachable by our id-selector QSS, so every sidebar —
all three pages use one — rendered black with barely legible text. apply_theme
must force a light appearance regardless of whatever palette was in effect
before it runs.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QScrollArea, QWidget

from vasp_analyzer.ui.style import apply_theme
from vasp_analyzer.ui.theme import PALETTE


def _luminance(color: QColor) -> float:
    return 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()


def test_apply_theme_overrides_a_pre_existing_dark_palette(qapp):
    dark = QPalette()
    for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base,
                 QPalette.ColorRole.WindowText):
        dark.setColor(role, QColor("#000000"))
    qapp.setPalette(dark)

    apply_theme(qapp)

    assert qapp.palette().color(QPalette.ColorRole.Window) == QColor(PALETTE.background)
    assert qapp.palette().color(QPalette.ColorRole.Base) == QColor(PALETTE.surface)
    assert qapp.palette().color(QPalette.ColorRole.WindowText) == QColor(PALETTE.text)


def test_scroll_area_viewport_renders_light_even_after_a_dark_palette(qapp, qtbot):
    # Simulate what macOS Dark Mode injects before the app gets a chance to
    # override it — this is the exact condition that produced the black
    # sidebars, so the test reproduces it rather than only checking the
    # palette values on paper.
    dark = QPalette()
    dark.setColor(QPalette.ColorRole.Base, QColor("#000000"))
    qapp.setPalette(dark)

    apply_theme(qapp)

    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(QWidget())
    area.resize(220, 220)
    qtbot.addWidget(area)
    area.show()
    qtbot.waitExposed(area)

    sampled = area.viewport().grab().toImage().pixelColor(110, 110)
    assert _luminance(sampled) > 200, (
        f"scroll area viewport rendered dark ({sampled.name()}) — "
        "the sidebar-black-background bug has regressed"
    )
