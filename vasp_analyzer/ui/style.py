"""Application stylesheet, generated from the design tokens."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette

from .theme import METRICS, PALETTE, RADIUS, SPACING, TYPE

ICON_DIR = Path(__file__).resolve().parent.parent / "resources" / "icons"


def icon_url(name: str) -> str:
    """Qt stylesheet URL for a bundled icon (forward slashes on every platform)."""
    return (ICON_DIR / name).as_posix()


def build_palette() -> QPalette:
    """An explicit light QPalette, independent of the OS appearance.

    The stylesheet alone is not enough: Fusion resolves any palette role the
    stylesheet doesn't touch from the system theme, and Qt's own widgets paint
    some of their internals from palette roles rather than QSS. The clearest
    case is QScrollArea's viewport, which every sidebar uses and which paints
    itself with the Base role — in macOS Dark Mode that made every sidebar
    render black with barely legible text. Setting every role explicitly here
    means the app looks identical regardless of the user's system appearance,
    which is deliberate: there is no dark theme yet, so half-dark is worse
    than consistently light.
    """
    p = PALETTE
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(p.background))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(p.text))
    palette.setColor(QPalette.ColorRole.Base, QColor(p.surface))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(p.surface_sunken))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(p.text))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Text, QColor(p.text))
    palette.setColor(QPalette.ColorRole.Button, QColor(p.surface))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(p.text))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(p.danger))
    palette.setColor(QPalette.ColorRole.Link, QColor(p.accent))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(p.accent))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(p.text_tertiary))
    for role in (
        QPalette.ColorRole.Text,
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(p.disabled_text))
    return palette


def apply_theme(app) -> None:
    """Install the application's look: style, palette and stylesheet, in the
    order Qt needs them applied so none of the three gets silently overridden
    by a later system-appearance change."""
    app.setStyle("Fusion")
    style_hints = getattr(app, "styleHints", None)
    if style_hints is not None:
        try:
            style_hints().setColorScheme(Qt.ColorScheme.Light)
        except AttributeError:
            pass  # Qt < 6.5: the palette override below still applies.
    app.setPalette(build_palette())
    app.setStyleSheet(build_stylesheet())


def build_stylesheet() -> str:
    p, r, s, t, m = PALETTE, RADIUS, SPACING, TYPE, METRICS
    check = icon_url("check.svg")
    return f"""
    QWidget {{
        color: {p.text};
        font-size: {t.body}px;
    }}
    QMainWindow, #Root {{
        background: {p.background};
    }}

    /* ---- Cards ------------------------------------------------------ */
    #Card {{
        background: {p.surface};
        border: 1px solid {p.border};
        border-radius: {r.lg}px;
    }}
    #Sidebar {{
        background: {p.surface};
        border-right: 1px solid {p.border};
    }}

    /* ---- Text styles ------------------------------------------------ */
    #TitleLabel {{
        font-size: {t.title}px;
        font-weight: 600;
        color: {p.text};
    }}
    #SubtitleLabel {{
        font-size: {t.subtitle}px;
        color: {p.text_secondary};
    }}
    #SectionLabel {{
        font-size: {t.section}px;
        font-weight: 600;
        color: {p.text_secondary};
    }}
    #CardTitle {{
        font-size: {t.body}px;
        font-weight: 600;
    }}
    #SecondaryLabel {{
        color: {p.text_secondary};
    }}
    #TertiaryLabel {{
        color: {p.text_tertiary};
        font-size: {t.small}px;
    }}
    #RangeLabel {{
        color: {p.text_tertiary};
        font-size: {t.tiny}px;
    }}

    /* ---- Segmented navigation --------------------------------------- */
    #SegmentedControl {{
        background: {p.surface};
        border: 1px solid {p.border};
        border-radius: {r.md + 2}px;
    }}
    #SegmentedControl QPushButton {{
        background: transparent;
        border: none;
        border-radius: {r.md}px;
        padding: 0 {s.xl}px;
        min-height: {m.nav_height}px;
        font-size: {t.body}px;
        font-weight: 500;
        color: {p.text};
    }}
    #SegmentedControl QPushButton:hover:!checked:enabled {{
        background: {p.surface_hover};
    }}
    #SegmentedControl QPushButton:checked {{
        background: {p.accent};
        color: #ffffff;
    }}
    #SegmentedControl QPushButton:disabled {{
        color: {p.disabled_text};
    }}

    /* ---- Toolbar buttons -------------------------------------------- */
    #ToolButton {{
        background: {p.surface};
        border: 1px solid {p.border};
        border-radius: {r.md}px;
        padding: 0 {s.md}px;
        min-height: {m.toolbar_button_height}px;
        font-size: {t.body}px;
        color: {p.text};
    }}
    #ToolButton:hover:enabled {{
        background: {p.surface_hover};
    }}
    #ToolButton:pressed:enabled {{
        background: {p.border};
    }}
    #ToolButton:disabled {{
        color: {p.disabled_text};
        border-color: {p.border};
    }}

    /* ---- Primary button --------------------------------------------- */
    #PrimaryButton {{
        background: {p.accent};
        color: #ffffff;
        border: none;
        border-radius: {r.md}px;
        padding: 0 {s.lg}px;
        min-height: {m.toolbar_button_height}px;
        font-weight: 500;
    }}
    #PrimaryButton:hover {{ background: {p.accent_hover}; }}

    /* Sliders are not QSlider — see ui/widgets/painted_slider.py. QSlider's
       ::groove/::add-page stylesheet subcontrols turn out to silently revert
       to native Fusion painting once nested inside a QScrollArea, which every
       sidebar in this app is; a custom-painted widget sidesteps that
       entirely and is what's actually in use everywhere. */

    /* ---- Value fields ------------------------------------------------ */
    /* Flat by default (a tinted pill, no outline) so a sidebar full of them
       doesn't read as a wall of boxes; the border only appears on hover and
       focus, where it signals "this is editable" right when it matters. */
    #ValueField {{
        background: {p.surface_sunken};
        border: 1px solid transparent;
        border-radius: {r.sm}px;
        padding: 2px {s.sm}px;
        min-height: 22px;
        font-size: {t.small}px;
        color: {p.text};
    }}
    #ValueField:hover {{
        border-color: {p.border};
    }}
    #ValueField:focus {{
        background: {p.surface};
        border-color: {p.accent};
    }}
    #ValueField:disabled {{
        color: {p.disabled_text};
        background: {p.surface_sunken};
    }}

    /* ---- Check boxes -------------------------------------------------- */
    QCheckBox {{
        spacing: {s.sm}px;
        color: {p.text};
    }}
    QCheckBox:disabled {{ color: {p.disabled_text}; }}
    QCheckBox::indicator {{
        width: 15px;
        height: 15px;
        border-radius: 4px;
        border: 1px solid {p.border_strong};
        background: {p.surface};
    }}
    QCheckBox::indicator:hover:enabled {{ border-color: {p.accent}; }}
    QCheckBox::indicator:checked {{
        background: {p.accent};
        border-color: {p.accent};
        image: url({check});
    }}
    QCheckBox::indicator:disabled {{
        background: {p.surface_sunken};
        border-color: {p.border};
    }}

    /* ---- Combo box ---------------------------------------------------- */
    QComboBox {{
        background: {p.surface};
        border: 1px solid {p.border};
        border-radius: {r.md}px;
        padding: 0 {s.md}px;
        min-height: 32px;
    }}
    QComboBox:hover {{ border-color: {p.border_strong}; }}
    QComboBox::drop-down {{ border: none; width: 22px; }}
    /* ---- Integer stepper ------------------------------------------------ */
    /* Not QSpinBox — see ui/widgets/stepper.py for why. Explicit +/- buttons,
       large and unambiguous, in the same #ToolButton spirit as the viewer's
       zoom controls. */
    #StepperButtonLeft, #StepperButtonRight {{
        background: {p.surface};
        border: 1px solid {p.border};
        color: {p.text_secondary};
        font-size: 15px;
        font-weight: 600;
    }}
    #StepperButtonLeft {{
        border-top-left-radius: {r.md}px;
        border-bottom-left-radius: {r.md}px;
        border-right: none;
    }}
    #StepperButtonRight {{
        border-top-right-radius: {r.md}px;
        border-bottom-right-radius: {r.md}px;
        border-left: none;
    }}
    #StepperButtonLeft:hover:enabled, #StepperButtonRight:hover:enabled {{
        background: {p.surface_hover}; color: {p.text};
    }}
    #StepperButtonLeft:pressed:enabled, #StepperButtonRight:pressed:enabled {{
        background: {p.border};
    }}
    #StepperButtonLeft:disabled, #StepperButtonRight:disabled {{
        color: {p.disabled_text};
    }}
    #StepperField {{
        background: {p.surface};
        border: 1px solid {p.border};
        border-left: none;
        border-right: none;
        font-size: {t.body}px;
        font-weight: 500;
    }}
    #StepperField:focus {{ border-color: {p.border}; }}
    #StepperField:disabled {{ color: {p.disabled_text}; background: {p.surface_sunken}; }}

    QComboBox QAbstractItemView {{
        background: {p.surface};
        border: 1px solid {p.border};
        border-radius: {r.md}px;
        selection-background-color: {p.accent_soft};
        selection-color: {p.text};
        padding: 4px;
        outline: none;
    }}

    /* ---- Scroll areas -------------------------------------------------- */
    QScrollArea {{ background: transparent; border: none; }}
    QScrollBar:vertical {{
        background: transparent; width: 10px; margin: 2px;
    }}
    QScrollBar::handle:vertical {{
        background: {p.border_strong};
        border-radius: 4px;
        min-height: 28px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {p.text_tertiary}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    QScrollBar:horizontal {{
        background: transparent; height: 10px; margin: 2px;
    }}
    QScrollBar::handle:horizontal {{
        background: {p.border_strong};
        border-radius: 4px;
        min-width: 28px;
    }}

    /* ---- Separators ----------------------------------------------------- */
    #Divider {{
        background: {p.border};
        max-height: 1px;
        min-height: 1px;
        border: none;
    }}

    /* ---- Tooltips -------------------------------------------------------- */
    QToolTip {{
        background: {p.text};
        color: #ffffff;
        border: none;
        border-radius: {r.sm}px;
        padding: 5px 8px;
        font-size: {t.small}px;
    }}

    /* ---- Menus ----------------------------------------------------------- */
    QMenu {{
        background: {p.surface};
        border: 1px solid {p.border};
        border-radius: {r.md}px;
        padding: 5px;
    }}
    QMenu::item {{
        padding: 6px 22px 6px 12px;
        border-radius: {r.sm}px;
    }}
    QMenu::item:selected {{ background: {p.accent_soft}; }}
    QMenu::item:disabled {{ color: {p.disabled_text}; }}
    QMenu::separator {{
        height: 1px;
        background: {p.border};
        margin: 4px 8px;
    }}
    """
