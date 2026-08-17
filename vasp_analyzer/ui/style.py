"""Application stylesheet, generated from the design tokens."""

from __future__ import annotations

from pathlib import Path

from .theme import METRICS, PALETTE, RADIUS, SPACING, TYPE

ICON_DIR = Path(__file__).resolve().parent.parent / "resources" / "icons"


def icon_url(name: str) -> str:
    """Qt stylesheet URL for a bundled icon (forward slashes on every platform)."""
    return (ICON_DIR / name).as_posix()


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

    /* ---- Sliders ----------------------------------------------------- */
    QSlider::groove:horizontal {{
        height: 4px;
        background: {p.border};
        border-radius: 2px;
    }}
    QSlider::sub-page:horizontal {{
        background: {p.accent};
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {p.surface};
        border: 1px solid {p.border_strong};
        width: 15px;
        height: 15px;
        margin: -6px 0;
        border-radius: 8px;
    }}
    QSlider::handle:horizontal:hover {{ border-color: {p.accent}; }}
    QSlider:disabled::sub-page:horizontal {{ background: {p.border_strong}; }}

    /* ---- Value fields ------------------------------------------------ */
    #ValueField {{
        background: {p.surface};
        border: 1px solid {p.border};
        border-radius: {r.sm}px;
        padding: 2px {s.sm}px;
        min-height: 22px;
        font-size: {t.small}px;
        color: {p.text};
    }}
    #ValueField:focus {{
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
