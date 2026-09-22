"""Design tokens.

Every colour, radius, spacing step and type size in the application comes from
here. Widgets never hard-code a hex value — that is what keeps six pages
looking like one product, and what will make a dark theme a single edit later.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    background: str = "#f5f5f5"
    surface: str = "#ffffff"
    surface_sunken: str = "#fafafa"
    surface_hover: str = "#f0f0f2"
    border: str = "#e0e0e0"
    border_strong: str = "#d0d0d4"
    text: str = "#1a1a1a"
    text_secondary: str = "#666666"
    text_tertiary: str = "#9a9a9f"
    accent: str = "#2563eb"
    accent_hover: str = "#1d4ed8"
    accent_soft: str = "#eff4ff"
    fermi: str = "#dc2626"
    wannier: str = "#1a1a1a"
    success: str = "#16a34a"
    warning: str = "#d97706"
    danger: str = "#dc2626"
    disabled_text: str = "#b4b4b8"
    shadow: str = "rgba(0, 0, 0, 0.06)"


@dataclass(frozen=True)
class Spacing:
    xs: int = 4
    sm: int = 8
    md: int = 12
    lg: int = 16
    xl: int = 24
    xxl: int = 32


@dataclass(frozen=True)
class Radius:
    sm: int = 6
    md: int = 8
    lg: int = 12
    pill: int = 999


@dataclass(frozen=True)
class Typography:
    # Qt resolves the empty family to the macOS system font (SF).
    ui_family: str = ""
    mono_family: str = "Menlo, Monaco, monospace"
    plot_family: str = "Helvetica Neue"

    title: int = 22
    subtitle: int = 12
    section: int = 11
    body: int = 13
    small: int = 11
    tiny: int = 10


@dataclass(frozen=True)
class Metrics:
    sidebar_width: int = 280
    dos_panel_width: int = 300
    info_bar_height: int = 64
    nav_height: int = 34
    control_height: int = 28
    toolbar_button_height: int = 32
    # The height a #ToolButton actually renders at: the stylesheet's
    # min-height is a content height, and the 1px border top and bottom
    # sits outside it. Every control floating over a 3D viewer -- tool
    # buttons, pop-out button, orientation bar -- is sized to this, so the
    # row along the top of a viewer is one height end to end.
    viewer_control_height: int = 34


PALETTE = Palette()
SPACING = Spacing()
RADIUS = Radius()
TYPE = Typography()
METRICS = Metrics()

# Series colours for band/DOS plots, used in order.
SERIES_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed"]
