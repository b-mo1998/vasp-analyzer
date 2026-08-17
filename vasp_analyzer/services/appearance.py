"""User-overridable element colours for the structure viewer.

A colour picked for an element is a display preference, not a fact about a
calculation — it belongs with the other persistent settings and applies to
every structure the user opens, the same way an editor's colour theme isn't
scoped to one file. This module is the single place that resolves "the colour
to actually draw this element" so the viewer, the legend, and any picker UI
agree without duplicating the override lookup.
"""

from __future__ import annotations

from ..resources.elements import color as default_color
from ..resources.elements import normalize
from .settings import Settings

_settings: Settings | None = None
_overrides: dict[str, str] = {}


def _store() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        for key, value in _settings.element_colors().items():
            _overrides[key] = value
    return _settings


def effective_color(symbol: str) -> str:
    """The colour to draw this element with: an override if one is set, else
    the default Jmol-style colour from :mod:`vasp_analyzer.resources.elements`."""
    _store()
    return _overrides.get(normalize(symbol), default_color(symbol))


def has_override(symbol: str) -> bool:
    _store()
    return normalize(symbol) in _overrides


def set_override(symbol: str, hex_color: str) -> None:
    # _store() must run — and finish its one-time disk seed — before the
    # in-memory dict is touched. Mutating first and priming second let the
    # seed's unconditional overwrite silently clobber this very call whenever
    # it happened to be the first appearance.py operation in the process: the
    # write reached disk correctly, but the in-memory override reverted right
    # back to whatever (nothing) was on disk a moment earlier.
    store = _store()
    key = normalize(symbol)
    _overrides[key] = hex_color
    store.set_element_color(key, hex_color)


def clear_override(symbol: str) -> None:
    store = _store()
    key = normalize(symbol)
    _overrides.pop(key, None)
    store.clear_element_color(key)
