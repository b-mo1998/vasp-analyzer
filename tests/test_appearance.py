"""Element-colour overrides: persistence and fallback to the default palette.

Settings is backed by QSettings, which defaults to the OS's native preference
store (a real plist under ~/Library/Preferences on macOS) — the same store the
installed app itself uses. Without isolation these tests would read and write
the developer's actual saved colour preferences. Every test here redirects
both the appearance module's cache and QSettings itself to a throwaway
temp-file-backed store first.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QSettings

from vasp_analyzer.resources.elements import color as default_color
from vasp_analyzer.services import appearance
from vasp_analyzer.services import settings as settings_module


@pytest.fixture(autouse=True)
def isolated_appearance(tmp_path, monkeypatch):
    ini_path = str(tmp_path / "test_settings.ini")
    monkeypatch.setattr(
        settings_module,
        "QSettings",
        lambda *a, **k: QSettings(ini_path, QSettings.IniFormat),
    )
    # The module caches one Settings instance and its override dict for the
    # process lifetime; force both to be rebuilt against the temp store.
    monkeypatch.setattr(appearance, "_settings", None)
    monkeypatch.setattr(appearance, "_overrides", {})
    yield


def test_effective_color_falls_back_to_default():
    assert appearance.effective_color("Xe") == default_color("Xe")


def test_set_override_takes_effect_immediately():
    appearance.set_override("Xe", "#123456")
    assert appearance.effective_color("Xe") == "#123456"
    assert appearance.has_override("Xe")


def test_clear_override_restores_default():
    appearance.set_override("Rn", "#abcdef")
    appearance.clear_override("Rn")
    assert not appearance.has_override("Rn")
    assert appearance.effective_color("Rn") == default_color("Rn")


def test_override_persists_through_settings():
    from vasp_analyzer.services.settings import Settings

    appearance.set_override("Kr", "#ff00ff")
    settings = Settings()
    assert settings.element_color("Kr") == "#ff00ff"
    assert "Kr" in settings.element_colors()


def test_set_override_survives_being_the_first_call_ever_made():
    """Regression: _store()'s one-time disk seed used to run *after* the
    in-memory dict was already mutated when set_override/clear_override was
    the very first appearance.py call in the process, and its unconditional
    overwrite silently reverted the mutation. The write still reached disk
    correctly, so this only showed up as a same-process read-your-own-write
    failure — exactly what calling set_override cold, with nothing having
    primed the cache yet, reproduces."""
    assert appearance._settings is None  # nothing has touched the cache yet

    appearance.set_override("Fr", "#112233")
    assert appearance.effective_color("Fr") == "#112233"
    assert appearance.has_override("Fr")


def test_clear_override_survives_being_the_first_call_ever_made(monkeypatch):
    from vasp_analyzer.services.settings import Settings

    Settings().set_element_color("Ra", "#445566")
    assert appearance._settings is None

    appearance.clear_override("Ra")
    assert not appearance.has_override("Ra")
    assert appearance.effective_color("Ra") == default_color("Ra")


def test_overrides_are_seeded_from_settings_on_first_use():
    """A colour saved in a prior session must be picked up on the next one —
    that's the entire point of persisting it."""
    from vasp_analyzer.services.settings import Settings

    Settings().set_element_color("Ar", "#00ffff")
    # Simulate a fresh process: nothing cached yet.
    appearance._settings = None
    appearance._overrides = {}

    assert appearance.effective_color("Ar") == "#00ffff"
