"""Test configuration.

Qt must be told to run without a display before it is imported anywhere.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")


import pytest


@pytest.fixture(autouse=True)
def _isolate_user_settings(monkeypatch, tmp_path):
    """Keep the developer's own saved preferences out of every test.

    services.appearance caches element-colour overrides in module globals,
    populated once from the real QSettings store the first time a colour is
    looked up. Without this, a test asserting that two species are drawn
    differently would fail on a machine where the user had personally
    recoloured both to the same colour -- which is a statement about their
    preferences, not about the code under test. Redirecting QSettings to a
    throwaway ini and clearing the caches makes every test start from
    documented defaults.
    """
    from PySide6.QtCore import QSettings

    from vasp_analyzer.services import appearance
    from vasp_analyzer.services import settings as settings_module

    ini_path = str(tmp_path / "isolated_settings.ini")
    monkeypatch.setattr(
        settings_module,
        "QSettings",
        lambda *a, **k: QSettings(ini_path, QSettings.IniFormat),
    )
    monkeypatch.setattr(appearance, "_settings", None)
    monkeypatch.setattr(appearance, "_overrides", {})
    yield
