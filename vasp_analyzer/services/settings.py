"""Persistent application settings, backed by QSettings."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QByteArray, QSettings

ORGANISATION = "VASPAnalyzer"
APPLICATION = "VASPAnalyzer"
MAX_RECENT = 8


class Settings:
    """Thin, typed wrapper so callers never touch raw QSettings keys."""

    def __init__(self) -> None:
        self._settings = QSettings(ORGANISATION, APPLICATION)

    # -- directories -------------------------------------------------------

    def last_directory(self) -> Path | None:
        value = self._settings.value("calculation/last_directory", "")
        if not value:
            return None
        path = Path(str(value))
        return path if path.is_dir() else None

    def set_last_directory(self, path: Path) -> None:
        self._settings.setValue("calculation/last_directory", str(path))
        self._push_recent(path)

    def recent_directories(self) -> list[Path]:
        raw = self._settings.value("calculation/recent", [])
        if isinstance(raw, str):
            raw = [raw]
        paths = [Path(str(item)) for item in (raw or [])]
        return [p for p in paths if p.is_dir()]

    def _push_recent(self, path: Path) -> None:
        recent = [str(p) for p in self.recent_directories() if p != path]
        recent.insert(0, str(path))
        self._settings.setValue("calculation/recent", recent[:MAX_RECENT])

    def clear_recent(self) -> None:
        self._settings.setValue("calculation/recent", [])

    # -- window ------------------------------------------------------------

    def window_geometry(self) -> QByteArray | None:
        value = self._settings.value("window/geometry")
        return value if isinstance(value, QByteArray) and not value.isEmpty() else None

    def set_window_geometry(self, geometry: QByteArray) -> None:
        self._settings.setValue("window/geometry", geometry)

    def window_state(self) -> QByteArray | None:
        value = self._settings.value("window/state")
        return value if isinstance(value, QByteArray) and not value.isEmpty() else None

    def set_window_state(self, state: QByteArray) -> None:
        self._settings.setValue("window/state", state)

    def sync(self) -> None:
        self._settings.sync()

    # -- element colours -----------------------------------------------------
    # A user override survives across calculations and sessions: once someone
    # picks a colour for carbon, every future structure with carbon in it uses
    # it, the same way an editor remembers a theme rather than a per-file
    # preference.

    def element_color(self, symbol: str) -> str | None:
        value = self._settings.value(f"structure/element_color/{symbol}")
        return str(value) if value else None

    def element_colors(self) -> dict[str, str]:
        """Every element with a stored override, keyed by symbol."""
        self._settings.beginGroup("structure/element_color")
        try:
            keys = self._settings.childKeys()
            return {key: str(self._settings.value(key)) for key in keys}
        finally:
            self._settings.endGroup()

    def set_element_color(self, symbol: str, hex_color: str) -> None:
        self._settings.setValue(f"structure/element_color/{symbol}", hex_color)

    def clear_element_color(self, symbol: str) -> None:
        self._settings.remove(f"structure/element_color/{symbol}")

    def clear_all_element_colors(self) -> None:
        self._settings.remove("structure/element_color")
