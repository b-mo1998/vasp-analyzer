"""Application logging.

Detailed technical output (tracebacks, per-file parse notes) goes to a rotating
log under ``~/Library/Logs/VASPAnalyzer``. The UI only ever shows the short
``Issue.message`` text.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

APP_NAME = "VASPAnalyzer"
_configured = False


def log_directory() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / APP_NAME
    return Path.home() / ".local" / "state" / APP_NAME


def log_file() -> Path:
    return log_directory() / "app.log"


def configure_logging(level: int = logging.INFO, console: bool = True) -> Path:
    """Install file + console handlers once. Returns the log file path."""
    global _configured
    path = log_file()
    if _configured:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("vasp_analyzer")
    root.setLevel(logging.DEBUG)
    root.propagate = False

    file_handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-7s %(name)-28s %(message)s")
    )
    root.addHandler(file_handler)

    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setLevel(level)
        stream.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
        root.addHandler(stream)

    _configured = True
    root.info("Logging initialised -> %s", path)
    return path


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"vasp_analyzer.{name}")
