"""Application entry point."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from .core.logging_setup import configure_logging, get_logger
from .ui.main_window import MainWindow
from .ui.plotting import apply_plot_style
from .ui.style import apply_theme


def main() -> int:
    configure_logging()
    log = get_logger("main")

    QApplication.setAttribute(Qt.AA_DontShowIconsInMenus, True)
    app = QApplication(sys.argv)
    app.setApplicationName("VASP Analyzer")
    app.setOrganizationName("VASPAnalyzer")
    apply_theme(app)
    apply_plot_style()

    window = MainWindow()
    window.show()

    # A directory argument here means "Reload Calculation" restarted this
    # process (see MainWindow.restart_argv) and wants the same calculation
    # reopened, rather than landing on a blank window. Scanned rather than
    # read from a fixed position: the launcher differs between a frozen
    # bundle and a `python -m` run, and a Finder "Open With" can prepend
    # arguments of its own.
    for argument in sys.argv[1:]:
        target = Path(argument)
        if target.is_dir():
            window.open_directory(target)
            break
        if target.is_file():
            window.open_file(target)
            break

    log.info("Application started")
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
