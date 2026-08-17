"""Application entry point."""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from .core.logging_setup import configure_logging, get_logger
from .ui.main_window import MainWindow
from .ui.plotting import apply_plot_style
from .ui.style import build_stylesheet


def main() -> int:
    configure_logging()
    log = get_logger("main")

    QApplication.setAttribute(Qt.AA_DontShowIconsInMenus, True)
    app = QApplication(sys.argv)
    app.setApplicationName("VASP Analyzer")
    app.setOrganizationName("VASPAnalyzer")
    app.setStyle("Fusion")  # a predictable base for the stylesheet
    app.setStyleSheet(build_stylesheet())

    apply_plot_style()

    window = MainWindow()
    window.show()
    log.info("Application started")
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
