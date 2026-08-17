"""Development helper: boot the app headless and save a PNG of the window.

Used to review layout changes without a display. Not part of the application.

    QT_QPA_PLATFORM=offscreen .venv/bin/python tools/screenshot.py <dir> <out.png>
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from vasp_analyzer.core.logging_setup import configure_logging  # noqa: E402
from vasp_analyzer.ui.main_window import MainWindow  # noqa: E402
from vasp_analyzer.ui.plotting import apply_plot_style  # noqa: E402
from vasp_analyzer.ui.style import build_stylesheet  # noqa: E402


def main() -> int:
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("window.png")
    page = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    width = int(sys.argv[4]) if len(sys.argv) > 4 else 1440
    height = int(sys.argv[5]) if len(sys.argv) > 5 else 900

    configure_logging(console=False)
    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    app.setStyleSheet(build_stylesheet())
    apply_plot_style()

    window = MainWindow()
    window.resize(width, height)
    window.show()

    if directory is not None:
        window.open_directory(directory)
    if page:
        window._nav.setCurrentIndex(page)

    def capture() -> None:
        # Let queued work (background parsing, deferred redraws) settle first.
        app.processEvents()
        window.grab().save(str(output))
        print(f"saved {output}")
        app.quit()

    QTimer.singleShot(1500, capture)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
