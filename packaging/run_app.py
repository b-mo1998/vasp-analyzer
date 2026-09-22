"""PyInstaller entry point.

Not vasp_analyzer/__main__.py directly: PyInstaller runs its entry script as
a bare top-level module with no package context, and __main__.py's relative
imports (``from .ui...``) need one. This tiny wrapper sits outside the
package and imports it properly instead.
"""

import sys

from vasp_analyzer.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
