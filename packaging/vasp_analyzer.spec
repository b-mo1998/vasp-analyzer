# PyInstaller build spec for VASP Analyzer.
#
# Run from the project root:
#   .venv/bin/pyinstaller packaging/vasp_analyzer.spec --noconfirm
#
# Produces dist/VASP Analyzer.app on macOS, or dist/VASP Analyzer/ on
# Windows/Linux. See install_guide.pdf for the full walkthrough (building,
# zipping, and what a recipient needs to do to run it).

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

PROJECT_ROOT = Path(SPECPATH).resolve().parent
ICON_ICNS = PROJECT_ROOT / "packaging" / "AppIcon.icns"
ICON_ICO = PROJECT_ROOT / "packaging" / "AppIcon.ico"

# The app resolves its own SVG icons relative to vasp_analyzer/resources at
# import time (see ui/style.py's ICON_DIR) -- bundling the folder at the same
# relative path means that lookup keeps working unchanged when frozen.
datas = [
    (str(PROJECT_ROOT / "vasp_analyzer" / "resources" / "icons"), "vasp_analyzer/resources/icons"),
]
# pymatgen ships its own reference data (periodic_table.json and friends)
# that vasp_analyzer's structure/symmetry code reads at runtime -- without
# this, imports succeed but element/symmetry lookups fail once frozen.
datas += collect_data_files("pymatgen")

a = Analysis(
    [str(PROJECT_ROOT / "packaging" / "run_app.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "matplotlib.backends.backend_qtagg",
        "matplotlib.backends.backend_qt5agg",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VASP Analyzer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(ICON_ICO) if sys.platform.startswith("win") and ICON_ICO.exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="VASP Analyzer",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="VASP Analyzer.app",
        icon=str(ICON_ICNS) if ICON_ICNS.exists() else None,
        bundle_identifier="com.vaspanalyzer.app",
        info_plist={
            "CFBundleName": "VASP Analyzer",
            "CFBundleDisplayName": "VASP Analyzer",
            "CFBundleShortVersionString": "0.1.0",
            "CFBundleVersion": "0.1.0",
            "NSHighResolutionCapable": True,
            "LSApplicationCategoryType": "public.app-category.education",
            "LSMinimumSystemVersion": "11.0",
        },
    )
