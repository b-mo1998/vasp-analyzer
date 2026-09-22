# Building and sharing VASP Analyzer

For the maintainer. `install_guide.pdf` is the recipient-facing document and
deliberately contains none of this.

The app is packaged with [PyInstaller](https://pyinstaller.org), which bundles
the Python interpreter, PySide6/Qt and every other dependency into a native
application. The build spec is `packaging/vasp_analyzer.spec`.

## macOS

```bash
cd /path/to/vasp-analyzer
.venv/bin/pip install -e ".[build]"
.venv/bin/pyinstaller packaging/vasp_analyzer.spec --noconfirm --clean
```

This produces `dist/VASP Analyzer.app`.

Zip it with `ditto`, not Finder's *Compress* — Finder's version mangles the
`.app`'s internal structure on some macOS releases, producing a bundle that
will not launch on the recipient's machine:

```bash
cd dist
ditto -c -k --sequesterRsrc --keepParent "VASP Analyzer.app" "VASP Analyzer-macOS.zip"
```

## Architecture: this is not a universal binary

PyInstaller builds for the architecture it runs on. A build made on an Apple
Silicon Mac produces an **arm64-only** app that will not launch on an Intel
Mac, and vice versa. Verify what you have built before sharing it:

```bash
file "dist/VASP Analyzer.app/Contents/MacOS/VASP Analyzer"
```

To support Intel Macs you have to build separately on (or for) x86_64. The
recipient-facing `install_guide.pdf` states the Apple Silicon requirement
explicitly, so update it if that ever changes.

## Windows

PyInstaller does not cross-compile. A Windows build must be produced on an
actual Windows machine or VM:

```
py -m venv .venv
.venv\Scripts\pip install -e ".[build]"
.venv\Scripts\pyinstaller packaging\vasp_analyzer.spec --noconfirm --clean
```

This produces `dist\VASP Analyzer\`. Zip the **whole folder** — the `.exe`
alone will not run without the files beside it.

On first launch Windows SmartScreen will warn about an unrecognised app;
*More info* → *Run anyway* proceeds.

## Version bumps

Two places, and they should agree:

- the `version` field in `pyproject.toml`
- `CFBundleVersion` and `CFBundleShortVersionString` in
  `packaging/vasp_analyzer.spec`

## Adding bundled resources

Anything read from disk at runtime (icons, data files) must be listed in the
spec's `datas` list, or it will be missing from the frozen app while continuing
to work fine when run from source — a failure mode that only shows up after
packaging.

## Signing and notarisation (optional)

Unsigned apps require the recipient to right-click → Open on first launch, and
occasionally show a misleading "damaged" message that `xattr -cr` clears. To
remove that friction entirely:

| Platform | What is needed | Approximate cost |
|---|---|---|
| macOS | Apple Developer ID certificate, then `codesign` + `notarytool` | $99/year |
| Windows | Authenticode code-signing certificate | $100–400/year |

## Release checklist

1. `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/ -q` — all green.
2. Rebuild the bundle (above).
3. Smoke-test the **frozen** app, not just the source tree:
   `"dist/VASP Analyzer.app/Contents/MacOS/VASP Analyzer" /path/to/a/calculation`
   then check `~/Library/Logs/VASPAnalyzer/app.log` for the expected parses and
   no tracebacks.
4. Re-zip with `ditto`.
5. Regenerate the PDFs if any feature or shortcut changed.
