# VASP Analyzer

A desktop application for inspecting and visualising VASP and Wannier90
calculations.

## Running it

```bash
.venv/bin/python -m vasp_analyzer
```

Open a calculation with **⌘O**, or drag a folder onto the window. The last
directory reopens automatically.

## Requirements

The environment is already set up in `.venv`. It uses Homebrew Python 3.13
(arm64 — deliberately *not* the Anaconda Python, which is x86_64 and would run
the whole GUI under Rosetta).

| | |
|---|---|
| GUI | PySide6 6.11 |
| Plots | matplotlib (QtAgg) |
| Structure/symmetry | pymatgen-core + spglib |
| Numerics | NumPy, SciPy |

To rebuild the environment from scratch:

```bash
/opt/homebrew/opt/python@3.13/bin/python3.13 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

## Architecture

Four layers, dependencies pointing one way only (`ui → services → parsers → core`):

```
core/       errors, logging, shared primitives          (no Qt)
models/     typed scientific data + capabilities        (no Qt)
parsers/    one module per file format                  (no Qt)
services/   discovery, validation, loading, symmetry    (Qt signals)
ui/         theme, shell, reusable widgets, pages
```

No parsing logic lives in a widget, and no widget imports a file format.

### Capability-based discovery

Opening a directory runs a cheap scan — stat plus a few header reads — that
produces a `FileIndex` and a `CapabilitySet`. **The UI binds to capabilities,
never to filenames.** Every unavailable capability carries a reason, which the
interface shows as a tooltip or an inline note rather than a dialog:

```
DOS unavailable — DOSCAR was not found in this calculation.
```

Heavy data (EIGENVAL, DOSCAR, POSCAR) is read on a worker thread when a page
first needs it, and cached against `(path, mtime, size)`.

### Errors

Parsers return a `ParseResult` and never raise into the GUI. A truncated file
yields the part that was readable plus a warning. Short messages go to the
window; full tracebacks go to `~/Library/Logs/VASPAnalyzer/app.log`.

### Cross-file consistency

Calculation folders accumulate stale files. Before using a file's metadata, the
app checks it against the others — band and k-point counts, electron count,
atom count, elements. A file describing a different calculation is marked
inconsistent and ignored, and the mismatch is reported.

This is not hypothetical: the OUTCAR in the sample silicon folder came from an
unrelated 118-atom C/Te run and reports a Fermi level 7 eV away from the true
one.

### Scientific integrity

A quantity that cannot be determined shows `—`. Nothing is guessed.

- **Fermi level** — EIGENVAL does not store one. Taken from DOSCAR, then
  OUTCAR, each only when consistent with the rest of the directory.
- **k-path labels** — only from line-mode KPOINTS. A Monkhorst-Pack or
  Γ-centred mesh has no path, so the axis shows the k-point index and the
  eigenvalues are drawn as points: joining them with lines would assert an
  adjacency in reciprocal space that does not exist.
- **Functional** — reported only when the INCAR states it. A run with no
  exchange-correlation tag is not labelled "PBE".

## Pages

| Page | Status |
|---|---|
| Bands | Band structure, optional DOS panel on a shared energy axis, export |
| Structure | Interactive 3D viewer, symmetry, lattice information |
| Inputs | File browser, syntax-highlighted viewer, search, guarded editing |
| Wannier / Topology / Calculation | Future modules |

## Development

```bash
.venv/bin/python -m pytest tests/ -q
```

72 tests, covering every partial-calculation scenario (complete, bands-only,
no EIGENVAL, structure-only, partial Wannier, malformed DOSCAR, empty folder)
at both the service and the window level.

To capture the window without a display:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python tools/screenshot.py <calc-dir> out.png <page>
```

`<page>` is 0 for Bands, 1 for Structure, 2 for Inputs.
