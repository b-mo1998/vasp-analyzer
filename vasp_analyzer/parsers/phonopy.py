"""Phonopy output files: ``band.yaml``, ``mesh.yaml``, DOS tables and metadata.

These are YAML, but they are not read with a YAML library. ``band.yaml`` with
eigenvectors is routinely hundreds of megabytes — a 96-atom cell over a
400-point path is 288 branches x 96 atoms x 3 components x 2 (complex) per
q-point — and a pure-Python YAML parser takes minutes over it while building
an intermediate dict of every scalar. Phonopy writes these files from a fixed
template, so a line-oriented reader over that known layout is both far faster
and, being strict about the shape it expects, no less safe: a file that does
not match raises rather than silently yielding half a dataset.

The layout consumed here is::

    phonon:
    - q-position: [ 0.0, 0.0, 0.0 ]
      distance: 0.0000000
      label: '\\Gamma'
      band:
      - # 1
        frequency: -0.0000021
        eigenvector:
        - # atom 1
          - [ 0.1, 0.0 ]      # x: real, imaginary
          - [ 0.0, 0.0 ]      # y
          - [ 0.0, 0.0 ]      # z

Eigenvectors from phonopy are mass-weighted; they are divided by sqrt(mass)
here so that what the rest of the application holds is a displacement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.errors import Issue, ParseResult, Severity, safe_parser
from ..core.logging_setup import get_logger
from ..models.phonon import PhononBand, PhononDos
from ..resources.elements import atomic_mass

log = get_logger("parsers.phonopy")

# Greek and other TeX-style labels phonopy writes into band.yaml.
_LABEL_REPLACEMENTS = {
    r"\Gamma": "Γ",
    r"$\Gamma$": "Γ",
    "GAMMA": "Γ",
    r"\Sigma": "Σ",
    r"\Delta": "Δ",
    r"\Lambda": "Λ",
}

_NUMBER = r"[-+0-9.eEdD]+"
_TRIPLE = re.compile(
    rf"\[\s*({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\]"
)
_PAIR = re.compile(rf"\[\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\]")
_LABEL_PAIR = re.compile(r"['\"]([^'\"]*)['\"]\s*,\s*['\"]([^'\"]*)['\"]")


def _clean_label(raw: str) -> str:
    label = raw.strip().strip("'\"").strip()
    for needle, replacement in _LABEL_REPLACEMENTS.items():
        if label == needle:
            return replacement
    return label


def _floats(text: str) -> list[float]:
    return [float(t.replace("D", "E").replace("d", "e")) for t in text]


@dataclass
class PhononBandFile:
    """What a band.yaml / mesh.yaml carried."""

    bands: list[PhononBand] = field(default_factory=list)
    n_atoms: int = 0
    lattice: np.ndarray | None = None
    species: list[str] = field(default_factory=list)
    masses: list[float] = field(default_factory=list)
    positions: list[list[float]] = field(default_factory=list)
    supercell: tuple[int, int, int] | None = None
    mesh: tuple[int, int, int] | None = None
    weights: list[float] = field(default_factory=list)


def _parse_header(lines: list[str]) -> dict:
    """Read the scalar preamble that precedes the ``phonon:`` block.

    Also reads the ``labels:`` / ``segment_nqpoint:`` pair phonopy uses to
    name high-symmetry points on a band path — one ``[start, end]`` label
    pair and a point count per segment, rather than a label on every
    q-point. Consecutive segments share their boundary q-point (real
    phonopy output duplicates that point's coordinates rather than omitting
    it), which is what lets a straight-through point like Γ-M-K's M end up
    labelled from *both* the segment that ends there and the one that
    starts there.
    """
    header: dict = {}
    species: list[str] = []
    masses: list[float] = []
    positions: list[list[float]] = []
    lattice: list[list[float]] = []
    segment_labels: list[tuple[str, str]] = []
    segment_counts: list[int] = []
    in_points = False
    in_lattice = False
    in_labels = False
    in_segments = False

    for raw in lines:
        line = raw.strip()
        if line.startswith("phonon:"):
            break

        if line.startswith("natom:"):
            header["natom"] = int(float(line.split(":", 1)[1]))
            in_labels = in_segments = False
        elif line.startswith("nqpoint:"):
            header["nqpoint"] = int(float(line.split(":", 1)[1]))
        elif line.startswith("segment_nqpoint:"):
            in_segments, in_labels = True, False
            segment_counts = []
        elif line.startswith("labels:"):
            in_labels, in_segments = True, False
            segment_labels = []
        elif line.startswith("supercell_matrix:"):
            in_lattice = in_labels = in_segments = False
        elif line.startswith("mesh:"):
            match = _TRIPLE.search(line)
            if match:
                header["mesh"] = tuple(int(float(v)) for v in match.groups())
            in_labels = in_segments = False
        elif line.startswith("reciprocal_lattice:"):
            in_labels = in_segments = False
        elif line.startswith("lattice:"):
            in_lattice, in_points = True, False
            in_labels = in_segments = False
            lattice = []
        elif line.startswith("points:") or line.startswith("atoms:"):
            in_points, in_lattice = True, False
            in_labels = in_segments = False
        elif in_segments and line.startswith("-"):
            try:
                segment_counts.append(int(float(line[1:].strip())))
            except ValueError:
                in_segments = False
        elif in_labels and line.startswith("-"):
            match = _LABEL_PAIR.search(line)
            if match:
                segment_labels.append((match.group(1), match.group(2)))
            else:
                in_labels = False
        elif in_lattice and line.startswith("- ["):
            match = _TRIPLE.search(line)
            if match:
                lattice.append(_floats(match.groups()))
            if len(lattice) == 3:
                in_lattice = False
        elif in_points:
            if "symbol:" in line:
                species.append(line.split("symbol:", 1)[1].split("#")[0].strip())
            elif line.startswith("mass:"):
                masses.append(float(line.split(":", 1)[1]))
            elif line.startswith("coordinates:") or line.startswith("position:"):
                match = _TRIPLE.search(line)
                if match:
                    positions.append(_floats(match.groups()))

    if lattice:
        header["lattice"] = np.array(lattice, dtype=float)
    header["species"] = species
    header["masses"] = masses
    header["positions"] = positions
    header["segment_labels"] = segment_labels
    header["segment_counts"] = segment_counts
    return header


def _supercell_from_matrix(lines: list[str]) -> tuple[int, int, int] | None:
    """The diagonal of ``supercell_matrix``, when it is diagonal."""
    rows: list[list[float]] = []
    collecting = False
    for raw in lines:
        line = raw.strip()
        if line.startswith("supercell_matrix:"):
            collecting = True
            continue
        if collecting:
            match = _TRIPLE.search(line)
            if match:
                rows.append(_floats(match.groups()))
                if len(rows) == 3:
                    break
            elif line and not line.startswith("-"):
                break
    if len(rows) != 3:
        return None
    matrix = np.array(rows)
    off_diagonal = matrix - np.diag(np.diag(matrix))
    if np.abs(off_diagonal).max() > 1e-9:
        # A non-diagonal supercell has no "a x b x c" description.
        return None
    return tuple(int(round(v)) for v in np.diag(matrix))  # type: ignore[return-value]


@safe_parser("band.yaml")
def parse_band_yaml(path: Path, want_eigenvectors: bool = True) -> ParseResult[PhononBandFile]:
    """Read a phonopy ``band.yaml`` or ``mesh.yaml``."""
    text = Path(path).read_text(errors="replace")
    lines = text.splitlines()

    header = _parse_header(lines)
    natom = int(header.get("natom", 0))
    masses = header.get("masses") or []
    issues: list[Issue] = []

    result = PhononBandFile(
        n_atoms=natom,
        lattice=header.get("lattice"),
        species=list(header.get("species") or []),
        masses=list(masses),
        positions=list(header.get("positions") or []),
        supercell=_supercell_from_matrix(lines),
        mesh=header.get("mesh"),
    )

    # Fall back to the periodic table when the file omitted masses, so
    # de-mass-weighting still produces real displacements.
    if len(masses) != natom and result.species:
        masses = [atomic_mass(symbol) for symbol in result.species]
    inv_sqrt_mass = (
        1.0 / np.sqrt(np.array(masses, dtype=float)) if len(masses) == natom else None
    )
    if inv_sqrt_mass is None and natom:
        issues.append(
            Issue(
                code="phonon_masses_missing",
                message=(
                    "Atomic masses were not found, so eigenvectors are shown "
                    "mass-weighted rather than as displacements."
                ),
                severity=Severity.WARNING,
                source=Path(path).name,
            )
        )

    bands: list[PhononBand] = []
    qpoint: np.ndarray | None = None
    distance = 0.0
    label = ""
    weight = 1.0
    frequencies: list[float] = []
    vectors: list[list[list[complex]]] = []
    atom_rows: list[list[complex]] = []
    current_atom: list[complex] = []
    in_eigenvector = False

    def flush_band() -> None:
        nonlocal current_atom, atom_rows
        if current_atom:
            atom_rows.append(list(current_atom))
            current_atom = []
        if atom_rows:
            vectors.append(list(atom_rows))
            atom_rows = []

    def flush_qpoint() -> None:
        nonlocal qpoint, frequencies, vectors, label, weight
        if qpoint is None or not frequencies:
            return
        eigen = None
        if want_eigenvectors and vectors and len(vectors) == len(frequencies):
            array = np.array(vectors, dtype=complex)   # (branch, atom, 3)
            if inv_sqrt_mass is not None and array.shape[1] == inv_sqrt_mass.size:
                array = array * inv_sqrt_mass[None, :, None]
            eigen = array
        bands.append(
            PhononBand(
                qpoint=qpoint,
                distance=distance,
                frequencies=np.array(frequencies, dtype=float),
                eigenvectors=eigen,
                label=label,
            )
        )
        result.weights.append(weight)
        qpoint, frequencies, vectors, label, weight = None, [], [], "", 1.0

    started = False
    for raw in lines:
        line = raw.strip()
        if not started:
            if line.startswith("phonon:"):
                started = True
            continue

        if line.startswith("- q-position:"):
            flush_band()
            flush_qpoint()
            match = _TRIPLE.search(line)
            if match:
                qpoint = np.array(_floats(match.groups()))
            in_eigenvector = False
        elif line.startswith("distance:"):
            distance = float(line.split(":", 1)[1])
        elif line.startswith("weight:"):
            weight = float(line.split(":", 1)[1])
        elif line.startswith("label:"):
            label = _clean_label(line.split(":", 1)[1])
        elif line.startswith("frequency:"):
            flush_band()
            in_eigenvector = False
            frequencies.append(float(line.split(":", 1)[1]))
        elif line.startswith("eigenvector:"):
            in_eigenvector = True
            atom_rows = []
            current_atom = []
        elif in_eigenvector and line.startswith("- # atom"):
            if current_atom:
                atom_rows.append(list(current_atom))
                current_atom = []
        elif in_eigenvector and line.startswith("- ["):
            match = _PAIR.search(line)
            if match:
                real, imaginary = _floats(match.groups())
                current_atom.append(complex(real, imaginary))
                if len(current_atom) == 3:
                    atom_rows.append(list(current_atom))
                    current_atom = []

    flush_band()
    flush_qpoint()

    if not bands:
        return ParseResult.failure(
            "phonon_no_qpoints",
            f"{Path(path).name} contains no q-points.",
            source=Path(path).name,
        )

    # No per-q-point `label:` lines were found (the modern phonopy format
    # never writes them) — fall back to the file's `labels:`/
    # `segment_nqpoint:` header, which names each segment's two endpoints
    # rather than every point.
    if not any(band.label for band in bands):
        segment_labels = header.get("segment_labels") or []
        segment_counts = header.get("segment_counts") or []
        if segment_labels and len(segment_labels) == len(segment_counts):
            index = 0
            for (start, end), count in zip(segment_labels, segment_counts):
                if count <= 0 or index >= len(bands):
                    break
                bands[index].label = _clean_label(start)
                end_index = min(index + count - 1, len(bands) - 1)
                bands[end_index].label = _clean_label(end)
                index += count

    # Every q-point must describe the same number of branches, or the arrays
    # the page builds from this would be ragged.
    counts = {band.frequencies.size for band in bands}
    if len(counts) > 1:
        return ParseResult.failure(
            "phonon_ragged_branches",
            (
                f"{Path(path).name} reports differing branch counts across "
                f"q-points ({sorted(counts)})."
            ),
            source=Path(path).name,
        )

    n_branches = bands[0].frequencies.size
    if natom and n_branches != 3 * natom:
        issues.append(
            Issue(
                code="phonon_branch_count",
                message=(
                    f"{Path(path).name} has {n_branches} branches for "
                    f"{natom} atoms; {3 * natom} were expected."
                ),
                severity=Severity.WARNING,
                source=Path(path).name,
            )
        )

    result.bands = bands
    log.info(
        "Parsed %s: %d q-points, %d branches, eigenvectors=%s",
        Path(path).name,
        len(bands),
        n_branches,
        bands[0].eigenvectors is not None,
    )
    return ParseResult.success(result, issues)


@safe_parser("total_dos.dat")
def parse_dos_dat(path: Path) -> ParseResult[PhononDos]:
    """Read phonopy's ``total_dos.dat`` / ``projected_dos.dat``.

    Both are plain whitespace columns: frequency first, then one column of
    DOS for total, or one per projection.
    """
    rows: list[list[float]] = []
    for raw in Path(path).read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rows.append([float(token) for token in line.split()])
        except ValueError:
            continue

    if not rows or len(rows[0]) < 2:
        return ParseResult.failure(
            "phonon_dos_empty",
            f"{Path(path).name} contains no usable DOS rows.",
            source=Path(path).name,
        )

    width = min(len(row) for row in rows)
    table = np.array([row[:width] for row in rows], dtype=float)
    frequencies = table[:, 0]
    columns = table[:, 1:]

    dos = PhononDos(frequencies=frequencies, total=columns.sum(axis=1))
    log.info(
        "Parsed %s: %d points, %d column(s)",
        Path(path).name,
        frequencies.size,
        columns.shape[1],
    )
    return ParseResult.success(dos)


def attach_projected_dos(dos: PhononDos, path: Path, species: list[str]) -> list[Issue]:
    """Fold ``projected_dos.dat`` into an existing DOS as per-element curves.

    Phonopy writes one column per *atom*; atoms of the same element are summed
    here, because a legend with 96 entries is not a legend.
    """
    issues: list[Issue] = []
    rows: list[list[float]] = []
    for raw in Path(path).read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rows.append([float(token) for token in line.split()])
        except ValueError:
            continue
    if not rows:
        return issues

    width = min(len(row) for row in rows)
    table = np.array([row[:width] for row in rows], dtype=float)
    columns = table[:, 1:]

    if columns.shape[1] != len(species):
        issues.append(
            Issue(
                code="phonon_pdos_mismatch",
                message=(
                    f"projected_dos.dat has {columns.shape[1]} columns but the "
                    f"structure has {len(species)} atoms; the projection was "
                    "not applied."
                ),
                severity=Severity.WARNING,
                source="projected_dos.dat",
            )
        )
        return issues

    if not np.allclose(table[:, 0], dos.frequencies, atol=1e-6):
        issues.append(
            Issue(
                code="phonon_pdos_grid",
                message=(
                    "projected_dos.dat is sampled on a different frequency grid "
                    "from total_dos.dat; the projection was not applied."
                ),
                severity=Severity.WARNING,
                source="projected_dos.dat",
            )
        )
        return issues

    for index, symbol in enumerate(species):
        current = dos.projected.get(symbol)
        contribution = columns[:, index]
        dos.projected[symbol] = (
            contribution if current is None else current + contribution
        )
    return issues


def read_calculation_kind(path: Path) -> str:
    """Describe how the force constants were obtained, from phonopy.yaml."""
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return ""
    lowered = text.lower()
    if "dfpt" in lowered or "linear_response" in lowered:
        return "DFPT (Phonopy)"
    if "displacement" in lowered:
        return "Finite displacement (Phonopy)"
    return "Phonopy"
