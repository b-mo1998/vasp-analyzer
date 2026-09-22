"""Discovering and assembling a phonon calculation.

A phonon run is a *directory* of phonopy outputs rather than a single file, and
which capabilities the page can offer depends on which of them are present:
``band.yaml`` alone gives a dispersion, ``total_dos.dat`` adds a DOS,
eigenvectors in the band file add mode animation. Each is checked separately so
a partial calculation loses only what it actually lacks.

Phonon files may live in the calculation directory itself or in a ``phonon``/
``phonons`` subdirectory, which is how they are usually kept — the force
constants come from a different run than the electronic structure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.errors import Issue, ParseResult, Severity
from ..core.logging_setup import get_logger
from ..models.phonon import PhononDataset, PhononDos
from ..models.structure import StructureData
from ..parsers.phonopy import (
    attach_projected_dos,
    parse_band_yaml,
    parse_dos_dat,
    read_calculation_kind,
)
from ..parsers.poscar import parse_poscar

log = get_logger("services.phonon")

# How many directory levels below the opened folder to look for phonon
# output. A real calculation commonly keeps the raw VASP DFPT run and
# Phonopy's post-processed results (dispersion, mesh, DOS) in separately
# named folders rather than at one guessable path — e.g. a "phonon"
# subdirectory holding the DFPT INCAR/OUTCAR, with a "dispersion"
# subdirectory *inside that* holding band.yaml — so a fixed list of expected
# subdirectory names is not enough; this searches broadly but shallowly
# instead.
PHONON_SEARCH_DEPTH = 2

BAND_FILES = ("band.yaml", "band.yml")
MESH_FILES = ("mesh.yaml", "mesh.yml")
DOS_FILES = ("total_dos.dat", "dos.dat")
PROJECTED_DOS_FILES = ("projected_dos.dat", "partial_dos.dat")
META_FILES = ("phonopy.yaml", "phonopy_disp.yaml")
FORCE_FILES = ("FORCE_CONSTANTS", "FORCE_SETS")


def _first_existing(directory: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = directory / name
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def _first_nearby(directory: Path, names: tuple[str, ...]) -> Path | None:
    """Like :func:`_first_existing`, but also checks immediate subdirectories.

    Phonopy output routinely ends up split across folders — a band.yaml
    copied up beside the VASP run while the mesh and DOS stay in the
    post-processing directory that produced them. The band file defines
    *which* directory is the phonon root; these supplementary files are only
    ever additive, so looking one level down for them can add capability but
    can never change which calculation is being shown.
    """
    direct = _first_existing(directory, names)
    if direct is not None:
        return direct
    try:
        children = sorted(
            p for p in directory.iterdir() if p.is_dir() and not p.name.startswith(".")
        )
    except OSError:
        return None
    for child in children:
        found = _first_existing(child, names)
        if found is not None:
            return found
    return None


def phonon_root(directory: Path) -> Path | None:
    """Where this calculation's phonon files live, if anywhere.

    The directory itself wins over anything nested inside it — a run whose
    band.yaml sits beside its POSCAR is describing itself, not a nested
    calculation. Failing that, subdirectories are searched breadth-first up
    to :data:`PHONON_SEARCH_DEPTH` levels down, so it does not matter what
    the phonon output folder happens to be named.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None
    return _search_for_phonons(directory, PHONON_SEARCH_DEPTH)


def _search_for_phonons(directory: Path, depth: int) -> Path | None:
    if _looks_like_phonon(directory):
        return directory
    if depth <= 0:
        return None
    try:
        children = sorted(
            p for p in directory.iterdir() if p.is_dir() and not p.name.startswith(".")
        )
    except OSError:
        return None
    for child in children:
        found = _search_for_phonons(child, depth - 1)
        if found is not None:
            return found
    return None


def _looks_like_phonon(directory: Path) -> bool:
    return any(
        _first_existing(directory, names) is not None
        for names in (BAND_FILES, MESH_FILES, DOS_FILES, FORCE_FILES)
    )


def describe_sources(directory: Path) -> list[str]:
    """Which phonon files were actually used, for the footer.

    Uses the same nearby lookup the loader does, so the footer lists the
    files the page is really reading rather than only those sitting directly
    in the root.
    """
    found: list[str] = []
    for names in (BAND_FILES, MESH_FILES, DOS_FILES, PROJECTED_DOS_FILES, META_FILES, FORCE_FILES):
        path = _first_nearby(directory, names)
        if path is not None:
            found.append(path.name)
    return found


def load_phonons(directory: Path) -> ParseResult[PhononDataset]:
    """Assemble everything the Phonons page needs from a directory."""
    directory = Path(directory)
    issues: list[Issue] = []
    dataset = PhononDataset(directory_name=directory.name)

    band_path = _first_existing(directory, BAND_FILES) or _first_existing(
        directory, MESH_FILES
    )
    if band_path is None:
        return ParseResult.failure(
            "phonon_no_band_file",
            "No band.yaml or mesh.yaml was found in this calculation.",
            source=directory.name,
        )

    result = parse_band_yaml(band_path)
    issues.extend(result.issues)
    if not result.ok or result.value is None:
        return ParseResult.failure(
            "phonon_band_unreadable",
            f"{band_path.name} could not be read.",
            source=band_path.name,
        )

    band_file = result.value
    dataset.bands = band_file.bands
    dataset.supercell = band_file.supercell
    dataset.mesh = band_file.mesh
    dataset.mesh_weights = np.array(band_file.weights) if band_file.weights else None

    # --- uniform mesh, for the DOS -----------------------------------------
    # A band path is a set of high-symmetry *lines*, which is the wrong
    # sample to build a density of states from — it over-weights those lines
    # and produces peaks the real DOS does not have. mesh.yaml is phonopy's
    # uniform Brillouin-zone sampling with per-point weights, so it is read
    # separately whenever it exists, purely as the DOS source. Eigenvectors
    # are skipped: only frequencies and weights are needed here, and asking
    # for them would multiply the parse cost for nothing.
    mesh_path = _first_nearby(directory, MESH_FILES)
    if mesh_path is not None and mesh_path != band_path:
        mesh_result = parse_band_yaml(mesh_path, want_eigenvectors=False)
        if mesh_result.ok and mesh_result.value is not None:
            mesh_file = mesh_result.value
            dataset.mesh_frequencies = np.array(
                [band.frequencies for band in mesh_file.bands]
            )
            if mesh_file.weights:
                dataset.mesh_weights = np.array(mesh_file.weights)
            if mesh_file.mesh:
                dataset.mesh = mesh_file.mesh

    # --- structure ---------------------------------------------------------
    # The band file carries its own cell; a POSCAR beside it is preferred only
    # when the two agree, because a mismatched POSCAR would put the animation's
    # atoms in different places from the eigenvectors describing them.
    structure = _structure_from_band(band_file)
    poscar = _first_existing(directory, ("POSCAR", "POSCAR-unitcell", "SPOSCAR"))
    if poscar is not None:
        parsed = parse_poscar(poscar, source=poscar.name)
        if parsed.ok and parsed.value is not None:
            if structure is None or parsed.value.n_atoms == band_file.n_atoms:
                structure = parsed.value
            else:
                issues.append(
                    Issue(
                        code="phonon_structure_mismatch",
                        message=(
                            f"{poscar.name} has {parsed.value.n_atoms} atoms but "
                            f"{band_path.name} describes {band_file.n_atoms}; the "
                            "phonon file's own cell was used."
                        ),
                        severity=Severity.WARNING,
                        source=poscar.name,
                    )
                )
    dataset.structure = structure

    # --- DOS ---------------------------------------------------------------
    dos_path = _first_nearby(directory, DOS_FILES)
    projected = _first_nearby(directory, PROJECTED_DOS_FILES)
    if dos_path is None and projected is not None:
        # phonopy's --pdos run writes projected_dos.dat without a separate
        # total_dos.dat alongside it — the total is just the per-atom columns
        # summed, which parse_dos_dat already does when read as a plain
        # table. Reading it as both the total *and* the projection source
        # means a phonon run that only asked for the projected DOS still
        # gets a total curve instead of showing no DOS at all.
        dos_path = projected

    if dos_path is not None:
        dos_result = parse_dos_dat(dos_path)
        issues.extend(dos_result.issues)
        if dos_result.ok and dos_result.value is not None:
            dataset.dos = dos_result.value
            species = list(structure.species) if structure is not None else band_file.species
            if projected is not None and species:
                issues.extend(
                    attach_projected_dos(dataset.dos, projected, species)
                )

    # --- metadata ----------------------------------------------------------
    meta = _first_nearby(directory, META_FILES)
    dataset.calculation = read_calculation_kind(meta) if meta else "Phonopy"
    dataset.source_files = describe_sources(directory)

    _validate(dataset, band_file.n_atoms, issues)
    log.info(
        "Loaded phonons from %s: %d q-points, %d branches, dos=%s, eigenvectors=%s",
        directory.name,
        dataset.n_qpoints,
        dataset.n_branches,
        dataset.dos is not None,
        dataset.has_eigenvectors,
    )
    return ParseResult.success(dataset, issues)


def _structure_from_band(band_file) -> StructureData | None:
    """The unit cell phonopy recorded in the band file itself."""
    if band_file.lattice is None or not band_file.species:
        return None
    # band.yaml lists coordinates per point; when they were not captured the
    # atoms are placed at the origin, which is useless for animation.
    positions = getattr(band_file, "positions", None)
    if positions is None:
        return None
    return StructureData(
        lattice=np.array(band_file.lattice, dtype=float),
        species=list(band_file.species),
        frac_coords=np.array(positions, dtype=float),
        comment="phonon cell",
        source="band.yaml",
    )


def _validate(dataset: PhononDataset, n_atoms: int, issues: list[Issue]) -> None:
    """Check the pieces describe the same calculation."""
    if dataset.structure is not None and n_atoms and dataset.structure.n_atoms != n_atoms:
        issues.append(
            Issue(
                code="phonon_atom_count",
                message=(
                    f"The structure has {dataset.structure.n_atoms} atoms but the "
                    f"phonon data describes {n_atoms}."
                ),
                severity=Severity.WARNING,
            )
        )
    if dataset.has_eigenvectors:
        eigen = dataset.bands[0].eigenvectors
        expected = (dataset.n_branches, dataset.n_atoms, 3)
        if eigen is not None and eigen.shape != expected:
            issues.append(
                Issue(
                    code="phonon_eigenvector_shape",
                    message=(
                        f"Eigenvectors have shape {eigen.shape}, expected {expected}; "
                        "mode animation was disabled."
                    ),
                    severity=Severity.WARNING,
                )
            )
            for band in dataset.bands:
                band.eigenvectors = None


# -- density of states ---------------------------------------------------------


def gaussian_dos(
    frequencies: np.ndarray,
    smearing: float,
    weights: np.ndarray | None = None,
    n_points: int = 400,
    padding: float = 1.0,
) -> PhononDos:
    """Build a DOS from sampled frequencies by Gaussian broadening.

    Recomputed from the frequencies rather than re-broadening a DOS that was
    already broadened: convolving twice would widen every peak by more than
    the width the user asked for, and the number on the slider would stop
    meaning what it says.
    """
    values = np.asarray(frequencies, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return PhononDos(frequencies=np.zeros(0), total=np.zeros(0))

    sigma = max(float(smearing), 1e-3)
    grid = np.linspace(values.min() - padding, values.max() + padding, n_points)

    if weights is None:
        sample_weights = np.ones(values.size)
    else:
        sample_weights = np.repeat(
            np.asarray(weights, dtype=float), values.size // max(len(weights), 1)
        )[: values.size]

    # (n_points, n_samples) would be large for a dense mesh; accumulate instead.
    total = np.zeros_like(grid)
    norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    for value, weight in zip(values, sample_weights):
        total += weight * norm * np.exp(-0.5 * ((grid - value) / sigma) ** 2)

    if sample_weights.sum() > 0:
        total /= sample_weights.sum() / values.size
    return PhononDos(frequencies=grid, total=total)


def project_dos_by_element(
    dataset: PhononDataset, smearing: float, n_points: int = 400
) -> PhononDos | None:
    """A DOS split by element, weighted by each element's share of |e|².

    Requires eigenvectors: without them there is no way to say which atoms a
    given mode moves, and a "projection" would just be the total scaled by
    stoichiometry, which says nothing.
    """
    if not dataset.has_eigenvectors or dataset.structure is None:
        return None

    species = dataset.structure.species
    frequencies = dataset.frequency_matrix.ravel()
    base = gaussian_dos(frequencies, smearing, n_points=n_points)
    grid = base.frequencies
    if grid.size == 0:
        return None

    sigma = max(float(smearing), 1e-3)
    norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    elements: list[str] = []
    for symbol in species:
        if symbol not in elements:
            elements.append(symbol)
    projected = {element: np.zeros_like(grid) for element in elements}

    for band in dataset.bands:
        if band.eigenvectors is None:
            continue
        # |e|² per atom, normalised per branch: the fraction of the mode that
        # lives on each atom.
        weight = np.abs(band.eigenvectors) ** 2          # (branch, atom, 3)
        per_atom = weight.sum(axis=2)                    # (branch, atom)
        totals = per_atom.sum(axis=1, keepdims=True)
        per_atom = np.divide(
            per_atom, totals, out=np.zeros_like(per_atom), where=totals > 1e-12
        )
        for branch, frequency in enumerate(band.frequencies):
            shape = norm * np.exp(-0.5 * ((grid - frequency) / sigma) ** 2)
            for index, symbol in enumerate(species):
                projected[symbol] += per_atom[branch, index] * shape

    return PhononDos(frequencies=grid, total=base.total, projected=projected)


# -- mode animation ------------------------------------------------------------


def mode_displacements(
    dataset: PhononDataset,
    qpoint_index: int,
    branch_index: int,
    phases: np.ndarray,
    amplitude: float = 1.0,
    supercell: tuple[int, int, int] = (1, 1, 1),
) -> np.ndarray | None:
    """Cartesian atom positions through one oscillation of the selected mode.

    Returns ``(n_phases, n_atoms_in_supercell, 3)`` in Å.

    The displacement of atom *j* in cell **R** is

        u = A · Re[ e_j · exp(i(q·R − ωt)) ]

    The ``q·R`` term is what makes a non-Γ mode a *wave*: without it every
    repeated cell would move identically and a zone-boundary mode would look
    exactly like a Γ mode. That is why a supercell is worth drawing at all.
    """
    if not dataset.has_eigenvectors or dataset.structure is None:
        return None
    if not (0 <= qpoint_index < dataset.n_qpoints):
        return None
    band = dataset.bands[qpoint_index]
    if band.eigenvectors is None or not (0 <= branch_index < band.eigenvectors.shape[0]):
        return None

    structure = dataset.structure
    eigenvector = band.eigenvectors[branch_index]      # (n_atoms, 3) complex
    qpoint = band.qpoint

    na, nb, nc = (max(1, int(v)) for v in supercell)
    offsets = np.array(
        [[i, j, k] for i in range(na) for j in range(nb) for k in range(nc)],
        dtype=float,
    )

    base_positions = []
    cell_phases = []
    for offset in offsets:
        base_positions.append((structure.frac_coords + offset) @ structure.lattice)
        cell_phases.append(np.exp(2j * np.pi * float(np.dot(qpoint, offset))))
    base = np.concatenate(base_positions)               # (n_cells*n_atoms, 3)
    cell_phase = np.repeat(np.array(cell_phases), structure.n_atoms)

    # Normalise so the largest displacement is exactly `amplitude`, making the
    # slider mean the same thing for every mode regardless of how phonopy
    # happened to normalise the eigenvector.
    largest = float(np.abs(eigenvector).max())
    scale = amplitude / largest if largest > 1e-12 else 0.0

    tiled = np.tile(eigenvector, (len(offsets), 1))     # (n_total, 3)
    weighted = tiled * cell_phase[:, None]

    frames = np.empty((len(phases), base.shape[0], 3))
    for index, phase in enumerate(phases):
        displacement = np.real(weighted * np.exp(-1j * phase)) * scale
        frames[index] = base + displacement
    return frames


def static_displacement(
    dataset: PhononDataset, qpoint_index: int, branch_index: int, amplitude: float = 1.0
) -> np.ndarray | None:
    """The instantaneous displacement vectors, for drawing arrows."""
    frames = mode_displacements(
        dataset, qpoint_index, branch_index, np.array([0.0]), amplitude
    )
    if frames is None or dataset.structure is None:
        return None
    return frames[0] - dataset.structure.cart_coords


def animation_supercell(
    dataset: PhononDataset, qpoint_index: int, limit: int = 400
) -> tuple[int, int, int]:
    """How many cells to draw so the selected mode reads as a wave.

    A mode with q = 1/2 along an axis repeats every two cells, so two cells
    must be drawn for the alternation to be visible at all. The repeat length
    is 1/q rounded to something small, capped so a long-wavelength mode does
    not ask for a hundred cells.
    """
    if not (0 <= qpoint_index < dataset.n_qpoints) or dataset.structure is None:
        return (1, 1, 1)
    qpoint = dataset.bands[qpoint_index].qpoint

    repeats = []
    for component in qpoint:
        value = abs(float(component))
        if value < 1e-6:
            repeats.append(1)          # uniform along this axis
        else:
            repeats.append(int(min(4, max(1, round(1.0 / value)))))

    # Keep the drawn cell within a size the viewer can animate smoothly.
    n_atoms = dataset.structure.n_atoms
    while n_atoms * repeats[0] * repeats[1] * repeats[2] > limit:
        largest = max(range(3), key=lambda i: repeats[i])
        if repeats[largest] <= 1:
            break
        repeats[largest] -= 1
    return (repeats[0], repeats[1], repeats[2])
