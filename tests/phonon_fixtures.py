"""Phonon fixtures built from an actual force-constant model.

No phonon calculation was available to check the parsers and the page against,
so rather than inventing plausible-looking numbers these fixtures solve a real
(if simple) lattice-dynamics problem: a mass-and-spring crystal whose dynamical
matrix is assembled and diagonalised exactly as phonopy would.

That makes the physics in the fixture genuine rather than decorative, and the
tests can therefore assert real invariants:

* three acoustic branches going to zero at Γ,
* 3N branches for N atoms,
* eigenvectors orthonormal at every q-point,
* a soft (imaginary) mode when a force constant is made negative.

A file that merely contained believable frequencies could not support any of
those checks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vasp_analyzer.resources.elements import atomic_mass


def _dynamical_matrix(
    qpoint: np.ndarray,
    positions: np.ndarray,
    masses: np.ndarray,
    lattice: np.ndarray,
    spring: float,
    neighbours: list[np.ndarray],
) -> np.ndarray:
    """Assemble D(q) for a central-force spring model.

    Each atom is bonded to its images along ``neighbours`` with a single
    force constant. The self term is the negative sum of the coupling terms,
    which is what enforces the acoustic sum rule and therefore what makes the
    three acoustic branches vanish at q = 0.
    """
    n_atoms = len(positions)
    dynamical = np.zeros((3 * n_atoms, 3 * n_atoms), dtype=complex)

    for i in range(n_atoms):
        for j in range(n_atoms):
            block = np.zeros((3, 3), dtype=complex)
            for shift in neighbours:
                separation = positions[j] + shift - positions[i]
                cartesian = separation @ lattice
                length = np.linalg.norm(cartesian)
                if length < 1e-8:
                    continue
                unit = cartesian / length
                # Central force: coupling only along the bond direction.
                coupling = spring * np.outer(unit, unit)
                phase = np.exp(2j * np.pi * np.dot(qpoint, shift))
                # Off-diagonal force constants are *negative* the coupling and
                # the on-site term is positive, so D(q) accumulates as
                # sum_R K (1 - cos q.R) and stays positive semi-definite.
                # With the signs the other way round every branch comes out
                # imaginary — a whole crystal that is unstable everywhere,
                # which is the giveaway that the convention is inverted.
                block -= coupling * phase
                dynamical[3 * i:3 * i + 3, 3 * i:3 * i + 3] += coupling
            dynamical[3 * i:3 * i + 3, 3 * j:3 * j + 3] += block

    # Mass-weight, as phonopy does before diagonalising.
    inv_sqrt = 1.0 / np.sqrt(np.repeat(masses, 3))
    dynamical = dynamical * inv_sqrt[:, None] * inv_sqrt[None, :]
    # Hermitise away accumulated rounding, so eigh sees an exactly Hermitian
    # matrix and returns real eigenvalues.
    return 0.5 * (dynamical + dynamical.conj().T)


def solve(
    qpoints: np.ndarray,
    positions: np.ndarray,
    species: list[str],
    lattice: np.ndarray,
    spring: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Frequencies (THz, imaginary as negative) and mass-weighted eigenvectors."""
    masses = np.array([atomic_mass(s) for s in species])
    neighbours = [
        np.array(shift, dtype=float)
        for shift in (
            (0, 0, 0), (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
        )
    ]

    frequencies = []
    eigenvectors = []
    for qpoint in qpoints:
        matrix = _dynamical_matrix(
            qpoint, positions, masses, lattice, spring, neighbours
        )
        values, vectors = np.linalg.eigh(matrix)
        # omega = sign(lambda) * sqrt(|lambda|): a negative eigenvalue is an
        # imaginary frequency, carried as a negative number.
        omega = np.sign(values) * np.sqrt(np.abs(values))
        frequencies.append(omega)
        # eigh returns columns; the rest of the code wants (branch, atom, 3).
        eigenvectors.append(vectors.T.reshape(len(omega), len(species), 3))
    return np.array(frequencies), np.array(eigenvectors)


def write_band_yaml(
    directory: Path,
    n_qpoints: int = 11,
    species: list[str] | None = None,
    spring: float = 2.0,
    with_eigenvectors: bool = True,
    labels: tuple[str, str] = (r"\Gamma", "X"),
    filename: str = "band.yaml",
) -> Path:
    """Write a phonopy-format band.yaml for a Γ→X path through the model."""
    species = species or ["Si", "Si"]
    lattice = np.eye(3) * 5.43
    # Spread N atoms along the cell diagonal so any atom count works and no
    # two sites coincide (a zero separation has no bond direction).
    n = len(species)
    positions = np.array([[i / (n + 1)] * 3 for i in range(n)], dtype=float)

    fractions = np.linspace(0.0, 0.5, n_qpoints)
    qpoints = np.array([[f, 0.0, 0.0] for f in fractions])
    frequencies, eigenvectors = solve(qpoints, positions, species, lattice, spring)
    distances = fractions / 5.43

    masses = [atomic_mass(s) for s in species]
    out = [
        f"nqpoint: {n_qpoints}",
        f"natom:   {len(species)}",
        "supercell_matrix:",
        "- [   2,   0,   0 ]",
        "- [   0,   2,   0 ]",
        "- [   0,   0,   1 ]",
        "",
        "lattice:",
    ]
    for vector in lattice:
        out.append(f"- [ {vector[0]:18.15f}, {vector[1]:18.15f}, {vector[2]:18.15f} ]")
    out.append("points:")
    for symbol, position, mass in zip(species, positions, masses):
        out.append(f"- symbol: {symbol}")
        out.append(
            f"  coordinates: [ {position[0]:18.15f}, "
            f"{position[1]:18.15f}, {position[2]:18.15f} ]"
        )
        out.append(f"  mass: {mass:.6f}")
    out.append("")
    out.append("phonon:")

    for index, qpoint in enumerate(qpoints):
        out.append(
            f"- q-position: [ {qpoint[0]:12.7f}, {qpoint[1]:12.7f}, {qpoint[2]:12.7f} ]"
        )
        out.append(f"  distance: {distances[index]:12.7f}")
        if index == 0:
            out.append(f"  label: '{labels[0]}'")
        elif index == n_qpoints - 1:
            out.append(f"  label: '{labels[1]}'")
        out.append("  band:")
        for branch, frequency in enumerate(frequencies[index]):
            out.append(f"  - # {branch + 1}")
            out.append(f"    frequency: {frequency:15.10f}")
            if with_eigenvectors:
                out.append("    eigenvector:")
                for atom in range(len(species)):
                    out.append(f"    - # atom {atom + 1}")
                    for component in range(3):
                        value = eigenvectors[index][branch][atom][component]
                        out.append(
                            f"      - [ {value.real:15.10f}, {value.imag:15.10f} ]"
                        )
        out.append("")

    path = directory / filename
    path.write_text("\n".join(out) + "\n")
    return path


def write_band_yaml_segmented(
    directory: Path,
    points_per_segment: int = 6,
    species: list[str] | None = None,
    spring: float = 2.0,
    path_labels: tuple[str, str, str] = (r"$\Gamma$", "M", "K"),
    filename: str = "band.yaml",
) -> Path:
    """A band.yaml in the modern phonopy layout: high-symmetry names live in a
    header ``labels:``/``segment_nqpoint:`` block, one pair per path segment,
    rather than inline on each q-point — real phonopy output never writes the
    inline form this module's other writer uses. Two segments share their
    boundary q-point (phonopy duplicates its coordinates rather than omitting
    it), so this also exercises a straight-through point labelled the same
    name from both sides — e.g. Γ-M-K's "M" — which must render as one tick,
    not "M|M".
    """
    species = species or ["C", "C"]
    lattice = np.eye(3) * 2.47
    n = len(species)
    positions = np.array([[i / (n + 1)] * 3 for i in range(n)], dtype=float)

    # Two straight segments in fractional-q space: Γ(0,0,0) -> M(0.5,0,0),
    # then M(0.5,0,0) -> K(0.5,0.25,0). Real path shapes don't matter here;
    # only the label/segment bookkeeping is under test.
    seg0 = np.linspace([0.0, 0.0, 0.0], [0.5, 0.0, 0.0], points_per_segment)
    seg1 = np.linspace([0.5, 0.0, 0.0], [0.5, 0.25, 0.0], points_per_segment)
    qpoints = np.vstack([seg0, seg1])
    frequencies, _eigenvectors = solve(qpoints, positions, species, lattice, spring)

    # Cumulative path length, continuous across the shared boundary point.
    steps = np.linalg.norm(np.diff(qpoints, axis=0) @ lattice, axis=1)
    distances = np.concatenate([[0.0], np.cumsum(steps)])

    masses = [atomic_mass(s) for s in species]
    out = [
        f"nqpoint: {len(qpoints)}",
        "npath: 2",
        "segment_nqpoint:",
        f"- {points_per_segment}",
        f"- {points_per_segment}",
        "labels:",
        f"- [ '{path_labels[0]}', '{path_labels[1]}' ]",
        f"- [ '{path_labels[1]}', '{path_labels[2]}' ]",
        f"natom:   {len(species)}",
        "lattice:",
    ]
    for vector in lattice:
        out.append(f"- [ {vector[0]:18.15f}, {vector[1]:18.15f}, {vector[2]:18.15f} ]")
    out.append("points:")
    for symbol, position, mass in zip(species, positions, masses):
        out.append(f"- symbol: {symbol}")
        out.append(
            f"  coordinates: [ {position[0]:18.15f}, "
            f"{position[1]:18.15f}, {position[2]:18.15f} ]"
        )
        out.append(f"  mass: {mass:.6f}")
    out.append("")
    out.append("phonon:")

    for index, qpoint in enumerate(qpoints):
        out.append(
            f"- q-position: [ {qpoint[0]:12.7f}, {qpoint[1]:12.7f}, {qpoint[2]:12.7f} ]"
        )
        out.append(f"  distance:    {distances[index]:12.7f}")
        out.append("  band:")
        for branch, frequency in enumerate(frequencies[index]):
            out.append(f"  - # {branch + 1}")
            out.append(f"    frequency: {frequency:15.10f}")
        out.append("")

    path = directory / filename
    path.write_text("\n".join(out) + "\n")
    return path


def write_total_dos(directory: Path, n_points: int = 60) -> Path:
    """A DOS table on the same footing as phonopy's total_dos.dat."""
    frequencies = np.linspace(-1.0, 12.0, n_points)
    dos = np.exp(-((frequencies - 5.0) ** 2) / 4.0)
    lines = ["# Frequency THz, DOS"]
    for frequency, value in zip(frequencies, dos):
        lines.append(f"{frequency:15.8f} {value:15.8f}")
    path = directory / "total_dos.dat"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_projected_dos(directory: Path, n_atoms: int = 2, n_points: int = 60) -> Path:
    """One column per atom, as phonopy writes projected_dos.dat."""
    frequencies = np.linspace(-1.0, 12.0, n_points)
    lines = ["# Frequency THz, then one column per atom"]
    for index, frequency in enumerate(frequencies):
        columns = [
            np.exp(-((frequency - (4.0 + atom)) ** 2) / 4.0) for atom in range(n_atoms)
        ]
        lines.append(
            f"{frequency:15.8f} " + " ".join(f"{c:15.8f}" for c in columns)
        )
    path = directory / "projected_dos.dat"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_phonopy_yaml(directory: Path, dfpt: bool = True) -> Path:
    path = directory / "phonopy.yaml"
    body = [
        "phonopy:",
        '  version: "2.21.0"',
        "  calculator: vasp",
        "  frequency_unit_conversion_factor: 15.633302",
        "physical_unit:",
        '  atomic_mass: "AMU"',
    ]
    body.append(
        "  create_displacements: false" if dfpt else "  create_displacements: true"
    )
    body.append("  dfpt: true" if dfpt else "  displacement_distance: 0.01")
    path.write_text("\n".join(body) + "\n")
    return path


def complete_phonon_calculation(directory: Path, species: list[str] | None = None) -> Path:
    """Everything a stable phonon run would leave behind."""
    directory.mkdir(parents=True, exist_ok=True)
    species = species or ["Si", "Si"]
    write_band_yaml(directory, species=species)
    write_total_dos(directory)
    write_projected_dos(directory, n_atoms=len(species))
    write_phonopy_yaml(directory)
    return directory


def unstable_phonon_calculation(directory: Path) -> Path:
    """A negative force constant, which really does produce imaginary modes."""
    directory.mkdir(parents=True, exist_ok=True)
    write_band_yaml(directory, spring=-1.5)
    write_phonopy_yaml(directory)
    return directory
