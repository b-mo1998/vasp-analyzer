"""Synthetic VASP files for testing.

Real calculation directories are large and can't be committed, so the test
suite builds minimal but format-accurate files on the fly. Every generator
produces output that the parsers must accept, and each can be perturbed to
produce the malformed cases the app has to survive.
"""

from __future__ import annotations

import math
from pathlib import Path

SILICON_LATTICE = 5.4667335224494948


def write_poscar(directory: Path, name: str = "POSCAR", atoms: int = 2) -> Path:
    """A cubic silicon cell with ``atoms`` sites."""
    positions = []
    for i in range(atoms):
        offset = i / max(atoms, 1)
        positions.append(f"  {offset:.8f}  {offset:.8f}  {offset:.8f}")
    body = "\n".join(positions)
    text = f"""Si{atoms}
   1.0000000000000000
     {SILICON_LATTICE:.10f}    0.0000000000000000    0.0000000000000000
     0.0000000000000000    {SILICON_LATTICE:.10f}    0.0000000000000000
     0.0000000000000000    0.0000000000000000    {SILICON_LATTICE:.10f}
   Si
     {atoms}
Direct
{body}
"""
    path = directory / name
    path.write_text(text)
    return path


def write_eigenval(
    directory: Path,
    n_kpoints: int = 12,
    n_bands: int = 6,
    n_spin: int = 1,
    n_electrons: float = 8.0,
    n_ions: int = 2,
    truncate_after: int | None = None,
) -> Path:
    """EIGENVAL with a smooth cosine dispersion.

    ``truncate_after`` cuts the file mid-way to emulate an interrupted job.
    """
    lines = [
        f"    {n_ions}    {n_ions}    1    {n_spin}",
        "  0.2042179E+02  0.5466734E-09  0.5466734E-09  0.5466734E-09  0.5000000E-15",
        "   1.0000000000000000E-004",
        "  CAR ",
        " Si",
        f"     {int(n_electrons)}    {n_kpoints}     {n_bands}",
    ]

    for k in range(n_kpoints):
        if truncate_after is not None and k >= truncate_after:
            break
        fraction = k / max(n_kpoints - 1, 1)
        lines.append("")
        lines.append(
            f"  {fraction * 0.5:.8E}  {0.0:.8E}  {0.0:.8E}  {1.0 / n_kpoints:.8E}"
        )
        for band in range(n_bands):
            base = -6.0 + 2.4 * band
            energy = base + 1.2 * math.cos(math.pi * fraction) * (1 + 0.2 * band)
            occupation = 1.0 if band < n_bands // 2 else 0.0
            if n_spin == 2:
                lines.append(
                    f"    {band + 1}  {energy:12.6f}  {energy + 0.15:12.6f}"
                    f"  {occupation:9.6f}  {occupation:9.6f}"
                )
            else:
                lines.append(f"    {band + 1}  {energy:12.6f}  {occupation:9.6f}")

    path = directory / "EIGENVAL"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_doscar(
    directory: Path,
    n_points: int = 60,
    n_spin: int = 1,
    e_fermi: float = 5.3068,
    n_ions: int = 2,
    malformed: bool = False,
) -> Path:
    """DOSCAR with a Gaussian-ish total DOS."""
    if malformed:
        (directory / "DOSCAR").write_text("this is not a DOSCAR\n")
        return directory / "DOSCAR"

    e_min, e_max = -10.0, 10.0
    lines = [
        f"   {n_ions}   {n_ions}   1   0",
        "  0.2042179E+02  0.5466734E-09  0.5466734E-09  0.5466734E-09  0.5000000E-15",
        "   1.0000000000000000E-004",
        "  CAR ",
        " Si",
        f"     {e_max:.8f}     {e_min:.8f} {n_points}      {e_fermi:.8f}      1.00000000",
    ]

    integrated = 0.0
    for i in range(n_points):
        energy = e_min + (e_max - e_min) * i / max(n_points - 1, 1)
        value = math.exp(-((energy - 2.0) ** 2) / 4.0) * 3.0
        integrated += value * (e_max - e_min) / n_points
        if n_spin == 2:
            lines.append(
                f"  {energy:10.3f}  {value:.4E}  {value * 0.8:.4E}"
                f"  {integrated:.4E}  {integrated * 0.8:.4E}"
            )
        else:
            lines.append(f"  {energy:10.3f}  {value:.4E}  {integrated:.4E}")

    path = directory / "DOSCAR"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_incar(directory: Path, **tags: str) -> Path:
    defaults = {
        "SYSTEM": "Si",
        "ENCUT": "520",
        "ISMEAR": "0",
        "SIGMA": "0.05",
    }
    defaults.update(tags)
    lines = ["# generated for tests"]
    lines += [f"{key} = {value}" for key, value in defaults.items()]
    path = directory / "INCAR"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_kpoints_mesh(directory: Path, mesh: tuple[int, int, int] = (8, 8, 6)) -> Path:
    text = f"""Automatic mesh
 0
Gamma
 {mesh[0]} {mesh[1]} {mesh[2]}
 0 0 0
"""
    path = directory / "KPOINTS"
    path.write_text(text)
    return path


def write_kpoints_line(directory: Path, divisions: int = 4) -> Path:
    """Line-mode KPOINTS describing Γ–X–M–Γ."""
    text = f"""Band path
 {divisions}
Line-mode
Reciprocal
  0.0 0.0 0.0 ! GAMMA
  0.5 0.0 0.0 ! X

  0.5 0.0 0.0 ! X
  0.5 0.5 0.0 ! M

  0.5 0.5 0.0 ! M
  0.0 0.0 0.0 ! GAMMA
"""
    path = directory / "KPOINTS"
    path.write_text(text)
    return path


def write_outcar(
    directory: Path,
    n_bands: int = 6,
    n_kpoints: int = 12,
    n_electrons: float = 8.0,
    n_ions: int = 2,
    e_fermi: float = 5.30,
    species: str = "Si",
) -> Path:
    text = f""" vasp.6.3.0 20Jan22 (build Jun  9 2026 22:37:21) complex
 executed on             LinuxNV date 2026.08.11  20:30:48

 INCAR:
   ENCUT = 520

   TITEL  = PAW_PBE {species} 08Apr2002
   ENCUT  =  520.0 eV  36.75 Ry
   ISPIN  =      1    spin polarized calculation?
   LNONCOLLINEAR =      F non collinear calculations
   LSORBIT =      F    spin-orbit coupling
   NIONS = {n_ions}
   k-points           NKPTS =     {n_kpoints}   k-points in BZ     NKDIM = \
{n_kpoints}   number of bands    NBANDS=    {n_bands}
   NELECT =     {n_electrons:.4f}    total number of electrons
 generate k-points for:    8    8    6

 E-fermi :  {e_fermi:.4f}     XC(G=0):  -3.4637     alpha+bet : -3.2482

                         Elapsed time (sec):     1429.468
"""
    path = directory / "OUTCAR"
    path.write_text(text)
    return path


def write_wannier_win(directory: Path, num_wann: int = 4) -> Path:
    text = f"""num_wann = {num_wann}
num_bands = 6
bands_plot = true

begin kpoint_path
G 0.0 0.0 0.0  X 0.5 0.0 0.0
X 0.5 0.0 0.0  M 0.5 0.5 0.0
end kpoint_path
"""
    path = directory / "wannier90.win"
    path.write_text(text)
    return path


def write_procar(
    directory: Path,
    n_kpoints: int = 12,
    n_bands: int = 6,
    n_ions: int = 2,
    n_spin: int = 1,
    orbitals: tuple[str, ...] = ("s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "x2-y2"),
    truncate_after_band: int | None = None,
) -> Path:
    """A collinear PROCAR with a deterministic, checkable weight pattern:
    ion 1's weight on orbital 0 equals the band index; every other entry is 0.
    That makes the parsed array trivial to assert against exactly."""
    header = (
        f"# of k-points:  {n_kpoints:4d}         # of bands:  {n_bands:4d}"
        f"         # of ions:  {n_ions:4d}"
    )
    lines = ["PROCAR lm decomposed", header, ""]
    header_row = "ion" + "".join(f"{o:>7s}" for o in orbitals) + "    tot"

    total_bands_written = 0
    for spin in range(n_spin):
        for k in range(n_kpoints):
            lines.append(
                f" k-point {k + 1:4d} :    0.00000000 0.00000000 0.00000000"
                f"     weight = {1.0 / n_kpoints:.8f}"
            )
            lines.append("")
            for band in range(n_bands):
                if truncate_after_band is not None and total_bands_written >= truncate_after_band:
                    path = directory / "PROCAR"
                    path.write_text("\n".join(lines) + "\n")
                    return path
                lines.append(
                    f"band {band + 1:4d} # energy {band - 3.0:12.6f} # occ.  1.00000000"
                )
                lines.append("")
                lines.append(header_row)
                weight = float(band + 1)
                for ion in range(n_ions):
                    values = [weight if (ion == 0 and col == 0) else 0.0 for col in range(len(orbitals))]
                    row = f"{ion + 1:4d}" + "".join(f"{v:7.3f}" for v in values)
                    row += f"{sum(values):7.3f}"
                    lines.append(row)
                tot_values = [weight if col == 0 else 0.0 for col in range(len(orbitals))]
                tot_row = "tot " + "".join(f"{v:7.3f}" for v in tot_values)
                tot_row += f"{sum(tot_values):7.3f}"
                lines.append(tot_row)
                lines.append("")
                total_bands_written += 1

    path = directory / "PROCAR"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_procar_noncollinear(
    directory: Path,
    n_kpoints: int = 4,
    n_bands: int = 4,
    n_ions: int = 2,
    orbitals: tuple[str, ...] = ("s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "x2-y2"),
) -> Path:
    """A non-collinear (LSORBIT=T) PROCAR: one ``ion ... tot`` header per
    band, followed by four stacked ion tables (charge, mx, my, mz) with no
    repeated header and no blank line between them — verified against a real
    noncollinear graphene PROCAR to be the actual VASP layout, not the
    "four separate headed tables" shape the docstring used to assume.

    Deterministic pattern: ion 1's charge weight on orbital 0 equals the band
    index (1-based); mz alternates sign by band, mimicking a spin-orbit
    Kramers pair; mx and my are zero throughout."""
    header = (
        f"# of k-points:  {n_kpoints:4d}         # of bands:  {n_bands:4d}"
        f"         # of ions:  {n_ions:4d}"
    )
    lines = ["PROCAR lm decomposed", header, ""]
    header_row = "ion" + "".join(f"{o:>7s}" for o in orbitals) + "    tot"

    def _table_lines(values_per_ion: list[list[float]]) -> list[str]:
        rows = []
        for ion, values in enumerate(values_per_ion):
            row = f"{ion + 1:4d}" + "".join(f"{v:7.3f}" for v in values)
            row += f"{sum(values):7.3f}"
            rows.append(row)
        tot_values = [sum(col) for col in zip(*values_per_ion)]
        tot_row = "tot " + "".join(f"{v:7.3f}" for v in tot_values)
        tot_row += f"{sum(tot_values):7.3f}"
        rows.append(tot_row)
        return rows

    for k in range(n_kpoints):
        lines.append(
            f" k-point {k + 1:4d} :    0.00000000 0.00000000 0.00000000"
            f"     weight = {1.0 / n_kpoints:.8f}"
        )
        lines.append("")
        for band in range(n_bands):
            lines.append(
                f"band {band + 1:4d} # energy {band - 3.0:12.6f} # occ.  1.00000000"
            )
            lines.append("")
            lines.append(header_row)

            charge_weight = float(band + 1)
            charge = [
                [
                    charge_weight if (ion == 0 and col == 0) else 0.0
                    for col in range(len(orbitals))
                ]
                for ion in range(n_ions)
            ]
            lines.extend(_table_lines(charge))

            mz_weight = (1.0 if band % 2 == 0 else -1.0) * 0.5
            mz = [
                [mz_weight if (ion == 0 and col == 0) else 0.0 for col in range(len(orbitals))]
                for ion in range(n_ions)
            ]
            zeros = [[0.0] * len(orbitals) for _ in range(n_ions)]
            lines.extend(_table_lines(zeros))  # mx
            lines.extend(_table_lines(zeros))  # my
            lines.extend(_table_lines(mz))  # mz

            lines.append("")

    path = directory / "PROCAR"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_wannier_eig(directory: Path, n_bands: int = 6, n_kpoints: int = 12) -> Path:
    lines = []
    for k in range(1, n_kpoints + 1):
        for band in range(1, n_bands + 1):
            lines.append(f"{band:5d}{k:5d}  {-6.0 + 2.0 * band:16.8f}")
    path = directory / "wannier90.eig"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_wannier_band_dat(directory: Path, n_bands: int = 4, n_points: int = 10) -> Path:
    """postw90's actual interpolated-band output: one blank-separated block
    per band, ``x  energy`` pairs. Only this file (or a real completed
    Wannierisation's equivalent) makes "Wannier Bands" genuinely plottable."""
    blocks = []
    for band in range(n_bands):
        rows = [
            f"{x:12.6f}{-4.0 + band * 1.5 + 0.3 * math.cos(x):14.6f}"
            for x in (i * 0.1 for i in range(n_points))
        ]
        blocks.append("\n".join(rows))
    path = directory / "wannier90_band.dat"
    path.write_text("\n\n".join(blocks) + "\n")
    return path


# --- Named scenarios from the specification -------------------------------------


def case_a_complete(directory: Path) -> Path:
    """Everything available."""
    write_eigenval(directory)
    write_doscar(directory)
    write_poscar(directory)
    write_incar(directory)
    write_kpoints_mesh(directory)
    write_outcar(directory)
    return directory


def case_b_bands_only(directory: Path) -> Path:
    """EIGENVAL + POSCAR: bands work, DOS disabled."""
    write_eigenval(directory)
    write_poscar(directory)
    return directory


def case_c_no_eigenval(directory: Path) -> Path:
    """DOSCAR + POSCAR: no band structure."""
    write_doscar(directory)
    write_poscar(directory)
    return directory


def case_d_structure_only(directory: Path) -> Path:
    """POSCAR + INCAR: structure and inputs only."""
    write_poscar(directory)
    write_incar(directory)
    return directory


def case_e_partial_wannier(directory: Path) -> Path:
    """EIGENVAL + wannier90.eig + wannier90.win."""
    write_eigenval(directory)
    write_wannier_win(directory)
    write_wannier_eig(directory)
    return directory


def case_f_malformed_doscar(directory: Path) -> Path:
    """A complete calculation whose DOSCAR is corrupt."""
    write_eigenval(directory)
    write_poscar(directory)
    write_incar(directory)
    write_doscar(directory, malformed=True)
    return directory


def case_g_empty(directory: Path) -> Path:
    """Nothing at all."""
    return directory


# --- molecular dynamics -----------------------------------------------------------


def write_xdatcar(
    directory: Path,
    n_frames: int = 10,
    n_atoms: int = 4,
    species: tuple[str, ...] = ("C",),
    lattice: float = 5.0,
    variable_cell: bool = False,
    drift: float = 0.0,
    truncate_last: bool = False,
) -> Path:
    """A fixed- or variable-cell XDATCAR with a deterministic drift pattern.

    Atom 0 advances by ``drift`` in fractional x every frame, so unwrapping and
    MSD both have an exactly predictable answer — including when the drift
    carries it across the periodic boundary.
    """
    counts = [n_atoms // len(species)] * len(species)
    counts[-1] += n_atoms - sum(counts)

    def header(scale: float) -> list[str]:
        return [
            "MD test cell",
            "           1",
            f"    {lattice * scale:.6f}    0.000000    0.000000",
            f"    0.000000    {lattice * scale:.6f}    0.000000",
            f"    0.000000    0.000000    {lattice * scale:.6f}",
            "   " + "  ".join(species),
            "   " + "  ".join(str(c) for c in counts),
        ]

    lines: list[str] = []
    if not variable_cell:
        lines += header(1.0)

    for frame in range(n_frames):
        if variable_cell:
            # A slowly expanding cell, as an NPT run would produce.
            lines += header(1.0 + 0.01 * frame)
        lines.append(f"Direct configuration=  {frame + 1:5d}")
        for atom in range(n_atoms):
            x = (0.1 * atom + drift * frame) % 1.0 if atom == 0 else 0.1 * atom
            if atom == 0:
                x = (0.1 + drift * frame) % 1.0
            lines.append(f"  {x:.8f}  {0.2 * atom % 1.0:.8f}  {0.25:.8f}")

    if truncate_last:
        # Drop half of the final configuration, as an interrupted job would.
        lines = lines[: -(n_atoms // 2)]

    path = directory / "XDATCAR"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_oszicar_md(
    directory: Path,
    n_steps: int = 10,
    temperature: float = 300.0,
    potential: float = -100.0,
) -> Path:
    """An MD OSZICAR: one ``T= … E= … F= … EK= …`` line per ionic step.

    Energies satisfy VASP's own identity ``E = F + EK + SP + SK`` exactly, so a
    test can assert the parser reads the right column rather than merely a
    number of the right magnitude.
    """
    lines = []
    for step in range(1, n_steps + 1):
        temp = temperature + step
        kinetic = 0.01 * step
        pot = potential - 0.001 * step
        total = pot + kinetic
        lines.append("DAV:   1    -0.100000000000E+03   -0.10000E+03   -0.1E+00  100")
        lines.append(
            f"{step:6d} T= {temp:6.0f}. E= {total:.8E} F= {pot:.8E} "
            f"E0= {pot:.8E}  EK= {kinetic:.5E} SP= 0.00E+00 SK= 0.00E+00"
        )
    path = directory / "OSZICAR"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_incar_md(
    directory: Path, potim: float = 1.0, tebeg: float = 300.0, **extra: str
) -> Path:
    tags = {
        "SYSTEM": "MD test",
        "IBRION": "0",
        "NSW": "10",
        "POTIM": f"{potim}",
        "TEBEG": f"{tebeg}",
        "MDALGO": "2",
        "SMASS": "1.0",
        "ISIF": "2",
    }
    tags.update(extra)
    return write_incar(directory, **tags)


def case_md(directory: Path, n_frames: int = 10, n_atoms: int = 4) -> Path:
    """XDATCAR + OSZICAR + INCAR + POSCAR: a complete little MD run."""
    write_xdatcar(directory, n_frames=n_frames, n_atoms=n_atoms)
    write_oszicar_md(directory, n_steps=n_frames)
    write_incar_md(directory)
    write_poscar(directory, atoms=n_atoms)
    return directory


# --- NEB ---------------------------------------------------------------------------


def write_neb_image(
    directory: Path,
    index: int,
    n_atoms: int = 2,
    shift: float = 0.0,
    energy: float | None = None,
    max_force: float | None = None,
    contcar: bool = False,
    lattice: float = 5.0,
) -> Path:
    """One NEB image folder: a POSCAR (optionally a CONTCAR) and an OUTCAR.

    ``shift`` displaces atom 0 along x, so the reaction coordinate between
    consecutive images is exactly ``shift`` times the lattice constant.
    """
    folder = directory / f"{index:02d}"
    folder.mkdir(parents=True, exist_ok=True)

    positions = []
    for atom in range(n_atoms):
        x = (0.1 + shift) if atom == 0 else 0.1 + 0.2 * atom
        positions.append(f"  {x:.8f}  0.10000000  0.10000000")
    structure = f"""NEB image {index}
   1.0000000000000000
     {lattice:.10f}    0.0000000000000000    0.0000000000000000
     0.0000000000000000    {lattice:.10f}    0.0000000000000000
     0.0000000000000000    0.0000000000000000    {lattice:.10f}
   C
     {n_atoms}
Direct
""" + "\n".join(positions) + "\n"

    (folder / "POSCAR").write_text(structure)
    if contcar:
        (folder / "CONTCAR").write_text(structure)

    if energy is not None:
        force = 0.0 if max_force is None else max_force
        rows = "\n".join(
            f"      0.100      0.100      0.100         "
            f"{force if atom == 0 else 0.0:.6f}      0.000000      0.000000"
            for atom in range(n_atoms)
        )
        (folder / "OUTCAR").write_text(
            f"""
  free  energy   TOTEN  =    {energy:12.8f} eV

  energy  without entropy=    {energy:12.8f}  energy(sigma->0) =    {energy:12.8f}

 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------
{rows}
 -----------------------------------------------------------------------------
"""
        )
    return folder


def write_neb_incar(directory: Path, climbing: bool = True, **extra: str) -> Path:
    tags = {
        "SYSTEM": "NEB test",
        "IMAGES": "3",
        "SPRING": "-5",
        "LCLIMB": ".TRUE." if climbing else ".FALSE.",
        "IBRION": "3",
        "EDIFFG": "-0.05",
        "ENCUT": "500",
        "ISIF": "2",
    }
    tags.update(extra)
    return write_incar(directory, **tags)


def case_neb(
    directory: Path,
    energies: tuple[float, ...] = (-10.0, -9.0, -8.0, -9.2, -9.8),
    forces: tuple[float, ...] = (0.001, 0.02, 0.03, 0.02, 0.001),
    climbing: bool = True,
) -> Path:
    """A complete five-image band with a barrier at image 2."""
    write_neb_incar(directory, climbing=climbing)
    for index, (energy, force) in enumerate(zip(energies, forces)):
        write_neb_image(
            directory,
            index,
            shift=0.05 * index,
            energy=energy,
            max_force=force,
            contcar=0 < index < len(energies) - 1,
        )
    return directory


# --- volumetric (CHGCAR / ELFCAR / PARCHG) ------------------------------------------


def write_volumetric(
    directory: Path,
    name: str = "CHGCAR",
    shape: tuple[int, int, int] = (4, 5, 6),
    lattice: list[list[float]] | None = None,
    species: tuple[str, ...] = ("C",),
    counts: tuple[int, ...] = (2,),
    values=None,
    n_blocks: int = 1,
    volume_scaled: bool = True,
    electrons: float | None = None,
    augmentation: bool = False,
    truncate: bool = False,
) -> Path:
    """A volumetric VASP file with an exactly known grid.

    ``values`` may be an explicit (nx, ny, nz) array; otherwise a ramp
    ``i + 10*j + 100*k`` is written, which makes an axis transposition
    impossible to miss — every grid point's value states its own index.

    ``electrons`` rescales the data so the file integrates to a known count,
    which is what lets a test assert the normalisation rather than merely the
    shape.
    """
    import numpy as np

    lattice = lattice or [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]]
    nx, ny, nz = shape
    n_atoms = sum(counts)

    if values is None:
        i, j, k = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
        )
        values = (i + 10 * j + 100 * k).astype(float)
    values = np.asarray(values, dtype=float)

    cell_volume = float(abs(np.linalg.det(np.array(lattice))))
    if electrons is not None:
        total = values.sum()
        if total != 0:
            # data = n * V, and N = sum(data)/Ngrid, so scale to hit N exactly.
            values = values * (electrons * values.size / total)

    lines = [f"{name} test", "   1.00000000000000"]
    for row in lattice:
        lines.append(f"  {row[0]:20.12f}{row[1]:20.12f}{row[2]:20.12f}")
    lines.append("   " + "  ".join(species))
    lines.append("   " + "  ".join(str(c) for c in counts))
    lines.append("Direct")
    for atom in range(n_atoms):
        offset = atom / max(n_atoms, 1)
        lines.append(f"  {offset:.10f}  {offset:.10f}  {0.25:.10f}")

    flat = values.ravel(order="F")          # first index fastest
    if truncate:
        flat = flat[: flat.size // 2]

    for block in range(n_blocks):
        lines.append("")
        lines.append(f" {nx:4d} {ny:4d} {nz:4d}")
        payload = flat if block == 0 else flat * 0.1
        for start in range(0, payload.size, 5):
            lines.append(
                "".join(f"{v:18.11E}" for v in payload[start : start + 5])
            )
        if augmentation:
            lines.append(f"augmentation occupancies   1  {4}")
            lines.append("  0.1000000E+00  0.2000000E+00  0.3000000E+00  0.4000000E+00")

    _ = volume_scaled, cell_volume
    path = directory / name
    path.write_text("\n".join(lines) + "\n")
    return path


def case_charge(directory: Path, **kwargs) -> Path:
    """A CHGCAR plus the POSCAR/OUTCAR a page would use alongside it."""
    write_volumetric(directory, **kwargs)
    write_poscar(directory, atoms=2)
    return directory
