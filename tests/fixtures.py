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
