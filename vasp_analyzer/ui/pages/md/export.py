"""Export for the Molecular Dynamics page.

Every writer takes the parsed trajectory rather than anything on screen, so an
exported file reflects the calculation rather than the current view state.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ....core.logging_setup import get_logger
from ....models.trajectory import MDTrajectory

log = get_logger("ui.md.export")


def _time_column(trajectory: MDTrajectory) -> tuple[str, np.ndarray]:
    times = trajectory.times()
    header = "time_ps" if trajectory.has_time_axis else "frame"
    return header, times


def save_trajectory_xyz(trajectory: MDTrajectory, path: Path) -> None:
    """Extended XYZ: one block per frame, cartesian Å, with the lattice.

    The ``Lattice=`` comment is the extended-XYZ convention that OVITO, ASE and
    VMD all read, so the file stays periodic rather than becoming a cloud of
    free atoms.
    """
    with Path(path).open("w", encoding="utf-8") as handle:
        for frame in range(trajectory.n_frames):
            lattice = trajectory.lattices[frame]
            cartesian = trajectory.cart_coords(frame)
            flat = " ".join(f"{value:.8f}" for value in lattice.reshape(-1))
            handle.write(f"{trajectory.n_atoms}\n")
            handle.write(
                f'Lattice="{flat}" Properties=species:S:1:pos:R:3 '
                f"Frame={frame}\n"
            )
            for symbol, position in zip(trajectory.species, cartesian):
                handle.write(
                    f"{symbol} {position[0]:.8f} {position[1]:.8f} {position[2]:.8f}\n"
                )
    log.info("Exported trajectory to %s", path)


def save_msd_csv(
    trajectory: MDTrajectory, curves: dict[str, np.ndarray], path: Path
) -> None:
    if not curves:
        raise ValueError("no MSD data to export")

    header, times = _time_column(trajectory)
    names = [e for e in trajectory.elements if e in curves]
    if "total" in curves:
        names.append("total")

    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([header] + [f"msd_{name}_A2" for name in names])
        for index in range(trajectory.n_frames):
            row = [f"{times[index]:.6f}"]
            row += [f"{curves[name][index]:.8f}" for name in names]
            writer.writerow(row)
    log.info("Exported MSD to %s", path)


def save_temperature_csv(trajectory: MDTrajectory, path: Path) -> None:
    series = trajectory.series_for("temperature")
    if series is None:
        raise ValueError("no temperature data to export")

    header, times = _time_column(trajectory)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([header, "temperature_K"])
        for index in range(trajectory.n_frames):
            writer.writerow([f"{times[index]:.6f}", f"{series.values[index]:.4f}"])
    log.info("Exported temperature to %s", path)


def save_energy_csv(trajectory: MDTrajectory, path: Path) -> None:
    keys = [k for k in ("total_energy", "potential_energy", "kinetic_energy")
            if trajectory.has(k)]
    if not keys:
        raise ValueError("no energy data to export")

    header, times = _time_column(trajectory)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([header] + [f"{key}_eV" for key in keys])
        for index in range(trajectory.n_frames):
            row = [f"{times[index]:.6f}"]
            row += [f"{trajectory.series[key].values[index]:.8f}" for key in keys]
            writer.writerow(row)
    log.info("Exported energies to %s", path)
