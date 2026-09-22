"""Export for the NEB page."""

from __future__ import annotations

import csv
from pathlib import Path

from ....core.logging_setup import get_logger
from ....models.neb import NEBPath

log = get_logger("ui.neb.export")


def save_profile_csv(path: NEBPath, target: Path) -> None:
    """The energy profile as a table, mirroring VTST's own neb.dat columns."""
    if not path.has_energies:
        raise ValueError("no image energies to export")

    relative = path.relative_energies()
    with Path(target).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image",
                "folder",
                "reaction_coordinate_A",
                "energy_eV",
                "relative_energy_eV",
                "max_force_eV_per_A",
                "force_kind",
            ]
        )
        for index, image in enumerate(path.images):
            coordinate = path.coordinate_of(index)
            writer.writerow(
                [
                    index,
                    image.folder,
                    "" if coordinate is None else f"{coordinate:.6f}",
                    "" if image.energy is None else f"{image.energy:.6f}",
                    "" if relative[index] is None else f"{relative[index]:.6f}",
                    "" if image.max_force is None else f"{abs(image.max_force):.6f}",
                    image.force_kind,
                ]
            )
    log.info("Exported NEB profile to %s", target)


def save_path_xyz(path: NEBPath, target: Path) -> None:
    """Every image as one extended-XYZ frame, in path order."""
    structures = [(i, image) for i, image in enumerate(path.images)
                  if image.structure is not None]
    if not structures:
        raise ValueError("no image structures to export")

    with Path(target).open("w", encoding="utf-8") as handle:
        for index, image in structures:
            structure = image.structure
            assert structure is not None
            flat = " ".join(f"{v:.8f}" for v in structure.lattice.reshape(-1))
            energy = "" if image.energy is None else f" Energy={image.energy:.8f}"
            handle.write(f"{structure.n_atoms}\n")
            handle.write(
                f'Lattice="{flat}" Properties=species:S:1:pos:R:3 '
                f"Image={image.folder}{energy}\n"
            )
            for symbol, position in zip(structure.species, structure.cart_coords):
                handle.write(
                    f"{symbol} {position[0]:.8f} {position[1]:.8f} {position[2]:.8f}\n"
                )
            _ = index
    log.info("Exported NEB path to %s", target)
