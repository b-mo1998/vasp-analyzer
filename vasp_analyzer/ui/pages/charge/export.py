"""Export for the Charge Density page.

Both writers take the full-resolution field, never the subsampled copy used
for rendering — an exported CSV must reproduce the calculation, not the
drawing performance setting that happened to be selected.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ....core.logging_setup import get_logger
from ....models.volumetric import VolumetricField
from ....services import volumetric as vol

log = get_logger("ui.charge.export")


def save_average_csv(field: VolumetricField, axis: int, path: Path) -> None:
    distance, values = vol.planar_average(field, axis)
    units = field.kind.units or "dimensionless"
    name = vol.AXIS_NAMES[axis]

    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"# {field.kind.label} planar average along {name}"])
        writer.writerow([f"# source: {field.source_file}"])
        writer.writerow([f"{name}_angstrom", f"value_{units}"])
        for position, value in zip(distance, values):
            writer.writerow([f"{position:.6f}", f"{value:.8e}"])
    log.info("Exported planar average to %s", path)


def save_slice_csv(
    field: VolumetricField, axis: int, fraction: float, path: Path
) -> None:
    """One slice as a grid of values, with its in-plane coordinates."""
    data = vol.slice_geometry(field, axis, fraction)
    units = field.kind.units or "dimensionless"

    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"# {field.kind.label} slice, source {field.source_file}"])
        writer.writerow(
            [
                f"# normal along {vol.AXIS_NAMES[axis]} at "
                f"{data.distance:.4f} A (fraction {data.fraction:.4f})"
            ]
        )
        writer.writerow(["x_angstrom", "y_angstrom", f"value_{units}"])

        # Cell centres, so each row is the coordinate the value belongs to.
        centres_x = 0.25 * (
            data.x_edges[:-1, :-1] + data.x_edges[1:, :-1]
            + data.x_edges[:-1, 1:] + data.x_edges[1:, 1:]
        )
        centres_y = 0.25 * (
            data.y_edges[:-1, :-1] + data.y_edges[1:, :-1]
            + data.y_edges[:-1, 1:] + data.y_edges[1:, 1:]
        )
        for i in range(data.values.shape[0]):
            for j in range(data.values.shape[1]):
                writer.writerow(
                    [
                        f"{centres_x[i, j]:.6f}",
                        f"{centres_y[i, j]:.6f}",
                        f"{data.values[i, j]:.8e}",
                    ]
                )
    log.info("Exported slice to %s", path)
