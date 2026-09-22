"""Assembling an :class:`MDTrajectory` from whatever the directory provides.

Geometry comes from XDATCAR; telemetry (temperature, energies) from OSZICAR;
timing and ensemble from INCAR. Each is optional except the geometry, and a
missing source removes only the features it would have fed — a trajectory with
no OSZICAR still plays, it just has no temperature plot.

Kept apart from the parsers because deciding *which* source wins, and how the
frame axis and the per-step axis are reconciled, is a policy question rather
than a file-format one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.errors import Issue, ParseResult, Severity
from ..core.logging_setup import get_logger
from ..models.files import FileIndex
from ..models.trajectory import MDSeries, MDTrajectory
from ..parsers.incar import as_float, as_int, parse_incar
from ..parsers.oszicar import parse_oszicar
from ..parsers.xdatcar import parse_xdatcar

log = get_logger("services.md")

# OSZICAR series that map onto a trajectory series of the same name.
_ENERGY_FIELDS = {
    "total_energy": ("Total Energy", "eV"),
    "potential_energy": ("Potential Energy", "eV"),
    "kinetic_energy": ("Kinetic Energy", "eV"),
    "temperature": ("Temperature", "K"),
}


def describe_ensemble(tags: dict[str, str]) -> str | None:
    """Name the MD ensemble from the INCAR, or None when it isn't an MD run.

    VASP does not record an "ensemble" tag; it is implied by the combination of
    IBRION, ISIF and the thermostat selection, so this reports only what those
    actually pin down and stays silent otherwise.
    """
    ibrion = as_int(tags.get("IBRION"))
    if ibrion != 0:
        return None

    isif = as_int(tags.get("ISIF"))
    mdalgo = as_int(tags.get("MDALGO"))
    smass = as_float(tags.get("SMASS"))

    # ISIF >= 3 relaxes the cell, which with a thermostat is NPT.
    variable_cell = isif is not None and isif >= 3

    thermostatted = False
    if mdalgo is not None:
        # 0 = standard (SMASS decides), 1 = Andersen, 2 = Nose-Hoover,
        # 3 = Langevin, 4 = NVE (Nose-Hoover chains), 5/13 = NPT variants.
        if mdalgo in (1, 2, 3, 5, 13):
            thermostatted = True
        elif mdalgo == 0:
            thermostatted = smass is not None and smass >= 0
    elif smass is not None:
        thermostatted = smass >= 0

    if mdalgo in (3, 5, 13) or variable_cell:
        return "NPT" if thermostatted else "NPH"
    return "NVT" if thermostatted else "NVE"


def target_temperature(tags: dict[str, str]) -> tuple[float, float] | None:
    """(TEBEG, TEEND) in K. TEEND defaults to TEBEG when only one is set."""
    begin = as_float(tags.get("TEBEG"))
    if begin is None:
        return None
    end = as_float(tags.get("TEEND"))
    return (begin, end if end is not None else begin)


def _attach_series(
    trajectory: MDTrajectory, oszicar, issues: list[Issue]
) -> None:
    """Align OSZICAR's per-ionic-step series onto the trajectory's frames.

    XDATCAR writes every NBLOCK-th step while OSZICAR writes every step, so the
    two axes coincide only when NBLOCK is 1. When they don't, the series is
    sub-sampled to match rather than silently plotted against the wrong axis;
    when neither interpretation fits, the series is dropped with a warning.
    """
    n_frames = trajectory.n_frames
    n_steps = oszicar.n_steps
    if n_steps == 0:
        return

    stride = 1
    if n_steps != n_frames:
        if n_frames > 0 and n_steps >= n_frames and n_steps % n_frames == 0:
            stride = n_steps // n_frames
            trajectory.frame_stride = stride
            log.info(
                "OSZICAR has %d steps for %d frames; sampling every %d",
                n_steps,
                n_frames,
                stride,
            )
        else:
            issues.append(
                Issue(
                    code="md_series_misaligned",
                    message=(
                        f"OSZICAR records {n_steps} steps but XDATCAR holds "
                        f"{n_frames} frames; the energy and temperature plots "
                        "were left out rather than shown against the wrong axis."
                    ),
                    severity=Severity.WARNING,
                    source="OSZICAR",
                )
            )
            return

    for key, (label, unit) in _ENERGY_FIELDS.items():
        values = oszicar.array(key)
        if values is None:
            continue
        sampled = values[stride - 1 :: stride] if stride > 1 else values
        sampled = sampled[:n_frames]
        if sampled.shape[0] != n_frames:
            continue
        trajectory.series[key] = MDSeries(
            name=label, values=sampled, unit=unit, source="OSZICAR"
        )

    if "OSZICAR" not in trajectory.source_files:
        trajectory.source_files.append("OSZICAR")


def load_trajectory(index: FileIndex) -> ParseResult[MDTrajectory]:
    """Read every MD source the directory offers and combine them."""
    xdatcar_path = index.path_for("xdatcar")
    if xdatcar_path is None:
        return ParseResult.failure(
            "md_no_trajectory",
            "No XDATCAR was found in this calculation.",
            source="XDATCAR",
        )

    result = parse_xdatcar(xdatcar_path)
    if not result.ok or result.value is None:
        return result

    trajectory = result.value
    issues = list(result.issues)

    # --- INCAR: timing and ensemble ---------------------------------------
    incar_path = index.path_for("incar")
    tags: dict[str, str] = {}
    if incar_path is not None:
        parsed = parse_incar(incar_path)
        if parsed.ok and parsed.value is not None:
            tags = parsed.value
            trajectory.source_files.append("INCAR")

    potim = as_float(tags.get("POTIM"))
    if potim is not None and potim > 0:
        trajectory.timestep = potim
    else:
        issues.append(
            Issue(
                code="md_no_timestep",
                message=(
                    "POTIM was not found, so the trajectory is shown against "
                    "frame number rather than physical time."
                ),
                severity=Severity.INFO,
                source="INCAR",
            )
        )

    nblock = as_int(tags.get("NBLOCK"))
    if nblock is not None and nblock > 1:
        trajectory.frame_stride = nblock

    trajectory.ensemble = describe_ensemble(tags)
    trajectory.target_temperature = target_temperature(tags)

    # --- OSZICAR: temperature and energies ---------------------------------
    oszicar_path = index.path_for("oszicar")
    if oszicar_path is not None:
        parsed_oszicar = parse_oszicar(oszicar_path)
        if parsed_oszicar.ok and parsed_oszicar.value is not None:
            _attach_series(trajectory, parsed_oszicar.value, issues)
        else:
            issues.extend(parsed_oszicar.issues)

    log.info(
        "Loaded MD trajectory: %d frames, %s, timestep %s fs, ensemble %s",
        trajectory.n_frames,
        trajectory.formula,
        trajectory.timestep,
        trajectory.ensemble or "unknown",
    )
    return ParseResult.success(trajectory, issues)


def frame_for_time(trajectory: MDTrajectory, value: float) -> int:
    """Nearest frame to a point on the plotted x-axis.

    Used when the user clicks a plot: the x-axis is picoseconds when POTIM is
    known and frame index otherwise, and this resolves either to a frame.
    """
    times = trajectory.times()
    if times.size == 0:
        return 0
    return int(np.argmin(np.abs(times - float(value))))


def summarise_source_files(index: FileIndex) -> str:
    """A compact list of the files an MD page is actually reading."""
    names = []
    for key in ("xdatcar", "oszicar", "incar", "poscar", "outcar"):
        path: Path | None = index.path_for(key)
        if path is not None:
            names.append(path.name)
    return ", ".join(names) if names else "—"
