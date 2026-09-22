"""VTST post-processing files: ``neb.dat``, ``nebef.dat`` and ``spline.dat``.

The VTST toolkit's ``nebbarrier.pl`` / ``nebef.pl`` scripts summarise a finished
band into two small tables. When they are present they are the most reliable
source available, because they carry two things a bare OUTCAR does not:

* the **reaction coordinate**, already accumulated with correct periodic
  handling, and
* the **projected NEB force** on each movable image — the quantity the
  optimiser actually drives to zero, which differs from the raw atomic force
  by the spring and tangent projection.

``neb.dat`` columns are: image, reaction coordinate (Å), energy relative to
image 0 (eV), *tangential* force (eV/Å, signed — it is the force along the path
tangent used to build the spline, so its sign says which way the image is being
pushed), image index again.

``nebef.dat`` columns are: image, force *magnitude* (eV/Å), absolute total
energy (eV), energy relative to image 0 (eV).

The distinction matters for convergence: a signed tangential force of -3.3 is
not "below 0.03", and only ``nebef.dat``'s magnitude may be compared against
EDIFFG.

``spline.dat`` is the densely sampled cubic spline ``nebspline.pl`` fits
through the band, with columns: image coordinate, reaction coordinate (Å),
energy relative to image 0 (eV), and the spline's derivative. When it is
present it is the curve to draw, because it is the one VTST itself computed
from the tangents and forces — an interpolation made here from the image
energies alone would be a different curve wearing the same shape, and would
disagree with the barrier VTST reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..core.errors import ParseResult, safe_parser
from ..core.logging_setup import get_logger

log = get_logger("parsers.vtst")


@dataclass
class VtstProfile:
    """Per-image values read from neb.dat and/or nebef.dat.

    ``force`` is a magnitude only when :attr:`force_is_magnitude` is set —
    ``neb.dat`` supplies a signed tangential component instead, which must not
    be compared against a convergence threshold.
    """

    reaction_coordinate: list[float] = field(default_factory=list)
    relative_energy: list[float] = field(default_factory=list)
    absolute_energy: list[float] = field(default_factory=list)
    force: list[float] = field(default_factory=list)
    force_is_magnitude: bool = False
    sources: list[str] = field(default_factory=list)

    @property
    def n_images(self) -> int:
        return max(
            len(self.reaction_coordinate),
            len(self.relative_energy),
            len(self.absolute_energy),
            len(self.force),
        )


def _numeric_rows(path: Path, columns: int) -> list[list[float]]:
    rows: list[list[float]] = []
    for line in Path(path).read_text(errors="replace").splitlines():
        tokens = line.split()
        if len(tokens) < columns:
            continue
        try:
            rows.append([float(t) for t in tokens[:columns]])
        except ValueError:
            continue
    return rows


@safe_parser("neb.dat")
def parse_neb_dat(path: Path) -> ParseResult[VtstProfile]:
    rows = _numeric_rows(path, 4)
    if not rows:
        return ParseResult.failure(
            "neb_dat_empty", "neb.dat contains no image rows.", source="neb.dat"
        )
    profile = VtstProfile(
        reaction_coordinate=[r[1] for r in rows],
        relative_energy=[r[2] for r in rows],
        force=[r[3] for r in rows],
        force_is_magnitude=False,
        sources=["neb.dat"],
    )
    log.info("Parsed neb.dat: %d images", len(rows))
    return ParseResult.success(profile)


@safe_parser("nebef.dat")
def parse_nebef_dat(path: Path) -> ParseResult[VtstProfile]:
    rows = _numeric_rows(path, 4)
    if not rows:
        return ParseResult.failure(
            "nebef_dat_empty", "nebef.dat contains no image rows.", source="nebef.dat"
        )
    profile = VtstProfile(
        force=[r[1] for r in rows],
        absolute_energy=[r[2] for r in rows],
        relative_energy=[r[3] for r in rows],
        force_is_magnitude=True,
        sources=["nebef.dat"],
    )
    log.info("Parsed nebef.dat: %d images", len(rows))
    return ParseResult.success(profile)


@dataclass
class SplineCurve:
    """The dense interpolated path VTST fitted through the band."""

    reaction_coordinate: list[float] = field(default_factory=list)
    relative_energy: list[float] = field(default_factory=list)

    @property
    def n_points(self) -> int:
        return len(self.reaction_coordinate)

    @property
    def barrier(self) -> float | None:
        """The spline's own maximum — the barrier VTST would quote, which can
        sit between two images rather than on one."""
        return max(self.relative_energy) if self.relative_energy else None


@safe_parser("spline.dat")
def parse_spline_dat(path: Path) -> ParseResult[SplineCurve]:
    rows = _numeric_rows(path, 3)
    if not rows:
        return ParseResult.failure(
            "spline_dat_empty", "spline.dat contains no rows.", source="spline.dat"
        )
    curve = SplineCurve(
        reaction_coordinate=[r[1] for r in rows],
        relative_energy=[r[2] for r in rows],
    )
    log.info("Parsed spline.dat: %d points", curve.n_points)
    return ParseResult.success(curve)


def merge_profiles(
    neb: VtstProfile | None, nebef: VtstProfile | None
) -> VtstProfile | None:
    """Combine the two tables, preferring whichever carries each column.

    ``neb.dat`` alone has no absolute energies and ``nebef.dat`` alone has no
    reaction coordinate, so a directory with both yields a strictly more
    complete picture than either.
    """
    if neb is None and nebef is None:
        return None
    if neb is None:
        return nebef
    if nebef is None:
        return neb

    # nebef.dat's force is a magnitude and neb.dat's is a signed tangential
    # component, so the magnitude is always preferred when both exist.
    if nebef.force:
        force, is_magnitude = list(nebef.force), True
    else:
        force, is_magnitude = list(neb.force), neb.force_is_magnitude

    merged = VtstProfile(
        reaction_coordinate=list(neb.reaction_coordinate),
        relative_energy=list(neb.relative_energy or nebef.relative_energy),
        absolute_energy=list(nebef.absolute_energy),
        force=force,
        force_is_magnitude=is_magnitude,
        sources=["neb.dat", "nebef.dat"],
    )
    return merged
