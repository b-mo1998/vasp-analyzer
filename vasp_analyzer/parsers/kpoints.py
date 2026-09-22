"""KPOINTS parser.

The important question for the Bands page is whether the run used *line mode*.
Only then do high-symmetry labels exist; a Monkhorst-Pack or automatic mesh has
no k-path, and the band plot must say so instead of inventing Γ–X–M.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..core.errors import ParseResult, safe_parser
from ..core.logging_setup import get_logger

log = get_logger("parsers.kpoints")

GREEK = {
    "GAMMA": "Γ",
    "G": "Γ",
    "\\GAMMA": "Γ",
    "DELTA": "Δ",
    "SIGMA": "Σ",
    "LAMBDA": "Λ",
}


@dataclass
class KpointsSpec:
    mode: str                      # "automatic" | "gamma" | "monkhorst" | "line" | "explicit"
    comment: str = ""
    mesh: tuple[int, int, int] | None = None
    shift: tuple[float, float, float] | None = None
    n_divisions: int = 0           # points per line segment (line mode)
    path_labels: list[str] = field(default_factory=list)
    path_coords: list[tuple[float, float, float]] = field(default_factory=list)
    n_explicit: int = 0

    @property
    def is_line_mode(self) -> bool:
        return self.mode == "line"

    @property
    def mesh_label(self) -> str | None:
        if self.mesh is None:
            return None
        a, b, c = self.mesh
        return f"{a}×{b}×{c}"


def _pretty_label(token: str) -> str:
    token = token.strip().strip("!#").strip()
    return GREEK.get(token.upper(), token)


@safe_parser("KPOINTS")
def parse_kpoints(path: Path) -> ParseResult[KpointsSpec]:
    lines = Path(path).read_text(errors="replace").splitlines()
    if len(lines) < 3:
        return ParseResult.failure(
            "kpoints_truncated",
            "KPOINTS is too short to describe a sampling scheme.",
            source="KPOINTS",
        )

    comment = lines[0].strip()
    try:
        count = int(float(lines[1].split()[0]))
    except (ValueError, IndexError):
        count = 0

    # Line mode inserts an extra flag line before the coordinate mode.
    third = lines[2].strip().upper()[:1] if lines[2].strip() else ""
    if third == "L":
        spec = KpointsSpec(mode="line", comment=comment, n_divisions=count)
        for raw in lines[4:]:
            body = raw.strip()
            if not body:
                continue
            parts = body.replace("!", " ! ").replace("#", " # ").split()
            numbers = []
            for token in parts[:3]:
                try:
                    numbers.append(float(token))
                except ValueError:
                    break
            if len(numbers) < 3:
                continue
            label = ""
            for marker in ("!", "#"):
                if marker in body:
                    label = _pretty_label(body.split(marker, 1)[1])
                    break
            else:
                # VASPKIT writes the label bare after the three coordinates,
                # with no comment marker at all:
                #     0.0000  0.0000 -0.5000    -Z
                # Without this the whole path comes back unlabelled and the
                # axis falls back to "Wave vector (Å⁻¹)" — which is exactly
                # what a VASPKIT-generated band structure was showing.
                if len(parts) > 3:
                    label = _pretty_label(parts[3])
            spec.path_coords.append((numbers[0], numbers[1], numbers[2]))
            spec.path_labels.append(label)
        log.info("Parsed KPOINTS: line mode, %d path points", len(spec.path_coords))
        return ParseResult.success(spec)

    if count == 0:
        mode = {"A": "automatic", "G": "gamma", "M": "monkhorst"}.get(third, "automatic")
        spec = KpointsSpec(mode=mode, comment=comment)
        if len(lines) > 3:
            try:
                grid = [int(float(t)) for t in lines[3].split()[:3]]
                if len(grid) == 3:
                    spec.mesh = (grid[0], grid[1], grid[2])
                elif len(grid) == 1:
                    spec.mesh = None  # fully automatic: a single length parameter
            except ValueError:
                pass
        if len(lines) > 4:
            try:
                shift = [float(t) for t in lines[4].split()[:3]]
                if len(shift) == 3:
                    spec.shift = (shift[0], shift[1], shift[2])
            except ValueError:
                pass
        log.info("Parsed KPOINTS: %s mesh %s", mode, spec.mesh_label)
        return ParseResult.success(spec)

    spec = KpointsSpec(mode="explicit", comment=comment, n_explicit=count)
    log.info("Parsed KPOINTS: %d explicit k-points", count)
    return ParseResult.success(spec)
