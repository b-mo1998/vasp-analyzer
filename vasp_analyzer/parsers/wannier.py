"""Wannier90 parsers.

Only what the Bands page needs for now: the ``.win`` input (to learn the number
of Wannier functions and any declared k-path) and interpolated band files. The
heavier ``.wout`` analysis belongs to the future Wannier page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.errors import ParseResult, safe_parser
from ..core.logging_setup import get_logger

log = get_logger("parsers.wannier")


@dataclass
class WannierInput:
    """Parsed wannier90.win."""

    tags: dict[str, str] = field(default_factory=dict)
    blocks: dict[str, list[str]] = field(default_factory=dict)

    @property
    def num_wann(self) -> int | None:
        value = self.tags.get("num_wann")
        return int(value) if value and value.isdigit() else None

    @property
    def num_bands(self) -> int | None:
        value = self.tags.get("num_bands")
        return int(value) if value and value.isdigit() else None

    @property
    def wants_bandplot(self) -> bool:
        return self.tags.get("bands_plot", "").lower().startswith(("t", ".t"))

    @property
    def kpath_labels(self) -> list[str]:
        labels: list[str] = []
        for line in self.blocks.get("kpoint_path", []):
            tokens = line.split()
            if len(tokens) >= 8:
                labels.extend([tokens[0], tokens[4]])
        return labels


@safe_parser("wannier90.win")
def parse_win(path: Path) -> ParseResult[WannierInput]:
    text = Path(path).read_text(errors="replace")
    result = WannierInput()
    current_block: str | None = None

    for raw in text.splitlines():
        line = raw.split("!")[0].split("#")[0].strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith("begin"):
            current_block = lowered.split(None, 1)[1].strip() if " " in lowered else ""
            result.blocks[current_block] = []
            continue
        if lowered.startswith("end"):
            current_block = None
            continue
        if current_block is not None:
            result.blocks[current_block].append(line)
            continue
        if "=" in line or ":" in line:
            separator = "=" if "=" in line else ":"
            key, _, value = line.partition(separator)
            result.tags[key.strip().lower()] = value.strip()

    log.info(
        "Parsed wannier90.win: %d tags, %d blocks", len(result.tags), len(result.blocks)
    )
    return ParseResult.success(result)


@dataclass
class WannierBands:
    """Interpolated band structure from wannier90_band.dat."""

    x: np.ndarray               # (n_points,) path coordinate as written by W90
    energies: np.ndarray        # (n_bands, n_points) absolute eV

    @property
    def n_bands(self) -> int:
        return int(self.energies.shape[0])


@safe_parser("wannier90_band.dat")
def parse_band_dat(path: Path) -> ParseResult[WannierBands]:
    """Read wannier90_band.dat: blank-line-separated blocks, one per band."""
    text = Path(path).read_text(errors="replace")
    blocks: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if current:
                blocks.append(current)
                current = []
            continue
        tokens = line.split()
        if len(tokens) < 2:
            continue
        try:
            current.append((float(tokens[0]), float(tokens[1])))
        except ValueError:
            continue
    if current:
        blocks.append(current)

    if not blocks:
        return ParseResult.failure(
            "w90_band_dat_empty",
            "wannier90_band.dat contains no band data.",
            source="wannier90_band.dat",
        )

    length = len(blocks[0])
    usable = [b for b in blocks if len(b) == length]
    x = np.array([point[0] for point in usable[0]], dtype=float)
    energies = np.array(
        [[point[1] for point in block] for block in usable], dtype=float
    )
    log.info("Parsed wannier90_band.dat: %d bands x %d points", len(usable), length)
    return ParseResult.success(WannierBands(x=x, energies=energies))
