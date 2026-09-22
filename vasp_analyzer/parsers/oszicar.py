"""OSZICAR parser: per-ionic-step energies, and MD temperature.

For a molecular-dynamics run VASP writes one summary line per ionic step::

    1 T=   300. E= -.45883648E+03 F= -.46073664E+03 E0= -.46073664E+03 \
        EK= 0.19002E+01 SP= 0.00E+00 SK= 0.00E+00

with

``T``   instantaneous ionic temperature (K)
``F``   free energy of the electronic system — the potential energy
``EK``  kinetic energy of the ions
``SP``/``SK``  Nose-Hoover thermostat potential and kinetic terms
``E``   the conserved total, ``F + EK + SP + SK``

A relaxation writes the same line without ``T``/``EK``. Both are read here; the
absence of a temperature is what tells the MD page it has a relaxation rather
than a thermostatted run, and it disables the temperature plot instead of
inventing one.

OSZICAR is chosen over OUTCAR for this deliberately: it carries the identical
per-step quantities in a few hundred kilobytes where the OUTCAR of the same run
is tens of megabytes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.errors import ParseResult, safe_parser
from ..core.logging_setup import get_logger

log = get_logger("parsers.oszicar")

# VASP writes Fortran-style floats that may omit the leading zero ("-.4588E+03"),
# which float() accepts but a naive [\d.] character class does not.
_NUMBER = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?)"
_RE_STEP = re.compile(r"^\s*(\d+)\s+(?:T=|F=)")
_RE_FIELD = {
    "temperature": re.compile(r"\bT=\s*" + _NUMBER),
    "total_energy": re.compile(r"\bE=\s*" + _NUMBER),
    "potential_energy": re.compile(r"\bF=\s*" + _NUMBER),
    "energy_sigma0": re.compile(r"\bE0=\s*" + _NUMBER),
    "kinetic_energy": re.compile(r"\bEK=\s*" + _NUMBER),
    "nose_potential": re.compile(r"\bSP=\s*" + _NUMBER),
    "nose_kinetic": re.compile(r"\bSK=\s*" + _NUMBER),
}


@dataclass
class OszicarData:
    """Per-ionic-step series. Every list may legitimately be empty."""

    steps: list[int] = field(default_factory=list)
    fields: dict[str, list[float]] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def is_molecular_dynamics(self) -> bool:
        """A thermostatted/velocity-carrying run reports T and EK per step."""
        return bool(self.fields.get("temperature")) and bool(
            self.fields.get("kinetic_energy")
        )

    def array(self, name: str) -> np.ndarray | None:
        values = self.fields.get(name)
        if not values or len(values) != self.n_steps:
            return None
        return np.array(values, dtype=float)


def _to_float(token: str) -> float:
    # Fortran sometimes emits a D exponent; Python only understands E.
    return float(token.replace("D", "E").replace("d", "e"))


@safe_parser("OSZICAR")
def parse_oszicar(path: Path) -> ParseResult[OszicarData]:
    path = Path(path)
    data = OszicarData()
    collected: dict[str, list[float]] = {name: [] for name in _RE_FIELD}

    with path.open("r", errors="replace") as handle:
        for line in handle:
            # Ionic summary lines start with the step number followed by T= or
            # F=; electronic iteration lines (DAV:, RMM:, ...) never do.
            match = _RE_STEP.match(line)
            if match is None:
                continue
            data.steps.append(int(match.group(1)))
            for name, regex in _RE_FIELD.items():
                found = regex.search(line)
                if found is not None:
                    collected[name].append(_to_float(found.group(1)))

    if not data.steps:
        return ParseResult.failure(
            "oszicar_no_steps",
            "OSZICAR contains no completed ionic steps.",
            source="OSZICAR",
        )

    # A field is kept only when it was present on every step: a partially
    # populated series would silently misalign against the frame axis.
    data.fields = {
        name: values for name, values in collected.items() if len(values) == len(data.steps)
    }

    log.info(
        "Parsed OSZICAR: %d ionic steps, fields: %s",
        data.n_steps,
        ", ".join(sorted(data.fields)) or "none",
    )
    return ParseResult.success(data)


def peek_oszicar_steps(path: Path) -> int | None:
    """Count ionic steps without building the arrays."""
    try:
        count = 0
        with Path(path).open("r", errors="replace") as handle:
            for line in handle:
                if _RE_STEP.match(line):
                    count += 1
        return count or None
    except OSError as exc:
        log.warning("Could not peek OSZICAR at %s: %s", path, exc)
        return None
