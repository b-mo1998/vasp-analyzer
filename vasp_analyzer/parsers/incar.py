"""INCAR parser.

Returns tags in file order with comments stripped. Multiple assignments on one
line separated by ``;`` are supported, as are ``#`` and ``!`` comments.
"""

from __future__ import annotations

from pathlib import Path

from ..core.errors import Issue, ParseResult, Severity, safe_parser
from ..core.logging_setup import get_logger

log = get_logger("parsers.incar")


def _strip_comment(line: str) -> str:
    for marker in ("#", "!"):
        pos = line.find(marker)
        if pos != -1:
            line = line[:pos]
    return line.strip()


@safe_parser("INCAR")
def parse_incar(path: Path) -> ParseResult[dict[str, str]]:
    text = Path(path).read_text(errors="replace")
    tags: dict[str, str] = {}
    issues: list[Issue] = []

    for number, raw in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw)
        if not line:
            continue
        for statement in line.split(";"):
            statement = statement.strip()
            if not statement:
                continue
            if "=" not in statement:
                issues.append(
                    Issue(
                        code="incar_unparsed_line",
                        message=f"INCAR line {number} is not a TAG = VALUE assignment.",
                        detail=raw,
                        severity=Severity.INFO,
                        source="INCAR",
                    )
                )
                continue
            key, _, value = statement.partition("=")
            key = key.strip().upper()
            if key:
                tags[key] = value.strip()

    log.info("Parsed INCAR: %d tags", len(tags))
    return ParseResult.success(tags, issues)


def as_bool(value: str | None) -> bool | None:
    """Interpret a VASP logical (.TRUE. / T / .FALSE. / F)."""
    if value is None:
        return None
    token = value.strip().strip(".").upper()
    if token.startswith("T"):
        return True
    if token.startswith("F"):
        return False
    return None


def as_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.split()[0])
    except (ValueError, IndexError):
        return None


def as_int(value: str | None) -> int | None:
    result = as_float(value)
    return int(result) if result is not None else None


def describe_functional(tags: dict[str, str]) -> str | None:
    """Human label for the exchange-correlation treatment, or None.

    Only reports what the INCAR actually states. A plain PBE run usually sets no
    GGA tag at all, so absence of every relevant tag yields None rather than a
    guess of "PBE".
    """
    metagga = tags.get("METAGGA")
    gga = tags.get("GGA")
    hybrid = as_bool(tags.get("LHFCALC"))

    label: str | None = None
    if metagga:
        label = metagga.strip().upper()
    elif hybrid:
        label = "HSE" if tags.get("HFSCREEN") else "Hartree-Fock hybrid"
    elif gga:
        gga_names = {
            "PE": "PBE",
            "PS": "PBEsol",
            "RP": "RPBE",
            "AM": "AM05",
            "91": "PW91",
            "B3": "B3LYP",
            "MK": "optB88",
        }
        token = gga.strip().strip("'\"").upper()[:2]
        label = gga_names.get(token, f"GGA {token}")

    if label is None:
        return None

    if as_bool(tags.get("LSORBIT")):
        label += " + SOC"
    return label


def describe_spin(tags: dict[str, str]) -> str | None:
    noncollinear = as_bool(tags.get("LNONCOLLINEAR"))
    soc = as_bool(tags.get("LSORBIT"))
    ispin = as_int(tags.get("ISPIN"))
    if soc or noncollinear:
        return "Non-collinear"
    if ispin == 2:
        return "Spin-polarised"
    if ispin == 1:
        return "Non-spin-polarised"
    return None
