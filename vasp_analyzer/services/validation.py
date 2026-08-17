"""Cross-file consistency checking.

Calculation folders are assembled by hand, and stale files from a previous run
routinely survive a copy. A leftover OUTCAR will happily report a Fermi level,
an ENCUT and a k-mesh belonging to a completely different system.

This module compares the cheap headers of the files that *are* present and
decides which ones actually describe the calculation in front of us. Anything
that disagrees is demoted rather than used, and the mismatch is reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.errors import Issue, Severity
from ..core.logging_setup import get_logger
from ..models.files import FileIndex, FileStatus
from ..models.metadata import Provenance, Sourced
from ..parsers.doscar import peek_doscar_header
from ..parsers.eigenval import peek_eigenval_header

log = get_logger("services.validation")

FERMI_DISAGREEMENT_EV = 0.05


@dataclass
class ConsistencyReport:
    outcar_trusted: bool = True
    doscar_trusted: bool = True
    fermi_candidate: Sourced[float] | None = None
    mismatches: list[str] = field(default_factory=list)


def _describe(name: str, expected, actual) -> str:
    return f"{name} {actual} vs {expected}"


def validate_consistency(
    index: FileIndex, context: dict, issues: list[Issue]
) -> ConsistencyReport:
    report = ConsistencyReport()

    eig_header = context.get("eigenval_header")
    if eig_header is None:
        eig_path = index.path_for("eigenval")
        if eig_path:
            eig_header = peek_eigenval_header(eig_path)

    dos_header = None
    dos_path = index.path_for("doscar")
    if dos_path:
        dos_header = peek_doscar_header(dos_path)
        if dos_header is None:
            entry = index.entries.get("doscar")
            if entry:
                entry.status = FileStatus.MALFORMED
                entry.note = "DOSCAR header could not be read."
            issues.append(
                Issue(
                    code="doscar_malformed",
                    message="DOSCAR was found but could not be parsed.",
                    severity=Severity.WARNING,
                    source="DOSCAR",
                )
            )

    structure = context.get("structure")
    outcar = context.get("outcar")

    # --- Does OUTCAR describe this calculation? ---------------------------
    if outcar is not None:
        mismatches: list[str] = []
        if eig_header is not None:
            if outcar.n_bands is not None and outcar.n_bands != eig_header.n_bands:
                mismatches.append(
                    _describe("NBANDS", outcar.n_bands, eig_header.n_bands)
                )
            if outcar.n_kpoints is not None and outcar.n_kpoints != eig_header.n_kpoints:
                mismatches.append(
                    _describe("NKPTS", outcar.n_kpoints, eig_header.n_kpoints)
                )
            if (
                outcar.n_electrons is not None
                and abs(outcar.n_electrons - eig_header.n_electrons) > 1e-6
            ):
                mismatches.append(
                    _describe(
                        "NELECT",
                        f"{outcar.n_electrons:g}",
                        f"{eig_header.n_electrons:g}",
                    )
                )
        if structure is not None:
            if outcar.n_ions is not None and outcar.n_ions != structure.n_atoms:
                mismatches.append(_describe("ion count", outcar.n_ions, structure.n_atoms))
            if outcar.potcar_species:
                expected = set(outcar.potcar_species)
                actual = set(structure.species)
                if expected and actual and not (expected & actual):
                    mismatches.append(
                        _describe(
                            "elements",
                            "/".join(sorted(expected)),
                            "/".join(sorted(actual)),
                        )
                    )

        if mismatches:
            report.outcar_trusted = False
            report.mismatches.extend(mismatches)
            entry = index.entries.get("outcar")
            if entry:
                entry.status = FileStatus.INCONSISTENT
                entry.note = "Describes a different calculation."
            issues.append(
                Issue(
                    code="outcar_inconsistent",
                    message=(
                        "OUTCAR describes a different calculation and was ignored."
                    ),
                    detail="Mismatched: " + "; ".join(mismatches),
                    severity=Severity.WARNING,
                    source="OUTCAR",
                )
            )
            log.warning("OUTCAR rejected as inconsistent: %s", "; ".join(mismatches))

    # --- Does DOSCAR describe this calculation? ---------------------------
    if dos_header is not None and structure is not None:
        if dos_header.n_ions and dos_header.n_ions != structure.n_atoms:
            report.doscar_trusted = False
            entry = index.entries.get("doscar")
            if entry:
                entry.status = FileStatus.INCONSISTENT
                entry.note = "Describes a different calculation."
            issues.append(
                Issue(
                    code="doscar_inconsistent",
                    message="DOSCAR describes a different structure and was ignored.",
                    detail=_describe("ion count", dos_header.n_ions, structure.n_atoms),
                    severity=Severity.WARNING,
                    source="DOSCAR",
                )
            )

    # --- Fermi level: pick the most trustworthy source ---------------------
    candidates: list[tuple[float, str]] = []
    if dos_header is not None and report.doscar_trusted and dos_header.e_fermi is not None:
        candidates.append((dos_header.e_fermi, "DOSCAR"))
    if outcar is not None and report.outcar_trusted and outcar.e_fermi is not None:
        candidates.append((outcar.e_fermi, "OUTCAR"))

    if candidates:
        value, source = candidates[0]
        report.fermi_candidate = Sourced(value, Provenance(source))
        if len(candidates) > 1:
            other_value, other_source = candidates[1]
            if abs(other_value - value) > FERMI_DISAGREEMENT_EV:
                issues.append(
                    Issue(
                        code="fermi_disagreement",
                        message=(
                            f"{source} and {other_source} report different Fermi "
                            f"levels; {source} was used."
                        ),
                        detail=(
                            f"{source}: {value:.4f} eV, "
                            f"{other_source}: {other_value:.4f} eV"
                        ),
                        severity=Severity.WARNING,
                    )
                )
    return report
