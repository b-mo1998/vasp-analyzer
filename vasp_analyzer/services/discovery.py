"""Directory scanning and capability determination.

The scan is deliberately cheap: it stats files, peeks at a few headers, and
parses only the small text inputs (INCAR, KPOINTS, POSCAR). EIGENVAL, DOSCAR
and friends are *not* read here — the loader reads them in the background when
a page first needs them.

The result is a :class:`Calculation` whose ``capabilities`` drive the entire UI.
"""

from __future__ import annotations

from pathlib import Path

from ..core.errors import Issue, Severity
from ..core.logging_setup import get_logger
from ..models.calculation import Calculation
from ..models.capabilities import Capability, CapabilitySet
from ..models.files import (
    ALL_ROLES,
    FileEntry,
    FileGroup,
    FileIndex,
    FileStatus,
)
from ..models.metadata import CalculationMetadata, Provenance, Sourced
from ..parsers.eigenval import peek_eigenval_header
from ..parsers.incar import (
    as_bool,
    as_float,
    as_int,
    describe_functional,
    describe_spin,
    parse_incar,
)
from ..parsers.kpoints import parse_kpoints
from ..parsers.outcar import parse_outcar
from ..parsers.poscar import parse_poscar
from ..parsers.wannier import parse_win
from .validation import validate_consistency

log = get_logger("services.discovery")

# Files whose presence we accept with a different case or a suffix, e.g.
# "INCAR.relax" is not INCAR, but "incar" is.
_MIN_USEFUL_BYTES = 8


def _match_role_filename(name: str, role_filename: str, wannier_prefix: str) -> bool:
    lowered = name.lower()
    target = role_filename.lower()
    if lowered == target:
        return True
    # Allow a custom Wannier90 seedname, e.g. "si.win" for "wannier90.win".
    if target.startswith("wannier90"):
        suffix = target[len("wannier90"):]
        return lowered == f"{wannier_prefix.lower()}{suffix}"
    return False


def _detect_wannier_prefix(names: list[str]) -> str:
    """Infer the Wannier90 seedname from whatever ``*.win`` file is present."""
    for name in names:
        if name.lower().endswith(".win"):
            return name[: -len(".win")]
    return "wannier90"


def scan_directory(directory: Path) -> FileIndex:
    """Index a directory: which recognised files exist and in what state."""
    directory = Path(directory)
    names = []
    if directory.is_dir():
        names = [p.name for p in directory.iterdir() if p.is_file()]

    prefix = _detect_wannier_prefix(names)
    index = FileIndex(directory=directory, wannier_prefix=prefix)
    claimed: set[str] = set()

    for role in ALL_ROLES:
        entry = FileEntry(role=role)
        for name in names:
            if _match_role_filename(name, role.filename, prefix):
                path = directory / name
                try:
                    stat = path.stat()
                    entry.path = path
                    entry.size = stat.st_size
                    entry.mtime = stat.st_mtime
                    if stat.st_size < _MIN_USEFUL_BYTES:
                        entry.status = FileStatus.MALFORMED
                        entry.note = "File is empty."
                    else:
                        entry.status = FileStatus.FOUND
                except OSError as exc:
                    entry.status = FileStatus.UNREADABLE
                    entry.note = "File could not be read."
                    log.warning("Cannot stat %s: %s", path, exc)
                claimed.add(name)
                break
        index.entries[role.key] = entry

    index.unrecognised = sorted(n for n in names if n not in claimed and not n.startswith("."))
    log.info(
        "Scanned %s: %d recognised files, %d other files",
        directory,
        sum(1 for e in index.entries.values() if e.exists),
        len(index.unrecognised),
    )
    return index


def _collect_metadata(index: FileIndex, issues: list[Issue]) -> tuple[CalculationMetadata, dict]:
    """Parse the small input files and assemble metadata with provenance."""
    metadata = CalculationMetadata()
    context: dict = {}

    # --- EIGENVAL header ---------------------------------------------------
    # Six lines, and the most direct statement of the calculation's dimensions.
    # Read before OUTCAR so these primary values are never overwritten by a
    # stale output file.
    eigenval_path = index.path_for("eigenval")
    if eigenval_path:
        header = peek_eigenval_header(eigenval_path)
        if header is None:
            entry = index.entries.get("eigenval")
            if entry:
                entry.status = FileStatus.MALFORMED
                entry.note = "EIGENVAL header could not be read."
            issues.append(
                Issue(
                    code="eigenval_malformed",
                    message="EIGENVAL was found but could not be parsed.",
                    severity=Severity.WARNING,
                    source="EIGENVAL",
                )
            )
        else:
            context["eigenval_header"] = header
            provenance = Provenance("EIGENVAL")
            metadata.n_bands = Sourced(header.n_bands, provenance)
            metadata.n_kpoints = Sourced(header.n_kpoints, provenance)
            metadata.n_electrons = Sourced(header.n_electrons, provenance)

    # --- INCAR -------------------------------------------------------------
    incar_path = index.path_for("incar")
    if incar_path:
        result = parse_incar(incar_path)
        issues.extend(result.issues)
        if result.ok:
            tags = result.value or {}
            context["incar"] = tags
            metadata.incar_tags = tags
            provenance = Provenance("INCAR")
            if tags.get("SYSTEM"):
                metadata.system = Sourced(tags["SYSTEM"], provenance)
            functional = describe_functional(tags)
            if functional:
                metadata.functional = Sourced(functional, provenance)
            spin = describe_spin(tags)
            if spin:
                metadata.spin_mode = Sourced(spin, provenance)
            encut = as_float(tags.get("ENCUT")) or as_float(tags.get("ENMAX"))
            if encut is not None:
                metadata.encut = Sourced(encut, provenance)

    # --- KPOINTS -----------------------------------------------------------
    kpoints_path = index.path_for("kpoints")
    if kpoints_path:
        result = parse_kpoints(kpoints_path)
        issues.extend(result.issues)
        if result.ok and result.value:
            context["kpoints"] = result.value
            label = result.value.mesh_label
            if label:
                metadata.kmesh = Sourced(label, Provenance("KPOINTS"))

    # --- Structure (POSCAR preferred, CONTCAR as fallback) ------------------
    structure_key = None
    if index.has("poscar"):
        structure_key = "poscar"
    elif index.has("contcar"):
        structure_key = "contcar"
    if structure_key:
        path = index.path_for(structure_key)
        assert path is not None
        result = parse_poscar(path, source=path.name)
        issues.extend(result.issues)
        if result.ok and result.value:
            context["structure"] = result.value
            metadata.formula = Sourced(result.value.formula, Provenance(path.name))
            if metadata.system is None and result.value.comment:
                metadata.system = Sourced(result.value.comment, Provenance(path.name))
        elif result.failed:
            entry = index.entries.get(structure_key)
            if entry:
                entry.status = FileStatus.MALFORMED
                entry.note = result.issues[0].message if result.issues else ""

    # --- OUTCAR ------------------------------------------------------------
    outcar_path = index.path_for("outcar")
    if outcar_path:
        result = parse_outcar(outcar_path)
        issues.extend(result.issues)
        if result.ok and result.value:
            context["outcar"] = result.value
        elif result.failed:
            entry = index.entries.get("outcar")
            if entry:
                entry.status = FileStatus.MALFORMED
                entry.note = result.issues[0].message if result.issues else ""

    # --- wannier90.win -----------------------------------------------------
    win_path = index.path_for("w90_win")
    if win_path:
        result = parse_win(win_path)
        issues.extend(result.issues)
        if result.ok:
            context["wannier"] = result.value

    return metadata, context


def _apply_outcar_metadata(
    metadata: CalculationMetadata, context: dict, trusted: bool
) -> None:
    """Fold OUTCAR facts into the metadata, but only if it matches this run."""
    info = context.get("outcar")
    if info is None or not trusted:
        return
    provenance = Provenance("OUTCAR")
    if info.vasp_version:
        metadata.vasp_version = Sourced(info.vasp_version, provenance)
    if info.encut is not None:
        metadata.encut = Sourced(info.encut, provenance)  # measured beats requested
    # EIGENVAL already stated the dimensions; only fill gaps it left.
    if info.n_bands is not None and metadata.n_bands is None:
        metadata.n_bands = Sourced(info.n_bands, provenance)
    if info.n_kpoints is not None and metadata.n_kpoints is None:
        metadata.n_kpoints = Sourced(info.n_kpoints, provenance)
    if info.n_electrons is not None and metadata.n_electrons is None:
        metadata.n_electrons = Sourced(info.n_electrons, provenance)
    if info.kmesh_label and metadata.kmesh is None:
        metadata.kmesh = Sourced(info.kmesh_label, provenance)
    if info.spin_mode and metadata.spin_mode is None:
        metadata.spin_mode = Sourced(info.spin_mode, provenance)


def _determine_capabilities(
    index: FileIndex, context: dict, consistency
) -> CapabilitySet:
    caps = CapabilitySet()

    # Bands ---------------------------------------------------------------
    eigenval = index.entries.get("eigenval")
    if eigenval and eigenval.usable:
        caps.enable(Capability.BANDS)
    elif eigenval and eigenval.status is FileStatus.MALFORMED:
        caps.disable(Capability.BANDS, "EIGENVAL could not be read.")
    else:
        caps.disable(Capability.BANDS, "EIGENVAL was not found in this calculation.")

    # DOS -----------------------------------------------------------------
    doscar = index.entries.get("doscar")
    if doscar and doscar.usable:
        caps.enable(Capability.DOS)
    elif doscar and doscar.status is FileStatus.MALFORMED:
        caps.disable(Capability.DOS, "DOSCAR could not be read.")
    else:
        caps.disable(Capability.DOS, "DOSCAR was not found in this calculation.")

    lorbit = (context.get("incar") or {}).get("LORBIT")
    caps.set(
        Capability.DOS_PROJECTED,
        caps.has(Capability.DOS) and lorbit is not None,
        "Projected DOS requires LORBIT in the INCAR.",
    )

    # Structure -----------------------------------------------------------
    if context.get("structure") is not None:
        caps.enable(Capability.STRUCTURE)
    elif index.entries.get("poscar", FileEntry(ALL_ROLES[0])).exists:
        caps.disable(Capability.STRUCTURE, "The structure file could not be read.")
    else:
        caps.disable(Capability.STRUCTURE, "No POSCAR or CONTCAR was found.")

    # k-path labels -------------------------------------------------------
    kpoints = context.get("kpoints")
    if kpoints is not None and kpoints.is_line_mode:
        caps.enable(Capability.KPATH_LABELS)
    elif kpoints is not None:
        caps.disable(
            Capability.KPATH_LABELS,
            f"KPOINTS describes a {kpoints.mode} mesh, not a band path.",
        )
    else:
        caps.disable(Capability.KPATH_LABELS, "KPOINTS was not found.")

    # Projections ---------------------------------------------------------
    incar = context.get("incar") or {}
    spin_polarised = as_int(incar.get("ISPIN")) == 2
    noncollinear = bool(as_bool(incar.get("LNONCOLLINEAR")) or as_bool(incar.get("LSORBIT")))
    outcar = context.get("outcar")
    if outcar is not None and consistency.outcar_trusted:
        spin_polarised = spin_polarised or outcar.ispin == 2
        noncollinear = noncollinear or bool(outcar.noncollinear or outcar.soc)

    caps.set(
        Capability.SPIN_PROJECTION,
        caps.has(Capability.BANDS) and (spin_polarised or noncollinear),
        "This calculation has a single spin channel.",
    )
    procar_entry = index.entries.get("procar")
    procar_present = bool(procar_entry and procar_entry.exists)
    procar_ok = index.has("procar") and consistency.procar_trusted
    caps.set(
        Capability.ELEMENT_PROJECTION,
        caps.has(Capability.BANDS) and (procar_ok or index.has("vasprun")),
        (
            "PROCAR describes a different calculation and was ignored."
            if procar_present and not consistency.procar_trusted
            else "Element projection needs PROCAR or vasprun.xml."
        ),
    )

    # Wannier -------------------------------------------------------------
    # wannier90.eig + wannier90.win alone are NOT enough: .eig is the raw DFT
    # eigenvalues on Wannier90's own construction mesh (mp_grid), a different
    # k-point set from the EIGENVAL/KPOINTS band path, and no interpolation
    # has actually happened without a completed run (.chk) or its output
    # (wannier90_band.dat). Treating that pair as sufficient would either plot
    # nothing meaningful or silently mismatch two different k-point sets.
    if index.has("w90_band_dat"):
        caps.enable(Capability.WANNIER_BANDS)
    elif index.has("w90_win"):
        caps.disable(
            Capability.WANNIER_BANDS,
            "No interpolated Wannier bands (wannier90_band.dat) were found. "
            "wannier90.win/.eig alone describe the Wannierisation setup, not "
            "a completed interpolation.",
        )
    else:
        caps.disable(
            Capability.WANNIER_BANDS, "No Wannier90 files were found in this directory."
        )

    # Inputs / metadata ---------------------------------------------------
    caps.set(
        Capability.INPUT_FILES,
        any(e.exists for e in index.entries.values() if not e.role.hidden),
        "No recognised input files were found.",
    )
    caps.set(
        Capability.METADATA,
        index.has("incar") or index.has("outcar") or index.has("vasprun"),
        "No metadata source (INCAR, OUTCAR or vasprun.xml) was found.",
    )
    return caps


def discover(directory: Path) -> Calculation:
    """Scan a directory and produce a fully described :class:`Calculation`."""
    directory = Path(directory)
    issues: list[Issue] = []

    if not directory.exists():
        index = FileIndex(directory=directory)
        calculation = Calculation(directory=directory, index=index)
        calculation.issues.append(
            Issue(
                code="directory_missing",
                message="That folder no longer exists.",
                severity=Severity.ERROR,
            )
        )
        calculation.capabilities = _determine_capabilities(
            index, {}, validate_consistency(index, {}, [])
        )
        return calculation

    index = scan_directory(directory)
    metadata, context = _collect_metadata(index, issues)
    consistency = validate_consistency(index, context, issues)
    _apply_outcar_metadata(metadata, context, consistency.outcar_trusted)

    # Fermi level: trust order is DOSCAR header, then OUTCAR, both only when the
    # file describes *this* calculation. EIGENVAL carries no Fermi level, so a
    # directory without either leaves the value undetermined on purpose.
    fermi = consistency.fermi_candidate
    if fermi is not None:
        metadata.e_fermi = fermi

    capabilities = _determine_capabilities(index, context, consistency)

    calculation = Calculation(
        directory=directory,
        index=index,
        capabilities=capabilities,
        metadata=metadata,
        issues=issues,
    )
    log.info(
        "Capabilities for %s: %s",
        directory.name,
        ", ".join(sorted(c.value for c in capabilities.available)) or "none",
    )
    return calculation
