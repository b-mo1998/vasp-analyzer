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
from ..parsers.oszicar import peek_oszicar_steps
from ..parsers.poscar import parse_poscar
from ..parsers.wannier import parse_win
from ..parsers.xdatcar import peek_xdatcar_frames
from .layout import ProjectLayout, resolve_layout
from .neb import find_image_directories, looks_like_neb
from .validation import validate_consistency
from .volumetric import VolumetricSource, find_volumetric_sources

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


def scan_file(path: Path) -> FileIndex:
    """Index a single file as if it were the only thing in its directory.

    Dropping one POSCAR or one CHGCAR onto the window is a real thing users
    do, and it must not quietly pull in the rest of whatever folder that file
    happens to live in — a CHGCAR sitting beside an unrelated EIGENVAL would
    otherwise light up the Bands page with a different calculation's data.
    Restricting the index to the dropped file makes every downstream
    capability check answer honestly on its own.
    """
    path = Path(path)
    prefix = _detect_wannier_prefix([path.name])
    index = FileIndex(directory=path.parent, wannier_prefix=prefix)
    matched = False

    for role in ALL_ROLES:
        entry = FileEntry(role=role)
        if not matched and _match_role_filename(path.name, role.filename, prefix):
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
                matched = True
            except OSError as exc:
                entry.status = FileStatus.UNREADABLE
                entry.note = "File could not be read."
                log.warning("Cannot stat %s: %s", path, exc)
                matched = True
        index.entries[role.key] = entry

    index.unrecognised = [] if matched else [path.name]
    return index


def _collect_metadata(
    index: FileIndex,
    issues: list[Issue],
    volumetric_sources: list | None = None,
) -> tuple[CalculationMetadata, dict]:
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

    # --- volumetric sources -------------------------------------------------
    # Only names and sizes are inspected. A CHGCAR can be hundreds of
    # megabytes; the Charge Density page reads it in the background when asked.
    sources = (
        find_volumetric_sources(index.directory)
        if volumetric_sources is None
        else list(volumetric_sources)
    )
    context["volumetric_sources"] = sources

    # A volumetric file opens with a complete POSCAR block, so a calculation
    # that has a CHGCAR but no POSCAR or CONTCAR still knows its own
    # structure — and the Structure and Editor pages have something real to
    # show instead of an empty state that is not actually true. Only ever a
    # fallback: an explicit POSCAR/CONTCAR is the authoritative statement of
    # the geometry and is never overridden by a density file's copy of it.
    if context.get("structure") is None and sources:
        from ..parsers.chgcar import peek_volumetric_structure  # noqa: PLC0415

        source_path = sources[0].path
        peeked = peek_volumetric_structure(source_path)
        if peeked is not None:
            context["structure"] = peeked
            provenance = Provenance(source_path.name)
            metadata.formula = Sourced(peeked.formula, provenance)
            if metadata.system is None and peeked.comment:
                metadata.system = Sourced(peeked.comment, provenance)

    return metadata, context


def _collect_md_context(md_root: Path) -> dict:
    """Scan the resolved MD root independently of the electronic-structure
    index — a project layout can point it at an entirely different directory.

    Only counts are taken here. Reading the coordinates of a multi-gigabyte
    XDATCAR at scan time would defeat the point of a cheap discovery pass;
    the loader reads them in the background when the MD page asks.
    """
    context: dict = {}
    md_index = scan_directory(md_root)
    context["md_index"] = md_index

    xdatcar_path = md_index.path_for("xdatcar")
    if xdatcar_path:
        context["xdatcar_frames"] = peek_xdatcar_frames(xdatcar_path)
    oszicar_path = md_index.path_for("oszicar")
    if oszicar_path:
        context["oszicar_steps"] = peek_oszicar_steps(oszicar_path)
    return context


def _collect_neb_context(neb_root: Path) -> dict:
    """Scan the resolved NEB root independently of the electronic-structure
    index, for the same reason as :func:`_collect_md_context`."""
    context: dict = {"neb_index": scan_directory(neb_root)}
    if looks_like_neb(neb_root):
        folders = find_image_directories(neb_root)
        context["neb_images"] = folders
        context["neb_has_output"] = any(
            (path / "OUTCAR").is_file() for _, path in folders
        ) or (neb_root / "nebef.dat").is_file()
    return context


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

    # Molecular dynamics --------------------------------------------------
    # A trajectory needs at least two configurations: a single-frame XDATCAR is
    # what a static or one-shot run leaves behind, and there is nothing to
    # animate there. Read from the MD-specific index — a project layout can
    # point it at a different directory than `index`/`directory` above.
    md_index = context.get("md_index") or index
    xdatcar = md_index.entries.get("xdatcar")
    n_frames = context.get("xdatcar_frames")
    if xdatcar and xdatcar.usable and n_frames and n_frames >= 2:
        caps.enable(Capability.MD_TRAJECTORY)
    elif xdatcar and xdatcar.exists and n_frames == 1:
        caps.disable(
            Capability.MD_TRAJECTORY,
            "XDATCAR holds a single configuration, so there is no trajectory "
            "to play.",
        )
    elif xdatcar and xdatcar.exists:
        caps.disable(
            Capability.MD_TRAJECTORY, "XDATCAR was found but contains no configurations."
        )
    else:
        caps.disable(
            Capability.MD_TRAJECTORY, "No XDATCAR was found in this calculation."
        )

    # Temperature and energies come from OSZICAR's per-ionic-step lines.
    oszicar_steps = context.get("oszicar_steps")
    caps.set(
        Capability.MD_THERMODYNAMICS,
        caps.has(Capability.MD_TRAJECTORY) and bool(oszicar_steps),
        (
            "No OSZICAR was found, so temperature and energy histories are "
            "unavailable."
            if not oszicar_steps
            else "This calculation has no trajectory to plot against."
        ),
    )

    # NEB -------------------------------------------------------------------
    neb_images = context.get("neb_images") or []
    if neb_images:
        caps.enable(Capability.NEB_PATH)
        caps.set(
            Capability.NEB_ENERGIES,
            bool(context.get("neb_has_output")),
            "No image folder contains an OUTCAR, so no energies can be read.",
        )
    else:
        reason = (
            "This directory does not contain numbered NEB image folders "
            "(00, 01, 02, …)."
        )
        caps.disable(Capability.NEB_PATH, reason)
        caps.disable(Capability.NEB_ENERGIES, reason)

    # Charge density / volumetric ------------------------------------------
    sources = context.get("volumetric_sources") or []
    if sources:
        caps.enable(Capability.CHARGE_DENSITY)
    else:
        caps.disable(
            Capability.CHARGE_DENSITY,
            "No volumetric file (CHGCAR, CHG, PARCHG or ELFCAR) was found in "
            "this calculation.",
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
    """Scan a directory and produce a fully described :class:`Calculation`.

    ``directory`` may itself be a flat calculation directory, or a project
    folder organised into ``band``/``md``/``neb`` subdirectories — see
    :mod:`services.layout`. Either way, exactly one :class:`Calculation`
    comes back, with each page family already pointed at the right place.
    """
    directory = Path(directory)
    if not directory.exists():
        return _discover_missing(directory)
    return discover_with_layout(directory, resolve_layout(directory))


def discover_with_layout(opened_directory: Path, layout: ProjectLayout) -> Calculation:
    """Like :func:`discover`, but with an already-resolved layout.

    Used to attach a single new sub-root — an md/neb/band folder dropped onto
    an already-open project — without re-deriving the other two roots from
    scratch, so the rest of the project keeps reading from exactly where it
    was.
    """
    opened_directory = Path(opened_directory)
    issues: list[Issue] = []

    index = scan_directory(layout.electronic)
    metadata, context = _collect_metadata(index, issues)
    consistency = validate_consistency(index, context, issues)
    _apply_outcar_metadata(metadata, context, consistency.outcar_trusted)

    # Fermi level: trust order is DOSCAR header, then OUTCAR, both only when the
    # file describes *this* calculation. EIGENVAL carries no Fermi level, so a
    # directory without either leaves the value undetermined on purpose.
    fermi = consistency.fermi_candidate
    if fermi is not None:
        metadata.e_fermi = fermi

    context.update(_collect_md_context(layout.md))
    context.update(_collect_neb_context(layout.neb))

    capabilities = _determine_capabilities(index, context, consistency)

    calculation = Calculation(
        directory=layout.electronic,
        index=index,
        capabilities=capabilities,
        metadata=metadata,
        issues=issues,
        opened_directory=(
            opened_directory if opened_directory != layout.electronic else None
        ),
        md_directory=layout.md,
        md_index=context.get("md_index"),
        neb_directory=layout.neb,
        neb_index=context.get("neb_index"),
        volumetric_sources=context.get("volumetric_sources") or [],
    )
    log.info(
        "Capabilities for %s: %s",
        opened_directory.name,
        ", ".join(sorted(c.value for c in capabilities.available)) or "none",
    )
    if not layout.is_flat:
        log.info(
            "Project layout: band=%s md=%s neb=%s",
            layout.electronic, layout.md, layout.neb,
        )
    return calculation


def discover_file(path: Path) -> Calculation:
    """Describe a single dropped file as a (very small) calculation.

    The point is that dropping a lone POSCAR or CHGCAR does something sensible
    instead of nothing: the pages that file can actually feed come up with
    real data, and every other page shows its ordinary "not found" empty state
    naming what is missing. That fallout is automatic — the pages already bind
    to capabilities rather than to filenames, so an index holding one file
    produces exactly the right set of enabled pages with no per-page special
    casing anywhere.

    Nothing else in the containing folder is read. See :func:`scan_file`.
    """
    from ..parsers.chgcar import kind_for_filename  # noqa: PLC0415 - avoids a cycle

    path = Path(path)
    if not path.is_file():
        return _discover_missing(path)

    issues: list[Issue] = []
    index = scan_file(path)

    # Recognised by name rather than by looking it up in the index, so a
    # PARCHG (which has no file role of its own, because its name varies with
    # the bands and k-points it holds) still opens the Charge Density page.
    sources: list[VolumetricSource] = []
    if path.name.upper().startswith(("CHGCAR", "CHG", "ELFCAR", "LOCPOT", "PARCHG")):
        sources = [VolumetricSource(path=path, kind=kind_for_filename(path.name))]

    metadata, context = _collect_metadata(index, issues, volumetric_sources=sources)
    consistency = validate_consistency(index, context, issues)
    _apply_outcar_metadata(metadata, context, consistency.outcar_trusted)

    fermi = consistency.fermi_candidate
    if fermi is not None:
        metadata.e_fermi = fermi

    # A single file is never an MD or NEB root, but the capability pass reads
    # these keys unconditionally and must find real (empty) indices rather
    # than fall back to the electronic one and mistake this file for a
    # trajectory.
    empty = FileIndex(directory=path.parent)
    context["md_index"] = empty
    context["neb_index"] = empty

    capabilities = _determine_capabilities(index, context, consistency)
    if not index.entries or all(not e.exists for e in index.entries.values()):
        issues.append(
            Issue(
                code="file_unrecognised",
                message=f"{path.name} is not a VASP file this app can read.",
                detail=(
                    "Recognised single files are POSCAR, CONTCAR, CHGCAR, CHG, "
                    "ELFCAR, LOCPOT, PARCHG, XDATCAR, EIGENVAL, DOSCAR, INCAR, "
                    "KPOINTS, OUTCAR and vasprun.xml. Drop a calculation "
                    "folder to open a full calculation."
                ),
                severity=Severity.WARNING,
                source=path.name,
            )
        )

    calculation = Calculation(
        directory=path.parent,
        index=index,
        capabilities=capabilities,
        metadata=metadata,
        issues=issues,
        md_directory=path.parent,
        md_index=empty,
        neb_directory=path.parent,
        neb_index=empty,
        volumetric_sources=sources,
    )
    log.info(
        "Opened single file %s: %s",
        path.name,
        ", ".join(sorted(c.value for c in capabilities.available)) or "no capabilities",
    )
    return calculation


def _discover_missing(directory: Path) -> Calculation:
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
