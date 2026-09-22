"""Discovering and assembling an NEB calculation.

Unlike every other analysis in this application, an NEB is a *directory shape*
rather than a file: numbered image folders ``00``, ``01``, … sitting beside a
shared INCAR. Detection and assembly therefore live here rather than in a
parser.

Where the data comes from, in order of preference:

* **Reaction coordinate** — VTST's ``neb.dat`` when present, otherwise computed
  as the cumulative minimum-image-corrected displacement between adjacent
  images. Both give identical numbers (verified against a real CI-NEB run to
  six decimal places), so the fallback is a genuine substitute rather than an
  approximation.
* **Energies** — ``nebef.dat``, otherwise each image's own OUTCAR via
  ``energy(sigma->0)``. VTST reports that same quantity, so the two agree.
* **Forces** — ``neb.dat``/``nebef.dat`` carry the *projected* NEB force, which
  is what the optimiser converges and what EDIFFG is tested against. Falling
  back to an OUTCAR gives only the raw atomic force, so the difference is
  recorded in :attr:`NEBImage.force_kind` rather than quietly conflated.
* **Geometry** — CONTCAR when it holds a real structure, else POSCAR. Fixed
  endpoints are never relaxed and often have no CONTCAR at all.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..core.errors import Issue, ParseResult, Severity
from ..core.logging_setup import get_logger
from ..models.neb import NEBImage, NEBPath
from ..models.structure import StructureData
from ..parsers.incar import as_bool, as_float, as_int, parse_incar
from ..parsers.outcar import parse_ionic_state, parse_outcar
from ..parsers.poscar import parse_poscar
from ..parsers.vtst import (
    merge_profiles,
    parse_neb_dat,
    parse_nebef_dat,
    parse_spline_dat,
)

log = get_logger("services.neb")

_RE_IMAGE_DIR = re.compile(r"^\d{1,3}$")
MIN_IMAGES = 3          # two endpoints plus at least one movable image
_MIN_USEFUL_BYTES = 10


def find_image_directories(directory: Path) -> list[tuple[int, Path]]:
    """Numbered image folders, sorted by numeric value.

    Sorting on the integer rather than the name matters: a ten-image band has
    folders 00…10, and lexicographic order would place 10 between 01 and 02.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for child in directory.iterdir():
        if child.is_dir() and _RE_IMAGE_DIR.match(child.name):
            found.append((int(child.name), child))
    found.sort(key=lambda item: item[0])
    return found


def _image_has_structure(path: Path) -> bool:
    for name in ("CONTCAR", "POSCAR"):
        candidate = path / name
        try:
            if candidate.is_file() and candidate.stat().st_size >= _MIN_USEFUL_BYTES:
                return True
        except OSError:
            continue
    return False


def looks_like_neb(directory: Path) -> bool:
    """Cheap structural test used at discovery time.

    Requires numbered folders that actually contain structures, and either a
    contiguous run starting at zero or an INCAR that declares IMAGES. A stray
    pair of numeric folders in an unrelated directory therefore does not turn
    the NEB page on.
    """
    images = find_image_directories(directory)
    if len(images) < MIN_IMAGES:
        return False
    if not all(_image_has_structure(path) for _, path in images):
        return False

    indices = [index for index, _ in images]
    contiguous = indices == list(range(indices[0], indices[0] + len(indices)))

    incar = directory / "INCAR"
    declares_images = False
    if incar.is_file():
        parsed = parse_incar(incar)
        if parsed.ok and parsed.value is not None:
            declares_images = as_int(parsed.value.get("IMAGES")) is not None

    return contiguous or declares_images


def _load_structure(folder: Path) -> tuple[StructureData | None, str, list[str]]:
    """Best available geometry for one image, with the file it came from.

    CONTCAR wins when it parses — it is the relaxed position of a movable
    image. An empty or truncated CONTCAR (a job killed mid-write) falls back to
    POSCAR rather than failing the image.
    """
    available: list[str] = []
    for name in ("CONTCAR", "POSCAR"):
        path = folder / name
        try:
            if not path.is_file() or path.stat().st_size < _MIN_USEFUL_BYTES:
                continue
        except OSError:
            continue
        available.append(name)

    for name in ("CONTCAR", "POSCAR"):
        if name not in available:
            continue
        result = parse_poscar(folder / name, source=f"{folder.name}/{name}")
        if result.ok and result.value is not None:
            return result.value, name, available
    return None, "", available


def reaction_coordinate(structures: list[StructureData | None]) -> np.ndarray | None:
    """Cumulative path length in Å along the band.

    Each step is the Euclidean norm of the full 3N-dimensional displacement
    between adjacent images, with fractional differences reduced by the
    minimum-image convention first so an atom crossing a cell face contributes
    its true small motion rather than a whole lattice vector.
    """
    if len(structures) < 2 or any(s is None for s in structures):
        return None

    coordinate = [0.0]
    total = 0.0
    for previous, current in zip(structures, structures[1:]):
        assert previous is not None and current is not None
        if previous.n_atoms != current.n_atoms:
            return None
        delta = current.frac_coords - previous.frac_coords
        delta -= np.round(delta)
        cartesian = delta @ current.lattice
        total += float(np.linalg.norm(cartesian))
        coordinate.append(total)
    return np.array(coordinate, dtype=float)


def _validate(images: list[NEBImage], issues: list[Issue]) -> None:
    """Flag images whose structure is incompatible with the rest of the band."""
    reference: StructureData | None = None
    for image in images:
        if image.structure is not None:
            reference = image.structure
            break
    if reference is None:
        return

    for image in images:
        if image.structure is None:
            continue
        if image.structure.n_atoms != reference.n_atoms:
            image.note = (
                f"{image.structure.n_atoms} atoms, but the band uses "
                f"{reference.n_atoms}"
            )
            issues.append(
                Issue(
                    code="neb_image_atom_mismatch",
                    message=(
                        f"NEB image {image.folder} contains "
                        f"{image.structure.n_atoms} atoms while the rest of the "
                        f"band has {reference.n_atoms}."
                    ),
                    severity=Severity.WARNING,
                    source=image.folder,
                )
            )
        elif image.structure.species != reference.species:
            image.note = "species order differs from the rest of the band"
            issues.append(
                Issue(
                    code="neb_image_species_mismatch",
                    message=(
                        f"NEB image {image.folder} lists its elements in a "
                        "different order from the rest of the band."
                    ),
                    severity=Severity.WARNING,
                    source=image.folder,
                )
            )


def _apply_incar(path: NEBPath, tags: dict[str, str]) -> None:
    climbing = as_bool(tags.get("LCLIMB"))
    images = as_int(tags.get("IMAGES"))
    spring = as_float(tags.get("SPRING"))
    ediffg = as_float(tags.get("EDIFFG"))

    path.n_movable = images
    path.spring_constant = spring
    # A negative EDIFFG is a force criterion in eV/Å; a positive one is an
    # energy criterion and says nothing about forces, so it is not reused here.
    if ediffg is not None and ediffg < 0:
        path.force_threshold = abs(ediffg)

    # VTST's own climbing switch (ICHAIN=0 selects NEB) is LCLIMB; VASP's
    # native implementation uses the same tag, so one test covers both.
    path.method = "NEB (Climbing Image)" if climbing else "NEB"


def _identify_climbing_image(path: NEBPath) -> None:
    """Record which image climbed, and whether that is known or deduced.

    Neither VASP nor VTST writes the chosen image to a machine-readable file,
    so for a climbing-image run it has to be deduced from the profile. The
    highest-energy *intermediate* image is the standard deduction, but it is
    marked as inferred so the UI never presents it as recorded fact.
    """
    if not path.is_climbing or path.n_images < 3:
        return
    interior = [
        (image.energy, i)
        for i, image in enumerate(path.images)
        if image.energy is not None and 0 < i < path.n_images - 1
    ]
    if not interior:
        return
    path.climbing_image = max(interior)[1]
    path.climbing_image_origin = "inferred"


def _read_potentials(
    directory: Path, folders: list[tuple[int, Path]], path: NEBPath
) -> None:
    """Recover pseudopotential provenance from an image's OUTCAR.

    An NEB directory has no top-level OUTCAR, so ordinary discovery finds no
    POTCAR titles for it. Any image's OUTCAR echoes them, and every image of a
    band shares the same potentials — so the first readable one answers for all.
    The functional is taken from the same titles ("PAW_PBE …" → PBE) rather
    than guessed, because a plain PBE INCAR sets no GGA tag at all.
    """
    for _, folder in folders:
        outcar = folder / "OUTCAR"
        if not outcar.is_file():
            continue
        result = parse_outcar(outcar)
        if not result.ok or result.value is None or not result.value.potcar_titles:
            continue
        path.potcar_titles = list(result.value.potcar_titles)
        first = path.potcar_titles[0].split()
        if first and "_" in first[0]:
            family, _, functional = first[0].partition("_")
            path.functional = f"{functional} ({family})" if functional else None
        return

    _ = directory


def load_neb_path(directory: Path) -> ParseResult[NEBPath]:
    """Assemble every source in an NEB directory into one path object."""
    directory = Path(directory)
    folders = find_image_directories(directory)
    if len(folders) < MIN_IMAGES:
        return ParseResult.failure(
            "neb_no_images",
            "No numbered NEB image folders were found in this directory.",
            source=directory.name,
        )

    issues: list[Issue] = []
    path = NEBPath(directory_name=directory.name)

    # --- INCAR ------------------------------------------------------------
    incar_file = directory / "INCAR"
    if incar_file.is_file():
        parsed = parse_incar(incar_file)
        if parsed.ok and parsed.value is not None:
            _apply_incar(path, parsed.value)
            path.source_files.append("INCAR")

    # --- images -------------------------------------------------------------
    last = len(folders) - 1
    for position, (number, folder) in enumerate(folders):
        image = NEBImage(
            index=position,
            folder=folder.name,
            is_endpoint=position in (0, last),
        )
        structure, used, available = _load_structure(folder)
        image.structure = structure
        if used:
            image.source_files.append(f"{folder.name}/{used}")
        elif available:
            image.note = "structure files present but unreadable"
        else:
            image.note = "no POSCAR or CONTCAR"

        outcar = folder / "OUTCAR"
        if outcar.is_file() and outcar.stat().st_size >= _MIN_USEFUL_BYTES:
            state = parse_ionic_state(outcar)
            if state.ok and state.value is not None:
                image.energy = state.value.energy
                if image.energy is None:
                    image.energy = state.value.free_energy
                    image.energy_source = f"{folder.name}/OUTCAR (TOTEN)"
                else:
                    image.energy_source = f"{folder.name}/OUTCAR"
                if state.value.max_force is not None:
                    image.max_force = state.value.max_force
                    image.force_kind = "atomic"
                image.source_files.append(f"{folder.name}/OUTCAR")

        path.images.append(image)
        _ = number  # folder name already carries it

    _validate(path.images, issues)

    # --- VTST tables override where they are richer ---------------------------
    neb_dat = directory / "neb.dat"
    nebef_dat = directory / "nebef.dat"
    profile = None
    neb_profile = None
    nebef_profile = None
    if neb_dat.is_file():
        result = parse_neb_dat(neb_dat)
        neb_profile = result.value if result.ok else None
    if nebef_dat.is_file():
        result = parse_nebef_dat(nebef_dat)
        nebef_profile = result.value if result.ok else None
    profile = merge_profiles(neb_profile, nebef_profile)

    spline_dat = directory / "spline.dat"
    if spline_dat.is_file():
        result = parse_spline_dat(spline_dat)
        if result.ok and result.value is not None and result.value.n_points > 1:
            path.spline = result.value
            path.source_files.append("spline.dat")

    if profile is not None and profile.n_images == path.n_images:
        path.source_files.extend(profile.sources)
        for i, image in enumerate(path.images):
            if i < len(profile.absolute_energy):
                image.energy = profile.absolute_energy[i]
                image.energy_source = "nebef.dat"
            if i < len(profile.force):
                image.max_force = profile.force[i]
                # Endpoints are fixed, so no projection is applied to them and
                # the reported value is the ordinary atomic force.
                image.force_kind = "atomic" if image.is_endpoint else "NEB projected"
        if profile.reaction_coordinate:
            path.reaction_coordinate = np.array(profile.reaction_coordinate, dtype=float)
    elif profile is not None:
        issues.append(
            Issue(
                code="neb_vtst_image_mismatch",
                message=(
                    f"neb.dat describes {profile.n_images} images but "
                    f"{path.n_images} image folders were found; it was ignored."
                ),
                severity=Severity.WARNING,
                source="neb.dat",
            )
        )

    if path.reaction_coordinate is None:
        path.reaction_coordinate = reaction_coordinate(path.structures)

    # Relative energies alone (neb.dat without nebef.dat) still give a usable
    # profile: the shape and the barrier are unchanged by the missing offset.
    if not path.has_energies and profile is not None and profile.relative_energy:
        for i, image in enumerate(path.images):
            if i < len(profile.relative_energy):
                image.energy = profile.relative_energy[i]
                image.energy_source = "neb.dat (relative)"

    _identify_climbing_image(path)
    _read_potentials(directory, folders, path)

    for image in path.images:
        path.source_files.extend(image.source_files)

    known = sum(1 for image in path.images if image.energy is not None)
    if 0 < known < path.n_images:
        issues.append(
            Issue(
                code="neb_partial_energies",
                message=(
                    f"Only {known} of {path.n_images} NEB images reported an "
                    "energy; the profile shows the images that did."
                ),
                severity=Severity.INFO,
            )
        )

    log.info(
        "Loaded NEB path: %d images, %d energies, method %s, barrier %s",
        path.n_images,
        known,
        path.method,
        f"{path.forward_barrier:.3f} eV" if path.forward_barrier is not None else "unknown",
    )
    if path.climbing_image is not None:
        log.info("CI image: %s (%s)", path.images[path.climbing_image].folder,
                 path.climbing_image_origin)
    return ParseResult.success(path, issues)
