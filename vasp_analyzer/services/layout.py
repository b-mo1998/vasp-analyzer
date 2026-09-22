"""Resolving a project folder into the directories each page family reads.

A calculation can be laid out two ways:

* **Flat** — one directory holding every file directly. This is the
  traditional layout and stays fully supported: every check below falls back
  to the opened directory itself when nothing more specific is found.
* **Project folder** — a parent directory organised into subdirectories named
  ``band``/``bands`` (Structure, Bands, Charge Density, Inputs), ``md``
  (Molecular Dynamics) and ``neb`` (NEB), matched case-insensitively. Each
  family resolves independently, so a folder can supply some of the three
  without the others.

Resolution never requires all three: a project with only a ``band``
subdirectory still works, it just leaves the MD and NEB pages reporting
"nothing found" the same way an ordinary directory without a trajectory does.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BAND_ALIASES = ("band", "bands", "structure")
MD_ALIASES = ("md",)
NEB_ALIASES = ("neb",)

# What a directory needs to directly contain to count as an electronic-
# structure calculation on its own, rather than a project folder that merely
# holds one as a subdirectory.
_ELECTRONIC_SIGNATURE_NAMES = frozenset(
    {
        "poscar",
        "contcar",
        "incar",
        "outcar",
        "eigenval",
        "doscar",
        "procar",
        "chgcar",
        "chg",
        "elfcar",
        "locpot",
        "kpoints",
        "vasprun.xml",
    }
)


@dataclass(frozen=True)
class ProjectLayout:
    """Which directory each page family should read from."""

    electronic: Path   # Structure, Bands, Charge Density, Inputs
    md: Path           # Molecular Dynamics
    neb: Path          # NEB

    @property
    def is_flat(self) -> bool:
        """True when all three families read from the same directory."""
        return self.electronic == self.md == self.neb


def _direct_subdirectories(directory: Path) -> dict[str, Path]:
    """Immediate subdirectories, keyed by lowercase name.

    Only the first match for a given lowercased name is kept — a directory
    cannot sensibly contain both "MD" and "md" as distinct project roots.
    """
    if not directory.is_dir():
        return {}
    found: dict[str, Path] = {}
    try:
        for entry in directory.iterdir():
            if entry.is_dir() and not entry.name.startswith("."):
                found.setdefault(entry.name.lower(), entry)
    except OSError:
        pass
    return found


def _first_alias(subdirs: dict[str, Path], aliases: tuple[str, ...]) -> Path | None:
    for alias in aliases:
        found = subdirs.get(alias)
        if found is not None:
            return found
    return None


def _listed_files(directory: Path) -> set[str]:
    if not directory.is_dir():
        return set()
    try:
        return {p.name.lower() for p in directory.iterdir() if p.is_file()}
    except OSError:
        return set()


def has_electronic_files(directory: Path) -> bool:
    """Does this directory itself hold VASP calculation files directly?"""
    return bool(_listed_files(directory) & _ELECTRONIC_SIGNATURE_NAMES)


def has_file_named(directory: Path, filename: str) -> bool:
    return filename.lower() in _listed_files(directory)


def resolve_layout(directory: Path) -> ProjectLayout:
    """Work out which directory each page family should read from.

    Each family resolves independently and in the same order: prefer the
    opened directory itself if it directly qualifies, otherwise fall back to
    a matching-named subdirectory, otherwise stay on the opened directory
    anyway — so discovery still runs there and reports a clear "not found"
    rather than silently picking nothing.
    """
    from .neb import looks_like_neb  # noqa: PLC0415 - avoids a cycle

    directory = Path(directory)
    subdirs = _direct_subdirectories(directory)

    band_dir = _first_alias(subdirs, BAND_ALIASES)
    electronic = (
        directory
        if has_electronic_files(directory) or band_dir is None
        else band_dir
    )

    md_dir = _first_alias(subdirs, MD_ALIASES)
    md = directory if has_file_named(directory, "XDATCAR") or md_dir is None else md_dir

    neb_dir = _first_alias(subdirs, NEB_ALIASES)
    neb = directory if looks_like_neb(directory) or neb_dir is None else neb_dir

    return ProjectLayout(electronic=electronic, md=md, neb=neb)


def classify_directory(path: Path) -> str | None:
    """Which sub-root a directory looks like it belongs to, if any.

    Returns ``"band"``, ``"md"``, ``"neb"`` or ``None``. Name match is
    checked first (a folder literally called "md" is MD data even if it's
    momentarily empty); an unrecognised name falls back to its contents, so a
    folder holding an XDATCAR still attaches to the right page even when it
    isn't named "md".
    """
    from .neb import looks_like_neb  # noqa: PLC0415 - avoids a cycle

    name = path.name.lower()
    if name in MD_ALIASES:
        return "md"
    if name in NEB_ALIASES:
        return "neb"
    if name in BAND_ALIASES:
        return "band"

    if looks_like_neb(path):
        return "neb"
    if has_file_named(path, "XDATCAR"):
        return "md"
    if has_electronic_files(path):
        return "band"
    return None
