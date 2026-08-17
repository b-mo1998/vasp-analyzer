"""File-level data model: what exists on disk and in what condition."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class FileGroup(Enum):
    VASP = "VASP"
    WANNIER90 = "Wannier90"


class FileStatus(Enum):
    FOUND = "found"          # present and readable
    MISSING = "missing"      # not present
    UNREADABLE = "unreadable"  # present but permissions/IO problem
    MALFORMED = "malformed"  # present, parsed, and rejected
    INCONSISTENT = "inconsistent"  # parses, but describes a different calculation


@dataclass(frozen=True)
class FileRole:
    """A recognised calculation file and what it can contribute."""

    key: str
    filename: str
    group: FileGroup
    description: str
    binary: bool = False
    # Roles the app never lists in the UI (e.g. POTCAR is deliberately hidden).
    hidden: bool = False


@dataclass
class FileEntry:
    """A discovered (or absent) file on disk."""

    role: FileRole
    path: Path | None = None
    status: FileStatus = FileStatus.MISSING
    size: int = 0
    mtime: float = 0.0
    note: str = ""

    @property
    def exists(self) -> bool:
        return self.status not in (FileStatus.MISSING,)

    @property
    def usable(self) -> bool:
        return self.status is FileStatus.FOUND

    @property
    def name(self) -> str:
        return self.path.name if self.path else self.role.filename

    @property
    def size_label(self) -> str:
        if not self.exists:
            return "—"
        size = float(self.size)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024 or unit == "GB":
                return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} GB"


# --- Registry of recognised files -------------------------------------------------

VASP_ROLES: list[FileRole] = [
    FileRole("poscar", "POSCAR", FileGroup.VASP, "Input crystal structure"),
    FileRole("contcar", "CONTCAR", FileGroup.VASP, "Relaxed crystal structure"),
    FileRole("incar", "INCAR", FileGroup.VASP, "Calculation parameters"),
    FileRole("kpoints", "KPOINTS", FileGroup.VASP, "k-point sampling"),
    FileRole("outcar", "OUTCAR", FileGroup.VASP, "Main output and metadata"),
    FileRole("eigenval", "EIGENVAL", FileGroup.VASP, "Band eigenvalues"),
    FileRole("doscar", "DOSCAR", FileGroup.VASP, "Density of states"),
    FileRole("vasprun", "vasprun.xml", FileGroup.VASP, "Complete run record"),
    FileRole("procar", "PROCAR", FileGroup.VASP, "Projected band character"),
    FileRole("chgcar", "CHGCAR", FileGroup.VASP, "Charge density"),
    FileRole("potcar", "POTCAR", FileGroup.VASP, "Pseudopotentials", hidden=True),
]

WANNIER_ROLES: list[FileRole] = [
    FileRole("w90_win", "wannier90.win", FileGroup.WANNIER90, "Wannier90 input"),
    FileRole("w90_wout", "wannier90.wout", FileGroup.WANNIER90, "Wannier90 output"),
    FileRole("w90_eig", "wannier90.eig", FileGroup.WANNIER90, "Eigenvalues for W90"),
    FileRole("w90_amn", "wannier90.amn", FileGroup.WANNIER90, "Projections", binary=True),
    FileRole("w90_mmn", "wannier90.mmn", FileGroup.WANNIER90, "Overlaps", binary=True),
    FileRole("w90_chk", "wannier90.chk", FileGroup.WANNIER90, "Checkpoint", binary=True),
    FileRole("w90_spn", "wannier90.spn", FileGroup.WANNIER90, "Spin matrices", binary=True),
    FileRole(
        "w90_band_dat", "wannier90_band.dat", FileGroup.WANNIER90, "Interpolated bands"
    ),
]

ALL_ROLES: list[FileRole] = VASP_ROLES + WANNIER_ROLES
ROLES_BY_KEY: dict[str, FileRole] = {r.key: r for r in ALL_ROLES}


@dataclass
class FileIndex:
    """Everything discovered in a calculation directory."""

    directory: Path
    entries: dict[str, FileEntry] = field(default_factory=dict)
    unrecognised: list[str] = field(default_factory=list)
    wannier_prefix: str = "wannier90"

    def get(self, key: str) -> FileEntry | None:
        return self.entries.get(key)

    def path_for(self, key: str) -> Path | None:
        entry = self.entries.get(key)
        return entry.path if entry and entry.usable else None

    def has(self, key: str) -> bool:
        entry = self.entries.get(key)
        return bool(entry and entry.usable)

    def in_group(self, group: FileGroup, include_hidden: bool = False) -> list[FileEntry]:
        return [
            e
            for e in self.entries.values()
            if e.role.group is group and (include_hidden or not e.role.hidden)
        ]
