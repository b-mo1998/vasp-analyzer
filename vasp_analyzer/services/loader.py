"""Background loading and caching of the expensive data sets.

Discovery is cheap and runs synchronously. Reading EIGENVAL or DOSCAR is not,
so it happens on a worker thread and the result arrives as a Qt signal. Results
are cached against ``(path, mtime, size)``, so re-opening a page is instant and
a file edited outside the app is re-read automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from ..core.errors import Issue, ParseResult, Severity
from ..core.logging_setup import get_logger
from ..models.bands import BandData
from ..models.calculation import Calculation
from ..models.dos import DosData
from ..models.structure import StructureData
from ..parsers.doscar import parse_doscar
from ..parsers.eigenval import parse_eigenval
from ..parsers.poscar import parse_poscar
from ..parsers.procar import parse_procar
from ..parsers.wannier import parse_band_dat
from .kpath import build_kpath

log = get_logger("services.loader")

DATASET_BANDS = "bands"
DATASET_DOS = "dos"
DATASET_STRUCTURE = "structure"
DATASET_PROCAR = "procar"
DATASET_WANNIER_BANDS = "wannier_bands"
DATASET_MD = "md_trajectory"
DATASET_NEB = "neb_path"
DATASET_VOLUMETRIC = "volumetric"

# Where a structure can come from, best first. POSCAR is what the run was
# given and CONTCAR what it produced; the volumetric files each embed a copy
# of the geometry in their header, which is the only structure available when
# a directory (or a single dropped file) has nothing else.
STRUCTURE_SOURCE_KEYS = ("poscar", "contcar", "chgcar", "chg", "elfcar", "locpot")
# Matched with str.startswith, so PARCHG.0012.ALLK and CHGCAR.relax are
# recognised alongside the bare names.
VOLUMETRIC_FILENAME_PREFIXES = ("CHGCAR", "CHG", "ELFCAR", "LOCPOT", "PARCHG")


def read_structure(path: Path) -> ParseResult[StructureData]:
    """Read a structure from whichever kind of file is holding it.

    One function so every caller — the background loader, the Editor's own
    source list — agrees on which files count as a structure and reads each
    of them the same way. A volumetric file gets its header peeked rather
    than parsed whole: ``parse_poscar`` reads the entire text, which for a
    CHGCAR means hundreds of megabytes of grid values read and thrown away to
    recover a few kilobytes of coordinates.
    """
    path = Path(path)
    if path.name.upper().startswith(VOLUMETRIC_FILENAME_PREFIXES):
        from ..parsers.chgcar import (  # noqa: PLC0415 - avoids a cycle
            peek_volumetric_structure,
        )

        structure = peek_volumetric_structure(path)
        if structure is None:
            return ParseResult.failure(
                "volumetric_header_unreadable",
                f"The structure block in {path.name} could not be read.",
                source=path.name,
            )
        return ParseResult.success(structure)
    return parse_poscar(path, source=path.name)


@dataclass(frozen=True)
class CacheKey:
    dataset: str
    path: str
    mtime: float
    size: int


class _WorkerSignals(QObject):
    finished = Signal(str, object)  # dataset, ParseResult
    failed = Signal(str, object)    # dataset, Issue


class _Worker(QRunnable):
    def __init__(
        self,
        dataset: str,
        func: Callable[[], ParseResult],
        pending_key: str | None = None,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.func = func
        # What to clear from the in-flight set; differs from ``dataset`` when
        # several files of one dataset can be loading independently.
        self.pending_key = pending_key or dataset
        self.signals = _WorkerSignals()

    def run(self) -> None:  # pragma: no cover - exercised via the GUI
        try:
            result = self.func()
        except Exception as exc:  # noqa: BLE001 - last-resort thread guard
            log.exception("Loader worker crashed for %s", self.dataset)
            self.signals.failed.emit(
                self.pending_key,
                Issue(
                    code="loader_crashed",
                    message="Reading this file failed unexpectedly.",
                    detail=f"{type(exc).__name__}: {exc}",
                    severity=Severity.ERROR,
                ),
            )
            return
        self.signals.finished.emit(self.pending_key, result)


class DataLoader(QObject):
    """Loads band/DOS/structure data for the current calculation on demand."""

    data_ready = Signal(str, object)   # dataset name, ParseResult
    busy_changed = Signal(bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._pool = QThreadPool.globalInstance()
        self._cache: dict[CacheKey, ParseResult] = {}
        self._pending: set[str] = set()
        self._calculation: Calculation | None = None

    # -- lifecycle ---------------------------------------------------------

    def set_calculation(self, calculation: Calculation | None) -> None:
        self._calculation = calculation
        self._pending.clear()

    def clear_cache(self) -> None:
        self._cache.clear()

    # -- requests ----------------------------------------------------------

    def request(self, dataset: str, key: str | None = None) -> ParseResult | None:
        """Ask for a dataset.

        Returns it immediately when cached; otherwise starts a worker and emits
        :attr:`data_ready` when it finishes. Returns None when the calculation
        cannot provide this dataset at all.

        ``key`` selects *which* file when a dataset has more than one candidate
        — a directory can hold a CHGCAR, an ELFCAR and several PARCHGs, and
        they are separate loads of the same dataset rather than separate
        datasets.
        """
        calculation = self._calculation
        if calculation is None:
            return None

        source = self._source_for(dataset, calculation, key)
        if source is None:
            return None
        path, cache_key = source

        cached = self._cache.get(cache_key)
        if cached is not None:
            # Deliver through the same signal as a background load, so callers
            # have exactly one code path for "the data arrived".
            self.data_ready.emit(dataset, cached)
            return cached

        pending_key = f"{dataset}:{key}" if key else dataset
        if pending_key in self._pending:
            return None

        self._pending.add(pending_key)
        self.busy_changed.emit(True)
        worker = _Worker(
            dataset, lambda: self._parse(dataset, path, calculation), pending_key
        )
        worker.signals.finished.connect(self._on_finished)
        worker.signals.failed.connect(self._on_failed)
        log.info("Loading %s from %s", dataset, path.name)
        self._pool.start(worker)
        return None

    def cached(self, dataset: str, key: str | None = None) -> ParseResult | None:
        calculation = self._calculation
        if calculation is None:
            return None
        source = self._source_for(dataset, calculation, key)
        if source is None:
            return None
        return self._cache.get(source[1])

    # -- internals ---------------------------------------------------------

    def _source_for(
        self, dataset: str, calculation: Calculation, key: str | None = None
    ) -> tuple[Path, CacheKey] | None:
        index = calculation.index

        if dataset == DATASET_VOLUMETRIC:
            # The caller names the file, because a directory can hold several
            # and they are all the same dataset.
            if not key:
                return None
            path = Path(key)
            try:
                stat = path.stat()
            except OSError:
                return None
            return path, CacheKey(dataset, str(path), stat.st_mtime, stat.st_size)

        if dataset == DATASET_BANDS:
            key_name = "eigenval"
        elif dataset == DATASET_DOS:
            key_name = "doscar"
        elif dataset == DATASET_STRUCTURE:
            # Volumetric files last, and only as a fallback: each one carries
            # a full POSCAR block in its header, so a calculation with a
            # CHGCAR but no POSCAR/CONTCAR still has a real structure to show
            # rather than an empty Structure and Editor page. An explicit
            # structure file always wins — it is the authoritative geometry,
            # and a density's embedded copy is a snapshot of it.
            key_name = next(
                (
                    candidate
                    for candidate in STRUCTURE_SOURCE_KEYS
                    if index.has(candidate)
                ),
                "contcar",
            )
        elif dataset == DATASET_PROCAR:
            key_name = "procar"
        elif dataset == DATASET_WANNIER_BANDS:
            key_name = "w90_band_dat"
        elif dataset == DATASET_MD:
            # An MD load reads several files; XDATCAR is the one that always
            # exists and always changes when the run does, so it carries the
            # cache key for the set. Reads from the calculation's own MD
            # index, which a project layout can point at a different
            # directory than the electronic-structure one above.
            index = calculation.md_index
            key_name = "xdatcar"
        elif dataset == DATASET_NEB:
            # An NEB has no single input file — it is a directory shape — so the
            # directory's own mtime keys the cache. Adding or rerunning an image
            # touches it, and Reload clears the cache outright. Uses the
            # calculation's own NEB directory, which a project layout can
            # point at a different directory than the electronic-structure one.
            directory = calculation.neb_directory
            try:
                stat = directory.stat()
            except OSError:
                return None
            return directory, CacheKey(
                dataset, str(directory), stat.st_mtime, stat.st_size
            )
        else:
            return None

        entry = index.entries.get(key_name)
        if entry is None or not entry.usable or entry.path is None:
            return None
        return entry.path, CacheKey(dataset, str(entry.path), entry.mtime, entry.size)

    def _parse(self, dataset: str, path: Path, calculation: Calculation) -> ParseResult:
        if dataset == DATASET_BANDS:
            result: ParseResult[Any] = parse_eigenval(path)
            if result.ok and result.value is not None:
                self._attach_kpath(result.value, calculation)
            return result
        if dataset == DATASET_DOS:
            return parse_doscar(path)
        if dataset == DATASET_STRUCTURE:
            return read_structure(path)
        if dataset == DATASET_PROCAR:
            return parse_procar(path)
        if dataset == DATASET_WANNIER_BANDS:
            return parse_band_dat(path)
        if dataset == DATASET_MD:
            from .md import load_trajectory  # noqa: PLC0415 - avoids a cycle

            return load_trajectory(calculation.md_index)
        if dataset == DATASET_NEB:
            from .neb import load_neb_path  # noqa: PLC0415 - avoids a cycle

            return load_neb_path(path)
        if dataset == DATASET_VOLUMETRIC:
            from ..parsers.chgcar import parse_volumetric  # noqa: PLC0415

            return parse_volumetric(path)
        return ParseResult.failure("unknown_dataset", f"Unknown dataset {dataset!r}.")

    def _attach_kpath(self, bands: BandData, calculation: Calculation) -> None:
        """Give the bands an x-axis using whatever context the directory offers."""
        from ..parsers.kpoints import parse_kpoints

        spec = None
        kpoints_path = calculation.index.path_for("kpoints")
        if kpoints_path:
            parsed = parse_kpoints(kpoints_path)
            spec = parsed.value if parsed.ok else None

        structure: StructureData | None = None
        structure_result = self.cached(DATASET_STRUCTURE)
        if structure_result is not None and structure_result.ok:
            structure = structure_result.value
        else:
            for key in ("poscar", "contcar"):
                path = calculation.index.path_for(key)
                if path:
                    parsed_structure = parse_poscar(path, source=path.name)
                    if parsed_structure.ok:
                        structure = parsed_structure.value
                    break

        build_kpath(bands, spec, structure)

    @staticmethod
    def _split_key(pending_key: str) -> tuple[str, str | None]:
        """Recover (dataset, key) from the combined in-flight identifier."""
        dataset, separator, key = pending_key.partition(":")
        return (dataset, key) if separator else (pending_key, None)

    def _on_finished(self, pending_key: str, result: ParseResult) -> None:
        self._pending.discard(pending_key)
        dataset, key = self._split_key(pending_key)

        calculation = self._calculation
        if calculation is not None:
            source = self._source_for(dataset, calculation, key)
            if source is not None:
                self._cache[source[1]] = result
        for issue in result.issues:
            log.log(
                {
                    Severity.ERROR: 40,
                    Severity.WARNING: 30,
                    Severity.INFO: 20,
                }[issue.severity],
                "%s: %s %s",
                dataset,
                issue.message,
                issue.detail,
            )
        self.data_ready.emit(dataset, result)
        if not self._pending:
            self.busy_changed.emit(False)

    def _on_failed(self, pending_key: str, issue: Issue) -> None:
        self._pending.discard(pending_key)
        dataset, _ = self._split_key(pending_key)
        self.data_ready.emit(dataset, ParseResult(value=None, issues=[issue]))
        if not self._pending:
            self.busy_changed.emit(False)


__all__ = [
    "DataLoader",
    "DATASET_BANDS",
    "DATASET_DOS",
    "DATASET_STRUCTURE",
    "DATASET_PROCAR",
    "DATASET_WANNIER_BANDS",
    "DATASET_MD",
    "DATASET_NEB",
    "DATASET_VOLUMETRIC",
]


# Re-exported for callers that only need the model types.
_ = (BandData, DosData, StructureData)
