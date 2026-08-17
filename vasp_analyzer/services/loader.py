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
from .kpath import build_kpath

log = get_logger("services.loader")

DATASET_BANDS = "bands"
DATASET_DOS = "dos"
DATASET_STRUCTURE = "structure"
DATASET_PROCAR = "procar"


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
    def __init__(self, dataset: str, func: Callable[[], ParseResult]) -> None:
        super().__init__()
        self.dataset = dataset
        self.func = func
        self.signals = _WorkerSignals()

    def run(self) -> None:  # pragma: no cover - exercised via the GUI
        try:
            result = self.func()
        except Exception as exc:  # noqa: BLE001 - last-resort thread guard
            log.exception("Loader worker crashed for %s", self.dataset)
            self.signals.failed.emit(
                self.dataset,
                Issue(
                    code="loader_crashed",
                    message="Reading this file failed unexpectedly.",
                    detail=f"{type(exc).__name__}: {exc}",
                    severity=Severity.ERROR,
                ),
            )
            return
        self.signals.finished.emit(self.dataset, result)


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

    def request(self, dataset: str) -> ParseResult | None:
        """Ask for a dataset.

        Returns it immediately when cached; otherwise starts a worker and emits
        :attr:`data_ready` when it finishes. Returns None when the calculation
        cannot provide this dataset at all.
        """
        calculation = self._calculation
        if calculation is None:
            return None

        source = self._source_for(dataset, calculation)
        if source is None:
            return None
        path, key = source

        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if dataset in self._pending:
            return None

        self._pending.add(dataset)
        self.busy_changed.emit(True)
        worker = _Worker(dataset, lambda: self._parse(dataset, path, calculation))
        worker.signals.finished.connect(self._on_finished)
        worker.signals.failed.connect(self._on_failed)
        log.info("Loading %s from %s", dataset, path.name)
        self._pool.start(worker)
        return None

    def cached(self, dataset: str) -> ParseResult | None:
        calculation = self._calculation
        if calculation is None:
            return None
        source = self._source_for(dataset, calculation)
        if source is None:
            return None
        return self._cache.get(source[1])

    # -- internals ---------------------------------------------------------

    def _source_for(
        self, dataset: str, calculation: Calculation
    ) -> tuple[Path, CacheKey] | None:
        index = calculation.index
        if dataset == DATASET_BANDS:
            key_name = "eigenval"
        elif dataset == DATASET_DOS:
            key_name = "doscar"
        elif dataset == DATASET_STRUCTURE:
            key_name = "poscar" if index.has("poscar") else "contcar"
        elif dataset == DATASET_PROCAR:
            key_name = "procar"
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
            return parse_poscar(path, source=path.name)
        if dataset == DATASET_PROCAR:
            return parse_procar(path)
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

    def _on_finished(self, dataset: str, result: ParseResult) -> None:
        self._pending.discard(dataset)
        calculation = self._calculation
        if calculation is not None:
            source = self._source_for(dataset, calculation)
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

    def _on_failed(self, dataset: str, issue: Issue) -> None:
        self._pending.discard(dataset)
        self.data_ready.emit(dataset, ParseResult(value=None, issues=[issue]))
        if not self._pending:
            self.busy_changed.emit(False)


__all__ = [
    "DataLoader",
    "DATASET_BANDS",
    "DATASET_DOS",
    "DATASET_STRUCTURE",
    "DATASET_PROCAR",
]


# Re-exported for callers that only need the model types.
_ = (BandData, DosData, StructureData)
