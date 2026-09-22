"""Whole-run statistics for an MD trajectory.

The MD page plots temperature and energy against time, which shows the shape
of a run but not whether it was a *good* run. Three questions decide that, and
none of them are answerable by eye from a noisy trace:

* Did the thermostat actually hold the target temperature? An instantaneous
  temperature swinging between 150 K and 750 K is normal for a small cell —
  the fluctuation scales as 1/sqrt(N) — so only the mean and its spread say
  whether the run sat where it was asked to.
* Did the total energy drift? In NVE a drift means the timestep is too large
  or the convergence too loose. Even in NVT a strong monotonic drift means the
  run never equilibrated.
* Where does equilibration end? Averaging over the initial transient biases
  every number taken from the run.

Statistics are computed over an *analysis window* that skips a leading
fraction of the run, because production averages should never include the
equilibration transient.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models.trajectory import MDTrajectory

# Fraction of the run treated as equilibration and excluded from averages.
# 20% is the usual rule of thumb; it is reported alongside the numbers so the
# choice is visible rather than hidden.
EQUILIBRATION_FRACTION = 0.2


def format_value(value: float) -> str:
    """A compact decimal rather than scientific notation.

    "± 1.4e+02 K" is a spread of 140 kelvin written in the least readable way
    available; at the magnitudes these quantities take, plain decimals are
    both shorter and instantly legible.
    """
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    if magnitude >= 0.1:
        return f"{value:.3f}"
    if magnitude == 0:
        return "0"
    return f"{value:.3g}"


@dataclass
class SeriesStats:
    """Mean, spread and drift of one scalar series over the analysis window."""

    name: str
    unit: str
    mean: float
    std: float
    minimum: float
    maximum: float
    drift: float          # end-to-end change from a linear fit, in `unit`
    n_samples: int

    @property
    def summary(self) -> str:
        return f"{format_value(self.mean)} ± {format_value(self.std)} {self.unit}".strip()


@dataclass
class TrajectoryStats:
    """Everything the statistics card reports."""

    start_index: int                      # first step included in the averages
    n_steps: int                          # total steps available
    temperature: SeriesStats | None = None
    total_energy: SeriesStats | None = None
    target_temperature: float | None = None

    @property
    def equilibration_steps(self) -> int:
        return self.start_index

    @property
    def temperature_offset(self) -> float | None:
        """Mean temperature minus the target, when a target is known.

        The number that answers "is my thermostat working": a well-behaved NVT
        run sits within a few kelvin of TEBEG however wildly the instantaneous
        value swings.
        """
        if self.temperature is None or self.target_temperature is None:
            return None
        return self.temperature.mean - self.target_temperature


def _series_stats(name: str, values: np.ndarray, unit: str) -> SeriesStats | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None

    # Drift as the end-to-end change of a least-squares line rather than
    # (last - first): a single noisy final step should not be read as a trend.
    steps = np.arange(values.size, dtype=float)
    slope, _intercept = np.polyfit(steps, values, 1)
    drift = float(slope * (values.size - 1))

    return SeriesStats(
        name=name,
        unit=unit,
        mean=float(values.mean()),
        std=float(values.std(ddof=1)),
        minimum=float(values.min()),
        maximum=float(values.max()),
        drift=drift,
        n_samples=int(values.size),
    )


def analyse(
    trajectory: MDTrajectory, equilibration: float = EQUILIBRATION_FRACTION
) -> TrajectoryStats | None:
    """Summarise a run, skipping its leading equilibration fraction.

    Returns None when there is no per-step telemetry to summarise — a
    trajectory from an XDATCAR with no OSZICAR alongside it has geometry but
    no temperature or energy at all.
    """
    temperature = trajectory.series.get("temperature")
    energy = trajectory.series.get("total_energy")
    if temperature is None and energy is None:
        return None

    lengths = [s.values.size for s in (temperature, energy) if s is not None]
    n_steps = max(lengths) if lengths else 0
    if n_steps < 2:
        return None

    fraction = min(max(equilibration, 0.0), 0.9)
    start = int(n_steps * fraction)
    # Never discard so much that nothing is left to average.
    if n_steps - start < 2:
        start = 0

    stats = TrajectoryStats(start_index=start, n_steps=n_steps)

    if temperature is not None:
        stats.temperature = _series_stats(
            "Temperature", temperature.values[start:], temperature.unit or "K"
        )
    if energy is not None:
        stats.total_energy = _series_stats(
            "Total Energy", energy.values[start:], energy.unit or "eV"
        )

    target = trajectory.target_temperature
    if target is not None:
        # TEBEG and TEEND are equal for an isothermal run; for a ramp the mean
        # is the only single number that means anything.
        stats.target_temperature = float(np.mean(target))
    return stats
