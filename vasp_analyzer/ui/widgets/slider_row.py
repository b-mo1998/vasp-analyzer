"""Labelled slider with an editable numeric field.

Layout::

    Y-Axis Min                -10.00 eV
    ──────●────────────────────────
    -20                              0

Qt sliders are integer-only, so the control keeps a float model internally and
maps it onto integer steps. The numeric field accepts typed input for exact
values, which the sliders alone cannot give.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..theme import SPACING


class SliderRow(QWidget):
    """A float slider with a title, an editable value box and range captions."""

    valueChanged = Signal(float)

    def __init__(
        self,
        title: str,
        minimum: float,
        maximum: float,
        value: float,
        step: float = 0.1,
        unit: str = "",
        decimals: int = 2,
        inline: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._min = float(minimum)
        self._max = float(maximum)
        self._step = float(step)
        self._unit = unit
        self._decimals = decimals
        self._value = float(value)
        self._emitting = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING.xs + 2)

        self._title = QLabel(title)
        self._field = QLineEdit()
        self._field.setObjectName("ValueField")
        self._field.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._field.setFixedWidth(86)
        self._field.setValidator(QDoubleValidator(self._min, self._max, decimals, self))
        self._field.editingFinished.connect(self._on_field_committed)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(self._steps())
        self._slider.setSingleStep(1)
        self._slider.setPageStep(max(1, self._steps() // 20))
        self._slider.valueChanged.connect(self._on_slider_moved)

        if inline:
            # Slider and value share one row; used where a caption already sits
            # above the control, as on the DOS panel.
            self._title.hide()
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(SPACING.sm)
            row.addWidget(self._slider, 1)
            row.addWidget(self._field, 0)
            layout.addLayout(row)
        else:
            header = QHBoxLayout()
            header.setContentsMargins(0, 0, 0, 0)
            header.setSpacing(SPACING.sm)
            header.addWidget(self._title)
            header.addStretch(1)
            header.addWidget(self._field)
            layout.addLayout(header)
            layout.addWidget(self._slider)

        captions = QHBoxLayout()
        captions.setContentsMargins(0, 0, 0, 0)
        self._min_label = QLabel(self._format_plain(self._min))
        self._min_label.setObjectName("RangeLabel")
        self._max_label = QLabel(self._format_plain(self._max))
        self._max_label.setObjectName("RangeLabel")
        captions.addWidget(self._min_label)
        captions.addStretch(1)
        captions.addWidget(self._max_label)
        layout.addLayout(captions)

        self._sync_widgets()

    # -- public API --------------------------------------------------------

    def value(self) -> float:
        return self._value

    def setValue(self, value: float, notify: bool = True) -> None:
        clamped = max(self._min, min(self._max, float(value)))
        if abs(clamped - self._value) < 1e-12:
            self._sync_widgets()
            return
        self._value = clamped
        self._sync_widgets()
        if notify:
            self.valueChanged.emit(self._value)

    def setRange(self, minimum: float, maximum: float) -> None:
        self._min, self._max = float(minimum), float(maximum)
        validator = self._field.validator()
        if isinstance(validator, QDoubleValidator):
            validator.setRange(self._min, self._max, self._decimals)
        self._slider.blockSignals(True)
        self._slider.setMaximum(self._steps())
        self._slider.blockSignals(False)
        self._min_label.setText(self._format_plain(self._min))
        self._max_label.setText(self._format_plain(self._max))
        self.setValue(self._value, notify=False)

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802 - Qt naming
        super().setEnabled(enabled)
        self._slider.setEnabled(enabled)
        self._field.setEnabled(enabled)

    # -- internals ---------------------------------------------------------

    def _steps(self) -> int:
        span = self._max - self._min
        return max(1, int(round(span / self._step)))

    def _to_step(self, value: float) -> int:
        return int(round((value - self._min) / self._step))

    def _from_step(self, step: int) -> float:
        return self._min + step * self._step

    def _format_value(self, value: float) -> str:
        text = f"{value:.{self._decimals}f}"
        return f"{text} {self._unit}".strip()

    def _format_plain(self, value: float) -> str:
        if abs(value - round(value)) < 1e-9:
            return f"{value:.0f}"
        return f"{value:.{self._decimals}f}"

    def _sync_widgets(self) -> None:
        self._emitting = True
        self._slider.setValue(self._to_step(self._value))
        self._field.setText(self._format_value(self._value))
        self._emitting = False

    def _on_slider_moved(self, step: int) -> None:
        if self._emitting:
            return
        self.setValue(self._from_step(step))

    def _on_field_committed(self) -> None:
        text = self._field.text().replace(self._unit, "").strip()
        try:
            self.setValue(float(text))
        except ValueError:
            self._sync_widgets()
