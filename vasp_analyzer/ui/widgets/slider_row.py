"""Labelled slider with an editable numeric field.

Layout::

    Y-Axis Min               [-10.00] eV
    ──────●────────────────────────
    -20                              0

The numeric field accepts typed input for exact values; the slider itself is
:class:`PaintedSlider`, a custom-painted control rather than a styled QSlider
(see that module for why).

The unit sits in its own label *outside* the editable box, and the field's
validator deliberately accepts any real number rather than being bounded to
the slider's range. Both details are load-bearing, not cosmetic: QLineEdit
suppresses ``editingFinished`` whenever its validator does not report
``Acceptable``, so text like ``-10.00 eV`` (Invalid — the unit is not part of
a number) or an out-of-range ``25`` (Intermediate) silently never commits,
and a typed value would appear to be accepted while the slider never moved.
Clamping happens in :meth:`setValue` instead, where an out-of-range entry
snaps to the nearest end of the range rather than being discarded.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

from ..theme import SPACING
from .painted_slider import PaintedSlider


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
        self._field.setFixedWidth(66 if unit else 82)
        # Unbounded on purpose — see the module docstring. Clamping is done in
        # setValue so an out-of-range entry still commits (to the nearest
        # limit) instead of being silently swallowed.
        self._field.setValidator(QDoubleValidator(self))
        self._field.editingFinished.connect(self._on_field_committed)

        # The unit lives outside the editable box so the box holds a bare
        # number the validator will accept.
        self._unit_label = QLabel(unit)
        self._unit_label.setObjectName("RangeLabel")
        self._unit_label.setVisible(bool(unit))

        self._slider = PaintedSlider(self._min, self._max, self._value, single_step=step)
        self._slider.valueChanged.connect(self._on_slider_moved)

        self._min_label = QLabel(self._format_plain(self._min))
        self._min_label.setObjectName("RangeLabel")
        self._max_label = QLabel(self._format_plain(self._max))
        self._max_label.setObjectName("RangeLabel")
        captions = QHBoxLayout()
        captions.setContentsMargins(0, 0, 0, 0)
        captions.addWidget(self._min_label)
        captions.addStretch(1)
        captions.addWidget(self._max_label)

        if inline:
            # Slider and value share one row; used where a caption already sits
            # above the control, as on the DOS panel. The min/max captions have
            # to be grouped with the slider in their own column here, sharing
            # its width exactly — added as a plain full-width row instead (the
            # non-inline layout below), they range over the *whole* row
            # including the value box and unit label beside the slider, so the
            # max caption lands under the value box's corner rather than under
            # the slider's own right edge.
            self._title.hide()
            slider_column = QVBoxLayout()
            slider_column.setContentsMargins(0, 0, 0, 0)
            slider_column.setSpacing(SPACING.xs + 2)
            slider_column.addWidget(self._slider)
            slider_column.addLayout(captions)

            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(SPACING.sm)
            row.addLayout(slider_column, 1)
            row.addWidget(self._field, 0)
            row.addWidget(self._unit_label, 0)
            layout.addLayout(row)
        else:
            header = QHBoxLayout()
            header.setContentsMargins(0, 0, 0, 0)
            header.setSpacing(SPACING.xs)
            header.addWidget(self._title)
            header.addStretch(1)
            header.addWidget(self._field)
            header.addWidget(self._unit_label)
            layout.addLayout(header)
            layout.addWidget(self._slider)
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
        self._slider.setRange(self._min, self._max)
        self._min_label.setText(self._format_plain(self._min))
        self._max_label.setText(self._format_plain(self._max))
        self.setValue(self._value, notify=False)

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802 - Qt naming
        super().setEnabled(enabled)
        self._slider.setEnabled(enabled)
        self._field.setEnabled(enabled)

    def set_unit(self, unit: str) -> None:
        """Change the unit shown beside the field.

        A setter rather than callers assigning ``_unit`` directly: the unit
        exists in two places (the string used to strip a pasted unit off typed
        input, and the label rendered next to the box), and setting only the
        first — which is what assigning the attribute does — leaves the unit
        permanently invisible.
        """
        self._unit = unit
        self._unit_label.setText(unit)
        self._unit_label.setVisible(bool(unit))
        self._field.setFixedWidth(66 if unit else 82)

    def set_title(self, title: str) -> None:
        self._title.setText(title)

    def set_single_step(self, step: float) -> None:
        """Arrow-key / scroll-wheel increment. Dragging stays continuous."""
        self._slider.setSingleStep(step)

    # -- internals ---------------------------------------------------------

    def _format_value(self, value: float) -> str:
        """The field's text: a bare number, never the unit — the unit is a
        separate label so the validator can accept what is typed here."""
        return f"{value:.{self._decimals}f}"

    def _format_plain(self, value: float) -> str:
        if abs(value - round(value)) < 1e-9:
            return f"{value:.0f}"
        return f"{value:.{self._decimals}f}"

    def _sync_widgets(self) -> None:
        self._emitting = True
        self._slider.setValue(self._value, notify=False)
        self._field.setText(self._format_value(self._value))
        self._emitting = False

    def _on_slider_moved(self, value: float) -> None:
        if self._emitting:
            return
        self.setValue(value)

    def _on_field_committed(self) -> None:
        # Units are tolerated on input even though they are never written
        # there — a value pasted from elsewhere, or left over from an older
        # session, should still commit rather than being thrown away.
        text = self._field.text()
        if self._unit:
            text = text.replace(self._unit, "")
        try:
            value = float(text.strip())
        except ValueError:
            self._sync_widgets()
            return
        # An out-of-range entry snaps to the nearest limit (setValue clamps).
        # If clamping means the number does not change, _sync_widgets still
        # has to run so the box shows the clamped value rather than the
        # rejected text the user typed.
        self.setValue(value)
        self._sync_widgets()
