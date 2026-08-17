"""A large, explicit integer stepper: minus button, number, plus button.

Not a QSpinBox. Its native up/down subcontrols turn out to silently lose
their arrow glyphs once styled via QSS on this app's Fusion base — the same
class of quirk that made QSlider unreliable inside a QScrollArea. Rather than
fight Qt's compound-widget rendering again, this uses two ordinary
push-buttons (the same #ToolButton pattern the viewer's zoom +/- controls
already use reliably) so the click targets are unambiguous and guaranteed
visible.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import QHBoxLayout, QLineEdit, QPushButton, QWidget


class IntegerStepper(QWidget):
    """A bounded integer value, editable by typing or by +/- buttons."""

    valueChanged = Signal(int)

    def __init__(
        self,
        minimum: int = 1,
        maximum: int = 99,
        value: int = 1,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._min = minimum
        self._max = maximum
        self._value = max(minimum, min(maximum, value))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._minus = QPushButton("−")
        self._minus.setObjectName("StepperButtonLeft")
        self._minus.setFixedSize(30, 34)
        self._minus.setCursor(Qt.PointingHandCursor)
        self._minus.setAutoRepeat(True)
        self._minus.clicked.connect(lambda: self.setValue(self._value - 1))
        layout.addWidget(self._minus)

        self._field = QLineEdit(str(self._value))
        self._field.setObjectName("StepperField")
        self._field.setAlignment(Qt.AlignCenter)
        self._field.setFixedHeight(34)
        self._field.setValidator(QIntValidator(minimum, maximum, self))
        self._field.editingFinished.connect(self._on_field_committed)
        layout.addWidget(self._field, 1)

        self._plus = QPushButton("+")
        self._plus.setObjectName("StepperButtonRight")
        self._plus.setFixedSize(30, 34)
        self._plus.setCursor(Qt.PointingHandCursor)
        self._plus.setAutoRepeat(True)
        self._plus.clicked.connect(lambda: self.setValue(self._value + 1))
        layout.addWidget(self._plus)

        self._sync_buttons()

    # -- public API (matches enough of QSpinBox to be a drop-in) -----------

    def value(self) -> int:
        return self._value

    def setValue(self, value: int, notify: bool = True) -> None:  # noqa: N802
        clamped = max(self._min, min(self._max, int(value)))
        self._field.setText(str(clamped))
        if clamped == self._value:
            return
        self._value = clamped
        self._sync_buttons()
        if notify:
            self.valueChanged.emit(self._value)

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802
        super().setEnabled(enabled)
        self._minus.setEnabled(enabled and self._value > self._min)
        self._plus.setEnabled(enabled and self._value < self._max)
        self._field.setEnabled(enabled)

    def _sync_buttons(self) -> None:
        if not self.isEnabled():
            return
        self._minus.setEnabled(self._value > self._min)
        self._plus.setEnabled(self._value < self._max)

    def _on_field_committed(self) -> None:
        text = self._field.text().strip()
        try:
            self.setValue(int(text))
        except ValueError:
            self._field.setText(str(self._value))
