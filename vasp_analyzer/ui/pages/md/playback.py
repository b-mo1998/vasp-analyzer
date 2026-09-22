"""Transport controls for trajectory playback.

Playback rate and physical simulation time are deliberately separate concepts
here. ``1x`` means :data:`BASE_FPS` frames drawn per second — a property of the
animation — while the femtoseconds each frame represents is a property of the
calculation. Conflating them would make a 0.5 fs run play at half the speed of
a 1 fs one for no reason the user could see.
"""

from __future__ import annotations

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ...theme import PALETTE, RADIUS, SPACING, TYPE
from ...widgets.painted_slider import PaintedSlider

BASE_FPS = 20.0
SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0]
DEFAULT_SPEED_INDEX = 2


class SpeedSelector(QWidget):
    """The 0.25x - 4x row of small exclusive buttons."""

    speedChanged = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        caption = QLabel("Playback Speed")
        caption.setAlignment(Qt.AlignCenter)
        caption.setStyleSheet(
            f"color: {PALETTE.text_secondary}; font-size: {TYPE.tiny}px;"
        )
        layout.addWidget(caption)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(3)
        self._buttons: list[QPushButton] = []
        for index, speed in enumerate(SPEEDS):
            label = f"{speed:g}×"
            button = QPushButton(label)
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.setFixedHeight(22)
            button.setMinimumWidth(38)
            button.setObjectName("SpeedButton")
            button.clicked.connect(lambda _=False, i=index: self._select(i))
            row.addWidget(button)
            self._buttons.append(button)
        layout.addLayout(row)

        self._index = DEFAULT_SPEED_INDEX
        self._buttons[self._index].setChecked(True)
        self._apply_styles()

    def _apply_styles(self) -> None:
        for index, button in enumerate(self._buttons):
            active = index == self._index
            button.setChecked(active)
            button.setStyleSheet(
                f"#SpeedButton {{"
                f" border-radius: {RADIUS.sm}px;"
                f" font-size: {TYPE.small}px;"
                f" border: 1px solid {PALETTE.accent if active else PALETTE.border};"
                f" background: {PALETTE.accent if active else PALETTE.surface};"
                f" color: {'#ffffff' if active else PALETTE.text_secondary};"
                f" }}"
            )

    def _select(self, index: int) -> None:
        self._index = index
        self._apply_styles()
        self.speedChanged.emit(SPEEDS[index])

    def speed(self) -> float:
        return SPEEDS[self._index]

    def reset(self) -> None:
        self._index = DEFAULT_SPEED_INDEX
        self._apply_styles()


class PlaybackBar(QWidget):
    """First/previous/play/next/last, a scrub bar, a speed selector and loop."""

    frameChanged = Signal(int)          # user-driven frame change
    playingChanged = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._n_frames = 0
        self._frame = 0
        self._playing = False
        self._loop = True
        self._times: list[float] | None = None
        self._time_unit = "ps"

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING.lg)

        layout.addWidget(self._build_transport())
        layout.addWidget(self._build_scrubber(), 1)
        layout.addWidget(self._build_right())

        self._sync()

    # -- construction --------------------------------------------------------

    def _build_transport(self) -> QWidget:
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING.xs)

        def button(text: str, tip: str, slot, primary: bool = False) -> QPushButton:
            widget = QPushButton(text)
            widget.setToolTip(tip)
            widget.setCursor(Qt.PointingHandCursor)
            size = 38 if primary else 30
            widget.setFixedSize(size, size)
            widget.setObjectName("PlayButton" if primary else "TransportButton")
            widget.clicked.connect(slot)
            row.addWidget(widget)
            return widget

        button("⏮", "First frame", self.go_first)
        button("◀", "Previous frame", self.step_back)
        self._play_button = button("▶", "Play", self.toggle_play, primary=True)
        button("▶", "Next frame", self.step_forward)
        button("⏭", "Last frame", self.go_last)
        self._style_transport()
        return holder

    def _style_transport(self) -> None:
        self.setStyleSheet(
            f"#TransportButton {{ border: 1px solid {PALETTE.border};"
            f" border-radius: {RADIUS.sm}px; background: {PALETTE.surface};"
            f" color: {PALETTE.text}; font-size: 12px; }}"
            f"#TransportButton:hover {{ background: {PALETTE.surface_hover}; }}"
            f"#TransportButton:disabled {{ color: {PALETTE.disabled_text}; }}"
            f"#PlayButton {{ border: none; border-radius: 19px;"
            f" background: {PALETTE.accent}; color: #ffffff; font-size: 14px; }}"
            f"#PlayButton:hover {{ background: {PALETTE.accent_hover}; }}"
            f"#PlayButton:disabled {{ background: {PALETTE.border}; }}"
        )

    def _build_scrubber(self) -> QWidget:
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(SPACING.xs)

        self._caption = QLabel("Frame — / —")
        self._caption.setAlignment(Qt.AlignCenter)
        self._caption.setStyleSheet(
            f"font-size: {TYPE.body}px; font-weight: 500; color: {PALETTE.text};"
        )
        column.addWidget(self._caption)

        self._slider = PaintedSlider(0.0, 1.0, 0.0, single_step=1.0)
        self._slider.valueChanged.connect(self._on_slider)
        column.addWidget(self._slider)

        ends = QHBoxLayout()
        ends.setContentsMargins(0, 0, 0, 0)
        self._start_label = QLabel("")
        self._start_label.setObjectName("RangeLabel")
        self._current_label = QLabel("")
        self._current_label.setObjectName("RangeLabel")
        self._end_label = QLabel("")
        self._end_label.setObjectName("RangeLabel")
        ends.addWidget(self._start_label)
        ends.addStretch(1)
        ends.addWidget(self._current_label)
        ends.addStretch(1)
        ends.addWidget(self._end_label)
        column.addLayout(ends)
        return holder

    def _build_right(self) -> QWidget:
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(SPACING.xs)

        loop_row = QHBoxLayout()
        loop_row.setContentsMargins(0, 0, 0, 0)
        loop_row.setSpacing(SPACING.sm)
        loop_row.addStretch(1)
        loop_label = QLabel("Loop")
        loop_label.setStyleSheet(f"font-size: {TYPE.small}px; color: {PALETTE.text};")
        loop_row.addWidget(loop_label)

        from ...widgets.toggle import ToggleSwitch

        self._loop_toggle = ToggleSwitch()
        self._loop_toggle.setChecked(True)
        self._loop_toggle.toggled.connect(self._on_loop)
        loop_row.addWidget(self._loop_toggle)
        loop_row.addStretch(1)
        column.addLayout(loop_row)

        self._speed = SpeedSelector()
        self._speed.speedChanged.connect(self._on_speed)
        column.addWidget(self._speed)
        return holder

    # -- state ---------------------------------------------------------------

    def set_frame_count(self, count: int, times: list[float] | None = None,
                        unit: str = "ps") -> None:
        self._n_frames = max(0, int(count))
        self._times = times
        self._time_unit = unit
        self._frame = 0
        self.stop()
        self._slider.setRange(0.0, float(max(self._n_frames - 1, 0)))
        self._slider.setValue(0.0, notify=False)
        self._sync()

    def frame(self) -> int:
        return self._frame

    def set_frame(self, frame: int, notify: bool = True) -> None:
        if self._n_frames == 0:
            return
        clamped = int(max(0, min(self._n_frames - 1, frame)))
        if clamped == self._frame:
            self._sync()
            return
        self._frame = clamped
        self._slider.setValue(float(clamped), notify=False)
        self._sync()
        if notify:
            self.frameChanged.emit(self._frame)

    def is_playing(self) -> bool:
        return self._playing

    # -- transport -----------------------------------------------------------

    def toggle_play(self) -> None:
        self.stop() if self._playing else self.play()

    def play(self) -> None:
        if self._n_frames < 2 or self._playing:
            return
        # Starting from the final frame would otherwise sit there doing nothing.
        if self._frame >= self._n_frames - 1:
            self.set_frame(0)
        self._playing = True
        self._timer.start(self._interval_ms())
        self._sync()
        self.playingChanged.emit(True)

    def stop(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
        if self._playing:
            self._playing = False
            self.playingChanged.emit(False)
        self._sync()

    def step_forward(self) -> None:
        self.stop()
        self.set_frame(self._frame + 1)

    def step_back(self) -> None:
        self.stop()
        self.set_frame(self._frame - 1)

    def go_first(self) -> None:
        self.stop()
        self.set_frame(0)

    def go_last(self) -> None:
        self.stop()
        self.set_frame(self._n_frames - 1)

    def _advance(self) -> None:
        if self._n_frames == 0:
            return
        if self._frame >= self._n_frames - 1:
            if self._loop:
                self.set_frame(0)
            else:
                self.stop()
            return
        self.set_frame(self._frame + 1)

    def _interval_ms(self) -> int:
        return max(8, int(1000.0 / (BASE_FPS * self._speed.speed())))

    # -- handlers -------------------------------------------------------------

    def _on_slider(self, value: float) -> None:
        self.stop()
        self.set_frame(int(round(value)))

    def _on_speed(self, _speed: float) -> None:
        if self._playing:
            self._timer.start(self._interval_ms())

    def _on_loop(self, enabled: bool) -> None:
        self._loop = bool(enabled)

    # -- display ---------------------------------------------------------------

    def _format_time(self, frame: int) -> str:
        if self._times is None or frame >= len(self._times):
            return f"{frame}"
        return f"{self._times[frame]:.2f} {self._time_unit}"

    def _sync(self) -> None:
        has_frames = self._n_frames > 0
        self._play_button.setText("❚❚" if self._playing else "▶")
        self._play_button.setToolTip("Pause" if self._playing else "Play")
        self._play_button.setEnabled(self._n_frames > 1)
        self._slider.setEnabled(self._n_frames > 1)

        if not has_frames:
            self._caption.setText("Frame — / —")
            for label in (self._start_label, self._current_label, self._end_label):
                label.setText("")
            return

        self._caption.setText(f"Frame {self._frame + 1} / {self._n_frames}")
        self._start_label.setText(self._format_time(0))
        self._current_label.setText(self._format_time(self._frame))
        self._end_label.setText(self._format_time(self._n_frames - 1))

    def reset(self) -> None:
        self.stop()
        self._speed.reset()
        self._loop_toggle.setChecked(True)
        self.set_frame(0)
