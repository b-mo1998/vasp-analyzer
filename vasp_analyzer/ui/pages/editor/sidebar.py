"""The Editor's left control column: source, tools, visualisation, filters."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...theme import PALETTE, RADIUS, SPACING, TYPE
from ...widgets.primitives import Divider, Label
from ...widgets.segmented import SegmentedControl
from ...widgets.toggle import ToggleSwitch

# (key, caption, tooltip with its shortcut)
#
# There is no "Add Atom" entry here. Adding an atom is a form (element,
# coordinates, Insert) in the AddAtomCard, not a click-in-viewport gesture —
# there was a tool button for it once, but it only ever switched to Select
# mode under a different label, since nothing ever wired up a placement
# click. A tool that does exactly what Select does is not a tool.
TOOLS = [
    ("select", "Select", "Select atoms  (S)"),
    ("delete", "Delete Atom", "Delete the selection  (Del)"),
    ("move", "Grab and Move", "Grab and move the selection  (G)"),
    ("duplicate", "Duplicate", "Duplicate the selection  (D)"),
    ("measure", "Measure", "Measure distance / angle  (M)"),
]

STYLES = ["Ball-and-Stick", "Space Filling", "Wireframe"]
COLORINGS = ["By Element", "Uniform"]


class ElementChip(QPushButton):
    """A per-element filter chip.

    Checked chips restrict what picking will select, which is what makes a
    dense multi-element cell workable — clicking through a lattice of oxygen
    to reach the niobium underneath is otherwise a fight.
    """

    def __init__(self, element: str, color: str, parent: QWidget | None = None) -> None:
        super().__init__(element, parent)
        self._element = element
        self._color = color
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(26)
        self.setToolTip(f"Restrict selection to {element}")
        self._apply_style()
        self.toggled.connect(lambda _c: self._apply_style())

    @property
    def element(self) -> str:
        return self._element

    def _apply_style(self) -> None:
        if self.isChecked():
            self.setStyleSheet(
                f"QPushButton {{ background: {self._color}; color: white;"
                f" border: 1px solid {self._color}; border-radius: {RADIUS.sm}px;"
                f" padding: 0 10px; font-size: {TYPE.small}px; font-weight: 600; }}"
            )
        else:
            self.setStyleSheet(
                f"QPushButton {{ background: transparent; color: {PALETTE.text};"
                f" border: 1px solid {self._color}; border-radius: {RADIUS.sm}px;"
                f" padding: 0 10px; font-size: {TYPE.small}px; }}"
            )


class EditorSidebar(QWidget):
    """Controls for the Editor page."""

    toolChanged = Signal(str)
    styleChanged = Signal(str)
    coloringChanged = Signal(str)
    coordinateModeChanged = Signal(bool)     # True = fractional
    snapChanged = Signal(bool)
    constrainChanged = Signal(bool)
    filterChanged = Signal(object)           # set[str] of allowed elements
    undoRequested = Signal()
    redoRequested = Signal()
    clearSelectionRequested = Signal()
    hideSelectionRequested = Signal()
    isolateSelectionRequested = Signal()
    showAllAtomsRequested = Signal()
    applyRequested = Signal()
    sourceChanged = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        # A sidebar is a fixed-width column; anything too wide for it is a
        # layout bug to fix at the source, not something to scroll sideways.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(SPACING.lg, SPACING.xl, SPACING.lg, SPACING.xl)
        layout.setSpacing(SPACING.md)
        scroll.setWidget(content)

        header = QVBoxLayout()
        header.setSpacing(SPACING.xs)
        header.addWidget(Label("Structure Editor", "TitleLabel"))
        subtitle = Label("Edit and manipulate POSCAR crystal structures.", "SubtitleLabel")
        subtitle.setWordWrap(True)
        header.addWidget(subtitle)
        layout.addLayout(header)
        layout.addWidget(Divider())

        # -- source ---------------------------------------------------------
        layout.addWidget(Label("FILE / STRUCTURE", "SectionLabel"))
        self._source = QComboBox()
        self._source.setFixedHeight(30)
        self._source.currentTextChanged.connect(self.sourceChanged)
        layout.addWidget(self._source)
        self._structure_label = QLabel("Structure: —")
        self._structure_label.setStyleSheet(
            f"color: {PALETTE.text_secondary}; font-size: {TYPE.small}px;"
        )
        self._structure_label.setWordWrap(True)
        layout.addWidget(self._structure_label)

        # -- tools ----------------------------------------------------------
        layout.addWidget(Label("EDIT TOOLS", "SectionLabel"))
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)
        self._tool_buttons: dict[str, QPushButton] = {}
        for index, (key, caption, tip) in enumerate(TOOLS):
            button = QPushButton(caption)
            button.setObjectName("ToolButton")
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.setFixedHeight(30)
            button.setToolTip(tip)
            button.setStyleSheet("QPushButton { text-align: left; padding-left: 10px; }")
            self._tool_group.addButton(button, index)
            layout.addWidget(button)
            self._tool_buttons[key] = button
        self._tool_buttons["select"].setChecked(True)
        self._tool_group.idClicked.connect(
            lambda index: self.toolChanged.emit(TOOLS[index][0])
        )

        # -- visualisation --------------------------------------------------
        layout.addWidget(Label("VISUALIZATION", "SectionLabel"))
        self._style = QComboBox()
        self._style.addItems(STYLES)
        self._style.setFixedHeight(30)
        self._style.currentTextChanged.connect(self.styleChanged)
        layout.addWidget(self._style)

        layout.addWidget(Label("COLORING", "SectionLabel"))
        self._coloring = QComboBox()
        self._coloring.addItems(COLORINGS)
        self._coloring.setFixedHeight(30)
        self._coloring.currentTextChanged.connect(self.coloringChanged)
        layout.addWidget(self._coloring)

        # -- coordinate mode ------------------------------------------------
        layout.addWidget(Label("COORDINATE MODE", "SectionLabel"))
        self._coords = SegmentedControl(["Cartesian", "Fractional"])
        self._coords.setCurrentIndex(1)
        self._coords.currentChanged.connect(
            lambda index: self.coordinateModeChanged.emit(index == 1)
        )
        layout.addWidget(self._coords)

        self._snap = self._toggle_row(layout, "Snap to Grid", True, self.snapChanged)
        self._constrain = self._toggle_row(
            layout, "Constrain to Cell", False, self.constrainChanged
        )

        # -- element filter -------------------------------------------------
        layout.addWidget(Label("SELECTION FILTER (ELEMENT)", "SectionLabel"))
        self._filter_field = QLineEdit()
        self._filter_field.setObjectName("ValueField")
        self._filter_field.setPlaceholderText("Filter elements…")
        self._filter_field.setFixedHeight(28)
        self._filter_field.textChanged.connect(self._on_filter_text)
        layout.addWidget(self._filter_field)

        self._chip_host = QWidget()
        self._chip_layout = QHBoxLayout(self._chip_host)
        self._chip_layout.setContentsMargins(0, 0, 0, 0)
        self._chip_layout.setSpacing(SPACING.xs)
        self._chip_layout.addStretch(1)
        layout.addWidget(self._chip_host)
        self._chips: list[ElementChip] = []

        # -- visibility ------------------------------------------------------
        # Deliberately its own section rather than another chip row: hiding
        # is about what is drawn, while the filter above is about what a
        # click can select, and conflating the two would make "I can't see
        # it" and "I can't click it" the same control when they are routinely
        # wanted apart.
        layout.addWidget(Label("VISIBILITY", "SectionLabel"))
        visibility = QHBoxLayout()
        visibility.setSpacing(SPACING.sm)
        self._hide = QPushButton("Hide")
        self._hide.setToolTip(
            "Hide the selected atoms from the viewer  (H).\n"
            "They stay in the structure, the table and any file you save."
        )
        self._isolate = QPushButton("Isolate")
        self._isolate.setToolTip(
            "Hide everything except the selection  (Shift+H) — for reading "
            "one layer or adsorbate out of a crowded cell."
        )
        for button, signal in (
            (self._hide, self.hideSelectionRequested),
            (self._isolate, self.isolateSelectionRequested),
        ):
            button.setObjectName("ToolButton")
            button.setCursor(Qt.PointingHandCursor)
            button.setFixedHeight(30)
            button.setEnabled(False)
            button.clicked.connect(signal.emit)
            visibility.addWidget(button)
        layout.addLayout(visibility)

        self._show_all = QPushButton("Show All Atoms")
        self._show_all.setObjectName("ToolButton")
        self._show_all.setCursor(Qt.PointingHandCursor)
        self._show_all.setFixedHeight(30)
        self._show_all.setEnabled(False)
        self._show_all.setToolTip("Bring every hidden atom back  (Alt+H)")
        self._show_all.clicked.connect(self.showAllAtomsRequested)
        layout.addWidget(self._show_all)

        self._hidden_caption = QLabel("")
        self._hidden_caption.setStyleSheet(
            f"font-size: {TYPE.small}px; color: {PALETTE.text_secondary};"
        )
        layout.addWidget(self._hidden_caption)

        self._clear = QPushButton("Clear Selection")
        self._clear.setObjectName("ToolButton")
        self._clear.setCursor(Qt.PointingHandCursor)
        self._clear.setFixedHeight(30)
        self._clear.setStyleSheet(f"QPushButton {{ color: {PALETTE.danger}; }}")
        self._clear.setToolTip("Deselect all atoms — this does not delete anything")
        self._clear.clicked.connect(self.clearSelectionRequested)
        layout.addWidget(self._clear)

        self._apply = QPushButton("✓  Apply Changes")
        self._apply.setObjectName("PrimaryButton")
        self._apply.setCursor(Qt.PointingHandCursor)
        self._apply.setFixedHeight(34)
        self._apply.setToolTip(
            "Commit the working structure to this session. Files are only "
            "written by Save / Export."
        )
        self._apply.clicked.connect(self.applyRequested)
        layout.addWidget(self._apply)

        layout.addStretch(1)

    def _toggle_row(
        self, layout: QVBoxLayout, caption: str, checked: bool, signal
    ) -> ToggleSwitch:
        row = QHBoxLayout()
        label = QLabel(caption)
        label.setStyleSheet(f"font-size: {TYPE.body}px;")
        row.addWidget(label)
        row.addStretch(1)
        toggle = ToggleSwitch()
        toggle.setChecked(checked)
        toggle.toggled.connect(signal)
        row.addWidget(toggle)
        layout.addLayout(row)
        return toggle

    # -- state ---------------------------------------------------------------

    def set_sources(self, names: list[str], current: str | None = None) -> None:
        self._source.blockSignals(True)
        self._source.clear()
        self._source.addItems(names)
        if current and current in names:
            self._source.setCurrentText(current)
        self._source.blockSignals(False)

    def set_hidden_count(self, count: int) -> None:
        """Report how many atoms are currently hidden.

        Stated in words rather than left implicit: a hidden atom looks
        exactly like an absent one, and a user who forgets they hid half a
        slab will otherwise read the viewer as having lost atoms.
        """
        self._show_all.setEnabled(count > 0)
        self._hidden_caption.setText(
            "" if count == 0 else f"{count} atom{'s' if count != 1 else ''} hidden"
        )

    def set_selection_present(self, present: bool) -> None:
        self._hide.setEnabled(present)
        self._isolate.setEnabled(present)

    def set_structure_caption(self, text: str) -> None:
        self._structure_label.setText(f"Structure: {text}" if text else "Structure: —")

    def set_elements(self, elements: list[str]) -> None:
        from ....services import appearance

        for chip in self._chips:
            self._chip_layout.removeWidget(chip)
            chip.setParent(None)
            chip.deleteLater()
        self._chips.clear()

        for element in elements:
            chip = ElementChip(element, appearance.effective_color(element))
            chip.toggled.connect(lambda _c: self._emit_filter())
            self._chip_layout.insertWidget(self._chip_layout.count() - 1, chip)
            self._chips.append(chip)
        self._on_filter_text(self._filter_field.text())

    def _on_filter_text(self, text: str) -> None:
        needle = text.strip().lower()
        for chip in self._chips:
            chip.setVisible(not needle or needle in chip.element.lower())

    def _emit_filter(self) -> None:
        allowed = {chip.element for chip in self._chips if chip.isChecked()}
        # No chip checked means no restriction, which is friendlier than
        # "nothing is selectable" and is what an empty filter usually means.
        self.filterChanged.emit(allowed)

    def style(self) -> str:
        return self._style.currentText()

    def coloring(self) -> str:
        return self._coloring.currentText()

    def set_history_state(self, can_undo: bool, can_redo: bool) -> None:
        """No-op: the sidebar no longer shows Undo/Redo buttons.

        Kept as a method rather than deleted outright so the page can keep
        reporting history state without knowing whether the sidebar happens
        to display it — the keyboard shortcuts still drive the same
        CommandStack, they just have no button to grey out.
        """

    def set_tool(self, key: str) -> None:
        button = self._tool_buttons.get(key)
        if button is not None:
            button.setChecked(True)

    def tool(self) -> str:
        index = self._tool_group.checkedId()
        return TOOLS[index][0] if 0 <= index < len(TOOLS) else "select"

    @property
    def snap_enabled(self) -> bool:
        return self._snap.isChecked()

    @property
    def constrain_enabled(self) -> bool:
        return self._constrain.isChecked()

    def reset(self) -> None:
        self._style.setCurrentIndex(0)
        self._coloring.setCurrentIndex(0)
        self._coords.setCurrentIndex(1)
        self._snap.setChecked(True)
        self._constrain.setChecked(False)
        self._filter_field.clear()
        for chip in self._chips:
            chip.setChecked(False)
        self.set_tool("select")
