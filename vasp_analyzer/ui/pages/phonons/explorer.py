"""The Phonon Mode Explorer: browse branches at the selected q-point.

Deliberately narrow in scope: a q-point stepper and a row of branch cards.
Everything else that once lived here — a q-path dropdown duplicating the
sidebar's, and a "Branch n / m" stepper duplicating the cards themselves —
was a second control for state already on screen, and two controls for one
piece of state is how they drift apart.

Thumbnails are static displacement snapshots rather than animations: six
simultaneously animating structures cost far more than they communicate, and
the arrows alone already distinguish one branch from the next. They are cached
per (q-point, branch), so stepping back and forth does not re-render the same
images.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ....models.phonon import PhononDataset, convert_frequency
from ....services import phonon as phonon_service
from ....services.appearance import effective_color
from ....services.bonds import build_sites_and_bonds
from ...theme import PALETTE, RADIUS, SPACING, TYPE
from ..structure.camera import Camera

# Larger than the original 96x84: the explorer no longer carries two redundant
# dropdowns, and the reclaimed space is worth more as legible thumbnails.
THUMBNAIL_SIZE = QSize(132, 116)
# Beyond this the thumbnail is a grey smear; the 3D viewer is the place to
# inspect a large cell.
MAX_THUMBNAIL_ATOMS = 60
ARROW_COLOR = "#4d8dff"
# Displacement arrows are scaled relative to the cell, so the same mode reads
# the same whatever the cell size.
ARROW_SCALE = 1.9
# Atoms are drawn small and hard-capped. The projection's per-point scale is
# tuned for a full-size viewport; reused unchanged at thumbnail size it turns
# a two-atom cell into a pair of discs that swallow the very arrows the
# thumbnail exists to show.
ATOM_RADIUS_SCALE = 0.11
# Pulled back from a tight fit so the displaced positions and arrowheads stay
# inside the frame rather than running off the edge.
THUMBNAIL_ZOOM = 0.62


def _draw_arrow(painter: QPainter, tail: QPointF, head: QPointF, colour: QColor) -> None:
    """A line with a solid head. Bare segments — what this drew before — leave
    the direction of motion ambiguous, which is the one thing the thumbnail
    exists to show."""
    direction = np.array([head.x() - tail.x(), head.y() - tail.y()])
    length = float(np.linalg.norm(direction))
    if length < 2.0:
        return
    unit = direction / length
    normal = np.array([-unit[1], unit[0]])

    pen = QPen(colour, 1.6)
    pen.setCapStyle(Qt.RoundCap)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawLine(tail, head)

    size = min(6.0, length * 0.45)
    base = np.array([head.x(), head.y()]) - unit * size
    left = base + normal * size * 0.5
    right = base - normal * size * 0.5
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(colour))
    painter.drawPolygon(
        QPolygonF(
            [
                head,
                QPointF(float(left[0]), float(left[1])),
                QPointF(float(right[0]), float(right[1])),
            ]
        )
    )


def render_mode_thumbnail(
    dataset: PhononDataset, qpoint_index: int, branch_index: int, camera: Camera
) -> QPixmap:
    """A small static snapshot of one mode's displacement pattern."""
    pixmap = QPixmap(THUMBNAIL_SIZE)
    # White rather than the dark canvas the rest of the page's 3D viewer
    # uses: several elements' Jmol colours (carbon among them) are near
    # black, and a near-black ball on a near-black background is exactly
    # the "can't see the atoms" complaint this answers.
    pixmap.fill(QColor(PALETTE.surface))
    structure = dataset.structure
    if structure is None or structure.n_atoms > MAX_THUMBNAIL_ATOMS:
        return pixmap

    displacement = phonon_service.static_displacement(
        dataset, qpoint_index, branch_index, amplitude=0.55
    )

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    try:
        bonds = build_sites_and_bonds(structure, include_ghosts=False)
        positions = np.array([site.position for site in bonds.sites])
        if positions.size == 0:
            return pixmap
        screen, depth, scale = camera.project(
            positions, THUMBNAIL_SIZE.width(), THUMBNAIL_SIZE.height()
        )

        # Bonds first, so atoms and arrows sit on top of them. A near-white,
        # near-transparent pen was chosen for the old dark background and is
        # invisible on this light one — a plain mid-grey reads clearly on
        # either.
        painter.setPen(QPen(QColor(PALETTE.text_tertiary), 1.0))
        for i, j in bonds.bonds:
            if i < len(screen) and j < len(screen):
                painter.drawLine(
                    QPointF(float(screen[i][0]), float(screen[i][1])),
                    QPointF(float(screen[j][0]), float(screen[j][1])),
                )

        for index in np.argsort(depth):
            index = int(index)
            site = bonds.sites[index]
            radius = min(9.0, max(2.0, float(scale[index]) * ATOM_RADIUS_SCALE))
            centre = QPointF(float(screen[index][0]), float(screen[index][1]))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(effective_color(site.symbol)))
            painter.drawEllipse(centre, radius, radius)

            if displacement is None or site.index >= len(displacement):
                continue
            tip = camera.project(
                np.array([site.position + displacement[site.index] * ARROW_SCALE]),
                THUMBNAIL_SIZE.width(),
                THUMBNAIL_SIZE.height(),
            )[0][0]
            _draw_arrow(
                painter,
                centre,
                QPointF(float(tip[0]), float(tip[1])),
                QColor(ARROW_COLOR),
            )
    finally:
        painter.end()
    return pixmap


class ModeCard(QWidget):
    """One branch, with its frequency and a displacement snapshot."""

    clicked = Signal(int)   # branch index

    def __init__(
        self,
        branch_index: int,
        frequency: float,
        units: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._branch = branch_index
        self._selected = False
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedWidth(THUMBNAIL_SIZE.width() + 14)
        # A plain QWidget does not paint its stylesheet background or border
        # unless told to, so without this the selected card's blue outline
        # simply never appears.
        self.setAttribute(Qt.WA_StyledBackground, True)

        column = QVBoxLayout(self)
        column.setContentsMargins(SPACING.xs, SPACING.xs, SPACING.xs, SPACING.xs)
        column.setSpacing(3)

        self._title = QLabel(f"Branch {branch_index + 1}")
        self._title.setAlignment(Qt.AlignCenter)
        self._title.setStyleSheet(f"font-size: {TYPE.small}px; font-weight: 600;")
        column.addWidget(self._title)

        self._image = QLabel()
        self._image.setFixedSize(THUMBNAIL_SIZE)
        self._image.setAlignment(Qt.AlignCenter)
        column.addWidget(self._image, 0, Qt.AlignHCenter)

        self._frequency = QLabel(f"{frequency:.2f} {units}")
        self._frequency.setAlignment(Qt.AlignCenter)
        colour = PALETTE.danger if frequency < -0.1 else PALETTE.text_secondary
        self._frequency.setStyleSheet(f"font-size: {TYPE.small}px; color: {colour};")
        column.addWidget(self._frequency)

        self._apply_style()

    @property
    def branch(self) -> int:
        return self._branch

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self._image.setPixmap(pixmap)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._apply_style()

    def _apply_style(self) -> None:
        if self._selected:
            self.setStyleSheet(
                f"ModeCard {{ background: {PALETTE.accent_soft};"
                f" border: 2px solid {PALETTE.accent};"
                f" border-radius: {RADIUS.md}px; }}"
            )
        else:
            self.setStyleSheet(
                f"ModeCard {{ background: transparent;"
                f" border: 1px solid {PALETTE.border};"
                f" border-radius: {RADIUS.md}px; }}"
            )

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.clicked.emit(self._branch)
        super().mousePressEvent(event)


class ModeExplorer(QWidget):
    """Q-point navigation with a centred strip of branch cards."""

    modeSelected = Signal(int, int)     # qpoint index, branch index

    VISIBLE_CARDS = 6

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dataset: PhononDataset | None = None
        self._qpoint = 0
        self._branch = 0
        self._units = "THz"
        self._cards: list[ModeCard] = []
        self._thumbnails: dict[tuple[int, int], QPixmap] = {}
        self._camera = Camera()

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(SPACING.sm)

        # --- header: title, then the q-point stepper centred in the row -----
        header = QHBoxLayout()
        header.setSpacing(SPACING.sm)
        title = QLabel("Phonon Mode Explorer")
        title.setObjectName("CardTitle")
        header.addWidget(title)
        header.addStretch(1)

        header.addWidget(self._caption("Q-point"))
        self._previous_q = self._arrow("‹", "Previous q-point")
        self._previous_q.clicked.connect(lambda: self._step_qpoint(-1))
        header.addWidget(self._previous_q)

        self._qpoint_label = QLabel("—")
        self._qpoint_label.setMinimumWidth(180)
        self._qpoint_label.setAlignment(Qt.AlignCenter)
        self._qpoint_label.setStyleSheet(
            f"font-size: {TYPE.body}px; font-weight: 500; color: {PALETTE.text};"
        )
        header.addWidget(self._qpoint_label)

        self._next_q = self._arrow("›", "Next q-point")
        self._next_q.clicked.connect(lambda: self._step_qpoint(1))
        header.addWidget(self._next_q)
        header.addStretch(1)
        column.addLayout(header)

        # --- branch cards ----------------------------------------------------
        strip = QHBoxLayout()
        strip.setSpacing(SPACING.sm)
        self._previous_page = self._arrow("‹", "Earlier branches")
        self._previous_page.clicked.connect(
            lambda: self._step_branch(-self.VISIBLE_CARDS)
        )
        strip.addWidget(self._previous_page)

        self._strip_host = QWidget()
        self._strip_layout = QHBoxLayout(self._strip_host)
        self._strip_layout.setContentsMargins(0, 0, 0, 0)
        self._strip_layout.setSpacing(SPACING.sm)
        self._strip_layout.setAlignment(Qt.AlignCenter)
        strip.addWidget(self._strip_host, 1)

        self._next_page = self._arrow("›", "Later branches")
        self._next_page.clicked.connect(lambda: self._step_branch(self.VISIBLE_CARDS))
        strip.addWidget(self._next_page)
        column.addLayout(strip)

    # -- small widgets ------------------------------------------------------

    def _arrow(self, text: str, tooltip: str) -> QPushButton:
        button = QPushButton(text)
        button.setCursor(Qt.PointingHandCursor)
        button.setFixedSize(28, 28)
        button.setToolTip(tooltip)
        # Not the shared ToolButton style: its horizontal padding exceeds a
        # 28px button and squeezes the chevron out of existence.
        button.setStyleSheet(
            f"QPushButton {{ background: {PALETTE.surface};"
            f" border: 1px solid {PALETTE.border};"
            f" border-radius: {RADIUS.sm}px; padding: 0px;"
            f" font-size: {TYPE.body}px; color: {PALETTE.text}; }}"
            f"QPushButton:hover:enabled {{ background: {PALETTE.surface_hover}; }}"
            f"QPushButton:disabled {{ color: {PALETTE.disabled_text}; }}"
        )
        return button

    @staticmethod
    def _caption(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(
            f"color: {PALETTE.text_secondary}; font-size: {TYPE.small}px;"
        )
        return label

    # -- state -------------------------------------------------------------

    def set_units(self, units: str) -> None:
        self._units = units
        self._rebuild_cards()

    def set_dataset(self, dataset: PhononDataset | None) -> None:
        self._dataset = dataset
        self._thumbnails.clear()
        if dataset is not None and dataset.structure is not None:
            self._camera = Camera()
            self._camera.fit(dataset.structure.cart_coords)
            self._camera.orbit(-40, -25)
            self._camera.zoom = THUMBNAIL_ZOOM
        self.set_selection(0, 0)

    def set_selection(self, qpoint_index: int, branch_index: int) -> None:
        dataset = self._dataset
        if dataset is None or dataset.n_qpoints == 0:
            self._qpoint_label.setText("—")
            self._clear_cards()
            return

        self._qpoint = max(0, min(qpoint_index, dataset.n_qpoints - 1))
        self._branch = max(0, min(branch_index, dataset.n_branches - 1))

        band = dataset.bands[self._qpoint]
        coords = ", ".join(f"{v:.3f}" for v in band.qpoint)
        label = f"{band.label}  " if band.label else ""
        self._qpoint_label.setText(f"{label}({coords})")
        self._previous_q.setEnabled(self._qpoint > 0)
        self._next_q.setEnabled(self._qpoint < dataset.n_qpoints - 1)
        self._rebuild_cards()

    def _step_qpoint(self, delta: int) -> None:
        if self._dataset is None:
            return
        target = self._qpoint + delta
        if 0 <= target < self._dataset.n_qpoints:
            self.modeSelected.emit(target, self._branch)

    def _step_branch(self, delta: int) -> None:
        if self._dataset is None:
            return
        target = max(0, min(self._branch + delta, self._dataset.n_branches - 1))
        if target != self._branch:
            self.modeSelected.emit(self._qpoint, target)

    def _clear_cards(self) -> None:
        """Empty the strip completely — every item, not just the cards.

        Removing only the card widgets left the layout's stretch spacers
        behind, and the next rebuild added another. They accumulated ahead of
        the cards, so every branch click nudged the whole strip further
        right.
        """
        while self._strip_layout.count():
            item = self._strip_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._cards.clear()

    def _rebuild_cards(self) -> None:
        dataset = self._dataset
        self._clear_cards()
        if dataset is None or dataset.n_branches == 0:
            return

        # Centre the window on the selection, clamped so it never runs past
        # either end — the selected card is always on screen.
        half = self.VISIBLE_CARDS // 2
        start = max(0, min(self._branch - half, dataset.n_branches - self.VISIBLE_CARDS))
        end = min(dataset.n_branches, start + self.VISIBLE_CARDS)

        frequencies = dataset.bands[self._qpoint].frequencies
        for branch in range(start, end):
            card = ModeCard(
                branch,
                float(convert_frequency(frequencies[branch], self._units)),
                self._units,
            )
            card.set_selected(branch == self._branch)
            card.clicked.connect(
                lambda b=branch: self.modeSelected.emit(self._qpoint, b)
            )
            card.set_pixmap(self._thumbnail(branch))
            self._strip_layout.addWidget(card)
            self._cards.append(card)

        self._previous_page.setEnabled(start > 0)
        self._next_page.setEnabled(end < dataset.n_branches)

    def _thumbnail(self, branch: int) -> QPixmap:
        key = (self._qpoint, branch)
        cached = self._thumbnails.get(key)
        if cached is not None:
            return cached
        assert self._dataset is not None
        pixmap = render_mode_thumbnail(self._dataset, self._qpoint, branch, self._camera)
        self._thumbnails[key] = pixmap
        return pixmap
