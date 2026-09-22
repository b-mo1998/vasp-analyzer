"""The Editor's 3D view: multi-selection and a translation gizmo.

Everything about drawing atoms, bonds, cells, ghosts and the camera comes from
:class:`~vasp_analyzer.ui.pages.structure.viewer.StructureView`. This subclass
adds only what editing needs — selecting more than one atom, and dragging a
selection either along a fixed axis or toward one of the selected atom's
nearest neighbours.

The gizmo is sized in **screen pixels**, not Angstroms. Earlier it used a
fixed fraction of the model's physical extent, which meant the handles
ballooned to fill the screen when zoomed in and shrank to an unusable speck
when zoomed out — exactly backwards, since precise dragging is most needed at
high zoom. Sizing to a constant pixel length (via the local pixels-per-Å scale
at the gizmo's own position) keeps the handles a comfortable, grabbable size
at any zoom, and as a side effect keeps drag sensitivity tied to the current
view rather than to how large the structure happens to be.

The arrows are cartesian x, y, z, not the lattice vectors, because a user
dragging "up" expects the atom to move up in space rather than along a
possibly oblique **c**. The drag is resolved by projecting the mouse motion
onto the axis as it appears on screen, then converting the resulting cartesian
displacement to fractional through the inverse lattice — so it stays correct
for triclinic cells and at any camera orientation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from ...theme import PALETTE
from ..sequence_view import VIEWER_BACKGROUND, lift_for_dark
from ..structure.viewer import StructureView

# Cartesian axis directions, their colours and their labels. The convention is
# the near-universal one for 3D transform gizmos.
AXES = [
    (np.array([1.0, 0.0, 0.0]), "#e5484d", "x"),
    (np.array([0.0, 1.0, 0.0]), "#46a758", "y"),
    (np.array([0.0, 0.0, 1.0]), "#3b82f6", "z"),
]

# A warm, neutral tone distinct from the R/G/B axes, the accent-blue
# selection ring and the white/text overlay colours — reads as "secondary,
# bond-shaped handle" rather than a fourth fixed axis.
NEIGHBOR_COLOR = "#f2b84b"
# The pair-move handle is one specific, user-chosen direction rather than a
# family of candidates — a different hue (violet) keeps it from being read as
# "just another neighbour handle" when both happen to be visible in explored
# code paths, though in practice the two are mutually exclusive on screen.
PAIR_COLOR = "#c264f0"

# Target on-screen length of the handles, in pixels — not Angstroms; see the
# module docstring. Neighbour handles are drawn shorter than the main axes so
# they read as secondary without competing for attention.
GIZMO_SCREEN_LENGTH = 108.0
NEIGHBOR_SCREEN_LENGTH = 76.0
# Click/hover tolerance in pixels. Larger than the old 9px along with the
# larger handles themselves — grabbing the right axis was the direct
# complaint this replaced.
GRAB_TOLERANCE = 13.0
# Target screen footprint, in pixels, of the tiny probe step used to measure
# a handle's true drag sensitivity (see _gizmo_handles). Small enough to stay
# a good local-derivative approximation, large enough to sit well above
# floating-point/projection noise at ordinary zoom levels.
PROBE_TARGET_PX = 6.0
# How much more sensitive a heavily foreshortened handle is allowed to be
# than a well-behaved, roughly-broadside one, before it gets capped. Above
# this, further obliqueness stops making the drag faster — a real effect,
# just one that reads as "broken" long before it reads as "fast."
MAX_SENSITIVITY_MULTIPLE = 8.0
# How many nearest-neighbour handles to offer at once. Enough to cover common
# coordination environments (3 for graphene's sp2 carbon, 4 for tetrahedral
# silicon) without the gizmo turning into a starburst.
MAX_NEIGHBOR_HANDLES = 4
# A neighbour more than this many times farther than the closest one is not
# really "nearby" — showing it would suggest a bond that is not there for an
# atom with genuinely low coordination (an adsorbate, a vacancy edge, ...).
NEIGHBOR_DISTANCE_RATIO = 1.6


@dataclass
class _Handle:
    """One draggable arm of the gizmo, already resolved to screen space.

    ``tip``/``arm_length`` describe how the handle is *drawn* — a length
    chosen to look right on screen, which for a handle pointing steeply
    toward or away from the camera is honestly foreshortened rather than
    forced to a fixed pixel length (see the long comment in
    ``_gizmo_handles``). ``screen_direction``/``sensitivity`` describe how a
    drag along it actually behaves, and are deliberately independent of the
    drawn length: they come from the true local derivative of the projection
    (a tiny physical step, not the full drawn arm), so they stay well-defined
    — and correctly near-zero, not wildly large — even for a handle that is
    nearly impossible to usefully drag from the current viewpoint.
    """

    tip: QPointF
    color: str
    kind: str              # "axis" | "neighbor"
    tooltip: str
    direction: np.ndarray   # unit vector, cartesian
    arm_length: float       # Å — this handle's current on-screen arm, in 3D
    screen_direction: np.ndarray   # unit 2D vector, screen space (may be ~0)
    sensitivity: float              # Å per screen pixel along screen_direction


class EditorView(StructureView):
    """Structure view with multi-selection and an axis/neighbour translation
    gizmo."""

    # The full selection after a click, as structure indices.
    selectionChanged = Signal(object)          # set[int]
    # Emitted continuously while dragging: cumulative cartesian delta (3,).
    dragMoved = Signal(object)
    # Emitted once when the drag ends, with the final cumulative delta.
    dragFinished = Signal(object)

    MODE_MOVE = "move"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._selection: set[int] = set()
        # Click order, structure-row indices — a set alone cannot say which
        # atom was selected *first*, which the pair-move handle needs to know
        # which one it is allowed to actually move.
        self._selection_order: list[int] = []
        self._gizmo_origin: np.ndarray | None = None   # cartesian
        # (neighbour_row, unit_direction, distance) — pushed in by the page
        # when exactly one atom is selected; see EditableStructure.nearest_neighbors.
        self._neighbor_handles: list[tuple[int, np.ndarray, float]] = []
        # (unit_direction, distance, target_row) toward the second of exactly
        # two selected atoms — pushed in by the page; see
        # EditableStructure.direction_to.
        self._pair_handle: tuple[np.ndarray, float, int] | None = None
        self._drag_axis: int | None = None
        self._drag_start: QPointF | None = None
        self._drag_delta = np.zeros(3)
        self._drag_target_rows: set[int] | None = None
        # Gizmo geometry snapshotted at mousePress and held for the whole
        # gesture — see _update_drag for why it must not be recomputed live.
        self._drag_geometry: tuple[QPointF, list[_Handle]] | None = None
        self._hovered_handle: int | None = None
        self._snap: float | None = None
        # The same dark canvas the MD, NEB and Charge viewers use, so the
        # editor reads as one of the 3D pages rather than a form.
        self.set_background(VIEWER_BACKGROUND)

    def element_color(self, symbol: str) -> QColor:
        return lift_for_dark(QColor(super().element_color(symbol)))

    # -- selection ---------------------------------------------------------

    def selection(self) -> set[int]:
        return set(self._selection)

    def drag_target_rows(self) -> set[int] | None:
        """Which rows the drag currently under way (or just finished) should
        actually move — None means "the page's own current selection", which
        is every handle except a pair handle. Valid for the page to read
        while handling ``dragMoved``/``dragFinished``; it is only replaced at
        the start of the *next* drag, never cleared out from under a signal
        that is still being handled.
        """
        return None if self._drag_target_rows is None else set(self._drag_target_rows)

    def selection_order(self) -> list[int]:
        """Structure-row indices in the order they were clicked. Only
        meaningful when the selection actually came from clicking in this
        view — a selection set from elsewhere (the table, undo, ...) falls
        back to a plain sorted order, which is a reasonable default but not a
        claim about what the user clicked first."""
        return list(self._selection_order)

    def set_selection(self, indices: set[int]) -> None:
        """Set the selection without emitting — used when the table or the
        inspector is the origin of the change, so the signal cannot loop."""
        indices = set(indices)
        if set(self._selection_order) != indices:
            self._selection_order = sorted(indices)
        self._selection = indices
        self._selected = next(iter(self._selection), None) if self._selection else None
        self._refresh_gizmo()
        self.update()

    def set_neighbor_handles(
        self, handles: list[tuple[int, np.ndarray, float]]
    ) -> None:
        """The nearest-neighbour bond directions to offer as extra drag
        handles — empty when the selection isn't exactly one atom, which is
        the only case a "toward this neighbour" direction is unambiguous."""
        self._neighbor_handles = list(handles)
        self.update()

    def set_pair_handle(
        self, direction: np.ndarray | None, distance: float | None, target_row: int | None
    ) -> None:
        """The direction toward the second of exactly two selected atoms, or
        None to clear it — set alongside :meth:`set_neighbor_handles` but
        mutually exclusive with it in practice, since one needs a single
        selected atom and the other needs two."""
        if direction is None or distance is None or target_row is None:
            self._pair_handle = None
        else:
            self._pair_handle = (direction, distance, target_row)
        self._refresh_gizmo()
        self.update()

    def set_snap(self, step: float | None) -> None:
        self._snap = step

    def refresh_gizmo(self) -> None:
        """Recompute the gizmo's on-screen anchor from the current structure
        and selection, without changing what is selected — used mid-drag,
        where atoms are moving but the selection itself is not. Calling
        :meth:`set_selection` there instead would, for a pair-move drag,
        wrongly collapse the two-atom selection down to whichever one row
        ended up passed in, losing the fixed second atom's highlight."""
        self._refresh_gizmo()
        self.update()

    def _refresh_gizmo(self) -> None:
        """Where the gizmo is grabbed from.

        Ordinarily the centroid of the selection, which is where a rigid
        multi-atom translation should be grabbed from. The one exception is
        an active pair handle: only the first-selected atom is going to move,
        so anchoring at the midpoint between it and the (stationary) second
        atom would visually suggest otherwise.
        """
        if not self._selection or self._structure is None:
            self._gizmo_origin = None
            return
        if (
            self._pair_handle is not None
            and len(self._selection) == 2
            and self._selection_order
            and self._selection_order[0] < self._structure.n_atoms
        ):
            self._gizmo_origin = self._structure.cart_coords[self._selection_order[0]]
            return
        rows = [i for i in sorted(self._selection) if i < self._structure.n_atoms]
        if not rows:
            self._gizmo_origin = None
            return
        self._gizmo_origin = self._structure.cart_coords[rows].mean(axis=0)

    # -- gizmo geometry ----------------------------------------------------

    def _gizmo_handles(self) -> tuple[QPointF, list[_Handle]] | None:
        """Origin and every handle's screen geometry, sized to a constant
        on-screen length rather than a fixed physical one."""
        if self._gizmo_origin is None or self._mode != self.MODE_MOVE:
            return None

        origin_screen, _depth, origin_scale = self._camera.project(
            self._gizmo_origin[None, :], self.width(), self.height()
        )
        origin = QPointF(float(origin_screen[0][0]), float(origin_screen[0][1]))
        # Pixels per Å at the gizmo's own depth — the local derivative of the
        # projection, valid for the small displacements a gizmo arm spans.
        pixels_per_angstrom = max(float(origin_scale[0]), 1e-6)

        specs: list[tuple[np.ndarray, str, float, str, str]] = [
            (direction, color, GIZMO_SCREEN_LENGTH, "axis", label)
            for direction, color, label in AXES
        ]
        if len(self._selection) == 1 and self._structure is not None:
            for row, direction, distance in self._neighbor_handles[:MAX_NEIGHBOR_HANDLES]:
                symbol = (
                    self._structure.species[row]
                    if row < len(self._structure.species)
                    else "?"
                )
                specs.append(
                    (
                        direction,
                        NEIGHBOR_COLOR,
                        NEIGHBOR_SCREEN_LENGTH,
                        "neighbor",
                        f"Move toward {symbol} #{row + 1} ({distance:.2f} Å)",
                    )
                )
        elif (
            self._pair_handle is not None
            and len(self._selection) == 2
            and self._structure is not None
        ):
            direction, distance, target_row = self._pair_handle
            symbol = (
                self._structure.species[target_row]
                if target_row < len(self._structure.species)
                else "?"
            )
            specs.append(
                (
                    direction,
                    PAIR_COLOR,
                    NEIGHBOR_SCREEN_LENGTH,
                    "pair",
                    f"Move toward {symbol} #{target_row + 1} ({distance:.2f} Å) — "
                    "only this atom moves",
                )
            )

        if not specs:
            return origin, []

        arm_lengths = [screen_len / pixels_per_angstrom for _d, _c, screen_len, _k, _t in specs]
        tip_points = np.array(
            [
                self._gizmo_origin + direction * arm
                for (direction, _color, _screen_len, _kind, _tooltip), arm in zip(
                    specs, arm_lengths
                )
            ]
        )
        projected, _depth2, _scale2 = self._camera.project(
            tip_points, self.width(), self.height()
        )

        # One correction pass: pixels_per_angstrom was measured at the
        # gizmo's origin, but under perspective a handle pointing toward or
        # away from the camera lands at a visibly different depth than the
        # origin, where that scale no longer quite applies — a handle end-on
        # to the view could land 10-20% short or long of the target length.
        # Re-scaling each arm by how far its first-pass tip actually missed
        # the target and projecting once more removes almost all of that
        # error without an iterative solver.
        #
        # The correction ratio itself is clamped to MAX_CORRECTION_RATIO.
        # A handle pointing *exactly* along the view axis has an on-screen
        # length that is not proportional to its 3D arm length at all — under
        # perspective it converges toward a vanishing point instead, so
        # "first-pass length came out near zero" does not mean "an arm
        # 1000x longer would reach the target," it means this handle's screen
        # length is fundamentally decoupled from its arm length. Left
        # unclamped, this single-shot linear extrapolation was inferring
        # arm lengths tens of times longer than the structure itself for
        # exactly this case — harmless for the static drawing (the point
        # still projects somewhere on screen), but ruinous once dragging
        # starts: distance-per-pixel in _update_drag scales directly with
        # arm_length, so grabbing a handle in this state turned a small mouse
        # move into a multi-angstrom jump. This is what "the gizmo breaks
        # under weird perspectives" was. Capping the ratio means a
        # near-view-axis handle simply looks foreshortened on screen —
        # correct, since that is what it actually is — rather than being
        # extrapolated into a physically meaningless length.
        MAX_CORRECTION_RATIO = 3.0
        actual_px = np.linalg.norm(
            projected[:, :2] - np.array([origin.x(), origin.y()]), axis=1
        )
        targets_px = np.array([screen_len for _d, _c, screen_len, _k, _t in specs])
        safe_actual = np.where(actual_px > 1e-6, actual_px, targets_px)
        ratios = np.clip(
            targets_px / safe_actual, 1.0 / MAX_CORRECTION_RATIO, MAX_CORRECTION_RATIO
        )
        arm_lengths = [a * float(r) for a, r in zip(arm_lengths, ratios)]
        tip_points = np.array(
            [
                self._gizmo_origin + direction * arm
                for (direction, _color, _screen_len, _kind, _tooltip), arm in zip(
                    specs, arm_lengths
                )
            ]
        )
        projected, _depth2, _scale2 = self._camera.project(
            tip_points, self.width(), self.height()
        )

        # Drag sensitivity, computed separately from the drawn arm above and
        # for a different reason than the correction pass: that pass is about
        # making the handle *look* a sensible length, bounded so it cannot
        # blow up. This is about how many Angstroms one pixel of mouse motion
        # should move the atom, which is not the same question — a bounded
        # but still-large arm length divided by a foreshortened (also small)
        # screen length can still yield a wildly oversensitive drag even
        # after the drawing itself looks reasonable. The true relationship is
        # the local derivative of the projection at the origin, found here
        # with a tiny probe step rather than the full drawn arm. For a handle
        # pointing straight along the view axis this derivative is genuinely
        # ~0 — correctly: a 2D mouse motion cannot express motion straight
        # into or out of the screen, so that handle should barely respond,
        # not respond unpredictably.
        #
        # The probe step is sized in *screen* terms (via pixels_per_angstrom,
        # already known at the origin), not as a fixed fraction of the
        # structure's extent — a fixed physical step becomes an
        # imperceptible sub-pixel probe once the view is zoomed out, which
        # made every handle read as "degenerate" and stop responding at all
        # the moment the camera zoomed out, not just the genuinely end-on
        # ones. Sizing it to a target pixel footprint keeps the probe
        # meaningful at any zoom; it is still capped as a fraction of the
        # structure's own extent so an extreme zoom-out cannot inflate it
        # into a step large enough for the local-derivative approximation
        # itself to stop being valid.
        extent = max(float(self._camera.extent), 1e-6)
        epsilon = min(PROBE_TARGET_PX / pixels_per_angstrom, extent * 0.05)
        epsilon = max(epsilon, extent * 1e-6)
        probe_points = np.array(
            [self._gizmo_origin + spec[0] * epsilon for spec in specs]
        )
        probe_screen, _depth3, _scale3 = self._camera.project(
            probe_points, self.width(), self.height()
        )
        origin_xy = np.array([origin.x(), origin.y()])
        screen_steps = probe_screen[:, :2] - origin_xy
        screen_step_px = np.linalg.norm(screen_steps, axis=1)
        # Below this, the on-screen direction of a whole-pixel drag is not
        # reliably distinguishable from projection noise at this precision —
        # treated the same as "cannot express this direction at all."
        MIN_SCREEN_STEP_PX = 1e-3
        screen_directions = np.zeros_like(screen_steps)
        sensitivities = np.zeros(len(specs))
        responsive = screen_step_px > MIN_SCREEN_STEP_PX
        screen_directions[responsive] = (
            screen_steps[responsive] / screen_step_px[responsive, None]
        )
        sensitivities[responsive] = epsilon / screen_step_px[responsive]
        # A handle that is merely *quite* oblique (not degenerate enough to
        # trip MIN_SCREEN_STEP_PX above) still has a small, correct, but very
        # large derivative — foreshortening is a continuous effect, not a
        # cliff. Physically honest is not the same as usable: a stress sweep
        # across many viewing angles turned up 50px drags moving an atom
        # 20-60x the size of the whole structure at these merely-oblique
        # angles, which is exactly what "the gizmo breaks" looks like from
        # the user's side, cap or no cap on the visual arm length. Capping
        # sensitivity itself at a bounded multiple of the well-behaved
        # baseline keeps every handle draggable — a very oblique one just
        # tops out at "quite fast" rather than "unusable."
        baseline_sensitivity = 1.0 / pixels_per_angstrom
        np.minimum(
            sensitivities,
            MAX_SENSITIVITY_MULTIPLE * baseline_sensitivity,
            out=sensitivities,
        )

        handles = [
            _Handle(
                tip=QPointF(float(projected[i][0]), float(projected[i][1])),
                color=specs[i][1],
                kind=specs[i][3],
                tooltip=specs[i][4],
                direction=specs[i][0],
                arm_length=arm_lengths[i],
                screen_direction=screen_directions[i],
                sensitivity=float(sensitivities[i]),
            )
            for i in range(len(specs))
        ]
        return origin, handles

    def _handle_at(self, position: QPointF) -> int | None:
        """Index into the current handle list the cursor is over, if any."""
        geometry = self._gizmo_handles()
        if geometry is None:
            return None
        origin, handles = geometry
        best: tuple[float, int] | None = None
        for index, handle in enumerate(handles):
            distance = _point_segment_distance(position, origin, handle.tip)
            if distance <= GRAB_TOLERANCE and (best is None or distance < best[0]):
                best = (distance, index)
        return None if best is None else best[1]

    # -- painting ----------------------------------------------------------

    def _draw_atom(self, painter, index, screen, radii) -> None:
        super()._draw_atom(painter, index, screen, radii)
        # The base class rings only its single `_selected`; ring the rest of a
        # multi-selection to match. Selected atoms keep their element colour
        # and gain a ring, rather than being recoloured — the colour is
        # information about the species, not about the selection.
        if self._bonds is None:
            return
        site = self._bonds.sites[index]
        if site.is_ghost or site.index not in self._selection:
            return
        if site.index == self._selected:
            return  # already ringed by the base class
        radius = max(1.5, float(radii[index]))
        centre = QPointF(float(screen[index][0]), float(screen[index][1]))
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(PALETTE.accent), 2.0))
        painter.drawEllipse(centre, radius + 3.5, radius + 3.5)

    def _extra_primitives(self) -> list[tuple[float, str, tuple]]:
        items = list(super()._extra_primitives())
        if self._gizmo_handles() is not None:
            # A very large depth so the sort puts it last: the gizmo is a
            # control surface and must never be occluded by the atoms it acts
            # on, or it becomes ungrabbable exactly when it is needed.
            items.append((1e9, "gizmo", ()))
        return items

    def _draw_extra(self, painter: QPainter, kind: str, payload: tuple) -> None:
        if kind != "gizmo":
            super()._draw_extra(painter, kind, payload)
            return
        geometry = self._gizmo_handles()
        if geometry is None:
            return
        origin, handles = geometry

        for index, handle in enumerate(handles):
            active = index == self._drag_axis
            hovered = index == self._hovered_handle and not active
            colour = QColor(handle.color)
            if handle.kind in ("neighbor", "pair"):
                width = 3.2 if active else (2.4 if hovered else 1.6)
                pen = QPen(colour, width)
                pen.setCapStyle(Qt.RoundCap)
                pen.setDashPattern([4.0, 3.0])
                painter.setPen(pen)
                painter.drawLine(origin, handle.tip)
                _draw_diamond_head(painter, origin, handle.tip, colour, active or hovered)
            else:
                width = 3.8 if active else (3.0 if hovered else 2.2)
                pen = QPen(colour, width)
                pen.setCapStyle(Qt.RoundCap)
                painter.setPen(pen)
                painter.drawLine(origin, handle.tip)
                _draw_arrow_head(painter, origin, handle.tip, colour, active or hovered)

        painter.setPen(QPen(QColor(255, 255, 255, 220), 1.2))
        painter.setBrush(QBrush(QColor(PALETTE.accent)))
        painter.drawEllipse(origin, 4.2, 4.2)

    # -- interaction -------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.button() == Qt.LeftButton and self._mode == self.MODE_MOVE:
            handle = self._handle_at(event.position())
            if handle is not None:
                geometry = self._gizmo_handles()
                assert geometry is not None
                # A pair handle only translates the atom it is anchored to
                # (the first-selected one) — the second atom is the fixed
                # reference the direction was measured toward, not something
                # this drag should also move. Every other handle keeps the
                # existing "move the whole current selection" behaviour.
                if geometry[1][handle].kind == "pair" and self._selection_order:
                    self._drag_target_rows = {self._selection_order[0]}
                else:
                    self._drag_target_rows = None
                self._drag_axis = handle
                self._drag_start = QPointF(event.position())
                self._drag_delta = np.zeros(3)
                self._drag_geometry = geometry
                self._last_mouse = QPointF(event.position())
                return
            # No handle under the cursor: a click in Move mode with nothing
            # to grab falls through to ordinary picking. Without this, hitting
            # G with no prior selection and then clicking an atom did nothing
            # at all — the gizmo has nowhere to anchor until something is
            # selected, so the click needs to select before it can be grabbed.
            self._select_click(event)
            return
        if event.button() == Qt.LeftButton and self._mode == self.MODE_SELECT:
            self._select_click(event)
            return
        super().mousePressEvent(event)

    def _select_click(self, event) -> None:
        """Click-to-select, shared by Select mode and by a Move-mode click
        that missed the gizmo. Ctrl/Cmd and Shift both extend the selection
        and both toggle a clicked atom back out if it was already in it —
        clicking the same atom twice under either modifier is how you
        deselect just that one atom without losing the rest."""
        self._last_mouse = QPointF(event.position())
        hit = self._pick(event.position())
        additive = bool(
            event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier | Qt.ShiftModifier)
        )
        if hit is None:
            if not additive:
                self._selection.clear()
                self._selection_order.clear()
        else:
            assert self._bonds is not None
            # Picking can land on a ghost (a periodic image drawn for
            # context). Its `.index` is the real atom it repeats, so
            # resolving through it selects the actual atom rather than
            # creating a phantom selection of something that does not
            # exist in the structure.
            index = self._bonds.sites[hit].index
            if not additive:
                self._selection = {index}
                self._selection_order = [index]
            elif index in self._selection:
                self._selection.discard(index)
                if index in self._selection_order:
                    self._selection_order.remove(index)
            else:
                self._selection.add(index)
                self._selection_order.append(index)
        self._selected = next(iter(self._selection), None) if self._selection else None
        self._refresh_gizmo()
        self.update()
        self.selectionChanged.emit(set(self._selection))

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._drag_axis is not None and self._drag_start is not None:
            self._update_drag(event.position())
            return
        if self._mode == self.MODE_MOVE and not event.buttons():
            hovered = self._handle_at(event.position())
            if hovered != self._hovered_handle:
                self._hovered_handle = hovered
                geometry = self._gizmo_handles()
                if hovered is not None and geometry is not None:
                    self.setToolTip(geometry[1][hovered].tooltip)
                else:
                    self.setToolTip("")
                self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._drag_axis is not None:
            delta = self._drag_delta.copy()
            self._drag_axis = None
            self._drag_start = None
            self._drag_delta = np.zeros(3)
            self._drag_geometry = None
            self.update()
            # One completed drag is one undo entry, so the signal that records
            # history fires here rather than on every mouse-move.
            if np.linalg.norm(delta) > 0:
                self.dragFinished.emit(delta)
            return
        super().mouseReleaseEvent(event)

    def _update_drag(self, position: QPointF) -> None:
        """Resolve pointer motion into a displacement along the grabbed axis
        or neighbour-bond direction."""
        assert self._drag_axis is not None and self._drag_start is not None
        # The geometry frozen at mousePress, *not* the live one. The live
        # gizmo is anchored to the atoms it is moving, so it travels with
        # them mid-drag — measuring pointer motion against a reference frame
        # that is itself moving feeds the displacement back into its own
        # input and the atom shoots off across the cell on the first few
        # pixels of movement. The frame the drag is measured in has to stay
        # fixed for the whole gesture.
        geometry = self._drag_geometry
        if geometry is None or self._drag_axis >= len(geometry[1]):
            return
        _origin, handles = geometry
        handle = handles[self._drag_axis]

        # screen_direction/sensitivity come from a small-epsilon local
        # derivative of the projection (computed once, in _gizmo_handles, and
        # frozen here along with the rest of the geometry) — not from the
        # drawn arm's length, which is chosen to look right on screen and is
        # a different question. A handle pointing straight at the camera
        # correctly has screen_direction ~ 0 here: a 2D mouse motion cannot
        # express that direction of travel, so the drag should do very
        # little, not something unpredictable.
        if not np.any(handle.screen_direction):
            return

        motion = np.array(
            [position.x() - self._drag_start.x(), position.y() - self._drag_start.y()]
        )
        pixels_along_handle = float(np.dot(motion, handle.screen_direction))
        distance = pixels_along_handle * handle.sensitivity

        if self._snap:
            distance = round(distance / self._snap) * self._snap

        self._drag_delta = handle.direction * distance
        self.dragMoved.emit(self._drag_delta.copy())


def _draw_arrow_head(
    painter: QPainter, origin: QPointF, tip: QPointF, colour: QColor, active: bool
) -> None:
    direction = np.array([tip.x() - origin.x(), tip.y() - origin.y()])
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return
    unit = direction / length
    normal = np.array([-unit[1], unit[0]])
    size = 11.0 if active else 8.5
    base = np.array([tip.x(), tip.y()]) - unit * size
    left = base + normal * size * 0.45
    right = base - normal * size * 0.45
    head = QPolygonF(
        [tip, QPointF(float(left[0]), float(left[1])), QPointF(float(right[0]), float(right[1]))]
    )
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(colour))
    painter.drawPolygon(head)


def _draw_diamond_head(
    painter: QPainter, origin: QPointF, tip: QPointF, colour: QColor, active: bool
) -> None:
    """The neighbour-handle tip: a small diamond, deliberately not the
    triangular arrowhead used for the fixed axes, so the two kinds of handle
    are distinguishable at a glance rather than only by colour."""
    direction = np.array([tip.x() - origin.x(), tip.y() - origin.y()])
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return
    unit = direction / length
    normal = np.array([-unit[1], unit[0]])
    size = 7.5 if active else 6.0
    centre = np.array([tip.x(), tip.y()])
    points = [
        centre + unit * size,
        centre + normal * size * 0.6,
        centre - unit * size * 0.4,
        centre - normal * size * 0.6,
    ]
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(colour))
    painter.drawPolygon(QPolygonF([QPointF(float(p[0]), float(p[1])) for p in points]))


def _point_segment_distance(point: QPointF, start: QPointF, end: QPointF) -> float:
    p = np.array([point.x(), point.y()])
    a = np.array([start.x(), start.y()])
    b = np.array([end.x(), end.y()])
    ab = b - a
    length_squared = float(np.dot(ab, ab))
    if length_squared < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / length_squared, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))
