"""The editable structure model behind the Editor page.

:class:`~vasp_analyzer.models.structure.StructureData` describes a structure
that was *read*: it is what parsers produce and what every read-only page
renders. Editing needs two things it deliberately does not have.

The first is **stable atom identity**. A row index is not an identity —
deleting atom 3, sorting by element, or building a supercell all renumber
everything after it. If selection, undo, and the atom table each remembered
"atom 27", they would silently come to mean three different atoms. Every atom
here carries an ``id`` that is assigned once and never reused, so those
systems can refer to the same atom across any structural change.

The second is **selective dynamics as first-class data**. VASP's per-axis
T/F flags are a property of the structure the user is authoring, not an
annotation on top of it, and they must survive every operation and round trip
to disk. They are stored as a plain ``(n_atoms, 3)`` boolean array where True
means *free to move*, which is what a ``T`` means in a POSCAR.

Coordinates are held as fractional, because that is the representation
periodic operations (wrapping, supercells, minimum-image distances) are
natural in, and the one POSCAR's ``Direct`` mode stores. Cartesian is derived
on demand through the full lattice matrix, so non-orthogonal cells are handled
correctly rather than by assuming a diagonal lattice.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np

from .structure import StructureData


@dataclass
class EditableStructure:
    """A structure being edited, with stable per-atom identity.

    All mutating helpers return *new* instances rather than modifying in
    place. Undo is implemented as snapshot restore, so a command can keep the
    previous value simply by holding the object it replaced — nothing can
    mutate it out from under the history.
    """

    lattice: np.ndarray                # (3, 3) row vectors in Å
    species: list[str]                 # per-atom element symbols
    frac_coords: np.ndarray            # (n_atoms, 3)
    selective: np.ndarray              # (n_atoms, 3) bool, True = free to move
    ids: list[int]                     # stable per-atom identity
    comment: str = ""
    source: str = "POSCAR"
    # Set when the source file carried an explicit Selective Dynamics block.
    # Writing the block back out for a structure that never had one would add
    # constraints the user did not ask for, so this is remembered rather than
    # inferred from the flags alone.
    has_selective_dynamics: bool = False
    _next_id: int = field(default=0, repr=False)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_structure(cls, structure: StructureData) -> EditableStructure:
        n = structure.n_atoms
        selective = np.ones((n, 3), dtype=bool)
        had_flags = structure.selective_dynamics is not None
        if had_flags:
            flags = np.asarray(structure.selective_dynamics, dtype=bool)
            # Tolerate a (n,) shaped array by broadcasting it across all axes.
            if flags.ndim == 1:
                flags = np.repeat(flags[:, None], 3, axis=1)
            selective[: len(flags)] = flags[:n]
        return cls(
            lattice=np.array(structure.lattice, dtype=float),
            species=list(structure.species),
            frac_coords=np.array(structure.frac_coords, dtype=float).reshape(-1, 3),
            selective=selective,
            ids=list(range(n)),
            comment=structure.comment,
            source=structure.source,
            has_selective_dynamics=had_flags,
            _next_id=n,
        )

    def to_structure(self) -> StructureData:
        """A plain :class:`StructureData` for the viewer, exporters and the
        symmetry service. Identity is dropped here on purpose — nothing
        downstream of the editor has a use for it."""
        return StructureData(
            lattice=self.lattice.copy(),
            species=list(self.species),
            frac_coords=self.frac_coords.copy(),
            comment=self.comment,
            source=self.source,
            selective_dynamics=(
                self.selective.copy() if self.has_selective_dynamics else None
            ),
        )

    def copy(self) -> EditableStructure:
        return replace(
            self,
            lattice=self.lattice.copy(),
            species=list(self.species),
            frac_coords=self.frac_coords.copy(),
            selective=self.selective.copy(),
            ids=list(self.ids),
        )

    # -- queries -----------------------------------------------------------

    @property
    def n_atoms(self) -> int:
        return len(self.species)

    @property
    def cart_coords(self) -> np.ndarray:
        return self.frac_coords @ self.lattice

    @property
    def composition(self) -> dict[str, int]:
        return dict(Counter(self.species))

    @property
    def elements(self) -> list[str]:
        """Distinct elements in first-appearance order."""
        seen: list[str] = []
        for symbol in self.species:
            if symbol not in seen:
                seen.append(symbol)
        return seen

    @property
    def formula(self) -> str:
        return self.to_structure().formula

    @property
    def lattice_parameters(self) -> tuple[float, float, float]:
        return tuple(float(np.linalg.norm(v)) for v in self.lattice)  # type: ignore[return-value]

    @property
    def lattice_angles(self) -> tuple[float, float, float]:
        a, b, c = self.lattice

        def angle(u, v):
            cos = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
            return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

        return angle(b, c), angle(a, c), angle(a, b)

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.lattice)))

    def index_of(self, atom_id: int) -> int | None:
        try:
            return self.ids.index(atom_id)
        except ValueError:
            return None

    def indices_of(self, atom_ids: set[int]) -> list[int]:
        """Row indices for a set of ids, in current structure order."""
        return [i for i, aid in enumerate(self.ids) if aid in atom_ids]

    def to_cartesian(self, frac: np.ndarray) -> np.ndarray:
        return np.asarray(frac, dtype=float) @ self.lattice

    def to_fractional(self, cart: np.ndarray) -> np.ndarray:
        return np.asarray(cart, dtype=float) @ np.linalg.inv(self.lattice)

    # -- mutations (all return a new instance) -----------------------------

    def with_coords(self, indices: list[int], frac: np.ndarray) -> EditableStructure:
        out = self.copy()
        out.frac_coords[indices] = np.asarray(frac, dtype=float).reshape(len(indices), 3)
        return out

    def translated(self, indices: list[int], delta_frac: np.ndarray) -> EditableStructure:
        out = self.copy()
        out.frac_coords[indices] = out.frac_coords[indices] + np.asarray(
            delta_frac, dtype=float
        )
        return out

    def with_atom_added(
        self, symbol: str, frac: np.ndarray, free: np.ndarray | None = None
    ) -> tuple[EditableStructure, int]:
        out = self.copy()
        atom_id = out._next_id
        out._next_id += 1
        out.species.append(symbol)
        out.frac_coords = np.vstack([out.frac_coords, np.asarray(frac, float).reshape(1, 3)])
        flags = np.ones((1, 3), dtype=bool) if free is None else np.asarray(free, bool).reshape(1, 3)
        out.selective = np.vstack([out.selective, flags])
        out.ids.append(atom_id)
        return out, atom_id

    def with_atoms_removed(self, indices: list[int]) -> EditableStructure:
        keep = [i for i in range(self.n_atoms) if i not in set(indices)]
        out = self.copy()
        out.species = [self.species[i] for i in keep]
        out.frac_coords = self.frac_coords[keep].copy() if keep else np.zeros((0, 3))
        out.selective = self.selective[keep].copy() if keep else np.zeros((0, 3), bool)
        out.ids = [self.ids[i] for i in keep]
        return out

    def with_atoms_duplicated(
        self, indices: list[int], offset_frac: np.ndarray | None = None
    ) -> tuple[EditableStructure, list[int]]:
        out = self.copy()
        new_ids: list[int] = []
        shift = np.zeros(3) if offset_frac is None else np.asarray(offset_frac, float)
        rows = []
        for i in indices:
            atom_id = out._next_id
            out._next_id += 1
            new_ids.append(atom_id)
            out.species.append(self.species[i])
            rows.append(self.frac_coords[i] + shift)
            out.ids.append(atom_id)
        if rows:
            out.frac_coords = np.vstack([out.frac_coords, np.array(rows)])
            out.selective = np.vstack([out.selective, self.selective[indices]])
        return out, new_ids

    def with_selective(self, indices: list[int], free: np.ndarray) -> EditableStructure:
        out = self.copy()
        out.selective[indices] = np.asarray(free, bool)
        # Constraining anything means the file now needs the block.
        if not out.selective.all():
            out.has_selective_dynamics = True
        return out

    def wrapped(self, indices: list[int] | None = None) -> EditableStructure:
        out = self.copy()
        rows = list(range(self.n_atoms)) if indices is None else indices
        out.frac_coords[rows] = out.frac_coords[rows] % 1.0
        return out

    def centred(self) -> EditableStructure:
        """Shift so the atoms' centroid sits at the middle of the cell.

        A rigid translation of every atom by the same vector — it never
        changes an interatomic distance, only where the structure sits inside
        the periodic box.
        """
        out = self.copy()
        if self.n_atoms:
            centroid = out.frac_coords.mean(axis=0)
            out.frac_coords = out.frac_coords + (np.array([0.5, 0.5, 0.5]) - centroid)
        return out

    def sorted_by_element(self) -> EditableStructure:
        """Group atoms by element, preserving first-appearance element order.

        This is the ordering POSCAR requires — its counts line describes runs
        of one element, so interleaved species cannot be represented.
        """
        order: list[int] = []
        for element in self.elements:
            order.extend(i for i, s in enumerate(self.species) if s == element)
        out = self.copy()
        out.species = [self.species[i] for i in order]
        out.frac_coords = self.frac_coords[order].copy()
        out.selective = self.selective[order].copy()
        out.ids = [self.ids[i] for i in order]
        return out

    def is_element_grouped(self) -> bool:
        """True when each element occupies one contiguous run, so the species
        list survives a POSCAR round trip."""
        seen: set[str] = set()
        previous: str | None = None
        for symbol in self.species:
            if symbol != previous:
                if symbol in seen:
                    return False
                seen.add(symbol)
                previous = symbol
        return True

    # -- periodic geometry -------------------------------------------------

    def minimum_image_distance(self, i: int, j: int) -> float:
        """Distance in Å between two atoms under the minimum-image convention."""
        delta = self.frac_coords[i] - self.frac_coords[j]
        delta -= np.round(delta)
        return float(np.linalg.norm(delta @ self.lattice))

    def direction_to(self, i: int, j: int) -> tuple[np.ndarray, float]:
        """Unit direction (cartesian) and distance from atom ``i`` to atom
        ``j``, using whichever periodic image of ``j`` is actually closest.

        Backs the Editor's "move the first atom toward the second" handle:
        a user connecting two specific selected atoms means the shortest path
        between them, which the simple ``round()`` minimum-image trick can
        get wrong in an oblique cell — this checks the same 27 neighbouring
        images :meth:`nearest_neighbors` does, for the same reason.
        """
        if i == j or not (0 <= i < self.n_atoms) or not (0 <= j < self.n_atoms):
            return np.zeros(3), 0.0
        origin = self.frac_coords[i] @ self.lattice
        offsets = np.array(
            [[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)],
            dtype=float,
        )
        shifts = offsets @ self.lattice
        candidates = (self.frac_coords[j] @ self.lattice) + shifts
        deltas = candidates - origin
        distances = np.linalg.norm(deltas, axis=1)
        best = int(np.argmin(distances))
        distance = float(distances[best])
        if distance < 1e-9:
            return np.zeros(3), 0.0
        return deltas[best] / distance, distance

    def nearest_neighbors(self, index: int, k: int = 3) -> list[tuple[int, np.ndarray, float]]:
        """The ``k`` closest neighbouring *bonds* to ``index``, periodic images
        included.

        Returns ``(neighbour_row, unit_direction, distance)`` nearest first,
        where ``unit_direction`` points from ``index`` toward the neighbour in
        cartesian Å — the direction the Editor's "move toward a neighbour"
        gizmo handles translate an atom along.

        Deliberately *not* a plain minimum-image search: that finds only the
        closest periodic copy of each other atom, which collapses to one
        result per row and misses real bonds. Graphene's two-atom cell is the
        clearest case — each carbon has three physical C-C bonds, all to
        periodic copies of the *same* other atom, not to three different
        atoms — so every one of the 27 neighbouring periodic images of every
        other atom is a candidate, and the same row can legitimately appear
        more than once with a different direction.
        """
        n = self.n_atoms
        if n < 2 or not (0 <= index < n):
            return []
        origin = self.frac_coords[index] @ self.lattice
        offsets = np.array(
            [[i, j, kk] for i in (-1, 0, 1) for j in (-1, 0, 1) for kk in (-1, 0, 1)],
            dtype=float,
        )
        shifts = offsets @ self.lattice                       # (27, 3)
        base = self.frac_coords @ self.lattice                # (n, 3)
        candidates = base[:, None, :] + shifts[None, :, :]     # (n, 27, 3)
        deltas = candidates - origin                            # (n, 27, 3)
        distances = np.linalg.norm(deltas, axis=2)              # (n, 27)
        distances[index, :] = np.inf                            # never bond to self

        n_shifts = len(shifts)
        order = np.argsort(distances, axis=None)
        results: list[tuple[int, np.ndarray, float]] = []
        for flat_index in order:
            distance = float(distances.flat[flat_index])
            if not np.isfinite(distance) or distance < 1e-6:
                continue
            row = int(flat_index) // n_shifts
            delta = deltas.reshape(-1, 3)[flat_index]
            results.append((row, delta / distance, distance))
            if len(results) >= k:
                break
        return results

    def close_pairs(self, threshold: float) -> list[tuple[int, int, float]]:
        """Atom pairs closer than ``threshold`` Å, minimum-image aware.

        Used for overlap detection. Reported rather than acted on — silently
        deleting an atom because two sites are close is exactly the kind of
        change that ruins a calculation without the user noticing.
        """
        pairs: list[tuple[int, int, float]] = []
        n = self.n_atoms
        if n < 2:
            return pairs
        delta = self.frac_coords[:, None, :] - self.frac_coords[None, :, :]
        delta -= np.round(delta)
        cart = delta @ self.lattice
        dist = np.linalg.norm(cart, axis=-1)
        iu = np.triu_indices(n, k=1)
        for i, j in zip(*iu):
            d = float(dist[i, j])
            if d < threshold:
                pairs.append((int(i), int(j), d))
        pairs.sort(key=lambda item: item[2])
        return pairs
