"""Undo/redo for the Editor page.

Each entry stores the whole structure before and after the change rather than
a description of how to reverse it. That is a deliberate trade: a structure of
a few thousand atoms is on the order of a hundred kilobytes, while
hand-written inverse operations are where undo systems get subtly wrong —
restoring a deleted atom to the wrong index, losing its constraint flags, or
failing to put back the element ordering a sort disturbed. Snapshots cannot
drift from the forward operation, because they *are* the forward operation's
input.

The stack is depth-limited so a long editing session cannot grow without
bound; the oldest entries are dropped, which is the behaviour every editor
has and the one users expect.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.logging_setup import get_logger
from ..models.editable import EditableStructure

log = get_logger("services.editing")

MAX_HISTORY = 100


@dataclass
class EditCommand:
    """One undoable change, as the structure either side of it.

    ``selection_before``/``selection_after`` travel with the change so undo
    restores what the user had highlighted, not just the atoms — landing back
    in a structure with an unrelated selection reads as a second, unexplained
    change.
    """

    label: str
    before: EditableStructure
    after: EditableStructure
    selection_before: set[int]
    selection_after: set[int]


class CommandStack:
    """Linear undo history."""

    def __init__(self, limit: int = MAX_HISTORY) -> None:
        self._done: list[EditCommand] = []
        self._undone: list[EditCommand] = []
        self._limit = limit

    def push(self, command: EditCommand) -> None:
        """Record a change that has already been applied.

        Recording a new change discards the redo branch — the future the user
        undid away is no longer reachable from where they now are.
        """
        self._done.append(command)
        self._undone.clear()
        if len(self._done) > self._limit:
            dropped = len(self._done) - self._limit
            del self._done[:dropped]
        log.info("Edit: %s", command.label)

    def undo(self) -> EditCommand | None:
        if not self._done:
            return None
        command = self._done.pop()
        self._undone.append(command)
        return command

    def redo(self) -> EditCommand | None:
        if not self._undone:
            return None
        command = self._undone.pop()
        self._done.append(command)
        return command

    def clear(self) -> None:
        self._done.clear()
        self._undone.clear()

    @property
    def can_undo(self) -> bool:
        return bool(self._done)

    @property
    def can_redo(self) -> bool:
        return bool(self._undone)

    @property
    def undo_label(self) -> str:
        return self._done[-1].label if self._done else ""

    @property
    def redo_label(self) -> str:
        return self._undone[-1].label if self._undone else ""
