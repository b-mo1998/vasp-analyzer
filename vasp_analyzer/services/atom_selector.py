"""Parsing a free-text atom/element selector.

One small grammar shared by every "project onto these atoms" control in the
app (Charge's density localization, Bands' per-atom orbital projection): a
comma- or space-separated list of

* a 1-based atom ID, matching the numbering the Editor and the atom table
  already show (row *N* in the table is atom ID *N*, POSCAR's own ion order)
* an inclusive ID range, ``12-15``
* an element symbol, expanding to every atom of that species

so a user who has been looking at atom IDs in the Editor can type the same
numbers back here without translating anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_RANGE = re.compile(r"^(\d+)-(\d+)$")


@dataclass(frozen=True)
class AtomSelection:
    """The result of parsing one selector string against one structure."""

    rows: set[int] = field(default_factory=set)     # 0-based structure rows
    invalid_tokens: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.rows


def parse_atom_selector(text: str, species: list[str]) -> AtomSelection:
    """Resolve a selector string to 0-based structure row indices.

    Unrecognised tokens (an ID outside the structure's range, an unknown
    element symbol, malformed text) are collected rather than raising — the
    caller decides how to surface them, and a typo in one token should never
    silently discard everything else in the field.
    """
    n_atoms = len(species)
    known_elements = {symbol.lower(): symbol for symbol in set(species)}

    rows: set[int] = set()
    invalid: list[str] = []

    for raw in re.split(r"[,\s]+", text.strip()):
        token = raw.strip()
        if not token:
            continue

        range_match = _RANGE.match(token)
        if range_match:
            low, high = int(range_match.group(1)), int(range_match.group(2))
            if low < 1 or high < low:
                invalid.append(token)
                continue
            found_any = False
            for atom_id in range(low, high + 1):
                if 1 <= atom_id <= n_atoms:
                    rows.add(atom_id - 1)
                    found_any = True
            if not found_any:
                invalid.append(token)
            continue

        if token.isdigit():
            atom_id = int(token)
            if 1 <= atom_id <= n_atoms:
                rows.add(atom_id - 1)
            else:
                invalid.append(token)
            continue

        symbol = known_elements.get(token.lower())
        if symbol is not None:
            rows.update(i for i, s in enumerate(species) if s == symbol)
        else:
            invalid.append(token)

    return AtomSelection(rows=rows, invalid_tokens=invalid)


__all__ = ["AtomSelection", "parse_atom_selector"]
