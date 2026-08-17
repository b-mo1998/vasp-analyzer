"""Element reference data: display colours and covalent radii.

Colours follow the Jmol/CPK convention that chemists expect. Covalent radii are
in ångström and are used both for drawing atoms and for inferring bonds.

Embedding this table keeps the structure viewer independent of pymatgen at draw
time, which matters because drawing happens on every frame and importing
pymatgen costs seconds.
"""

from __future__ import annotations

# symbol: (covalent radius Å, Jmol colour)
ELEMENTS: dict[str, tuple[float, str]] = {
    "H": (0.31, "#ffffff"), "He": (0.28, "#d9ffff"),
    "Li": (1.28, "#cc80ff"), "Be": (0.96, "#c2ff00"), "B": (0.84, "#ffb5b5"),
    "C": (0.76, "#909090"), "N": (0.71, "#3050f8"), "O": (0.66, "#ff0d0d"),
    "F": (0.57, "#90e050"), "Ne": (0.58, "#b3e3f5"),
    "Na": (1.66, "#ab5cf2"), "Mg": (1.41, "#8aff00"), "Al": (1.21, "#bfa6a6"),
    "Si": (1.11, "#f0c8a0"), "P": (1.07, "#ff8000"), "S": (1.05, "#ffff30"),
    "Cl": (1.02, "#1ff01f"), "Ar": (1.06, "#80d1e3"),
    "K": (2.03, "#8f40d4"), "Ca": (1.76, "#3dff00"), "Sc": (1.70, "#e6e6e6"),
    "Ti": (1.60, "#bfc2c7"), "V": (1.53, "#a6a6ab"), "Cr": (1.39, "#8a99c7"),
    "Mn": (1.39, "#9c7ac7"), "Fe": (1.32, "#e06633"), "Co": (1.26, "#f090a0"),
    "Ni": (1.24, "#50d050"), "Cu": (1.32, "#c88033"), "Zn": (1.22, "#7d80b0"),
    "Ga": (1.22, "#c28f8f"), "Ge": (1.20, "#668f8f"), "As": (1.19, "#bd80e3"),
    "Se": (1.20, "#ffa100"), "Br": (1.20, "#a62929"), "Kr": (1.16, "#5cb8d1"),
    "Rb": (2.20, "#702eb0"), "Sr": (1.95, "#00ff00"), "Y": (1.90, "#94ffff"),
    "Zr": (1.75, "#94e0e0"), "Nb": (1.64, "#73c2c9"), "Mo": (1.54, "#54b5b5"),
    "Tc": (1.47, "#3b9e9e"), "Ru": (1.46, "#248f8f"), "Rh": (1.42, "#0a7d8c"),
    "Pd": (1.39, "#006985"), "Ag": (1.45, "#c0c0c0"), "Cd": (1.44, "#ffd98f"),
    "In": (1.42, "#a67573"), "Sn": (1.39, "#668080"), "Sb": (1.39, "#9e63b5"),
    "Te": (1.38, "#d47a00"), "I": (1.39, "#940094"), "Xe": (1.40, "#429eb0"),
    "Cs": (2.44, "#57178f"), "Ba": (2.15, "#00c900"), "La": (2.07, "#70d4ff"),
    "Ce": (2.04, "#ffffc7"), "Pr": (2.03, "#d9ffc7"), "Nd": (2.01, "#c7ffc7"),
    "Pm": (1.99, "#a3ffc7"), "Sm": (1.98, "#8fffc7"), "Eu": (1.98, "#61ffc7"),
    "Gd": (1.96, "#45ffc7"), "Tb": (1.94, "#30ffc7"), "Dy": (1.92, "#1fffc7"),
    "Ho": (1.92, "#00ff9c"), "Er": (1.89, "#00e675"), "Tm": (1.90, "#00d452"),
    "Yb": (1.87, "#00bf38"), "Lu": (1.87, "#00ab24"),
    "Hf": (1.75, "#4dc2ff"), "Ta": (1.70, "#4da6ff"), "W": (1.62, "#2194d6"),
    "Re": (1.51, "#267dab"), "Os": (1.44, "#266696"), "Ir": (1.41, "#175487"),
    "Pt": (1.36, "#d0d0e0"), "Au": (1.36, "#ffd123"), "Hg": (1.32, "#b8b8d0"),
    "Tl": (1.45, "#a6544d"), "Pb": (1.46, "#575961"), "Bi": (1.48, "#9e4fb5"),
    "Po": (1.40, "#ab5c00"), "At": (1.50, "#754f45"), "Rn": (1.50, "#428296"),
    "Fr": (2.60, "#420066"), "Ra": (2.21, "#007d00"), "Ac": (2.15, "#70abfa"),
    "Th": (2.06, "#00baff"), "Pa": (2.00, "#00a1ff"), "U": (1.96, "#008fff"),
    "Np": (1.90, "#0080ff"), "Pu": (1.87, "#006bff"), "Am": (1.80, "#545cf2"),
    "Cm": (1.69, "#785ce3"),
}

DEFAULT_RADIUS = 1.5
DEFAULT_COLOR = "#dd77ff"

# Non-metals and metalloids. Used to decide whether a short metal-metal
# contact in a compound should be drawn as a bond — in an oxide or a halide it
# should not, whereas in an elemental metal or an alloy it should.
NON_METALS = {
    "H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar",
    "Ge", "As", "Se", "Br", "Kr", "Sb", "Te", "I", "Xe", "At", "Rn",
}


def is_metal(symbol: str) -> bool:
    return normalize(symbol) not in NON_METALS


def covalent_radius(symbol: str) -> float:
    return ELEMENTS.get(normalize(symbol), (DEFAULT_RADIUS, DEFAULT_COLOR))[0]


def color(symbol: str) -> str:
    return ELEMENTS.get(normalize(symbol), (DEFAULT_RADIUS, DEFAULT_COLOR))[1]


def is_known(symbol: str) -> bool:
    return normalize(symbol) in ELEMENTS


def normalize(symbol: str) -> str:
    """Normalise labels such as ``Si1``, ``O_s`` or ``FE`` to ``Si``/``O``/``Fe``.

    Only the leading run of letters is considered: ``O_s`` is an oxygen with a
    site suffix, not osmium, and taking letters from across a separator would
    silently pick the wrong element.
    """
    letters = ""
    for char in symbol.strip():
        if char.isalpha():
            letters += char
        else:
            break
    if not letters:
        return symbol
    if len(letters) >= 2:
        two = letters[0].upper() + letters[1].lower()
        if two in ELEMENTS:
            return two
    return letters[0].upper()
