"""The shared atom/element selector grammar: IDs, ranges, element symbols."""

from __future__ import annotations

from vasp_analyzer.services.atom_selector import parse_atom_selector

SPECIES = ["C", "C", "C", "Te", "Te", "O"]  # rows 0-5, ids 1-6


def test_bare_ids_resolve_to_zero_based_rows():
    result = parse_atom_selector("1, 4", SPECIES)
    assert result.rows == {0, 3}
    assert not result.invalid_tokens


def test_element_symbol_expands_to_every_matching_atom():
    result = parse_atom_selector("Te", SPECIES)
    assert result.rows == {3, 4}


def test_case_insensitive_element_match():
    result = parse_atom_selector("te, o", SPECIES)
    assert result.rows == {3, 4, 5}


def test_range_expands_inclusive():
    result = parse_atom_selector("1-3", SPECIES)
    assert result.rows == {0, 1, 2}


def test_mixed_ids_ranges_and_elements():
    result = parse_atom_selector("1, 4-5, O", SPECIES)
    assert result.rows == {0, 3, 4, 5}


def test_out_of_range_id_is_invalid_not_silently_dropped():
    result = parse_atom_selector("1, 99", SPECIES)
    assert result.rows == {0}
    assert result.invalid_tokens == ["99"]


def test_unknown_element_symbol_is_invalid():
    result = parse_atom_selector("Xx", SPECIES)
    assert result.is_empty
    assert result.invalid_tokens == ["Xx"]


def test_empty_text_yields_an_empty_selection():
    result = parse_atom_selector("   ", SPECIES)
    assert result.is_empty
    assert not result.invalid_tokens


def test_whitespace_and_comma_separators_both_work():
    a = parse_atom_selector("1 4 6", SPECIES)
    b = parse_atom_selector("1,4,6", SPECIES)
    assert a.rows == b.rows == {0, 3, 5}


def test_malformed_range_is_invalid():
    result = parse_atom_selector("5-2", SPECIES)  # backwards
    assert result.is_empty
    assert result.invalid_tokens == ["5-2"]
