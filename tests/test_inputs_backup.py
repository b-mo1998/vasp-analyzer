"""The Inputs page backup never writes through a link planted in the folder."""

from __future__ import annotations

import datetime as real_dt
import os

from vasp_analyzer.ui.pages.inputs import page as inputs_page


class _FrozenClock:
    class datetime:  # noqa: N801 - mirrors the ``datetime`` module attribute
        @staticmethod
        def now():
            return real_dt.datetime(2026, 1, 2, 3, 4, 5)


def test_backup_does_not_follow_a_planted_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr(inputs_page, "dt", _FrozenClock)
    calc = tmp_path / "calc"
    calc.mkdir()
    incar = calc / "INCAR"
    incar.write_bytes(b"ENCUT = 500\n")
    victim = tmp_path / "victim"
    victim.write_text("untouched\n")
    missing = tmp_path / "not-yet-there"
    os.symlink(victim, calc / "INCAR.backup-20260102-030405")
    os.symlink(missing, calc / "INCAR.backup-20260102-030405-1")

    backup = inputs_page._write_backup(incar)

    assert victim.read_text() == "untouched\n"
    assert not missing.exists()
    assert backup.name == "INCAR.backup-20260102-030405-2"
    assert not backup.is_symlink()
    assert backup.read_bytes() == b"ENCUT = 500\n"


def test_backup_keeps_an_existing_backup_of_the_same_second(tmp_path, monkeypatch):
    monkeypatch.setattr(inputs_page, "dt", _FrozenClock)
    incar = tmp_path / "INCAR"
    incar.write_bytes(b"first\n")
    first = inputs_page._write_backup(incar)
    incar.write_bytes(b"second\n")
    second = inputs_page._write_backup(incar)

    assert first != second
    assert first.read_bytes() == b"first\n"
    assert second.read_bytes() == b"second\n"
