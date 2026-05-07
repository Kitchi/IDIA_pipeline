"""Tests for read_ms.auto_detect_spw — uses a mock msmd, no real MS required."""

from unittest.mock import MagicMock

import pytest

from processMeerKAT import read_ms


def _make_msmd(spw_freqs_hz):
    """Return a mock msmd with the supplied per-SPW chanfreq arrays."""
    msmd = MagicMock()
    msmd.nspw.return_value = len(spw_freqs_hz)
    msmd.chanfreqs.side_effect = lambda i: spw_freqs_hz[i]
    return msmd


def test_auto_detect_spw_single():
    msmd = _make_msmd([[880e6, 881e6, 882e6, 1080e6]])
    spw_str, nspw = read_ms.auto_detect_spw(msmd)
    assert nspw == 1
    assert spw_str == '0:880.000~1080.000MHz'


def test_auto_detect_spw_multi():
    msmd = _make_msmd([
        [880e6, 933e6],
        [960e6, 1010e6],
        [1010e6, 1060e6],
        [1060e6, 1110e6],
    ])
    spw_str, nspw = read_ms.auto_detect_spw(msmd)
    assert nspw == 4
    assert spw_str == (
        '0:880.000~933.000MHz,'
        '1:960.000~1010.000MHz,'
        '2:1010.000~1060.000MHz,'
        '3:1060.000~1110.000MHz'
    )


def test_auto_detect_spw_descending_chanfreqs():
    """Some MSes (USB/LSB) have descending chanfreq arrays — we still want low~high."""
    msmd = _make_msmd([[1080e6, 1079e6, 1000e6, 880e6]])
    spw_str, nspw = read_ms.auto_detect_spw(msmd)
    assert nspw == 1
    assert spw_str == '0:880.000~1080.000MHz'


def test_auto_detect_spw_uses_global_indices():
    """SPW IDs in the output string must be the actual MS SPW indices, not '*'."""
    msmd = _make_msmd([[100e6, 200e6], [300e6, 400e6]])
    spw_str, _ = read_ms.auto_detect_spw(msmd)
    parts = spw_str.split(',')
    assert parts[0].startswith('0:')
    assert parts[1].startswith('1:')
    assert '*:' not in spw_str


# ---------------------------------------------------------------------------
# resolve_spw_for_build policy
# ---------------------------------------------------------------------------

def test_resolve_multi_spw_honors_ms_structure_overrides_user_nspw():
    msmd = _make_msmd([[880e6, 933e6], [960e6, 1010e6], [1010e6, 1060e6]])
    spw_str, nspw = read_ms.resolve_spw_for_build(msmd, requested_nspw=11)
    assert nspw == 3  # MS structure wins
    assert spw_str.count(',') == 2


def test_resolve_single_spw_preserves_user_nspw_for_parallelism():
    """A single wide native SPW must keep the user's nspw so spw_split can subdivide it."""
    msmd = _make_msmd([[880e6, 1680e6]])
    spw_str, nspw = read_ms.resolve_spw_for_build(msmd, requested_nspw=11)
    assert nspw == 11  # user's choice preserved
    assert spw_str == '0:880.000~1680.000MHz'  # single bounds emitted; spw_split subdivides at -R time


def test_resolve_single_spw_with_nspw_1():
    msmd = _make_msmd([[880e6, 1080e6]])
    spw_str, nspw = read_ms.resolve_spw_for_build(msmd, requested_nspw=1)
    assert nspw == 1
    assert spw_str == '0:880.000~1080.000MHz'


def test_resolve_single_spw_with_nspw_zero_defaults_to_1():
    msmd = _make_msmd([[880e6, 1080e6]])
    spw_str, nspw = read_ms.resolve_spw_for_build(msmd, requested_nspw=0)
    assert nspw == 1


def test_resolve_multi_spw_matching_nspw_no_warning(caplog):
    """When requested_nspw already matches MS structure, no warning is emitted."""
    msmd = _make_msmd([[880e6, 933e6], [960e6, 1010e6]])
    with caplog.at_level('WARNING', logger='processMeerKAT.processMeerKAT'):
        _, nspw = read_ms.resolve_spw_for_build(msmd, requested_nspw=2)
    assert nspw == 2
    assert not any('overriding' in rec.message for rec in caplog.records)
