"""Tests for caltable-concat discovery & orchestration (mocked CASA tb)."""

import os
from unittest.mock import MagicMock, patch

import pytest

import processMeerKAT.bookkeeping as bookkeeping
from processMeerKAT.crosscal_scripts import concat_caltables as cc


# ---------------------------------------------------------------------------
# bookkeeping.get_all_spw_caldirs
# ---------------------------------------------------------------------------

CONFIG_TMPL = """\
crosscal:
  spw: '{spw}'

data:
  vis: foo.ms
"""


def _make_spw_dir(parent, dirname, spw_string, with_caldir=True):
    spw_dir = parent / dirname
    spw_dir.mkdir()
    (spw_dir / 'default_config.yaml').write_text(CONFIG_TMPL.format(spw=spw_string))
    if with_caldir:
        (spw_dir / 'caltables').mkdir()
    return spw_dir


def test_get_all_spw_caldirs_orders_by_global_id(tmp_path):
    _make_spw_dir(tmp_path, '880~933MHz', '0:880~933MHz')
    _make_spw_dir(tmp_path, '1010~1060MHz', '2:1010~1060MHz')
    _make_spw_dir(tmp_path, '960~1010MHz', '1:960~1010MHz')

    entries = bookkeeping.get_all_spw_caldirs(str(tmp_path))
    assert [e['spw_id'] for e in entries] == [0, 1, 2]
    assert entries[0]['caldir'].endswith('880~933MHz/caltables')


def test_get_all_spw_caldirs_skips_non_spw_dirs(tmp_path):
    _make_spw_dir(tmp_path, '880~933MHz', '0:880~933MHz')
    # A bare directory with no config — should be ignored.
    (tmp_path / 'logs').mkdir()
    # A caltables/ at top level — should be ignored.
    (tmp_path / 'caltables').mkdir()

    entries = bookkeeping.get_all_spw_caldirs(str(tmp_path))
    assert len(entries) == 1
    assert entries[0]['spw_id'] == 0


def test_get_all_spw_caldirs_skips_legacy_wildcard_spw(tmp_path):
    """Legacy '*:LO~HI' SPW strings have no parseable global ID — must be skipped."""
    _make_spw_dir(tmp_path, '880~933MHz', '*:880~933MHz')
    _make_spw_dir(tmp_path, '960~1010MHz', '1:960~1010MHz')

    entries = bookkeeping.get_all_spw_caldirs(str(tmp_path))
    assert [e['spw_id'] for e in entries] == [1]


def test_parse_global_spw_id():
    assert bookkeeping._parse_global_spw_id('0:880~933MHz') == 0
    assert bookkeeping._parse_global_spw_id('11:1450~1500MHz') == 11
    assert bookkeeping._parse_global_spw_id('*:880~933MHz') is None
    assert bookkeeping._parse_global_spw_id("'2:980~1020MHz'") == 2


# ---------------------------------------------------------------------------
# _expected_caltable_paths
# ---------------------------------------------------------------------------

def test_expected_caltable_paths_finds_present(tmp_path):
    caldir = tmp_path / 'caltables'
    caldir.mkdir()
    (caldir / 'foo.bcal').mkdir()
    (caldir / 'foo.kcal').mkdir()
    # gcal absent

    found = cc._expected_caltable_paths(str(caldir), 'foo')
    assert set(found.keys()) == {'bcal', 'kcal'}
    assert found['bcal'].endswith('foo.bcal')


def test_expected_caltable_paths_empty_when_none(tmp_path):
    caldir = tmp_path / 'caltables'
    caldir.mkdir()
    found = cc._expected_caltable_paths(str(caldir), 'foo')
    assert found == {}


# ---------------------------------------------------------------------------
# concat_one_caltable orchestration (mocked tb)
# ---------------------------------------------------------------------------

class _FakeTb:
    """Minimal CASA tb stand-in. Records every operation so we can assert on it."""

    def __init__(self):
        self.opened = []
        self.copied = []
        self.removed_rows = []
        self.added_rows = []
        self.put_cols = []
        self._nrows = 0
        self._cols = ['TIME', 'ANTENNA1', 'SPECTRAL_WINDOW_ID', 'CPARAM']

    def open(self, path, nomodify=True):
        self.opened.append((path, nomodify))
        # Source caltables have rows; freshly-templated dst starts empty after we removerows.
        self._nrows = 1 if 'src' in path else self._nrows

    def copy(self, dst, deep=True, valuecopy=True, returnobject=False):
        self.copied.append(dst)
        # Simulate copying — dst is created with same row count.
        self._nrows = 1

    def close(self):
        pass

    def nrows(self):
        return self._nrows

    def removerows(self, rows):
        self.removed_rows.append(rows)
        self._nrows = 0

    def addrows(self, n):
        self.added_rows.append(n)
        self._nrows += n

    def colnames(self):
        return self._cols

    def getcol(self, col):
        import numpy as np
        return np.array([0])

    def putcol(self, col, vals, startrow=0, nrow=0):
        self.put_cols.append((col, startrow, nrow))


def test_concat_one_caltable_validates_input_lengths(tmp_path):
    with pytest.raises(ValueError, match='equal length'):
        cc.concat_one_caltable(['a', 'b'], [0], str(tmp_path / 'out'))


def test_concat_one_caltable_raises_on_empty(tmp_path):
    with pytest.raises(ValueError, match='Nothing to concatenate'):
        cc.concat_one_caltable([], [], str(tmp_path / 'out'))
