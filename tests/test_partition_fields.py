"""Tests for cal/target field selection helpers in the partition scripts."""

import pytest

from processMeerKAT.crosscal_scripts.partition import cal_field_selection, _strip_quotes
from processMeerKAT.crosscal_scripts.partition_target import target_field_selection


# ---------------------------------------------------------------------------
# cal_field_selection — for partition.py (calibrators only)
# ---------------------------------------------------------------------------

def test_cal_field_selection_distinct_fields():
    fields = {
        'bpassfield': "'0'",
        'fluxfield': "'0'",
        'phasecalfield': "'1'",
        'targetfields': "'2'",
        'extrafields': "''",
    }
    sel = cal_field_selection(fields)
    # bpass and flux are the same field — dedup; target excluded
    assert sel == '0,1'


def test_cal_field_selection_excludes_target():
    fields = {
        'bpassfield': "'J0408-6545'",
        'fluxfield': "'J0408-6545'",
        'phasecalfield': "'J1939-6342'",
        'targetfields': "'NGC1234,NGC5678'",
        'extrafields': "''",
    }
    sel = cal_field_selection(fields)
    assert 'NGC1234' not in sel
    assert 'NGC5678' not in sel
    assert 'J0408-6545' in sel
    assert 'J1939-6342' in sel


def test_cal_field_selection_includes_extras():
    fields = {
        'bpassfield': "'0'",
        'fluxfield': "'0'",
        'phasecalfield': "'1'",
        'targetfields': "'2'",
        'extrafields': "'3,4'",
    }
    sel = cal_field_selection(fields)
    parts = sel.split(',')
    assert '3' in parts
    assert '4' in parts


def test_cal_field_selection_dedups_preserving_order():
    fields = {
        'bpassfield': "'0'",
        'fluxfield': "'0'",
        'phasecalfield': "'0'",
        'targetfields': "'2'",
        'extrafields': "'1,0'",
    }
    sel = cal_field_selection(fields)
    assert sel.split(',') == ['0', '1']


def test_cal_field_selection_handles_unquoted_strings():
    """Some configs strip quotes during parsing; the helper must cope with both."""
    fields = {
        'bpassfield': '0',
        'fluxfield': '0',
        'phasecalfield': '1',
        'targetfields': '2',
        'extrafields': '',
    }
    sel = cal_field_selection(fields)
    assert sel == '0,1'


def test_cal_field_selection_all_calibrators_missing():
    """No cal fields at all → empty string (caller should raise)."""
    fields = {
        'bpassfield': "''",
        'fluxfield': "''",
        'phasecalfield': "''",
        'targetfields': "'2'",
        'extrafields': "''",
    }
    sel = cal_field_selection(fields)
    assert sel == ''


# ---------------------------------------------------------------------------
# target_field_selection — for partition_target.py
# ---------------------------------------------------------------------------

def test_target_field_selection_single():
    assert target_field_selection({'targetfields': "'NGC1234'"}) == 'NGC1234'


def test_target_field_selection_multiple_kept_as_csv():
    assert target_field_selection({'targetfields': "'A,B,C'"}) == 'A,B,C'


def test_target_field_selection_missing_raises():
    with pytest.raises(ValueError, match='No targetfields'):
        target_field_selection({'targetfields': "''"})


def test_target_field_selection_absent_key_raises():
    with pytest.raises(ValueError, match='No targetfields'):
        target_field_selection({})


# ---------------------------------------------------------------------------
# _strip_quotes
# ---------------------------------------------------------------------------

def test_strip_quotes_single():
    assert _strip_quotes("'foo'") == 'foo'


def test_strip_quotes_double():
    assert _strip_quotes('"bar"') == 'bar'


def test_strip_quotes_unquoted():
    assert _strip_quotes('baz') == 'baz'


def test_strip_quotes_passthrough_non_string():
    assert _strip_quotes(42) == 42
    assert _strip_quotes(None) is None
