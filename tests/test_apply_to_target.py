"""Tests for apply_to_target.main — per-SPW local apply, mocks CASA applycal.

apply_to_target now runs inside a per-SPW directory: it applies that SPW's own
caltables (under ``caltables/``, derived from the calibrator MMS name in
[state].cal_vis) to that SPW's target MMS ([state].target_vis). The fluxscale
table is mandatory — a missing one raises rather than silently applying raw
gains.
"""

import os
from unittest.mock import patch

import pytest

from processMeerKAT.crosscal_scripts import apply_to_target as apt
from processMeerKAT.bookkeeping import ScriptContext, get_field_ids, get_calfiles


def _taskvals(target_vis='target.mms', cal_vis='foo.mms'):
    return {
        'data': {'vis': "'{0}'".format(cal_vis)},
        'fields': {
            'targetfields': "'NGC1234'",
            'fluxfield': "'0'",
            'bpassfield': "'0'",
            'phasecalfield': "'1'",
            'extrafields': "''",
        },
        'state': {
            'cal_vis': "'{0}'".format(cal_vis),
            'target_vis': "'{0}'".format(target_vis) if target_vis else '',
            'dopol': False,
        },
    }


def _build_context(config_dict):
    vis = config_dict['data']['vis'].strip("'")
    caldir = 'caltables'
    fields = get_field_ids(config_dict['fields'])
    calfiles = get_calfiles(vis, caldir)
    return ScriptContext(
        input_vis=vis,
        cal_vis=vis,
        target_vis=config_dict['state'].get('target_vis', '').strip("'") or None,
        fields=fields,
        calfiles=calfiles,
        caldir=caldir,
        config=config_dict,
        config_path='cfg.txt',
    )


def _make_caltables(tmp_path, base='foo', exts=('kcal', 'bcal', 'fluxscale')):
    """Create empty caltable directories under caltables/ for the given base."""
    caldir = tmp_path / 'caltables'
    caldir.mkdir(exist_ok=True)
    for ext in exts:
        (caldir / '{0}.{1}'.format(base, ext)).mkdir()


def _common_setup(monkeypatch, tmp_path):
    monkeypatch.setenv('SLURM_JOB_NAME', 'test')
    monkeypatch.setenv('SLURM_JOB_ID', '0')
    (tmp_path / 'logs').mkdir()
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'target.mms').mkdir()  # target must exist


def test_apply_to_target_calls_applycal_with_local_tables(tmp_path, monkeypatch):
    _common_setup(monkeypatch, tmp_path)
    _make_caltables(tmp_path)  # kcal + bcal + fluxscale

    captured = {}
    with patch.object(apt, 'applycal', side_effect=lambda **kw: captured.update(kw)), \
         patch.object(apt, 'casalog'):
        apt.main(_build_context(_taskvals()))

    cal = str(tmp_path / 'caltables')
    assert captured['vis'] == 'target.mms'
    assert captured['field'] == 'NGC1234'
    assert captured['gaintable'] == [
        os.path.join(cal, 'foo.kcal'),
        os.path.join(cal, 'foo.bcal'),
        os.path.join(cal, 'foo.fluxscale'),
    ]
    assert captured['gainfield'] == ['1', '0', '0']  # kcorr=phasecal, bpass=0, flux=0
    assert captured['parang'] is False
    assert captured['calwt'] is False
    assert captured['interp'] == 'linear,linearflag'


def test_apply_to_target_raises_when_target_vis_missing(monkeypatch, tmp_path):
    _common_setup(monkeypatch, tmp_path)
    _make_caltables(tmp_path)

    with patch.object(apt, 'applycal'), patch.object(apt, 'casalog'):
        with pytest.raises(RuntimeError, match='target_vis'):
            apt.main(_build_context(_taskvals(target_vis='')))


def test_apply_to_target_raises_when_fluxscale_missing(monkeypatch, tmp_path):
    """Missing fluxscale must fail loudly — no silent gcal fallback."""
    _common_setup(monkeypatch, tmp_path)
    _make_caltables(tmp_path, exts=('kcal', 'bcal', 'gcal'))  # no fluxscale

    with patch.object(apt, 'applycal'), patch.object(apt, 'casalog'):
        with pytest.raises(RuntimeError, match='fluxscale'):
            apt.main(_build_context(_taskvals()))


def test_apply_to_target_skips_missing_caltable_types(monkeypatch, tmp_path):
    """If only bpass+fluxscale exist (no kcorr), applycal is still invoked."""
    _common_setup(monkeypatch, tmp_path)
    _make_caltables(tmp_path, exts=('bcal', 'fluxscale'))

    captured = {}
    with patch.object(apt, 'applycal', side_effect=lambda **kw: captured.update(kw)), \
         patch.object(apt, 'casalog'):
        apt.main(_build_context(_taskvals()))

    cal = str(tmp_path / 'caltables')
    assert captured['gaintable'] == [
        os.path.join(cal, 'foo.bcal'),
        os.path.join(cal, 'foo.fluxscale'),
    ]
    assert captured['gainfield'] == ['0', '0']
