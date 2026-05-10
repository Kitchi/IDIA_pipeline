"""Tests for apply_to_target.main — mocks CASA applycal."""

from unittest.mock import patch

import pytest

from processMeerKAT.crosscal_scripts import apply_to_target as apt


def _taskvals(target_vis='target.mms', combined_kcorr='cal/foo.kcal',
              combined_bpass='cal/foo.bcal', combined_flux='cal/foo.fluxscale'):
    return {
        'data': {'vis': "'foo.ms'"},
        'fields': {
            'targetfields': "'NGC1234'",
            'fluxfield': "'0'",
            'bpassfield': "'0'",
            'phasecalfield': "'1'",
            'extrafields': "''",
        },
        'state': {
            'target_vis': "'{0}'".format(target_vis) if target_vis else '',
            'combined_kcorr': "'{0}'".format(combined_kcorr) if combined_kcorr else '',
            'combined_bpass': "'{0}'".format(combined_bpass) if combined_bpass else '',
            'combined_flux': "'{0}'".format(combined_flux) if combined_flux else '',
        },
    }


def test_apply_to_target_calls_applycal_with_combined_tables(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_NAME', 'test')
    monkeypatch.setenv('SLURM_JOB_ID', '0')
    (tmp_path / 'logs').mkdir()
    monkeypatch.chdir(tmp_path)

    captured = {}

    def fake_applycal(**kwargs):
        captured.update(kwargs)

    with patch.object(apt, 'applycal', side_effect=fake_applycal), \
         patch.object(apt, 'casalog'):
        apt.main({'config': 'cfg.txt'}, _taskvals())

    assert captured['vis'] == 'target.mms'
    assert captured['field'] == 'NGC1234'
    assert captured['gaintable'] == ['cal/foo.kcal', 'cal/foo.bcal', 'cal/foo.fluxscale']
    assert captured['gainfield'] == ['1', '0', '0']  # kcorr=phasecal, bpass=0, flux=0
    assert captured['parang'] is False
    assert captured['calwt'] is False
    assert captured['interp'] == 'linear,linearflag'


def test_apply_to_target_raises_when_target_vis_missing(monkeypatch, tmp_path):
    monkeypatch.setenv('SLURM_JOB_NAME', 'test')
    monkeypatch.setenv('SLURM_JOB_ID', '0')
    (tmp_path / 'logs').mkdir()
    monkeypatch.chdir(tmp_path)

    with patch.object(apt, 'applycal'), patch.object(apt, 'casalog'):
        with pytest.raises(RuntimeError, match='target_vis'):
            apt.main({'config': 'cfg.txt'}, _taskvals(target_vis=''))


def test_apply_to_target_raises_when_no_caltables(monkeypatch, tmp_path):
    monkeypatch.setenv('SLURM_JOB_NAME', 'test')
    monkeypatch.setenv('SLURM_JOB_ID', '0')
    (tmp_path / 'logs').mkdir()
    monkeypatch.chdir(tmp_path)

    with patch.object(apt, 'applycal'), patch.object(apt, 'casalog'):
        with pytest.raises(RuntimeError, match='no combined caltables'):
            apt.main({'config': 'cfg.txt'},
                     _taskvals(combined_kcorr='', combined_bpass='', combined_flux=''))


def test_apply_to_target_skips_missing_caltable_types(monkeypatch, tmp_path):
    """If only bpass+flux exist (no kcorr), applycal is still invoked with what's available."""
    monkeypatch.setenv('SLURM_JOB_NAME', 'test')
    monkeypatch.setenv('SLURM_JOB_ID', '0')
    (tmp_path / 'logs').mkdir()
    monkeypatch.chdir(tmp_path)

    captured = {}
    with patch.object(apt, 'applycal', side_effect=lambda **kw: captured.update(kw)), \
         patch.object(apt, 'casalog'):
        apt.main({'config': 'cfg.txt'}, _taskvals(combined_kcorr=''))

    assert captured['gaintable'] == ['cal/foo.bcal', 'cal/foo.fluxscale']
    assert captured['gainfield'] == ['0', '0']
