"""Tests for flag_round_1 — verifies badfreq_uvrange threads into flagdata.

Imports casatasks at module load (via the crosscal script), so these run only
in a CASA-enabled environment; collection is skipped otherwise.
"""

import pytest

pytest.importorskip("casatasks")

from unittest.mock import patch

from processMeerKAT.crosscal_scripts import flag_round_1 as fr1
from processMeerKAT.bookkeeping import ScriptContext, get_field_ids, get_calfiles


def _ctx(badfreq_uvrange='', badfreqranges=None):
    if badfreqranges is None:
        badfreqranges = ['933~960MHz']
    config = {
        'data': {'vis': 'foo.ms'},
        'fields': {
            'fluxfield': '0', 'bpassfield': '0', 'phasecalfield': '1',
            'targetfields': '2', 'extrafields': '',
        },
        'crosscal': {
            'badfreqranges': badfreqranges,
            'badants': [],
            'badfreq_uvrange': badfreq_uvrange,
        },
    }
    return ScriptContext(
        input_vis='foo.ms', cal_vis='foo.ms', target_vis=None,
        fields=get_field_ids(config['fields']),
        calfiles=get_calfiles('foo.ms', 'caltables'),
        caldir='caltables', config=config, config_path='cfg.toml',
    )


def _badfreq_flag_call(ctx):
    """Run main() with flagdata mocked; return the manual call carrying 'spw'
    (the badfreqranges flag)."""
    calls = []
    with patch.object(fr1, 'flagdata', side_effect=lambda **kw: calls.append(kw)):
        fr1.main(ctx)
    return next(c for c in calls if 'spw' in c)


def test_badfreq_uvrange_passed_to_flagdata():
    call = _badfreq_flag_call(_ctx(badfreq_uvrange='<1klambda'))
    assert call['spw'] == '*:933~960MHz'
    assert call['uvrange'] == '<1klambda'


def test_badfreq_uvrange_defaults_to_all_baselines():
    call = _badfreq_flag_call(_ctx(badfreq_uvrange=''))
    assert call['uvrange'] == ''
