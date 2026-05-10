import pytest
from unittest.mock import patch, MagicMock, mock_open
import os
import sys

from processMeerKAT.bookkeeping import (
    get_calfiles, get_field_ids, run_script, get_selfcal_params, get_imaging_params,
    Calfiles, FieldIDs,
)


@pytest.fixture
def mock_fields_config():
    return {
        'targetfields': '3C286',
        'extrafields': '3C138,3C48',
        'fluxfield': '3C286',
        'bpassfield': '3C138',
        'phasecalfield': '3C48',
    }


# --- Tests for get_calfiles ---

def test_get_calfiles_structure():
    visname = 'test_vis'
    caldir = '/fake/caldir'
    result = get_calfiles(visname, caldir)

    assert isinstance(result, Calfiles)
    assert result.kcorrfile == '/fake/caldir/test_vis.kcal'
    assert result.bpassfile == '/fake/caldir/test_vis.bcal'
    assert result.gainfile == '/fake/caldir/test_vis.gcal'
    assert result.dpolfile == '/fake/caldir/test_vis.pcal'
    assert result.xpolfile == '/fake/caldir/test_vis.xcal'
    assert result.xdelfile == '/fake/caldir/test_vis.xdel'
    assert result.fluxfile == '/fake/caldir/test_vis.fluxscale'


def test_get_calfiles_strips_extension():
    result = get_calfiles('test_vis.ms', '/cals')
    assert result.kcorrfile == '/cals/test_vis.kcal'


# --- Tests for get_field_ids ---

def test_get_field_ids_standard(mock_fields_config):
    result = get_field_ids(mock_fields_config)
    assert isinstance(result, FieldIDs)
    assert result.targetfield == '3C286'
    assert result.fluxfield == '3C286'
    assert result.bpassfield == '3C138'
    assert result.secondaryfield == '3C48'
    assert result.gainfields == '3C286,3C48'
    assert result.extrafields == '3C138,3C48'


def test_get_field_ids_identical_flux_and_secondary():
    fields = {
        'targetfields': '3C286',
        'extrafields': '3C138',
        'fluxfield': '3C286',
        'bpassfield': '3C138',
        'phasecalfield': '3C286',
    }
    result = get_field_ids(fields)
    assert result.gainfields == '3C286'


def test_get_field_ids_pol_fields_all_secondary():
    fields = {
        'targetfields': 'T',
        'extrafields': '',
        'fluxfield': 'F',
        'bpassfield': 'B',
        'phasecalfield': 'S',
    }
    result = get_field_ids(fields)
    assert result.kcorrfield == 'S'
    assert result.xdelfield == 'S'
    assert result.dpolfield == 'S'
    assert result.xpolfield == 'S'


# --- Tests for get_selfcal_params ---

_SELFCAL_TASKVALS = {
    'selfcal': {
        'nloops': 1,
        'loop': 10,
        'discard_nloops': 0,
        'outlier_threshold': 3,
        'outlier_radius': 1.0,
    },
    'data': {'vis': 'test.ms'},
    'crosscal': {'refant': 'cal.refant'},
    'state': {'dopol': False},
}


@patch('processMeerKAT.bookkeeping.config_parser.parse_config')
@patch('processMeerKAT.bookkeeping.logger')
def test_get_selfcal_params_success(mock_logger, mock_parse_config):
    mock_parse_config.return_value = (_SELFCAL_TASKVALS, None)

    _, params = get_selfcal_params(config_path='/fake/config.ini')

    assert params['vis'] == 'test.ms'
    assert params['refant'] == 'cal.refant'
    assert params['dopol'] is False
    assert isinstance(params['nloops'], int)
    assert params['loop'] == 10


@patch('processMeerKAT.bookkeeping.config_parser.parse_config')
@patch('processMeerKAT.bookkeeping.logger')
def test_get_selfcal_params_input_error_list_violation(mock_logger, mock_parse_config):
    taskvals = {
        'selfcal': {
            'nloops': 1,
            'loop': [10, 11],  # list — must be scalar
            'discard_nloops': 0,
            'outlier_threshold': 3,
            'outlier_radius': 1.0,
        },
        'data': {'vis': 'test.ms'},
        'crosscal': {'refant': 'cal.refant'},
        'state': {'dopol': False},
    }
    mock_parse_config.return_value = (taskvals, None)

    with pytest.raises(SystemExit):
        get_selfcal_params(config_path='/fake/config.ini')


@patch('processMeerKAT.bookkeeping.config_parser.parse_config')
@patch('processMeerKAT.bookkeeping.logger')
def test_get_selfcal_params_length_mismatch(mock_logger, mock_parse_config):
    # nloops=2 → gaincal_args need length 2, others need length 3
    # solint is a gaincal_arg: providing length 1 triggers mismatch
    taskvals = {
        'selfcal': {
            'nloops': 2,
            'loop': 10,
            'discard_nloops': 0,
            'outlier_threshold': 3,
            'outlier_radius': 1.0,
            'solint': ['inf'],  # length 1, but nloops=2 requires length 2
        },
        'data': {'vis': 'test.ms'},
        'crosscal': {'refant': 'cal.refant'},
        'state': {'dopol': False},
    }
    mock_parse_config.return_value = (taskvals, None)

    with pytest.raises(SystemExit):
        get_selfcal_params(config_path='/fake/config.ini')


# --- Tests for get_imaging_params ---

_IMAGING_TASKVALS = {
    'image': {'outlierfile': 'outliers.txt'},
    'data': {'vis': 'test.ms'},
    'crosscal': {'keepmms': True},
}


@patch('processMeerKAT.bookkeeping.config_parser.parse_config')
@patch('processMeerKAT.bookkeeping.os.path.exists')
@patch('processMeerKAT.bookkeeping.open', new_callable=mock_open, read_data='imagename=A\nimagename=B\n')
def test_get_imaging_params_mask_file_reused(mock_file_open, mock_exists, mock_parse_config):
    mock_parse_config.return_value = (_IMAGING_TASKVALS, None)
    mock_exists.return_value = True

    with patch('processMeerKAT.bookkeeping.os.rename') as mock_rename:
        get_imaging_params(config_path='/fake/config.ini')
        mock_rename.assert_any_call('A.mask', 'A.mask.old')
        mock_rename.assert_any_call('B.mask', 'B.mask.old')


# --- Tests for run_script ---

_RUN_TASKVALS_OK = {
    'state': {'continue': True, 'cal_vis': '', 'target_vis': None},
    'crosscal': {'spw': '*:880~1080MHz', 'nspw': 1},
    'data': {'vis': '/path/to/data.ms'},
    'fields': {
        'targetfields': '0',
        'extrafields': '',
        'fluxfield': '1',
        'bpassfield': '1',
        'phasecalfield': '2',
    },
}

_RUN_TASKVALS_STOPPED = {
    'state': {'continue': False, 'cal_vis': '', 'target_vis': None},
    'crosscal': {'spw': '*:880~1080MHz', 'nspw': 1},
    'data': {'vis': '/path/to/data.ms'},
    'fields': {
        'targetfields': '0',
        'extrafields': '',
        'fluxfield': '1',
        'bpassfield': '1',
        'phasecalfield': '2',
    },
}


@patch('processMeerKAT.bookkeeping.config_parser.parse_config')
@patch('processMeerKAT.bookkeeping.logger')
@patch('processMeerKAT.bookkeeping.sys.exit')
def test_run_script_successful_run(mock_exit, mock_logger, mock_parse_config):
    mock_parse_config.return_value = (_RUN_TASKVALS_OK, None)
    mock_func = MagicMock()

    run_script(mock_func, logfile='')

    mock_func.assert_called_once()
    mock_exit.assert_not_called()


@patch('processMeerKAT.bookkeeping.config_parser.parse_config')
@patch('processMeerKAT.bookkeeping.logger')
def test_run_script_skipped_due_to_continue_false(mock_logger, mock_parse_config):
    mock_parse_config.return_value = (_RUN_TASKVALS_STOPPED, None)
    mock_func = MagicMock()

    with pytest.raises(SystemExit) as exc_info:
        run_script(mock_func, logfile='')

    mock_func.assert_not_called()
    assert exc_info.value.code == 1


@patch('processMeerKAT.bookkeeping.config_parser.overwrite_config')
@patch('processMeerKAT.bookkeeping.config_parser.parse_config')
@patch('processMeerKAT.bookkeeping.logger')
@patch('processMeerKAT.bookkeeping.sys.exit')
def test_run_script_failed_and_set_continue_false(mock_exit, mock_logger, mock_parse_config, mock_overwrite):
    mock_parse_config.return_value = (_RUN_TASKVALS_OK, None)
    mock_func = MagicMock(side_effect=RuntimeError("Bad CAL file detected"))

    run_script(mock_func, logfile='')

    mock_exit.assert_called_once_with(1)
    mock_overwrite.assert_called_once()
    # Verify the overwrite targeted the 'state' section with continue=False
    _, kwargs = mock_overwrite.call_args
    assert kwargs['conf_sec'] == 'state'
    assert kwargs['conf_dict'] == {'continue': False}
