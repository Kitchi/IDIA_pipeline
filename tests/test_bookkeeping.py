import pytest
from unittest.mock import patch, MagicMock, mock_open
import os
from collections import namedtuple
import sys

# Assuming bookkeeping is importable from the pipeline structure
from processMeerKAT.bookkeeping import (
    get_calfiles, get_field_ids, run_script, get_selfcal_params, get_imaging_params,
    FieldIDs,
)

# Fixture for a basic configuration dictionary for testing
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
    
    assert isinstance(result, namedtuple)
    assert result.kcorrfile == '/fake/caldir/test_vis.kcal'
    assert result.bpassfile == '/fake/caldir/test_vis.bcal'
    assert result.gainfile == '/fake/caldir/test_vis.gcal'
    assert result.dpolfile == '/fake/caldir/test_vis.pcal'
    assert result.xpolfile == '/fake/caldir/test_vis.xcal'
    assert result.xdelfile == '/fake/caldir/test_vis.xdel'
    assert result.fluxfile == '/fake/caldir/test_vis.fluxscale'

# --- Tests for get_field_ids ---

def test_get_field_ids_standard(mock_fields_config):
    # Test case where fluxfield != secondaryfield
    result = get_field_ids(mock_fields_config)
    assert isinstance(result, FieldIDs)
    assert result.targetfield == '3C286'
    assert result.fluxfield == '3C286'
    assert result.bpassfield == '3C138'
    assert result.secondaryfield == '3C48'
    assert result.gainfields == '3C286,3C48' # Check combined field string
    assert result.extrafields == '3C138,3C48'

def test_get_field_ids_identical_flux_and_secondary():
    fields = {
        'targetfields': '3C286',
        'extrafields': '3C138',
        'fluxfield': '3C286',
        'bpassfield': '3C138',
        'phasecalfield': '3C286', # Flux and Secondary are the same
    }
    result = get_field_ids(fields)
    # Should only list the field once in gainfields
    assert result.gainfields == '3C286'

# --- Tests for get_selfcal_params (Input Intent) ---

@patch('processMeerKAT.bookkeeping.config_parser.parse_config')
@patch('processMeerKAT.bookkeeping.logger')
def test_get_selfcal_params_success(mock_logger, mock_parse_config):
    # Mock a successful config load for a single loop run
    mock_config_data = {
        'selfcal': {'nloops': 1, 'loop': [10], 'discard_nloops': [0], 'outlier_threshold': [3], 'outlier_radius': [1.0]},
        'data': {'vis': 'test.ms'},
        'crosscal': {'refant': 'cal.refant'},
        'run': {'dopol': True}
    }
    mock_parse_config.return_value = (mock_config_data, None)
    
    _, params = get_selfcal_params()
    
    # Check that parameters were correctly extracted and structured
    assert params['vis'] == 'test.ms'
    assert params['refant'] == 'cal.refant'
    assert params['dopol'] == True
    assert params['gaintype'] == 'T' # Should default if not in selfcal section
    
    # Check that parameters were wrapped in lists (as per implementation logic)
    assert isinstance(params['nloops'], int)
    assert isinstance(params['loop'], list)

@patch('processMeerKAT.bookkeeping.config_parser.parse_config')
@patch('processMeerKAT.bookkeeping.logger')
def test_get_selfcal_params_input_error_list_violation(mock_logger, mock_parse_config):
    # Test intent: Fail if a single argument is provided as a list
    mock_config_data = {
        'selfcal': {'nloops': 1, 'loop': [10, 11], 'discard_nloops': [0], 'outlier_threshold': [3], 'outlier_radius': [1.0]}, # Loop is a list of more than 1
        'data': {'vis': 'test.ms'},
        'crosscal': {'refant': 'cal.refant'},
        'run': {'dopol': True}
    }
    mock_parse_config.return_value = (mock_config_data, None)
    
    with pytest.raises(SystemExit):
        get_selfcal_params()

@patch('processMeerKAT.bookkeeping.config_parser.parse_config')
@patch('processMeerKAT.bookkeeping.logger')
def test_get_selfcal_params_length_mismatch(mock_logger, mock_parse_config):
    # Test intent: Fail if an argument list length doesn't match nloops + 1
    mock_config_data = {
        'selfcal': {'nloops': 2, 'loop': [10], 'discard_nloops': [0], 'outlier_threshold': [3], 'outlier_radius': [1.0]}, # nloops=2, expects length 3
        'data': {'vis': 'test.ms'},
        'crosscal': {'refant': 'cal.refant'},
        'run': {'dopol': True}
    }
    mock_parse_config.return_value = (mock_config_data, None)
    
    with pytest.raises(SystemExit):
        get_selfcal_params()


# --- Tests for get_imaging_params (File System Intent) ---

@patch('processMeerKAT.bookkeeping.os.path.exists')
@patch('processMeerKAT.bookkeeping.open', new_callable=mock_open)
def test_get_imaging_params_mask_file_reused(mock_file_open, mock_exists):
    # Test intent: If the outlierfile exists, check for masks and rename old masks
    mock_exists.return_value = True
    
    # Mock the file content to trigger the renaming logic
    mock_file_open.return_value.read.return_value = "imagename=A\nimagename=B\n"
    
    # Patch os.rename to track calls
    with patch('processMeerKAT.bookkeeping.os.rename') as mock_rename:
        get_imaging_params(config_path='/fake/config.ini')
        
        # Ensure renaming happened for both A and B
        mock_rename.assert_any_call('A.mask', 'A.mask.old')
        mock_rename.assert_any_call('B.mask', 'B.mask.old')


# --- Tests for run_script (Control Flow Intent) ---

@patch('processMeerKAT.bookkeeping.config_parser.parse_config')
@patch('processMeerKAT.bookkeeping.logger')
@patch('processMeerKAT.bookkeeping.sys.exit')
def test_run_script_successful_run(mock_exit, mock_logger, mock_parse_config):
    # Test intent: Normal execution path
    mock_parse_config.return_value = ({'run': {'continue': True}}, {'crosscal': {'spw': '1,2'}})
    mock_func = MagicMock()
    
    run_script(mock_func, logfile='log.mpi')
    
    mock_func.assert_called_once()
    mock_exit.assert_not_called()

@patch('processMeerKAT.bookkeeping.config_parser.parse_config')
@patch('processMeerKAT.bookkeeping.logger')
@patch('processMeerKAT.bookkeeping.sys.exit')
def test_run_script_skipped_due_to_continue_false(mock_exit, mock_logger, mock_parse_config):
    # Test intent: Pipeline handles previous failure state
    mock_parse_config.return_value = ({'run': {'continue': False}}, {'crosscal': {'spw': '1,2'}})
    mock_func = MagicMock()
    
    run_script(mock_func, logfile='log.mpi')
    
    mock_func.assert_not_called() # Function should not be called
    mock_exit.assert_called_once_with(1) # Should exit with failure code
    # Check if log was renamed (implies failure handling)
    with patch('processMeerKAT.bookkeeping.os.rename') as mock_rename:
        mock_rename.assert_called_once_with('log.mpi', 'logs/SLURM_JOB_NAME-SLURM_ARRAY_JOB_ID_SLURM_ARRAY_TASK_ID.mpi')


@patch('processMeerKAT.bookkeeping.config_parser.parse_config')
@patch('processMeerKAT.bookkeeping.logger')
@patch('processMeerKAT.bookkeeping.sys.exit')
def test_run_script_failed_and_set_continue_false(mock_exit, mock_logger, mock_parse_config):
    # Test intent: Pipeline catches exceptions and marks config as failed for subsequent jobs
    mock_parse_config.return_value = ({'run': {'continue': True}}, {'crosscal': {'spw': '1,2'}})
    mock_func = MagicMock()
    
    # Simulate an internal error during job execution
    mock_func.side_effect = RuntimeError("Bad CAL file detected")
    
    run_script(mock_func, logfile='log.mpi')
    
    mock_exit.assert_called_once_with(1) # Should exit with failure code
    
    # Check if config was marked 'continue=False' for the main job
    with patch('processMeerKAT.bookkeeping.config_parser.overwrite_config') as mock_overwrite:
        mock_overwrite.assert_called_once_with(
            '/fake/config.ini',
            {'continue': False},
            'run',
            '# Internal variables for pipeline execution',
        )
    # Check if config was marked 'continue=False' for all SPWs
    with patch('processMeerKAT.bookkeeping.config_parser.overwrite_config') as mock_overwrite_spw:
        self.assertEqual(mock_overwrite_spw.call_count, 2)


# Note: get_selfcal_args requires CASA dependency (casatools, quanta) and complex file setup,
# making it a candidate for an Integration Test, as noted in CLAUDE.md.
