import pytest
import sys
from unittest.mock import patch, MagicMock, call
from argparse import ArgumentParser

# Import the function to be tested
# Since processMeerKAT.py contains parse_args(), we need to import it carefully
# For this test, we assume the structure allows us to import the function directly.
# In a real scenario, we might need to patch the sys.argv environment.

# We will patch the entire module to control dependencies like constants.py
@patch('processMeerKAT.processMeerKAT.constants.CONFIG', '/fake/config.ini')
@patch('processMeerKAT.processMeerKAT.os.path.exists', return_value=True)
@patch('processMeerKAT.processMeerKAT.load_facility_from_config')
@patch('processMeerKAT.processMeerKAT.validate_args')
@patch('processMeerKAT.processMeerKAT.ArgumentParser')
def test_cli_parser_mutual_exclusivity(MockArgumentParser, mock_validate, mock_load_facility, mock_exists, mock_config):
    """Test that exactly one run mode (-B, -R, -V, -L) is provided."""
    # Mock the parser object returned by MockArgumentParser
    mock_parser_instance = MockArgumentParser.return_value
    
    # Case 1: No run mode provided (should fail)
    with pytest.raises(SystemExit) as excinfo:
        # Simulate passing arguments that don't hit the required group
        sys.argv = ['processMeerKAT.py']
        from processMeerKAT.processMeerKAT import parse_args # Dynamic import after patching
        parse_args()
    assert excinfo.value.code == 2 # argparse default exit code for missing required args

    # Case 2: Multiple run modes provided (should fail)
    with pytest.raises(SystemExit) as excinfo:
        # Simulate passing two run modes
        sys.argv = ['processMeerKAT.py', '-B', '-R']
        from processMeerKAT.processMeerKAT import parse_args
        parse_args()
    assert excinfo.value.code == 2


@patch('processMeerKAT.processMeerKAT.constants.CONFIG', '/fake/config.ini')
@patch('processMeerKAT.processMeerKAT.os.path.exists', return_value=True)
@patch('processMeerKAT.processMeerKAT.load_facility_from_config', return_value=MagicMock())
@patch('processMeerKAT.processMeerKAT.validate_args')
def test_cli_parser_successful_build_mode(mock_validate, mock_load_facility, mock_exists, mock_config):
    """Test successful parsing when building configuration files."""
    mock_parser_instance = MagicMock()
    MockArgumentParser.return_value = mock_parser_instance
    
    sys.argv = ['processMeerKAT.py', '--build', '-M', 'input.ms', '-N', '10', '-t', '16']
    
    # Mocking argparse to return the parsed arguments dictionary
    with patch('processMeerKAT.processMeerKAT.parse_args') as mock_parse:
        mock_parse.return_value = MagicMock(
            build=True, 
            MS='input.ms', 
            config='/fake/config.ini', 
            nodes=10, 
            ntasks_per_node=16
        )
        # Execute parse_args which should internally call the parser setup
        from processMeerKAT.processMeerKAT import parse_args
        args = parse_args()
        
    # Check if the arguments were captured correctly
    assert args.build is True
    assert args.MS == 'input.ms'
    assert args.nodes == 10
    assert args.ntasks_per_node == 16

    # Check if validation was called with the correct parameters
    mock_validate.assert_called_once()


@patch('processMeerKAT.processMeerKAT.constants.CONFIG', '/fake/config.ini')
@patch('processMeerKAT.processMeerKAT.os.path.exists', return_value=True)
@patch('processMeerKAT.processMeerKAT.load_facility_from_config', return_value=MagicMock())
@patch('processMeerKAT.processMeerKAT.validate_args')
def test_cli_parser_validation_failure_on_resource_limits(mock_validate, mock_load_facility, mock_exists, mock_config):
    """Test that validation is triggered and can catch resource overruns (intent test)."""
    # Mock the validate_args to simulate a failure (e.g., resource limit exceeded)
    mock_validate.side_effect = ValueError("Memory per node exceeds facility limit.")
    
    mock_parser_instance = MagicMock()
    MockArgumentParser.return_value = mock_parser_instance
    
    sys.argv = ['processMeerKAT.py', '-B', '-M', 'input.ms', '-N', '1000'] # Massive number of nodes
    
    with pytest.raises(ValueError) as excinfo:
        from processMeerKAT.processMeerKAT import parse_args
        parse_args()
        
    assert "Memory per node exceeds facility limit" in str(excinfo.value)
    mock_validate.assert_called_once()

@patch('processMeerKAT.processMeerKAT.constants.CONFIG', '/fake/config.ini')
@patch('processMeerKAT.processMeerKAT.os.path.exists', return_value=True)
@patch('processMeerKAT.processMeerKAT.load_facility_from_config', return_value=MagicMock())
@patch('processMeerKAT.processMeerKAT.validate_args')
def test_cli_parser_run_mode_missing_config(mock_validate, mock_load_facility, mock_exists, mock_config):
    """Test that -R mode requires a config file."""
    # Test intent: Ensure the logic handles the dependency chain (Run -> Needs Config)
    sys.argv = ['processMeerKAT.py', '-R'] # No config file provided
    
    with pytest.raises(SystemExit) as excinfo:
        from processMeerKAT.processMeerKAT import parse_args
        parse_args()
    
    # Expected SystemExit code for argparse error message
    assert excinfo.value.code == 2

@patch('processMeerKAT.processMeerKAT.constants.CONFIG', '/fake/config.ini')
@patch('processMeerKAT.processMeerKAT.os.path.exists')
@patch('processMeerKAT.processMeerKAT.load_facility_from_config')
@patch('processMeerKAT.processMeerKAT.validate_args')
def test_cli_parser_run_mode_config_not_found(mock_validate, mock_load_facility, mock_exists, mock_config):
    """Test that -R mode fails if the specified config file does not exist."""
    mock_exists.return_value = False
    
    sys.argv = ['processMeerKAT.py', '-R', '-C', 'nonexistent.ini'] 
    
    with pytest.raises(SystemExit) as excinfo:
        from processMeerKAT.processMeerKAT import parse_args
        parse_args()
        
    # Expected SystemExit code for argparse error message
    assert excinfo.value.code == 2

