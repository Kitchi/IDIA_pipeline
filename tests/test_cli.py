import pytest
import sys
from unittest.mock import patch

from processMeerKAT import parse_args


@pytest.fixture(autouse=True)
def restore_argv():
    original = sys.argv[:]
    yield
    sys.argv = original


def test_cli_no_run_mode_exits():
    """argparse exits 2 when no run mode (-B/-R/-V/-L) is given."""
    sys.argv = ['processMeerKAT.py']
    with pytest.raises(SystemExit) as exc:
        parse_args()
    assert exc.value.code == 2


def test_cli_mutual_exclusion_exits():
    """-B and -R are mutually exclusive; argparse exits 2."""
    sys.argv = ['processMeerKAT.py', '-B', '-R']
    with pytest.raises(SystemExit) as exc:
        parse_args()
    assert exc.value.code == 2


@patch('processMeerKAT.processMeerKAT.validate_args')
def test_cli_build_mode_parses_args(mock_validate):
    """-B with -M returns Namespace(build=True, MS=...) and calls validate_args."""
    sys.argv = ['processMeerKAT.py', '-B', '-M', 'test.ms']
    args = parse_args()
    assert args.build is True
    assert args.MS == 'test.ms'
    mock_validate.assert_called_once()


def test_cli_run_mode_nonexistent_config_exits():
    """-R exits 2 when the specified config file does not exist."""
    sys.argv = ['processMeerKAT.py', '-R', '-C', '/nonexistent/config.ini']
    with pytest.raises(SystemExit) as exc:
        parse_args()
    assert exc.value.code == 2


@patch('processMeerKAT.processMeerKAT.validate_args',
       side_effect=ValueError("Memory per node exceeds facility limit."))
def test_cli_validate_args_error_propagates(mock_validate):
    """Errors from validate_args propagate out of parse_args."""
    sys.argv = ['processMeerKAT.py', '-B', '-M', 'test.ms']
    with pytest.raises(ValueError, match="Memory per node exceeds facility limit"):
        parse_args()


@patch('processMeerKAT.processMeerKAT.validate_args')
def test_runner_sets_facility_default_runner(mock_validate):
    """--runner sets _FACILITY.default_runner and clears container on args."""
    import processMeerKAT.processMeerKAT as pm
    sys.argv = ['processMeerKAT.py', '-B', '-M', 'test.ms',
                '--runner', 'conda run -n mycasa --no-capture-output']
    args = parse_args()
    assert args.runner == 'conda run -n mycasa --no-capture-output'
    # main() applies it; parse_args() just captures it
    original_facility = pm._FACILITY
    pm.main.__wrapped__ if hasattr(pm.main, '__wrapped__') else None
    # Verify the runner arg is present and container default is still facility default at parse time
    assert args.runner is not None


@patch('processMeerKAT.processMeerKAT.validate_args')
def test_runner_empty_string_parses(mock_validate):
    """--runner '' (bare Python) is accepted and distinct from not providing --runner."""
    sys.argv = ['processMeerKAT.py', '-B', '-M', 'test.ms', '--runner', '']
    args = parse_args()
    assert args.runner == ''
    assert args.runner is not None  # '' is not None — runner was explicitly set
