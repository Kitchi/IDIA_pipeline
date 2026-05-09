"""Tests for config_parser.py — pure Python, no CASA required."""

import os
import pytest

import processMeerKAT.config_parser as config_parser


# ---------------------------------------------------------------------------
# parse_config
# ---------------------------------------------------------------------------

def test_parse_config_returns_dict(minimal_config):
    taskvals, config = config_parser.parse_config(minimal_config)
    assert isinstance(taskvals, dict)
    assert 'data' in taskvals
    assert 'slurm' in taskvals
    assert 'crosscal' in taskvals


def test_parse_config_types(minimal_config):
    taskvals, _ = config_parser.parse_config(minimal_config)
    assert isinstance(taskvals['slurm']['nodes'], int)
    assert isinstance(taskvals['slurm']['mem'], int)
    assert isinstance(taskvals['slurm']['submit'], bool)
    assert isinstance(taskvals['slurm']['modules'], list)
    assert isinstance(taskvals['crosscal']['badants'], list)
    assert isinstance(taskvals['data']['vis'], str)


def test_parse_config_string_values(minimal_config):
    taskvals, _ = config_parser.parse_config(minimal_config)
    assert taskvals['data']['vis'] == 'test.ms'
    assert taskvals['slurm']['partition'] == 'Main'
    assert taskvals['crosscal']['refant'] == 'm059'


def test_parse_config_missing_file():
    """A missing config file returns empty dict."""
    taskvals, _ = config_parser.parse_config('/nonexistent/path/config.toml')
    assert taskvals == {}


def test_parse_config_bad_value(tmp_path):
    """Malformed TOML should raise ValueError."""
    cfg = tmp_path / 'bad.toml'
    cfg.write_text('[data\nvis = "test.ms"\n')
    with pytest.raises(ValueError, match="Cannot parse TOML"):
        config_parser.parse_config(str(cfg))


def test_parse_config_script_entries(minimal_config):
    """Script entries are parsed as lists of dicts with 'script' and 'mpi' keys."""
    taskvals, _ = config_parser.parse_config(minimal_config)
    scripts = taskvals['slurm']['scripts']
    assert isinstance(scripts, list)
    assert isinstance(scripts[0], dict)
    assert 'script' in scripts[0]
    assert 'mpi' in scripts[0]
    assert scripts[0]['script'] == 'validate_input.py'
    assert scripts[0]['mpi'] is False
    assert scripts[1]['mpi'] is True


# ---------------------------------------------------------------------------
# has_key / has_section / get_key
# ---------------------------------------------------------------------------

def test_has_section_true(minimal_config):
    assert config_parser.has_section(minimal_config, 'data') is True
    assert config_parser.has_section(minimal_config, 'crosscal') is True


def test_has_section_false(minimal_config):
    assert config_parser.has_section(minimal_config, 'nonexistent') is False


def test_has_key_true(minimal_config):
    assert config_parser.has_key(minimal_config, 'data', 'vis') is True
    assert config_parser.has_key(minimal_config, 'slurm', 'nodes') is True


def test_has_key_false(minimal_config):
    assert config_parser.has_key(minimal_config, 'data', 'nokey') is False
    assert config_parser.has_key(minimal_config, 'nosection', 'vis') is False


def test_get_key_returns_value(minimal_config):
    assert config_parser.get_key(minimal_config, 'data', 'vis') == 'test.ms'
    assert config_parser.get_key(minimal_config, 'slurm', 'nodes') == 1


def test_get_key_missing_returns_empty(minimal_config):
    assert config_parser.get_key(minimal_config, 'data', 'nokey') == ''


# ---------------------------------------------------------------------------
# overwrite_config
# ---------------------------------------------------------------------------

def test_overwrite_config_new_section(tmp_path):
    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('[data]\nvis = "test.ms"\n')
    config_parser.overwrite_config(str(cfg), conf_dict={'mykey': 42}, conf_sec='newsec')
    taskvals, _ = config_parser.parse_config(str(cfg))
    assert 'newsec' in taskvals
    assert taskvals['newsec']['mykey'] == 42


def test_overwrite_config_existing_section(minimal_config):
    config_parser.overwrite_config(minimal_config, conf_dict={'nodes': 4}, conf_sec='slurm')
    assert config_parser.get_key(minimal_config, 'slurm', 'nodes') == 4


def test_overwrite_config_string_value(tmp_path):
    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('[data]\nvis = "old.ms"\n')
    config_parser.overwrite_config(str(cfg), conf_dict={'vis': 'new.ms'}, conf_sec='data')
    assert config_parser.get_key(str(cfg), 'data', 'vis') == 'new.ms'


def test_overwrite_config_script_list_roundtrip(tmp_path):
    """Script lists survive a write/read roundtrip as lists of dicts."""
    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('[data]\nvis = "test.ms"\n')
    scripts = [{'script': 'foo.py', 'mpi': True}, {'script': 'bar.py', 'mpi': False}]
    config_parser.overwrite_config(str(cfg), conf_dict={'scripts': scripts}, conf_sec='slurm')
    taskvals, _ = config_parser.parse_config(str(cfg))
    assert taskvals['slurm']['scripts'] == scripts


# ---------------------------------------------------------------------------
# remove_section
# ---------------------------------------------------------------------------

def test_remove_section(minimal_config):
    assert config_parser.has_section(minimal_config, 'crosscal') is True
    config_parser.remove_section(minimal_config, 'crosscal')
    assert config_parser.has_section(minimal_config, 'crosscal') is False
    assert config_parser.has_section(minimal_config, 'data') is True  # others intact


# ---------------------------------------------------------------------------
# validate_args
# ---------------------------------------------------------------------------

def test_validate_args_str(minimal_config):
    taskvals, _ = config_parser.parse_config(minimal_config)
    result = config_parser.validate_args(taskvals, 'slurm', 'partition', str)
    assert result == 'Main'


def test_validate_args_str_strips_trailing_slash(tmp_path):
    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('[slurm]\npartition = "Main/"\n')
    taskvals, _ = config_parser.parse_config(str(cfg))
    result = config_parser.validate_args(taskvals, 'slurm', 'partition', str)
    assert result == 'Main'


def test_validate_args_int(minimal_config):
    taskvals, _ = config_parser.parse_config(minimal_config)
    result = config_parser.validate_args(taskvals, 'slurm', 'nodes', int)
    assert result == 1
    assert isinstance(result, int)


def test_validate_args_bool(minimal_config):
    taskvals, _ = config_parser.parse_config(minimal_config)
    result = config_parser.validate_args(taskvals, 'slurm', 'submit', bool)
    assert result is False


def test_validate_args_default_used_when_key_missing(minimal_config):
    taskvals, _ = config_parser.parse_config(minimal_config)
    result = config_parser.validate_args(taskvals, 'slurm', 'nonexistent_key', int, default=99)
    assert result == 99


def test_validate_args_invalid_dtype(minimal_config):
    taskvals, _ = config_parser.parse_config(minimal_config)
    with pytest.raises(NotImplementedError):
        config_parser.validate_args(taskvals, 'slurm', 'nodes', list)


# ---------------------------------------------------------------------------
# typed_get — new API (same semantics, non-mutating, better errors)
# ---------------------------------------------------------------------------

def test_typed_get_str(minimal_config):
    taskvals, _ = config_parser.parse_config(minimal_config)
    assert config_parser.typed_get(taskvals, 'slurm', 'partition', str) == 'Main'


def test_typed_get_int(minimal_config):
    taskvals, _ = config_parser.parse_config(minimal_config)
    result = config_parser.typed_get(taskvals, 'slurm', 'nodes', int)
    assert result == 1 and isinstance(result, int)


def test_typed_get_bool(minimal_config):
    taskvals, _ = config_parser.parse_config(minimal_config)
    assert config_parser.typed_get(taskvals, 'slurm', 'submit', bool) is False


def test_typed_get_float(tmp_path):
    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('[crosscal]\nrobust = -0.5\n')
    taskvals, _ = config_parser.parse_config(str(cfg))
    result = config_parser.typed_get(taskvals, 'crosscal', 'robust', float)
    assert result == -0.5 and isinstance(result, float)


def test_typed_get_default_when_missing(minimal_config):
    taskvals, _ = config_parser.parse_config(minimal_config)
    result = config_parser.typed_get(taskvals, 'slurm', 'nonexistent', int, default=42)
    assert result == 42


def test_typed_get_missing_required_raises(minimal_config):
    taskvals, _ = config_parser.parse_config(minimal_config)
    with pytest.raises(KeyError, match="nonexistent"):
        config_parser.typed_get(taskvals, 'slurm', 'nonexistent', int)


def test_typed_get_does_not_mutate(minimal_config):
    """typed_get must not remove keys from the config dict (old validate_args used pop)."""
    taskvals, _ = config_parser.parse_config(minimal_config)
    config_parser.typed_get(taskvals, 'slurm', 'nodes', int, default=99)
    assert 'nodes' in taskvals['slurm']


def test_typed_get_strips_trailing_slash(tmp_path):
    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('[slurm]\ncontainer = "/some/path/"\n')
    taskvals, _ = config_parser.parse_config(str(cfg))
    assert config_parser.typed_get(taskvals, 'slurm', 'container', str) == '/some/path'


def test_typed_get_bad_int_raises(tmp_path):
    cfg = tmp_path / 'cfg.toml'
    # TOML won't parse "notanint" as int natively; write it as a string
    cfg.write_text('[slurm]\nnodes = "notanint"\n')
    taskvals, _ = config_parser.parse_config(str(cfg))
    with pytest.raises(ValueError, match="cannot be converted to int"):
        config_parser.typed_get(taskvals, 'slurm', 'nodes', int)


def test_typed_get_bad_bool_raises(tmp_path):
    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('[slurm]\nsubmit = "yes"\n')
    taskvals, _ = config_parser.parse_config(str(cfg))
    with pytest.raises(ValueError, match="is not a bool"):
        config_parser.typed_get(taskvals, 'slurm', 'submit', bool)


def test_typed_get_unsupported_dtype(minimal_config):
    taskvals, _ = config_parser.parse_config(minimal_config)
    with pytest.raises(NotImplementedError):
        config_parser.typed_get(taskvals, 'slurm', 'nodes', list)
