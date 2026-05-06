#Copyright (C) 2022 Inter-University Institute for Data Intensive Astronomy
#See processMeerKAT.py for license details.

"""Config file I/O for the MeerKAT pipeline.

Public API
----------
parse_config(filename)              Parse an INI config → (dict, ConfigParser)
has_section(filename, section)
has_key(filename, section, key)
get_key(filename, section, key)
overwrite_config(filename, ...)
remove_section(filename, section)
parse_spw(filename)                 Parse SPW bounds from [crosscal] section
typed_get(config_dict, section, key, dtype, default=None)
    Extract and type-coerce a value from a parsed config dict.
    Use this instead of the old validate_args alias.
"""

import configparser
import ast
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core parse
# ---------------------------------------------------------------------------

def parse_config(filename):
    """Parse *filename* and return (taskvals_dict, ConfigParser).

    Values are evaluated with ast.literal_eval so Python literals (int, bool,
    list, str, …) come back as their proper types.  Strings must be quoted in
    the config file.
    """
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(filename)

    taskvals = {}
    for section in config.sections():
        taskvals[section] = {}
        for option in config.options(section):
            raw = config.get(section, option)
            try:
                taskvals[section][option] = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                raise ValueError(
                    f"Cannot format field '{option}' in config file '{filename}', "
                    f"which is currently set to {raw!r}. "
                    "Ensure strings are in 'quotes'."
                )

    return taskvals, config


# ---------------------------------------------------------------------------
# Query helpers  (each re-parses; callers that need performance should call
# parse_config once and work with the returned dict directly)
# ---------------------------------------------------------------------------

def has_section(filename, section):
    taskvals, _ = parse_config(filename)
    return section in taskvals


def has_key(filename, section, key):
    taskvals, _ = parse_config(filename)
    return section in taskvals and key in taskvals[section]


def get_key(filename, section, key):
    taskvals, _ = parse_config(filename)
    if section in taskvals and key in taskvals[section]:
        return taskvals[section][key]
    return ''


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def overwrite_config(filename, conf_dict=None, conf_sec='', sec_comment=''):
    """Write *conf_dict* into *conf_sec* of *filename*, creating the section if needed."""
    if conf_dict is None:
        conf_dict = {}
    _, config = parse_config(filename)

    if conf_sec not in config.sections():
        logger.debug("Writing [%s] in '%s': %s", conf_sec, filename, conf_dict)
        config.add_section(conf_sec)
    else:
        logger.debug("Overwriting [%s] in '%s': %s", conf_sec, filename, conf_dict)

    if sec_comment:
        config.set(conf_sec, sec_comment)

    for key, value in conf_dict.items():
        config.set(conf_sec, key, str(value))

    with open(filename, 'w') as fh:
        config.write(fh)


def remove_section(filename, section):
    _, config = parse_config(filename)
    config.remove_section(section)
    with open(filename, 'w') as fh:
        config.write(fh)


# ---------------------------------------------------------------------------
# Typed value extraction
# ---------------------------------------------------------------------------

def typed_get(config_dict, section, key, dtype, default=None):
    """Extract and type-coerce a value from a parsed config dict.

    Parameters
    ----------
    config_dict : dict
        The dict returned by parse_config()[0].
    section : str
        Config section name.
    key : str
        Key within the section.
    dtype : type
        Target type: str, int, float, or bool.
    default : optional
        Returned (without coercion) when the key is absent.
        If omitted and the key is missing, KeyError is raised.

    Notes
    -----
    Does NOT mutate config_dict.  Each call is a read-only operation.
    """
    if default is not None:
        val = config_dict.get(section, {}).get(key, default)
    else:
        try:
            val = config_dict[section][key]
        except KeyError:
            raise KeyError(
                f"Missing required key '{key}' in [{section}]"
            )

    if dtype is str:
        return str(val).rstrip('/ ')
    elif dtype is int:
        try:
            return int(val)
        except (ValueError, TypeError):
            raise ValueError(
                f"[{section}] {key} = {val!r} cannot be converted to int"
            )
    elif dtype is float:
        try:
            return float(val)
        except (ValueError, TypeError):
            raise ValueError(
                f"[{section}] {key} = {val!r} cannot be converted to float"
            )
    elif dtype is bool:
        if isinstance(val, bool):
            return val
        if isinstance(val, int):
            return bool(val)
        raise ValueError(
            f"[{section}] {key} = {val!r} is not a bool (True/False)"
        )
    else:
        raise NotImplementedError(
            f"Unsupported dtype {dtype!r}. Use str, int, float, or bool."
        )


# Backward-compatible alias — prefer typed_get in new code.
validate_args = typed_get


# ---------------------------------------------------------------------------
# SPW parsing  (reads [crosscal] section for spw/nspw)
# ---------------------------------------------------------------------------

def parse_spw(filename):
    """Return (low, high, unit, dirs) parsed from the [crosscal] spw key."""
    from spw import get_spw_bounds

    config_dict, _ = parse_config(filename)
    spw = config_dict['crosscal']['spw']

    if ',' in spw:
        SPWs = spw.split(',')
        low = [0] * len(SPWs)
        high = [0] * len(SPWs)
        unit = [''] * len(SPWs)
        dirs = [''] * len(SPWs)
        for i, SPW in enumerate(SPWs):
            low[i], high[i], unit[i], _ = get_spw_bounds(SPW)
            dirs[i] = '{0}~{1}{2}'.format(low[i], high[i], unit[i])
        return low, high, unit, dirs
    else:
        low, high, unit, _ = get_spw_bounds(spw)
        return low, high, unit, []
