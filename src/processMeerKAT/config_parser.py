#Copyright (C) 2022 Inter-University Institute for Data Intensive Astronomy
#See processMeerKAT.py for license details.

"""Config file I/O for the MeerKAT pipeline.

Public API
----------
parse_config(filename)              Parse a TOML config → (dict, dict)
has_section(filename, section)
has_key(filename, section, key)
get_key(filename, section, key)
overwrite_config(filename, ...)
remove_section(filename, section)
parse_spw(filename)                 Parse SPW bounds from crosscal section
typed_get(config_dict, section, key, dtype, default=None)
    Extract and type-coerce a value from a parsed config dict.
"""

import logging

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError(
            "Python < 3.11 requires the 'tomli' package: pip install tomli"
        )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TOML writer (tomllib is read-only; we use a focused custom serializer)
# ---------------------------------------------------------------------------

def _toml_scalar(v):
    if v is None:
        return '""'
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        escaped = v.replace('\\', '\\\\').replace('"', '\\"')
        return f'"{escaped}"'
    raise TypeError(f"Cannot serialize {type(v).__name__} as TOML scalar: {v!r}")


def _toml_value(v):
    if isinstance(v, list):
        if not v:
            return '[]'
        if all(isinstance(item, dict) for item in v):
            rows = []
            for item in v:
                pairs = ', '.join(f'{k} = {_toml_scalar(iv)}' for k, iv in item.items())
                rows.append(f'  {{{pairs}}}')
            return '[\n' + ',\n'.join(rows) + ',\n]'
        return '[' + ', '.join(_toml_scalar(i) for i in v) + ']'
    return _toml_scalar(v)


def _dump_toml(data):
    """Serialize a two-level {section: {key: value}} dict to a TOML string."""
    parts = []
    for section, vals in data.items():
        parts.append(f'[{section}]')
        for key, val in vals.items():
            parts.append(f'{key} = {_toml_value(val)}')
        parts.append('')
    return '\n'.join(parts)


# ---------------------------------------------------------------------------
# Core parse
# ---------------------------------------------------------------------------

def parse_config(filename):
    """Parse *filename* and return (taskvals_dict, taskvals_dict).

    Returns a 2-tuple for API compatibility; both elements are the same dict.
    A missing file returns ({}, {}) to match legacy configparser behaviour.
    Invalid TOML raises ValueError.
    """
    try:
        with open(filename, 'rb') as fh:
            data = tomllib.load(fh)
    except FileNotFoundError:
        return {}, {}
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(
            f"Cannot parse TOML config '{filename}': {exc}"
        )
    if data is None:
        data = {}
    return data, data


# ---------------------------------------------------------------------------
# Query helpers
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
    taskvals, _ = parse_config(filename)

    if conf_sec not in taskvals:
        logger.debug("Writing [%s] in '%s': %s", conf_sec, filename, conf_dict)
        taskvals[conf_sec] = {}
    else:
        logger.debug("Overwriting [%s] in '%s': %s", conf_sec, filename, conf_dict)

    taskvals[conf_sec].update(conf_dict)

    with open(filename, 'w', encoding='utf-8') as fh:
        fh.write(_dump_toml(taskvals))


def remove_section(filename, section):
    taskvals, _ = parse_config(filename)
    taskvals.pop(section, None)
    with open(filename, 'w', encoding='utf-8') as fh:
        fh.write(_dump_toml(taskvals))


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
# SPW parsing  (reads crosscal section for spw/nspw)
# ---------------------------------------------------------------------------

def parse_spw(filename):
    """Return (low, high, unit, dirs) parsed from the crosscal spw key."""
    from .spw import get_spw_bounds

    config_dict, _ = parse_config(filename)
    spw = config_dict['crosscal']['spw']
    SPWs = spw.split(',') if ',' in spw else [spw]

    low, high, unit, dirs = [], [], [], []
    for SPW in SPWs:
        l, h, u, _ = get_spw_bounds(SPW)
        low.append(l)
        high.append(h)
        unit.append(u)
        if ',' in spw:
            dirs.append('{0}~{1}{2}'.format(l, h, u))

    return low, high, unit, dirs
