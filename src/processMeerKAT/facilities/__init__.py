"""Facility registry — auto-discovered from this package and PROCESSMEERKAT_FACILITIES_PATH."""

import dataclasses
import importlib
import importlib.util
import os
import pkgutil
from pathlib import Path

from .base import FacilityConfig, DEFAULT_FACILITY_CONFIG


def _iter_facility_module(mod):
    """Yield every FacilityConfig instance with a non-empty name from a module."""
    for attr in vars(mod).values():
        if isinstance(attr, FacilityConfig) and attr.name:
            yield attr.name, attr


def _scan_package_dir(package_dir, package_name):
    """Scan built-in facilities/ package for FacilityConfig instances."""
    facilities = {}
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if module_info.name == 'base':
            continue
        try:
            mod = importlib.import_module(f'.{module_info.name}', package=package_name)
        except Exception:
            continue
        for name, fac in _iter_facility_module(mod):
            facilities[name] = fac
    return facilities


def _scan_external_dirs(path_str):
    """Scan colon-separated directories from PROCESSMEERKAT_FACILITIES_PATH."""
    facilities = {}
    for dirpath in filter(None, path_str.split(':')):
        p = Path(dirpath)
        if not p.is_dir():
            continue
        for pyfile in sorted(p.glob('*.py')):
            spec = importlib.util.spec_from_file_location(pyfile.stem, pyfile)
            if spec is None:
                continue
            try:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
            except Exception:
                continue
            for name, fac in _iter_facility_module(mod):
                facilities[name] = fac
    return facilities


def _build_facilities():
    package_dir = Path(__file__).parent
    facilities = _scan_package_dir(package_dir, __name__)
    extra = os.environ.get('PROCESSMEERKAT_FACILITIES_PATH', '')
    if extra:
        # External dirs take precedence — lets local installs override built-ins.
        facilities.update(_scan_external_dirs(extra))
    return facilities


FACILITIES = _build_facilities()


def get_facility(name='ilifu', **overrides):
    """Return a FacilityConfig instance by name, with optional field overrides.

    For a known facility, all defaults are pre-validated.  Pass keyword
    arguments to override individual fields (e.g. total_nodes_limit=50).
    The validate_account / validate_reservation callables are always inherited
    from the named facility and cannot be overridden via config.
    """
    if name not in FACILITIES:
        raise ValueError(
            f"Unknown facility '{name}'. Known facilities: {list(FACILITIES)}"
        )
    base = FACILITIES[name]
    if overrides:
        return dataclasses.replace(base, **overrides)
    return base
