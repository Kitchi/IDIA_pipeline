import dataclasses

from .base import FacilityConfig
from .ilifu import ILIFU
from .generic_slurm import GENERIC_SLURM

FACILITIES = {
    'ilifu': ILIFU,
    'generic_slurm': GENERIC_SLURM,
}


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
