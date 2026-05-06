from .base import FacilityConfig
from .ilifu import IlifuFacility
from .generic_slurm import GenericSlurmFacility

FACILITIES = {
    'ilifu': IlifuFacility,
    'generic_slurm': GenericSlurmFacility,
}


def get_facility(name='ilifu', **kwargs):
    """Return a FacilityConfig instance by name."""
    if name not in FACILITIES:
        raise ValueError(
            f"Unknown facility '{name}'. Choose one of: {list(FACILITIES)}"
        )
    return FACILITIES[name](**kwargs)
