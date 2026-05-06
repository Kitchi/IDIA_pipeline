"""Facility (cluster) configuration dataclass."""

from dataclasses import dataclass, field
from typing import Callable


def _noop_validate_account(account, config, parser=None):
    return account


def _noop_validate_reservation(reservation, args, config, parser=None):
    pass


@dataclass
class FacilityConfig:
    """Everything that differs between compute facilities.

    To add a new facility, construct a FacilityConfig instance with your
    cluster's limits and defaults.  Optionally supply validate_account and
    validate_reservation callables if your cluster requires it.
    """

    # Resource limits (required — every facility must specify these)
    total_nodes_limit: int
    cpus_per_node_limit: int
    mem_per_node_gb_limit: int
    mem_per_node_gb_limit_highmem: int

    # Identity and defaults
    name: str = ''
    default_container: str = ''
    default_mpi_wrapper: str = 'mpirun'
    default_account: str = ''
    default_partition: str = 'normal'
    default_modules: list = field(default_factory=list)

    # Optional validation hooks
    validate_account: Callable = field(default=_noop_validate_account, repr=False)
    validate_reservation: Callable = field(default=_noop_validate_reservation, repr=False)
