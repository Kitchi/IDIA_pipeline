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
    # Command prefix injected before every script invocation. Replaces the
    # legacy hard-coded `singularity exec {container}`. Examples:
    #   - Container facility: "singularity exec /idia/.../casa.sif"
    #   - Conda env on a bare cluster: "conda run -n py312 --no-capture-output"
    #   - Native Python with casatasks pip-installed: "" (empty)
    # If a per-script `container` override is set in [slurm].containers, that
    # wins (built as `singularity exec {container}` for backward compat).
    default_runner: str = ''

    # Optional validation hooks
    validate_account: Callable = field(default=_noop_validate_account, repr=False)
    validate_reservation: Callable = field(default=_noop_validate_reservation, repr=False)
