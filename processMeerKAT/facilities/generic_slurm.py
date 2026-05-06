"""Generic SLURM facility — user supplies all cluster-specific parameters."""

import os
import logging

from .base import FacilityConfig

logger = logging.getLogger(__name__)


class GenericSlurmFacility(FacilityConfig):
    """A facility where the user is responsible for all cluster limits.

    Use this when running on any SLURM cluster that isn't Ilifu.  Supply all
    parameters at construction time (or via the ``[facility]`` config section).

    Example config::

        [facility]
        name = generic_slurm
        total_nodes_limit = 100
        cpus_per_node_limit = 64
        mem_per_node_gb_limit = 512
        mem_per_node_gb_limit_highmem = 512
        default_container = /path/to/container.sif
        default_mpi_wrapper = mpirun
        default_account = myproject
        default_partition = normal
        default_modules = ['openmpi/4.1.0']
    """

    def __init__(
        self,
        total_nodes_limit=100,
        cpus_per_node_limit=64,
        mem_per_node_gb_limit=256,
        mem_per_node_gb_limit_highmem=256,
        default_container='',
        default_mpi_wrapper='mpirun',
        default_account='',
        default_partition='normal',
        default_modules=None,
    ):
        self.total_nodes_limit = total_nodes_limit
        self.cpus_per_node_limit = cpus_per_node_limit
        self.mem_per_node_gb_limit = mem_per_node_gb_limit
        self.mem_per_node_gb_limit_highmem = mem_per_node_gb_limit_highmem
        self.default_container = default_container
        self.default_mpi_wrapper = default_mpi_wrapper
        self.default_account = default_account
        self.default_partition = default_partition
        self.default_modules = default_modules or []

    def validate_account(self, account, config, parser=None):
        """No account validation — just warn if empty and return as-is."""
        if not account:
            logger.warning(
                "No SLURM account specified and none set as default for "
                "generic_slurm facility. Jobs may fail to submit."
            )
        return account or self.default_account

    def validate_reservation(self, reservation, args, config, parser=None):
        """No reservation validation for generic facility — log and continue."""
        if reservation:
            logger.warning(
                f"Reservation '{reservation}' specified but cannot be validated "
                "on generic_slurm facility. Proceeding anyway."
            )
