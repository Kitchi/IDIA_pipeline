"""Generic SLURM facility — user supplies all cluster-specific parameters."""

import logging

from .base import FacilityConfig

logger = logging.getLogger(__name__)


def _validate_account(account, config, parser=None):
    """Warn if no account given; return as-is."""
    if not account:
        logger.warning(
            "No SLURM account specified. Jobs may fail to submit."
        )
    return account


GENERIC_SLURM = FacilityConfig(
    name='generic_slurm',
    total_nodes_limit=100,
    cpus_per_node_limit=64,
    mem_per_node_gb_limit=256,
    mem_per_node_gb_limit_highmem=256,
    validate_account=_validate_account,
)
