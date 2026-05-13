"""Ilifu cluster facility configuration."""

import os
import platform
import logging

from .base import FacilityConfig

logger = logging.getLogger(__name__)


def _on_slurm_node():
    hostname = platform.node()
    return any(x in hostname for x in ('slurm-login', 'slwrk', 'compute'))


def _validate_account(account, config, parser=None):
    """Validate SLURM account via sacctmgr. Auto-detects default if None."""

    from processMeerKAT import raise_error  # avoid circular import at module level

    if not _on_slurm_node():
        logger.warning(
            f"Not on a Slurm node. Skipping account validation for '{account}'."
        )
        return account or ILIFU.default_account

    if os.environ.get('SLURM_JOB_ID'):
        logger.info(
            f"Running inside Slurm job {os.environ['SLURM_JOB_ID']}. "
            "Skipping account validation."
        )
        return account

    user = os.environ.get('USER', '')

    if not account or str(account).lower() == 'none':
        default = os.popen(
            f"sacctmgr show user {user} --noheader format=DefaultAccount%30"
        ).read().strip()
        available = os.popen(
            f"sacctmgr show user {user} --noheader -s format=account%30"
        ).read().split()
        if default:
            logger.info(
                f"No account specified. Using default '{default}'. "
                f"All authorized groups: {', '.join(available)}."
            )
            return default
        if available:
            logger.warning(
                f"No default account set for your user. "
                f"Using first available group '{available[0]}'. "
                f"All authorized groups: {', '.join(available)}. "
                f"Override with -A / --account."
            )
            return available[0]
        raise_error(config, "No Slurm account found for your user.", parser)

    check = os.popen(
        f"sacctmgr show associations user={user} account={account} --noheader"
    ).read().strip()
    if not check:
        available = os.popen(
            f"sacctmgr show user {user} --noheader -s format=account%30"
        ).read().split()
        default = os.popen(
            f"sacctmgr show user {user} --noheader format=DefaultAccount%30"
        ).read().strip()
        msg = f"Accounting group '{account}' is not recognized for user {user}."
        if available:
            msg += f" Your authorized groups are: {', '.join(available)}."
        if default:
            msg += f" Your default is '{default}'."
        raise_error(config, msg, parser)

    logger.info(f"Using Slurm account '{account}'")
    return account


def _validate_reservation(reservation, args, config, parser=None):
    """Validate that reservation exists via scontrol."""

    from processMeerKAT import raise_error

    if not reservation:
        return

    if not _on_slurm_node():
        msg = (
            f"Reservation '{reservation}' not recognised. "
            "You're not on a SLURM node, so cannot query reservations."
        )
        raise_error(config, msg, parser)
        return

    reservations = (
        os.popen(
            "scontrol show reservation | grep ReservationName "
            "| awk '{print $1}' | cut -d = -f2"
        )
        .read()[:-1]
        .split('\n')
    )
    if reservation not in reservations:
        msg = f"Reservation '{reservation}' not recognised."
        if reservations == ['']:
            msg += ' There are no active reservations.'
        else:
            msg += f' Active reservations: {reservations}.'
        raise_error(config, msg, parser)


ILIFU = FacilityConfig(
    name='ilifu',
    total_nodes_limit=79,
    cpus_per_node_limit=32,
    mem_per_node_gb_limit=232,
    mem_per_node_gb_limit_highmem=480,
    default_container='/idia/software/containers/casa-6.5.0-modular.sif',
    default_runner='singularity exec',
    default_mpi_wrapper='mpirun',
    default_account='',
    default_partition='Main',
    default_modules=['openmpi/4.0.3'],
    validate_account=_validate_account,
    validate_reservation=_validate_reservation,
)
