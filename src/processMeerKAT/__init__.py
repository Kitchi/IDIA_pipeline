"""processMeerKAT — IDIA MeerKAT calibration pipeline.

Public API: import processMeerKAT and use the symbols below directly.
"""

from .spw import get_spw_bounds, linspace, spw_split
from .slurm_jobs import (
    check_path, check_bash_path,
    write_command, write_sbatch, srun,
    write_master, write_spw_master,
    write_all_bash_jobs_scripts, write_bash_job_script,
    write_jobs,
)
from .pipeline import get_config_kwargs, get_slurm_dict, pop_script, format_args, default_config
from .processMeerKAT import (
    load_facility_from_config, raise_error, validate_args, parse_args, setup_logger, main,
)
from .constants import (
    THIS_PROG, SCRIPT_DIR, LOG_DIR,
    CONFIG, PIPELINE_STATE, MASTER_SCRIPT, SPW_PREFIX,
    FIELDS_CONFIG_KEYS, CROSSCAL_CONFIG_KEYS, SELFCAL_CONFIG_KEYS,
    IMAGING_CONFIG_KEYS, SLURM_CONFIG_STR_KEYS, SLURM_CONFIG_KEYS,
    PRECAL_SCRIPTS, POSTCAL_SCRIPTS, SCRIPTS, TARGET_SCRIPTS,
)

__version__ = '2.0'
