"""Pipeline-wide constants, extracted from processMeerKAT.py.

Facility-specific limits (node counts, memory) now live in facilities/.
These constants cover file/directory names and calibration script lists
that are facility-independent.
"""

import os

# Paths relative to the pipeline installation directory.
# THIS_PROG = the user-facing CLI command name (entry point in pyproject.toml).
# SCRIPT_DIR = the package directory, used for locating bundled scripts.
THIS_PROG = 'processMeerKAT'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Well-known directory / file names (relative to the working directory)
LOG_DIR = 'logs'
CALIB_SCRIPTS_DIR = 'crosscal_scripts'
AUX_SCRIPTS_DIR = 'aux_scripts'
SELFCAL_SCRIPTS_DIR = 'selfcal_scripts'
CONFIG = 'default_config.toml'
PIPELINE_STATE = 'pipeline_state.toml'
MASTER_SCRIPT = 'submit_pipeline.sh'
SPW_PREFIX = '*:'

# Config section key lists
FIELDS_CONFIG_KEYS = [
    'fluxfield', 'bpassfield', 'phasecalfield', 'targetfields', 'extrafields',
]
CROSSCAL_CONFIG_KEYS = [
    'minbaselines', 'chanbin', 'width', 'timeavg', 'createmms', 'keepmms',
    'spw', 'nspw', 'calcrefant', 'refant', 'standard', 'badants', 'badfreqranges',
    'badfreq_uvrange',
]
SELFCAL_CONFIG_KEYS = [
    'nloops', 'loop', 'cell', 'robust', 'imsize', 'wprojplanes', 'niter',
    'threshold', 'uvrange', 'nterms', 'gridder', 'deconvolver', 'solint',
    'calmode', 'discard_nloops', 'gaintype', 'outlier_threshold', 'flag',
    'outlier_radius',
]
IMAGING_CONFIG_KEYS = [
    'cell', 'robust', 'imsize', 'wprojplanes', 'niter', 'threshold',
    'multiscale', 'nterms', 'gridder', 'deconvolver', 'restoringbeam',
    'stokes', 'mask', 'rmsmap', 'outlierfile', 'pbthreshold', 'pbband',
]
SLURM_CONFIG_STR_KEYS = [
    'container', 'runner', 'mpi_wrapper', 'partition', 'time', 'name',
    'dependencies', 'exclude', 'account', 'reservation',
]
SLURM_CONFIG_KEYS = [
    'nodes', 'ntasks_per_node', 'mem', 'submit',
    'precal_scripts', 'postcal_scripts', 'scripts', 'target_scripts',
    'verbose', 'modules',
] + SLURM_CONFIG_STR_KEYS

# Default calibration script lists.
# Each entry is a dict with keys: script (str), mpi (bool), and optionally
# container (str) to override the global slurm.container for that step.
#
# DAG (multi-SPW): precal_scripts run once at top level (just the calibrator
# partition). Each per-SPW directory then runs a single linear chain: the
# calibrator solve `scripts`, followed by the target sub-chain
# (TARGET_SUBCHAIN) — split that SPW's target, flag it, and apply that SPW's
# own caltables to it. Because every SPW directory works on independent MSes,
# all 11 applies run in parallel with no shared-column contention, and a
# missing fluxscale fails only that SPW's job. postcal_scripts then run once at
# top level, concatenating the corrected per-SPW targets along frequency before
# selfcal / imaging.
PRECAL_SCRIPTS = [
    {'script': 'partition.py', 'mpi': True},
]
POSTCAL_SCRIPTS = [
    {'script': 'concat_target.py', 'mpi': False},
    {'script': 'selfcal_part1.py', 'mpi': True},
    {'script': 'selfcal_part2.py', 'mpi': False},
    {'script': 'science_image.py', 'mpi': True},
]
# Per-SPW target sub-chain, injected into each per-SPW directory's
# postcal_scripts by spw.spw_split. partition_target splits this SPW's target
# (and switches [data].vis to it), the flag steps clean it, and apply_to_target
# applies this SPW's caltables. Runs after the calibrator solve `scripts`.
TARGET_SUBCHAIN = [
    {'script': 'partition_target.py', 'mpi': True},
    {'script': 'validate_input.py', 'mpi': False},
    {'script': 'flag_round_1.py', 'mpi': True},
    {'script': 'apply_to_target.py', 'mpi': True},
]
SCRIPTS = [
    {'script': 'validate_input.py', 'mpi': False},
    {'script': 'flag_round_1.py', 'mpi': True},
    {'script': 'setjy.py', 'mpi': True},
    {'script': 'xx_yy_solve.py', 'mpi': False},
    {'script': 'xx_yy_apply.py', 'mpi': True},
    {'script': 'flag_round_2.py', 'mpi': True},
    {'script': 'xx_yy_solve.py', 'mpi': False},
    {'script': 'xx_yy_apply.py', 'mpi': True},
]
# The old top-level monolithic target branch is retired: target flagging now
# runs per-SPW as part of TARGET_SUBCHAIN inside each SPW directory. Kept as an
# empty list so callers that still pass target_scripts get a no-op branch.
TARGET_SCRIPTS = []

# Facility-specific defaults (these match IlifuFacility; overridden at runtime)
# Kept here only so legacy code that imports directly still works.
CONTAINER = ''
MPI_WRAPPER = 'mpirun'
TOTAL_NODES_LIMIT = 79
CPUS_PER_NODE_LIMIT = 32
NTASKS_PER_NODE_LIMIT = CPUS_PER_NODE_LIMIT
MEM_PER_NODE_GB_LIMIT = 232
MEM_PER_NODE_GB_LIMIT_HIGHMEM = 480
