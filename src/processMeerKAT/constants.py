"""Pipeline-wide constants, extracted from processMeerKAT.py.

Facility-specific limits (node counts, memory) now live in facilities/.
These constants cover file/directory names and calibration script lists
that are facility-independent.
"""

import os

# Paths relative to the pipeline installation directory
THIS_PROG = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(THIS_PROG)

# Well-known directory / file names (relative to the working directory)
LOG_DIR = 'logs'
CALIB_SCRIPTS_DIR = 'crosscal_scripts'
AUX_SCRIPTS_DIR = 'aux_scripts'
SELFCAL_SCRIPTS_DIR = 'selfcal_scripts'
CONFIG = 'default_config.txt'
TMP_CONFIG = '.config.tmp'
MASTER_SCRIPT = 'submit_pipeline.sh'
SPW_PREFIX = '*:'

# Config section key lists
FIELDS_CONFIG_KEYS = [
    'fluxfield', 'bpassfield', 'phasecalfield', 'targetfields', 'extrafields',
]
CROSSCAL_CONFIG_KEYS = [
    'minbaselines', 'chanbin', 'width', 'timeavg', 'createmms', 'keepmms',
    'spw', 'nspw', 'calcrefant', 'refant', 'standard', 'badants', 'badfreqranges',
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
    'container', 'mpi_wrapper', 'partition', 'time', 'name',
    'dependencies', 'exclude', 'account', 'reservation',
]
SLURM_CONFIG_KEYS = [
    'nodes', 'ntasks_per_node', 'mem', 'plane', 'submit',
    'precal_scripts', 'postcal_scripts', 'scripts', 'verbose', 'modules',
] + SLURM_CONFIG_STR_KEYS

# Default calibration script lists
# Each entry: (script_name, threadsafe, container_override)
PRECAL_SCRIPTS = [
    ('calc_refant.py', False, ''),
    ('partition.py', True, ''),
]
POSTCAL_SCRIPTS = [
    ('concat.py', False, ''),
    ('plotcal_spw.py', False, ''),
    ('selfcal_part1.py', True, ''),
    ('selfcal_part2.py', False, ''),
    ('science_image.py', True, ''),
]
SCRIPTS = [
    ('validate_input.py', False, ''),
    ('flag_round_1.py', True, ''),
    ('calc_refant.py', False, ''),
    ('setjy.py', True, ''),
    ('xx_yy_solve.py', False, ''),
    ('xx_yy_apply.py', True, ''),
    ('flag_round_2.py', True, ''),
    ('xx_yy_solve.py', False, ''),
    ('xx_yy_apply.py', True, ''),
    ('split.py', True, ''),
    ('quick_tclean.py', True, ''),
]

# Facility-specific defaults (these match IlifuFacility; overridden at runtime)
# Kept here only so legacy code that imports directly still works.
CONTAINER = '/idia/software/containers/casa-6.5.0-modular.sif'
MPI_WRAPPER = 'mpirun'
TOTAL_NODES_LIMIT = 79
CPUS_PER_NODE_LIMIT = 32
NTASKS_PER_NODE_LIMIT = CPUS_PER_NODE_LIMIT
MEM_PER_NODE_GB_LIMIT = 232
MEM_PER_NODE_GB_LIMIT_HIGHMEM = 480
