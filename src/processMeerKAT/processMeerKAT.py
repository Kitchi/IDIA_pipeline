#!/usr/bin/env python3

__version__ = '2.0'

license = """
    Process MeerKAT data via CASA MeasurementSet.
    Copyright (C) 2022 Inter-University Institute for Data Intensive Astronomy.
    support@ilifu.ac.za

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import os
import logging
from time import gmtime

from . import config_parser
from .constants import (
    THIS_PROG, SCRIPT_DIR, LOG_DIR,
    CONFIG, PIPELINE_STATE, MASTER_SCRIPT, SPW_PREFIX,
    FIELDS_CONFIG_KEYS, CROSSCAL_CONFIG_KEYS, SELFCAL_CONFIG_KEYS,
    IMAGING_CONFIG_KEYS, SLURM_CONFIG_STR_KEYS, SLURM_CONFIG_KEYS,
    PRECAL_SCRIPTS, POSTCAL_SCRIPTS, SCRIPTS, TARGET_SCRIPTS,
    TOTAL_NODES_LIMIT, NTASKS_PER_NODE_LIMIT, CPUS_PER_NODE_LIMIT,
)
from .spw import get_spw_bounds, linspace, spw_split
from .slurm_jobs import (
    check_path, check_bash_path,
    write_command, write_sbatch, srun,
    write_master, write_spw_master,
    write_all_bash_jobs_scripts, write_bash_job_script,
    write_jobs,
)
from .pipeline import (
    get_config_kwargs, get_slurm_dict, pop_script,
    format_args, default_config,
)
from .facilities import get_facility
from .facilities.ilifu import ILIFU

logging.Formatter.converter = gmtime
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)-15s %(levelname)s: %(message)s")

# Active facility — default to Ilifu; overridden at -R time via [facility] config section.
_FACILITY = ILIFU


# ---------------------------------------------------------------------------
# Facility loading
# ---------------------------------------------------------------------------

def load_facility_from_config(config_path):
    """Read the [facility] section from config and return a FacilityConfig.

    If the section is absent the current default facility is returned unchanged.
    Keys other than 'name' are treated as field overrides on the named facility.
    """
    if not config_parser.has_section(config_path, 'facility'):
        return _FACILITY

    fac_dict = dict(config_parser.parse_config(config_path)[0].get('facility', {}))
    name = str(fac_dict.pop('name', _FACILITY.name)).strip("'\"")
    return get_facility(name, **fac_dict)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def raise_error(config, msg, parser=None):
    """Raise parser error or ValueError depending on context."""
    if parser is None:
        raise ValueError("Bad input found in '{0}' -- {1}".format(config, msg))
    else:
        parser.error(msg)


# ---------------------------------------------------------------------------
# Argument validation (pure Python — no SLURM calls)
# ---------------------------------------------------------------------------

def validate_args(args, config, parser=None):
    """Validate CLI / config arguments against the active facility's limits.

    Account and reservation validation (which require SLURM CLI tools) are
    NOT performed here — they happen in format_args() during the -R step only.
    """
    if parser is None or args.get('build'):
        if args.get('MS') is None and not args.get('nofields'):
            raise_error(config, "You must input an MS [-M --MS] to build the config file.", parser)
        if args.get('MS') not in [None, 'None'] and not os.path.isdir(args['MS']):
            raise_error(config, "Input MS '{0}' not found.".format(args['MS']), parser)

    if parser is not None and not args.get('build') and args.get('MS'):
        raise_error(config, "Only input an MS [-M --MS] during [-B --build] step. Otherwise input is ignored.", parser)

    if args['ntasks_per_node'] > _FACILITY.cpus_per_node_limit:
        raise_error(
            config,
            "The number of tasks per node [-t --ntasks-per-node] must not exceed {0}. You input {1}.".format(
                _FACILITY.cpus_per_node_limit, args['ntasks_per_node']
            ),
            parser,
        )

    if args['nodes'] > _FACILITY.total_nodes_limit:
        raise_error(
            config,
            "The number of nodes [-N --nodes] must not exceed {0}. You input {1}.".format(
                _FACILITY.total_nodes_limit, args['nodes']
            ),
            parser,
        )

    if args['mem'] > _FACILITY.mem_per_node_gb_limit:
        if args.get('partition') != 'HighMem':
            raise_error(
                config,
                "Memory per node [-m --mem] must not exceed {0} GB. You input {1} GB.".format(
                    _FACILITY.mem_per_node_gb_limit, args['mem']
                ),
                parser,
            )
        elif args['mem'] > _FACILITY.mem_per_node_gb_limit_highmem:
            raise_error(
                config,
                "Memory per node [-m --mem] must not exceed {0} GB for HighMem partition. You input {1} GB.".format(
                    _FACILITY.mem_per_node_gb_limit_highmem, args['mem']
                ),
                parser,
            )

    if args['plane'] > args['ntasks_per_node']:
        raise_error(
            config,
            "[-D --plane] cannot be greater than ntasks-per-node ({0}). You input {1}.".format(
                args['ntasks_per_node'], args['plane']
            ),
            parser,
        )


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    """Parse and validate command-line arguments."""

    def parse_scripts(val):
        if val.lower() in ('true', 'false'):
            return val.lower() == 'true'
        return check_path(val)

    parser = argparse.ArgumentParser(
        prog=THIS_PROG,
        description='Process MeerKAT data via CASA MeasurementSet. Version: {0}'.format(__version__),
    )

    parser.add_argument("-M", "--MS", metavar="path", required=False, type=str,
                        help="Path to MeasurementSet.")
    parser.add_argument("-C", "--config", metavar="path", default=CONFIG, required=False, type=str,
                        help="Relative (not absolute) path to config file.")
    parser.add_argument("-N", "--nodes", metavar="num", required=False, type=int, default=1,
                        help="Use this number of nodes [default: 1; max: {0}].".format(_FACILITY.total_nodes_limit))
    parser.add_argument("-t", "--ntasks-per-node", metavar="num", required=False, type=int, default=8,
                        help="Use this number of tasks (per node) [default: 8; max: {0}].".format(_FACILITY.cpus_per_node_limit))
    parser.add_argument("-D", "--plane", metavar="num", required=False, type=int, default=1,
                        help="Distribute tasks of this block size before moving onto next node [default: 1].")
    parser.add_argument("-m", "--mem", metavar="num", required=False, type=int,
                        default=_FACILITY.mem_per_node_gb_limit,
                        help="Use this many GB of memory per node [default: {0}].".format(_FACILITY.mem_per_node_gb_limit))
    parser.add_argument("-p", "--partition", metavar="name", required=False, type=str,
                        default=_FACILITY.default_partition,
                        help="SLURM partition to use [default: '{0}'].".format(_FACILITY.default_partition))
    parser.add_argument("-T", "--time", metavar="time", required=False, type=str, default="12:00:00",
                        help="Time limit for all jobs [default: '12:00:00'].")
    parser.add_argument("-S", "--scripts", action='append', nargs=3,
                        metavar=('script', 'threadsafe', 'container'), required=False,
                        type=parse_scripts, default=SCRIPTS,
                        help="Pipeline scripts to run in order with threadsafe and container flags.")
    parser.add_argument("-b", "--precal_scripts", action='append', nargs=3,
                        metavar=('script', 'threadsafe', 'container'), required=False,
                        type=parse_scripts, default=PRECAL_SCRIPTS,
                        help="Scripts run before calibration (nspw > 1 only).")
    parser.add_argument("-a", "--postcal_scripts", action='append', nargs=3,
                        metavar=('script', 'threadsafe', 'container'), required=False,
                        type=parse_scripts, default=POSTCAL_SCRIPTS,
                        help="Scripts run after calibration (nspw > 1 only).")
    parser.add_argument("-g", "--target_scripts", action='append', nargs=3,
                        metavar=('script', 'threadsafe', 'container'), required=False,
                        type=parse_scripts, default=TARGET_SCRIPTS,
                        help="Scripts run on the monolithic target MMS in parallel with per-SPW calibrator solves (nspw > 1 only).")
    parser.add_argument("--modules", nargs='*', metavar='module', required=False,
                        default=_FACILITY.default_modules,
                        help="Load these modules within each sbatch script.")
    parser.add_argument("-w", "--mpi_wrapper", metavar="path", required=False, type=str,
                        default=_FACILITY.default_mpi_wrapper,
                        help="MPI wrapper for threadsafe scripts [default: '{0}'].".format(_FACILITY.default_mpi_wrapper))
    parser.add_argument("-c", "--container", metavar="path", required=False, type=str,
                        default=_FACILITY.default_container,
                        help="Singularity container [default: '{0}'].".format(_FACILITY.default_container))
    parser.add_argument("--runner", metavar="prefix", required=False, type=str, default=None,
                        help=(
                            "Command prefix for every script invocation, overriding the facility "
                            "default. Use '' for bare Python (CASA installed in the active env), "
                            "'conda run -n mycasa --no-capture-output' for a conda env, or "
                            "'singularity exec /path/to.sif' for an explicit container. "
                            "When set, --container is cleared."
                        ))
    parser.add_argument("-n", "--name", metavar="unique", required=False, type=str, default='',
                        help="Unique run name prefix for all job names.")
    parser.add_argument("-d", "--dependencies", metavar="list", required=False, type=str, default='',
                        help="Comma-separated SLURM job dependencies (nspw=1 only).")
    parser.add_argument("-e", "--exclude", metavar="nodes", required=False, type=str, default='',
                        help="SLURM nodes to exclude.")
    parser.add_argument("-A", "--account", metavar="group", required=False, type=str, default=None,
                        help="SLURM accounting group (auto-detected if omitted).")
    parser.add_argument("-r", "--reservation", metavar="name", required=False, type=str, default='',
                        help="SLURM reservation to use.")
    parser.add_argument("-F", "--facility", metavar="name", required=False, type=str, default=None,
                        help="Facility to target during -B (e.g. ilifu, generic_slurm). At -R time the [facility] section in the config file takes precedence.")
    parser.add_argument("-l", "--local", action="store_true", required=False, default=False,
                        help="Build config locally without srun.")
    parser.add_argument("-s", "--submit", action="store_true", required=False, default=False,
                        help="Submit jobs immediately to SLURM queue.")
    parser.add_argument("-v", "--verbose", action="store_true", required=False, default=False,
                        help="Verbose output.")
    parser.add_argument("-q", "--quiet", action="store_true", required=False, default=False,
                        help="Suppress output.")
    parser.add_argument("-P", "--dopol", action="store_true", required=False, default=False,
                        help="Perform polarization calibration.")
    parser.add_argument("-2", "--do2GC", action="store_true", required=False, default=False,
                        help="Perform self-calibration (2GC).")
    parser.add_argument("-I", "--science_image", action="store_true", required=False, default=False,
                        help="Create a science image.")
    parser.add_argument("-x", "--nofields", action="store_true", required=False, default=False,
                        help="Do not read the input MS to extract field IDs.")
    parser.add_argument("-j", "--justrun", action="store_true", required=False, default=False,
                        help="Do not rebuild existing job scripts.")

    run_args = parser.add_mutually_exclusive_group(required=True)
    run_args.add_argument("-B", "--build", action="store_true", required=False, default=False,
                          help="Build config file using input MS.")
    run_args.add_argument("-R", "--run", action="store_true", required=False, default=False,
                          help="Run pipeline with input config file.")
    run_args.add_argument("-V", "--version", action="store_true", required=False, default=False,
                          help="Display the version of this pipeline and quit.")
    run_args.add_argument("-L", "--license", action="store_true", required=False, default=False,
                          help="Display this program's license and quit.")

    args, unknown = parser.parse_known_args()

    if unknown:
        parser.error('Unknown input argument(s) present - {0}'.format(unknown))

    if args.run:
        if args.config is None:
            parser.error("You must input a config file [--config] to run the pipeline.")
        if not os.path.exists(args.config):
            parser.error("Input config file '{0}' not found. Please set [-C --config] or write a new one with [-B --build].".format(args.config))

    # Remove default lists if user provided custom ones
    if len(args.scripts) > len(SCRIPTS):
        [args.scripts.pop(0) for _ in range(len(SCRIPTS))]
    if len(args.precal_scripts) > len(PRECAL_SCRIPTS):
        [args.precal_scripts.pop(0) for _ in range(len(PRECAL_SCRIPTS))]
    if len(args.postcal_scripts) > len(POSTCAL_SCRIPTS):
        [args.postcal_scripts.pop(0) for _ in range(len(POSTCAL_SCRIPTS))]
    if len(args.target_scripts) > len(TARGET_SCRIPTS):
        [args.target_scripts.pop(0) for _ in range(len(TARGET_SCRIPTS))]

    validate_args(vars(args), args.config, parser=parser)
    return args


# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

def setup_logger(config, verbose=False):
    if not verbose:
        config_dict = config_parser.parse_config(config)[0]
        if 'slurm' in config_dict and 'verbose' in config_dict['slurm']:
            verbose = config_dict['slurm']['verbose']
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    setup_logger(args.config, args.verbose)

    if args.version:
        logger.info('This is version {0}'.format(__version__))
    if args.license:
        logger.info(license)
    global _FACILITY
    if args.build:
        if args.facility:
            _FACILITY = get_facility(args.facility)
        if args.runner is not None:
            _FACILITY = get_facility(_FACILITY.name, default_runner=args.runner, default_container='')
            args.container = ''
        default_config(vars(args))
    if args.run:
        _FACILITY = load_facility_from_config(args.config)
        kwargs = format_args(args.config, args.submit, args.quiet, args.dependencies, args.justrun)
        write_jobs(args.config, **kwargs)


if __name__ == "__main__":
    main()
