"""Pipeline orchestration utilities: config validation, format_args, default_config."""

import os
import logging
from shutil import copyfile
from copy import deepcopy
from datetime import datetime

from . import config_parser
from . import bookkeeping
from .constants import (
    FIELDS_CONFIG_KEYS, CROSSCAL_CONFIG_KEYS, SELFCAL_CONFIG_KEYS,
    IMAGING_CONFIG_KEYS, SLURM_CONFIG_KEYS, SLURM_CONFIG_STR_KEYS,
    TMP_CONFIG, CONFIG, SCRIPT_DIR,
)
from .slurm_jobs import srun, write_command, write_jobs, check_path

logger = logging.getLogger(__name__)


def get_config_kwargs(config, section, expected_keys):
    """Return kwargs from a config section, validating expected keys exist."""
    config_dict = config_parser.parse_config(config)[0]

    if section not in config_dict:
        raise KeyError(
            "Config file '{0}' has no section [{1}]. "
            "Please insert section or build new config with [-B --build].".format(config, section)
        )

    kwargs = config_dict[section]

    unknown_keys = list(set(kwargs) - set(expected_keys))
    if unknown_keys:
        logger.warning("Unknown keys {0} present in section [{1}] in '{2}'.".format(
            unknown_keys, section, config
        ))

    missing_keys = list(set(expected_keys) - set(kwargs))
    if missing_keys:
        raise KeyError(
            "Keys {0} missing from section [{1}] in '{2}'. "
            "Please add these keywords to '{2}', or run [-B --build] step again.".format(
                missing_keys, section, config
            )
        )

    return kwargs


def get_slurm_dict(arg_dict, slurm_config_keys):
    """Extract SLURM-relevant keys from arg_dict for insertion into config."""
    return {key: arg_dict[key] for key in slurm_config_keys}


def pop_script(kwargs, script):
    """Remove a script from the parallel lists in kwargs. Returns True if popped."""
    if script in kwargs['scripts']:
        index = kwargs['scripts'].index(script)
        kwargs['scripts'].pop(index)
        kwargs['threadsafe'].pop(index)
        kwargs['containers'].pop(index)
        return True
    return False


def format_args(config, submit, quiet, dependencies, justrun):
    """Validate config and build kwargs dict for write_jobs()."""

    from .processMeerKAT import validate_args, _FACILITY
    from .spw import spw_split

    kwargs = get_config_kwargs(config, 'slurm', SLURM_CONFIG_KEYS)
    data_kwargs = get_config_kwargs(config, 'data', ['vis'])
    field_kwargs = get_config_kwargs(config, 'fields', FIELDS_CONFIG_KEYS)
    crosscal_kwargs = get_config_kwargs(config, 'crosscal', CROSSCAL_CONFIG_KEYS)

    if submit:
        kwargs['submit'] = True

    if type(crosscal_kwargs['nspw']) is not int:
        logger.warning("Argument 'nspw'={0} in '{1}' is not an integer. Will set to integer ({2}).".format(
            crosscal_kwargs['nspw'], config, int(crosscal_kwargs['nspw'])
        ))
        crosscal_kwargs['nspw'] = int(crosscal_kwargs['nspw'])

    spw = crosscal_kwargs['spw']
    nspw = crosscal_kwargs['nspw']
    mem = int(kwargs['mem'])

    if nspw > 1 and len(kwargs['scripts']) == 0:
        logger.warning(
            'Setting nspw=1, since "scripts" parameter in "{0}" is empty.'.format(config)
        )
        config_parser.overwrite_config(config, conf_dict={'nspw': 1}, conf_sec='crosscal')
        nspw = 1

    def _names(lst):
        return [s[0] for s in lst]

    if config_parser.has_section(config, 'selfcal') and (
        'selfcal_part1.py' in _names(kwargs['postcal_scripts'])
        or 'selfcal_part1.py' in _names(kwargs['scripts'])
    ):
        selfcal_kwargs = get_config_kwargs(config, 'selfcal', SELFCAL_CONFIG_KEYS)
        bookkeeping.get_selfcal_params(config)
        if selfcal_kwargs['loop'] > 0:
            logger.warning("Starting with loop={0}. Only valid if previous loops were run.".format(selfcal_kwargs['loop']))
        elif selfcal_kwargs['outlier_threshold'] != 0 and selfcal_kwargs['outlier_threshold'] != '':
            outlierfile = 'outliers.txt'
            outliers_loop0 = 'outliers_loop0.txt'
            CWD = os.path.split(os.getcwd())[1]
            if os.path.exists(outlierfile) and os.path.exists(outliers_loop0):
                logger.warning("Using existing outlier files from '{0}'.".format(CWD))
            elif os.path.exists('../{0}'.format(outlierfile)) and os.path.exists('../{0}'.format(outliers_loop0)):
                logger.warning("Copying outlier files from parent directory to '{0}'.".format(CWD))
                copyfile('../{0}'.format(outlierfile), outlierfile)
                copyfile('../{0}'.format(outliers_loop0), outliers_loop0)
            else:
                txt = ('within {0} degrees'.format(selfcal_kwargs['outlier_radius'])
                       if selfcal_kwargs['outlier_radius'] not in ('', 0.0)
                       else 'within calculated search radius')
                logger.info('Populating sky model for selfcal using outlier_threshold={0} {1}'.format(
                    selfcal_kwargs['outlier_threshold'], txt
                ))
                sky_model_kwargs = deepcopy(kwargs)
                sky_model_kwargs['partition'] = 'Devel'
                mpi_wrapper = srun(sky_model_kwargs, qos=True, time=2, mem=0)
                command = write_command('set_sky_model.py', '-C {0}'.format(config),
                                        mpi_wrapper=mpi_wrapper, container=kwargs['container'],
                                        default_runner=getattr(_FACILITY, 'default_runner', ''),
                                        logfile=False)
                logger.debug('Running: {0}'.format(command))
                os.system(command)

    if config_parser.has_section(config, 'image'):
        imaging_kwargs = get_config_kwargs(config, 'image', IMAGING_CONFIG_KEYS)
        valid_pbbands = ['LBand', 'SBand', 'UHF']
        if not any(pb.lower() in imaging_kwargs['pbband'].lower() for pb in valid_pbbands):
            logger.warning('Invalid pbband found. Must be one of {}.'.format(valid_pbbands))

    target_scripts = kwargs.get('target_scripts', []) or []

    if nspw == 1:
        if len(kwargs['precal_scripts']) > 0 or len(kwargs['postcal_scripts']) > 0:
            logger.warning('Appending pre/postcal_scripts to scripts since nspw=1.')
            if ('calc_refant.py' in _names(kwargs['precal_scripts'])
                    and 'calc_refant.py' in _names(kwargs['scripts'])):
                kwargs['precal_scripts'].pop(_names(kwargs['precal_scripts']).index('calc_refant.py'))

            scripts = kwargs['precal_scripts'] + kwargs['scripts'] + kwargs['postcal_scripts']
            config_parser.overwrite_config(config, conf_dict={'scripts': scripts}, conf_sec='slurm')
            config_parser.overwrite_config(config, conf_dict={'precal_scripts': []}, conf_sec='slurm')
            config_parser.overwrite_config(config, conf_dict={'postcal_scripts': []}, conf_sec='slurm')
            kwargs = get_config_kwargs(config, 'slurm', SLURM_CONFIG_KEYS)
        else:
            scripts = kwargs['scripts']
        # nspw == 1: no parallel target branch — there's a single MS containing everything.
        target_scripts = []
    else:
        scripts = kwargs['precal_scripts'] + kwargs['postcal_scripts']

    kwargs['num_precal_scripts'] = len(kwargs['precal_scripts'])
    kwargs['MS'] = data_kwargs['vis']
    validate_args(kwargs, config)

    # Facility validation runs only at -R time (requires SLURM node + sacctmgr/scontrol)
    kwargs['account'] = _FACILITY.validate_account(kwargs.get('account'), config)
    _FACILITY.validate_reservation(kwargs.get('reservation', ''), kwargs, config)

    kwargs['scripts'] = [check_path(i[0]) for i in scripts]
    kwargs['threadsafe'] = [i[1] for i in scripts]
    kwargs['containers'] = [check_path(i[2]) for i in scripts]
    kwargs['target_scripts'] = [
        (check_path(i[0]), i[1], check_path(i[2])) for i in target_scripts
    ]

    if not crosscal_kwargs['createmms']:
        logger.info("'createmms = False' in '{0}', forcing 'keepmms = False'.".format(config))
        config_parser.overwrite_config(config, conf_dict={'keepmms': False}, conf_sec='crosscal')
        kwargs['threadsafe'] = [False] * len(scripts)
    elif not crosscal_kwargs['keepmms']:
        if 'split.py' in kwargs['scripts']:
            kwargs['threadsafe'][kwargs['scripts'].index('split.py')] = False
        if nspw != 1:
            kwargs['threadsafe'][kwargs['num_precal_scripts']:] = [False] * len(kwargs['postcal_scripts'])

    for threadsafe_script in ['quick_tclean.py', 'selfcal_part1.py', 'science_image.py']:
        if threadsafe_script in kwargs['scripts']:
            kwargs['threadsafe'][kwargs['scripts'].index(threadsafe_script)] = True

    if kwargs['ntasks_per_node'] < _FACILITY.cpus_per_node_limit and nspw > 1:
        mem = int(mem // (nspw / 2))

    dopol = config_parser.get_key(config, 'run', 'dopol')
    if not dopol and ('xy_yx_solve.py' in kwargs['scripts'] or 'xy_yx_apply.py' in kwargs['scripts']):
        logger.warning("Cross-hand calibration scripts found. Forcing dopol=True.")
        config_parser.overwrite_config(
            config, conf_dict={'dopol': True}, conf_sec='run',
            sec_comment='# Internal variables for pipeline execution'
        )

    includes_partition = any('partition' in script for script in kwargs['scripts'])
    if nspw > 1:
        kwargs['timestamp'] = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        config_parser.overwrite_config(
            config,
            conf_dict={'timestamp': "'{0}'".format(kwargs['timestamp'])},
            conf_sec='run',
            sec_comment='# Internal variables for pipeline execution',
        )
        nspw = spw_split(spw, nspw, config, mem, crosscal_kwargs['badfreqranges'],
                         kwargs['MS'], includes_partition,
                         createmms=crosscal_kwargs['createmms'], fields=field_kwargs)
        config_parser.overwrite_config(config, conf_dict={'nspw': "{0}".format(nspw)}, conf_sec='crosscal')

    if not crosscal_kwargs['calcrefant']:
        if pop_script(kwargs, 'calc_refant.py'):
            kwargs['num_precal_scripts'] -= 1

    for i in range(len(kwargs['containers'])):
        if kwargs['containers'][i] == '':
            kwargs['containers'][i] = kwargs['container']
    kwargs.pop('container')
    kwargs.pop('MS')
    kwargs.pop('precal_scripts')
    kwargs.pop('postcal_scripts')
    kwargs['quiet'] = quiet
    kwargs['justrun'] = justrun

    if dependencies != '':
        kwargs['dependencies'] = dependencies

    if len(kwargs['scripts']) == 0 and nspw == 1:
        logger.error('Nothing to do. Please insert scripts into "scripts" in "{0}".'.format(config))

    logger.debug("Copying '{0}' to '{1}'.".format(config, TMP_CONFIG))
    copyfile(config, TMP_CONFIG)
    if not quiet:
        logger.warning("Changing [slurm] section in your config will have no effect unless you [-R --run] again.")

    return kwargs


def default_config(arg_dict):
    """Generate default config file from MS metadata."""

    filename = arg_dict['config']
    MS = arg_dict['MS']

    copyfile('{0}/{1}'.format(SCRIPT_DIR, CONFIG), filename)

    slurm_dict = get_slurm_dict(arg_dict, SLURM_CONFIG_KEYS)
    for key in SLURM_CONFIG_STR_KEYS:
        if key in slurm_dict:
            slurm_dict[key] = "'{0}'".format(slurm_dict[key])

    config_parser.overwrite_config(filename, conf_dict=slurm_dict, conf_sec='slurm')
    config_parser.overwrite_config(filename, conf_dict={'vis': "'{0}'".format(MS)}, conf_sec='data')

    from .processMeerKAT import _FACILITY
    config_parser.overwrite_config(
        filename,
        conf_dict={'name': "'{0}'".format(_FACILITY.name)},
        conf_sec='facility',
        sec_comment=(
            '# Known facilities: ilifu, generic_slurm\n'
            '# For a known facility, defaults are pre-validated — override only what differs.'
        ),
    )
    config_parser.overwrite_config(
        filename,
        conf_dict={'dopol': arg_dict['dopol']},
        conf_sec='run',
        sec_comment='# Internal variables for pipeline execution',
    )

    if not arg_dict['do2GC'] or not arg_dict['science_image']:
        remove_scripts = []
        if not arg_dict['do2GC']:
            config_parser.remove_section(filename, 'selfcal')
            remove_scripts = ['selfcal_part1.py', 'selfcal_part2.py']
        if not arg_dict['science_image']:
            config_parser.remove_section(filename, 'image')
            remove_scripts += ['science_image.py']

        scripts = [s for s in arg_dict['postcal_scripts'] if s[0] not in remove_scripts]
        config_parser.overwrite_config(filename, conf_dict={'postcal_scripts': scripts}, conf_sec='slurm')

    if not arg_dict['nofields']:
        if arg_dict['local']:
            mpi_wrapper = ''
        else:
            mpi_wrapper = srun(arg_dict)

        params = '-B -M {MS} -C {config} -N {nodes} -t {ntasks_per_node}'.format(**arg_dict)
        if arg_dict['dopol']:
            params += ' -P'
        if arg_dict['verbose']:
            params += ' -v'
        command = write_command('read_ms.py', params, mpi_wrapper=mpi_wrapper,
                                container=arg_dict['container'],
                                default_runner=getattr(_FACILITY, 'default_runner', ''),
                                logfile=False)
        logger.info('Extracting field IDs from "{0}" using CASA.'.format(MS))
        logger.debug('Using command:\n\t{0}'.format(command))
        os.system(command)
    else:
        logger.info('Skipping extraction of field IDs and assuming nspw=1.')
        config_parser.overwrite_config(filename, conf_dict={'nspw': 1}, conf_sec='crosscal')

    dopol = config_parser.get_key(filename, 'run', 'dopol')
    if dopol:
        count = 0
        for ind, ss in enumerate(arg_dict['scripts']):
            if ss[0] in ('xx_yy_solve.py', 'xx_yy_apply.py'):
                count += 1
            if count > 2:
                if ss[0] == 'xx_yy_solve.py':
                    arg_dict['scripts'][ind] = ('xy_yx_solve.py', arg_dict['scripts'][ind][1], arg_dict['scripts'][ind][2])
                if ss[0] == 'xx_yy_apply.py':
                    arg_dict['scripts'][ind] = ('xy_yx_apply.py', arg_dict['scripts'][ind][1], arg_dict['scripts'][ind][2])
        config_parser.overwrite_config(filename, conf_dict={'scripts': arg_dict['scripts']}, conf_sec='slurm')

    logger.info('Config "{0}" generated.'.format(filename))
