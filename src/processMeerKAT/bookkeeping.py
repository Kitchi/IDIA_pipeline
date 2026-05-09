#Copyright (C) 2022 Inter-University Institute for Data Intensive Astronomy
#See processMeerKAT.py for license details.

"""Pipeline bookkeeping: cal table paths, field IDs, script runner, selfcal/imaging helpers."""

import sys
import os
import glob
import re
import traceback
import argparse
from collections import namedtuple

from . import config_parser
from .config_parser import typed_get

import logging
from time import gmtime
logging.Formatter.converter = gmtime
logger = logging.getLogger(__name__)

Calfiles = namedtuple('calfiles', [
    'kcorrfile', 'bpassfile', 'gainfile', 'dpolfile',
    'xpolfile', 'xdelfile', 'fluxfile',
])

FieldIDs = namedtuple('FieldIDs', [
    'targetfield', 'fluxfield', 'bpassfield', 'secondaryfield',
    'kcorrfield', 'xdelfield', 'dpolfield', 'xpolfield',
    'gainfields', 'extrafields',
])


# ---------------------------------------------------------------------------
# Cal table paths
# ---------------------------------------------------------------------------

def get_calfiles(visname, caldir):
    base = os.path.splitext(visname)[0]
    return Calfiles(
        kcorrfile=os.path.join(caldir, base + '.kcal'),
        bpassfile=os.path.join(caldir, base + '.bcal'),
        gainfile=os.path.join(caldir, base + '.gcal'),
        dpolfile=os.path.join(caldir, base + '.pcal'),
        xpolfile=os.path.join(caldir, base + '.xcal'),
        xdelfile=os.path.join(caldir, base + '.xdel'),
        fluxfile=os.path.join(caldir, base + '.fluxscale'),
    )


def bookkeeping(visname):
    caldir = os.path.join(os.getcwd(), 'caltables')
    return get_calfiles(visname, caldir), caldir


def get_all_spw_caldirs(top_dir, config_name='default_config.toml'):
    """Discover the per-SPW caldirs sitting under a top-level run directory.

    Used by postcal scripts (concat_caltables, apply_to_target) to find every
    per-SPW pipeline subdir, read its global SPW ID from its config, and
    locate its ``caltables/`` directory.

    Parameters
    ----------
    top_dir : str
        Top-level pipeline directory (the parent of the per-SPW subdirs).
    config_name : str
        Filename of the per-SPW config file (defaults to ``default_config.toml``).

    Returns
    -------
    list[dict]
        One entry per discovered SPW, sorted by global SPW ID. Each entry has:
          - ``dir``: absolute path to the SPW subdir
          - ``caldir``: absolute path to the SPW's caltables/ directory
          - ``spw_id``: the global SPW ID parsed from ``[crosscal] spw``
          - ``spw_string``: the raw SPW string from the per-SPW config
    """
    from . import config_parser as _cp

    entries = []
    for name in sorted(os.listdir(top_dir)):
        spw_dir = os.path.join(top_dir, name)
        spw_config = os.path.join(spw_dir, config_name)
        if not os.path.isdir(spw_dir) or not os.path.isfile(spw_config):
            continue
        spw_string = _cp.get_key(spw_config, 'crosscal', 'spw')
        if not isinstance(spw_string, str):
            continue
        spw_id = _parse_global_spw_id(spw_string)
        if spw_id is None:
            continue
        entries.append({
            'dir': spw_dir,
            'caldir': os.path.join(spw_dir, 'caltables'),
            'spw_id': spw_id,
            'spw_string': spw_string,
        })

    entries.sort(key=lambda e: e['spw_id'])
    return entries


def _parse_global_spw_id(spw_string):
    """Extract the global SPW ID from an SPW selection string like '0:880~933MHz'.

    Returns None if the string has no explicit numeric ID prefix (e.g. '*:..').
    """
    head = spw_string.split(':', 1)[0].strip().strip("'\"")
    if head.isdigit():
        return int(head)
    return None


# ---------------------------------------------------------------------------
# Field IDs
# ---------------------------------------------------------------------------

def get_field_ids(fields):
    """Build a FieldIDs namedtuple from the [fields] config dict."""
    targetfield    = fields['targetfields']
    extrafields    = fields['extrafields']
    fluxfield      = fields['fluxfield']
    bpassfield     = fields['bpassfield']
    secondaryfield = fields['phasecalfield']

    gainfields = (
        str(fluxfield) + ',' + str(secondaryfield)
        if fluxfield != secondaryfield
        else str(fluxfield)
    )

    return FieldIDs(
        targetfield=targetfield,
        fluxfield=fluxfield,
        bpassfield=bpassfield,
        secondaryfield=secondaryfield,
        kcorrfield=secondaryfield,
        xdelfield=secondaryfield,
        dpolfield=secondaryfield,
        xpolfield=secondaryfield,
        gainfields=gainfields,
        extrafields=extrafields,
    )


# ---------------------------------------------------------------------------
# Polarization field detection (requires CASA)
# ---------------------------------------------------------------------------

def polfield_name(visname):
    from casatools import msmetadata
    msmd = msmetadata()
    msmd.open(visname)
    fieldnames = msmd.fieldnames()
    msmd.done()

    candidates = [
        ["3C286", "1328+307", "1331+305", "J1331+3030"],
        ["3C138", "0518+165", "0521+166", "J0521+1638"],
        ["3C48",  "0134+329", "0137+331", "J0137+3309"],
    ]
    for group in candidates:
        match = set(group) & set(fieldnames)
        if match:
            return match.pop()

    if "J1130-1449" in fieldnames:
        return "J1130-1449"

    logger.warning(
        "No valid polarization field found. "
        "Defaulting to the phase calibrator for XY phase — check results carefully."
    )
    return ''


# ---------------------------------------------------------------------------
# File existence check
# ---------------------------------------------------------------------------

def check_file(filepath):
    if not os.path.exists(filepath):
        logger.error(
            'Calibration table "%s" was not written. '
            'Check the CASA output and whether a solution was found.', filepath
        )
        raise FileNotFoundError(filepath)
    logger.info('Calibration table "%s" successfully written.', filepath)


# ---------------------------------------------------------------------------
# Log renaming
# ---------------------------------------------------------------------------

def rename_logs(logfile=''):
    if not logfile or not os.path.exists(logfile):
        return
    if 'SLURM_ARRAY_JOB_ID' in os.environ:
        IDs = '{SLURM_JOB_NAME}-{SLURM_ARRAY_JOB_ID}_{SLURM_ARRAY_TASK_ID}'.format(**os.environ)
    else:
        IDs = '{SLURM_JOB_NAME}-{SLURM_JOB_ID}'.format(**os.environ)
    os.rename(logfile, 'logs/{0}.mpi'.format(IDs))
    for log in glob.glob('*.last'):
        os.rename(log, 'logs/{0}-{1}.last'.format(os.path.splitext(log)[0], IDs))


# ---------------------------------------------------------------------------
# Script runner — entry point for all SLURM calibration jobs
# ---------------------------------------------------------------------------

def _parse_script_config():
    """Parse -C/--config from the calibration script's argv."""
    from .constants import CONFIG
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-C', '--config', default=CONFIG)
    args, _ = parser.parse_known_args()
    return args.config


def run_script(func, logfile=''):
    """Bootstrap wrapper for calibration scripts running as SLURM jobs.

    Reads the config path from -C/--config in sys.argv, parses the config,
    checks the 'continue' flag, calls func(args, taskvals), and handles
    errors by setting continue=False and exiting non-zero.
    """
    config_path = _parse_script_config()
    taskvals, _ = config_parser.parse_config(config_path)

    continue_run = typed_get(taskvals, 'run', 'continue', bool, default=True)
    spw          = typed_get(taskvals, 'crosscal', 'spw', str)
    nspw         = typed_get(taskvals, 'crosscal', 'nspw', int)

    args = {'config': config_path}

    if not continue_run:
        script = os.path.split(sys.argv[0])[1] if sys.argv else '?'
        logger.error(
            'A previous job set continue=False in "%s". Skipping "%s".',
            config_path, script,
        )
        rename_logs(logfile)
        sys.exit(1)

    try:
        func(args, taskvals)
        rename_logs(logfile)
    except Exception as err:
        logger.error('Exception in pipeline (%s): %s', type(err).__name__, err)
        logger.error(traceback.format_exc())
        config_parser.overwrite_config(
            config_path,
            conf_dict={'continue': False},
            conf_sec='run',
            sec_comment='# Internal variables for pipeline execution',
        )
        if nspw > 1:
            for SPW in spw.split(','):
                spw_config = '{0}/{1}'.format(SPW.replace('*:', ''), config_path)
                config_parser.overwrite_config(
                    spw_config,
                    conf_dict={'continue': False},
                    conf_sec='run',
                    sec_comment='# Internal variables for pipeline execution',
                )
        rename_logs(logfile)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Selfcal parameter loading and validation
# ---------------------------------------------------------------------------

def get_selfcal_params(config_path=None):
    """Load and validate selfcal parameters from config.

    Parameters
    ----------
    config_path : str, optional
        Path to config file.  If omitted, read from -C/--config in sys.argv.

    Returns
    -------
    (args_dict, params_dict)
    """
    if config_path is None:
        config_path = _parse_script_config()

    taskvals, _ = config_parser.parse_config(config_path)
    params = dict(taskvals['selfcal'])
    other_params = list(params.keys())

    params['vis']    = taskvals['data']['vis']
    params['refant'] = taskvals['crosscal']['refant']
    params['dopol']  = taskvals['run']['dopol']

    if params['dopol'] and 'G' in params.get('gaintype', ''):
        logger.warning(
            "dopol=True but gaintype includes 'G'. "
            "Use gaintype='T' for linear feeds (e.g. MeerKAT)."
        )

    single_args  = ['nloops', 'loop', 'discard_nloops', 'outlier_threshold', 'outlier_radius']
    gaincal_args = ['solint', 'calmode', 'gaintype', 'flag']
    list_args    = ['imsize']
    errors = []

    for arg in single_args:
        other_params = [p for p in other_params if p != arg]

    for arg in single_args:
        if isinstance(params[arg], list) or (isinstance(params[arg], str) and ',' in params[arg]):
            errors.append(
                f"'{arg}' must be a single value, not a list."
            )

    for arg in other_params:
        if isinstance(params[arg], str) and ',' in params[arg]:
            errors.append(
                f"'{arg}' cannot use comma-separated values; use a Python list."
            )

        if arg in list_args:
            if isinstance(params[arg], list):
                if len(params[arg]) == 0 or not isinstance(params[arg][0], list):
                    params[arg] = [params[arg]] * (params['nloops'] + 1)
                elif len(params[arg]) == 1:
                    params[arg] = [params[arg][0]] * (params['nloops'] + 1)
            else:
                params[arg] = [[params[arg]]] * (params['nloops'] + 1)
        elif not isinstance(params[arg], list):
            n = params['nloops'] if arg in gaincal_args else params['nloops'] + 1
            params[arg] = [params[arg]] * n

    for arg in other_params:
        expected = params['nloops'] if arg in gaincal_args else params['nloops'] + 1
        if len(params[arg]) != expected:
            errors.append(
                f"'{arg}' is length {len(params[arg])} but must be "
                f"{'nloops' if arg in gaincal_args else 'nloops+1'} = {expected}."
            )

    if errors:
        for msg in errors:
            logger.error("Selfcal config error in '%s': %s", config_path, msg)
        sys.exit(1)

    return {'config': config_path}, params


def get_imaging_params(config_path=None):
    """Load imaging parameters from config.

    Parameters
    ----------
    config_path : str, optional
        Path to config file.  If omitted, read from -C/--config in sys.argv.
    """
    if config_path is None:
        config_path = _parse_script_config()

    taskvals, _ = config_parser.parse_config(config_path)
    params = dict(taskvals['image'])
    params['vis']     = taskvals['data']['vis']
    params['keepmms'] = taskvals['crosscal']['keepmms']

    if params.get('outlierfile') and os.path.exists(params['outlierfile']):
        outliers = open(params['outlierfile']).read()
        for name in re.findall(r'imagename=(.*)\n', outliers):
            mask = '{0}.mask'.format(name)
            if os.path.exists(mask):
                newname = '{0}.old'.format(mask)
                logger.info(
                    'Re-using old mask for "%s". Renaming "%s" → "%s".',
                    name, mask, newname,
                )
                os.rename(mask, newname)

    return {'config': config_path}, params


# ---------------------------------------------------------------------------
# Selfcal runtime argument builder (requires CASA)
# ---------------------------------------------------------------------------

def get_selfcal_args(vis, loop, nloops, nterms, deconvolver, discard_nloops,
                     calmode, outlier_threshold, outlier_radius, threshold, step):
    from casatools import msmetadata, quanta
    from read_ms import check_spw
    msmd = msmetadata()
    qa = quanta()

    tmpvis = glob.glob('{0}/SUBMSS/*'.format(vis))[0] if os.path.exists('{0}/SUBMSS'.format(vis)) else vis
    msmd.open(tmpvis)

    visbase = os.path.split(vis.rstrip('/ '))[1]
    visbase = re.sub(r'\.\d+\.*\d*\~\d+\.*\d*[a-z,A-Z]?[Hz,hz,hZ,HZ]*\.', '.', visbase)

    config_path = _parse_script_config()
    targetfields = config_parser.get_key(config_path, 'fields', 'targetfields')

    if isinstance(targetfields, str) and ',' in targetfields:
        targetfield = targetfields.split(',')[0]
        logger.warning(
            'Multiple target fields ("%s"); using "%s" for outlier identification.',
            targetfields, targetfield,
        )
    else:
        targetfield = targetfields

    try:
        targetfield = int(targetfield)
    except (ValueError, TypeError):
        targetfield = msmd.fieldsforname(targetfield)[0]

    target_str = msmd.namesforfields(targetfield)[0]

    if '.ms' in visbase and target_str not in visbase:
        basename = visbase.replace('.ms', '.{0}'.format(target_str))
    else:
        basename = visbase.replace('.mms', '')

    imbase     = basename + '_im_%d'
    imagename  = imbase % loop
    outimage   = imagename + '.image'
    pixmask    = imagename + '.pixmask'
    maskfile   = imagename + '.islmask'
    rmsfile    = imagename + '.rms'
    caltable   = basename + '.gcal%d' % loop
    prev_caltables = sorted(glob.glob('*.gcal?'))
    cfcache    = basename + '.cf'
    thresh     = 10

    if deconvolver[loop] == 'mtmfs':
        outimage += '.tt0'

    if step not in ['tclean', 'sky'] and not os.path.exists(outimage):
        logger.error(
            "Image '%s' doesn't exist — selfcal loop %d failed. Terminating.", outimage, loop
        )
        sys.exit(1)

    if step in ['tclean', 'predict']:
        pixmask = imbase % (loop - 1) + '.pixmask'
        rmsfile = imbase % (loop - 1) + '.rms'
    if step in ['tclean', 'predict', 'sky'] and (
        (loop == 0 and not os.path.exists(pixmask))
        or (0 < loop < nloops and calmode[loop] == '')
    ):
        pixmask = ''

    for i in range(loop):
        if calmode[i] != '' and not os.path.exists(basename + '.gcal%d' % i):
            logger.error(
                "Cal table '%s' missing — selfcal loop %d failed. Terminating.",
                basename + '.gcal%d' % i, i,
            )
            sys.exit(1)
    for _ in range(discard_nloops):
        prev_caltables.pop(0)

    outlierfile = ''
    sky_model_radius = 0.0
    if outlier_threshold not in ('', 0):
        if step in ['tclean', 'predict', 'sky']:
            outlierfile = 'outliers_loop{0}.txt'.format(loop)
        else:
            outlierfile = 'outliers_loop{0}.txt'.format(loop + 1)

        if (outlier_radius in (0.0, '') and step == 'sky'):
            SPW = check_spw(config_path, msmd)
            low_freq = float(SPW.replace('*:', '').split('~')[0]) * 1e6
            rads = (1.025 * qa.constants(v='c')['value'] / low_freq
                    / msmd.antennadiameter()['0']['value'])
            FWHM = qa.convert(qa.quantity(rads, 'rad'), 'deg')['value']
            sky_model_radius = 1.5 * FWHM
            logger.warning('Using calculated search radius of %.1f degrees.', sky_model_radius)
        else:
            if step == 'sky':
                logger.info('Using preset search radius of %s degrees.', outlier_radius)
            sky_model_radius = outlier_radius

    msmd.done()

    if not (isinstance(threshold[loop], str) and 'Jy' in threshold[loop]) and threshold[loop] > 1:
        if step in ['tclean', 'predict']:
            if os.path.exists(rmsfile):
                from casatasks import imstat
                stats = imstat(imagename=rmsfile)
                threshold[loop] *= stats['min'][0]
            else:
                logger.error(
                    "'%s' doesn't exist — can't threshold at S/N>%s. "
                    "Loop 0 must use an absolute threshold. Check logs.",
                    rmsfile, threshold[loop],
                )
                sys.exit(1)
        elif step == 'bdsf':
            thresh = threshold[loop]

    return (imbase, imagename, outimage, pixmask, rmsfile,
            caltable, prev_caltables, threshold, outlierfile,
            cfcache, thresh, maskfile, targetfield, sky_model_radius)
