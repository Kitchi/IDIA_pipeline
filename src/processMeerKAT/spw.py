"""SPW (spectral window) splitting and frequency-range utilities."""

import os
import re
import logging
from shutil import copyfile

from . import config_parser
from .constants import SPW_PREFIX

logger = logging.getLogger(__name__)


def linspace(lower, upper, length):
    """Basically np.linspace, but without needing to import numpy."""
    return [lower + x * (upper - lower) / float(length - 1) for x in range(length)]


def get_spw_bounds(spw):
    """Get upper and lower frequency bounds of a single SPW string.

    Parameters
    ----------
    spw : str
        CASA spectral window string, e.g. ``'*:880~1080MHz'``.

    Returns
    -------
    (low, high, unit, func) or None
        Returns None for comma-separated or malformed SPW strings.
    """
    bounds = spw.split(':')[-1].split('~')
    if ',' not in spw and ':' in spw and '~' in spw and len(bounds) == 2 and bounds[1] != '':
        high, unit = re.search(r'(\d+\.*\d*)(\w*)', bounds[1]).groups()
        func = int if unit == '' or '.' not in bounds[0] else float
        low = func(bounds[0])
        func = int if unit == '' or '.' not in high else float
        high = func(high)

        if unit != 'MHz':
            logger.warning(
                'Please use SPW unit "MHz" for best performance '
                '(e.g. not processing entirely flagged frequency ranges).'
            )
    else:
        return None

    return low, high, unit, func


def spw_split(spw, nspw, config, mem, badfreqranges, MS, partition,
              createmms=True, remove=True, fields={}):
    """Split into N SPWs, placing a pipeline instance into N directories.

    Each directory receives 1/N of the total bandwidth.

    Parameters
    ----------
    spw : str
        SPW string from config (single range or comma-separated).
    nspw : int
        Requested number of spectral windows.
    config : str
        Path to config file (relative to CWD).
    mem : int
        Memory in GB per pipeline instance.
    badfreqranges : list
        Bad frequency ranges to exclude.
    MS : str
        Path to input MeasurementSet.
    partition : bool
        Whether this run includes a partition step.
    createmms : bool
        Create MMS (True) or MS (False).
    remove : bool
        Remove SPWs completely inside a bad frequency range.
    fields : dict
        Field names for vis-renaming logic.

    Returns
    -------
    int
        Actual nspw after removing bad SPWs.
    """
    if get_spw_bounds(spw) is not None:
        low, high, unit, func = get_spw_bounds(spw)
        interval = func((high - low) / float(nspw))
        lo = linspace(low, high - interval, nspw)
        hi = linspace(low + interval, high, nspw)
        SPWs = [
            '{0}{1}~{2}{3}'.format(SPW_PREFIX, func(lo[i]), func(hi[i]), unit)
            for i in range(len(lo))
        ]

    elif ',' in spw:
        SPWs = spw.split(',')
        unit = get_spw_bounds(SPWs[0])[2]
        if len(SPWs) != nspw:
            logger.error(
                "nspw ({0}) not equal to number of separate SPWs ({1} in '{2}') "
                "from '{3}'. Setting to nspw={1}.".format(nspw, len(SPWs), spw, config)
            )
            nspw = len(SPWs)
    else:
        logger.error(
            "Can't split into {0} SPWs using SPW format '{1}'. "
            "Using nspw=1 in '{2}'.".format(nspw, spw, config)
        )
        return 1

    # Remove any SPWs completely encompassed by bad frequency ranges
    i = 0
    while i < nspw:
        badfreq = False
        low, high = get_spw_bounds(SPWs[i])[0:2]
        if unit == 'MHz' and remove:
            for freq in badfreqranges:
                bad_low, bad_high = get_spw_bounds('{0}{1}'.format(SPW_PREFIX, freq))[0:2]
                if low >= bad_low and high <= bad_high:
                    logger.info(
                        "Won't process spw '{0}{1}~{2}{3}', since it's completely "
                        "encompassed by bad frequency range '{3}'.".format(
                            SPW_PREFIX, low, high, unit, freq
                        )
                    )
                    badfreq = True
                    break
        if badfreq:
            SPWs.pop(i)
            i -= 1
            nspw -= 1
        i += 1

    config_parser.overwrite_config(
        config, conf_dict={'spw': ','.join(SPWs)}, conf_sec='crosscal'
    )

    logger.info(
        "Making {0} directories for SPWs ({1}) and copying '{2}' to each of them.".format(
            nspw, SPWs, config
        )
    )
    for spw in SPWs:
        # Dir name = SPW string minus any prefix selector. SPW_PREFIX is '*:'
        # (legacy linspace path); auto_detect_spw emits explicit IDs like '0:'
        # which we also strip so the dir name has no colons.
        spw_dir = re.sub(r'^(\*|\d+):', '', spw.replace(SPW_PREFIX, ''))
        spw_config = '{0}/{1}'.format(spw_dir, config)
        if not os.path.exists(spw_dir):
            os.mkdir(spw_dir)
        copyfile(config, spw_config)
        config_parser.overwrite_config(spw_config, conf_dict={'spw': spw}, conf_sec='crosscal')
        config_parser.overwrite_config(spw_config, conf_dict={'nspw': 1}, conf_sec='crosscal')
        config_parser.overwrite_config(spw_config, conf_dict={'mem': mem}, conf_sec='slurm')
        config_parser.overwrite_config(spw_config, conf_dict={'calcrefant': False}, conf_sec='crosscal')
        config_parser.overwrite_config(spw_config, conf_dict={'precal_scripts': []}, conf_sec='slurm')
        config_parser.overwrite_config(spw_config, conf_dict={'postcal_scripts': []}, conf_sec='slurm')

        # Look 1 directory up when using relative path
        if MS[0] != '/':
            config_parser.overwrite_config(
                spw_config, conf_dict={'vis': '../{0}'.format(MS)}, conf_sec='data'
            )

        if not partition:
            basename, ext = os.path.splitext(MS.rstrip('/ '))
            filebase = os.path.split(basename)[1]
            extn = 'mms' if createmms else 'ms'

            prefix, suffix = os.path.splitext(filebase)
            if suffix[1:] != '' and suffix[1:] in fields.values():
                extn = '{0}.{1}'.format(suffix[1:], extn)
                filebase = prefix

            vis = '{0}.{1}.{2}'.format(filebase, spw.replace(SPW_PREFIX, ''), extn)
            logger.warning(
                "Since script with 'partition' in its name isn't present in '{0}', "
                "assuming partition has already been done, and setting vis='{1}' in '{2}'. "
                "If '{1}' doesn't exist, please update '{2}'.".format(config, vis, spw_config)
            )
            orig_vis = config_parser.get_key(spw_config, 'data', 'vis')
            config_parser.overwrite_config(
                spw_config,
                conf_dict={'orig_vis': orig_vis},
                conf_sec='state',
            )
            config_parser.overwrite_config(
                spw_config, conf_dict={'vis': vis}, conf_sec='data'
            )

    return nspw
