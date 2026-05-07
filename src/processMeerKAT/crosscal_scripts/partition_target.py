#Copyright (C) 2022 Inter-University Institute for Data Intensive Astronomy
#See processMeerKAT.py for license details.

"""
Split the target field(s) out of the input MS into a single monolithic MMS
spanning all SPWs.

Runs once at top level (no SPW job-array) so the target flag chain can proceed
in parallel with the per-SPW calibrator solve chains. The MMS layout
(``createmms=True``, ``numsubms`` capped) lets downstream ``flagdata`` and
``applycal`` exploit MPI parallelism.
"""
import sys
import os

from .. import config_parser
from ..config_parser import validate_args as va
from .. import processMeerKAT
from .. import bookkeeping
from .partition import _strip_quotes

from casatasks import *
logfile = casalog.logfile()
from casatools import msmetadata
import casampi
msmd = msmetadata()

# Cap the per-target subms count so a small input MS (few scans) doesn't end up
# with trivially-tiny subms that hurt more than help.
MAX_TARGET_NUMSUBMS = 8


def target_field_selection(fields_section):
    raw = _strip_quotes(fields_section.get('targetfields', ''))
    if not raw:
        raise ValueError("No targetfields entry in [fields] section of config — partition_target.py has nothing to split.")
    return str(raw)


def do_partition_target(visname, target_field, preavg, include_crosshand, createmms):
    basename, ext = os.path.splitext(visname)
    filebase = os.path.split(basename)[1]
    extn = 'mms' if createmms else 'ms'

    target_vis = '{0}.target.{1}'.format(filebase, extn)
    nscan = msmd.nscans() if createmms else 1
    numsubms = min(nscan, MAX_TARGET_NUMSUBMS) if createmms else 1
    chanaverage = preavg > 1
    correlation = '' if include_crosshand else 'XX,YY'

    mstransform(
        vis=visname, outputvis=target_vis, field=target_field, spw='',
        createmms=createmms, datacolumn='DATA',
        chanaverage=chanaverage, chanbin=preavg,
        numsubms=numsubms, separationaxis='scan',
        keepflags=True, usewtspectrum=True, antenna='*&',
        correlation=correlation,
    )
    return target_vis


def main(args, taskvals):
    visname = va(taskvals, 'data', 'vis', str)
    preavg = va(taskvals, 'crosscal', 'chanbin', int, default=1)
    include_crosshand = va(taskvals, 'run', 'dopol', bool, default=False)
    createmms = va(taskvals, 'crosscal', 'createmms', bool, default=True)

    casalog.setlogfile('logs/{SLURM_JOB_NAME}-{SLURM_JOB_ID}.casa'.format(**os.environ))

    target_field = target_field_selection(taskvals.get('fields', {}))

    msmd.open(visname)
    target_vis = do_partition_target(visname, target_field, preavg, include_crosshand, createmms)
    msmd.done()

    config_parser.overwrite_config(
        args['config'],
        conf_sec='run',
        sec_comment='# Internal variables for pipeline execution',
        conf_dict={'target_vis': "'{0}'".format(target_vis)},
    )

    # Also stamp out a sibling config file used by the target branch sbatches.
    # We copy the main config and override [data].vis so target scripts run
    # against the target MMS rather than the cal MMS.
    import shutil
    main_cfg = args['config']
    target_cfg = os.path.join(os.path.dirname(main_cfg) or '.', 'target_config.txt')
    shutil.copyfile(main_cfg, target_cfg)
    config_parser.overwrite_config(
        target_cfg,
        conf_sec='data',
        conf_dict={'vis': "'{0}'".format(target_vis)},
    )


if __name__ == '__main__':
    bookkeeping.run_script(main, logfile)
