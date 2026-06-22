#Copyright (C) 2022 Inter-University Institute for Data Intensive Astronomy
#See processMeerKAT.py for license details.

"""
Split the target field(s) out of the input MS into a per-SPW target MMS.

Runs *inside each per-SPW directory* (as a job array, mirroring partition.py),
splitting the target field of the original input MS down to this directory's
single spectral-window frequency range. The resulting per-SPW target carries a
single SPW (id 0), aligned to that directory's calibrator caltables, so
``apply_to_target.py`` can index-match caltable SPW 0 → target SPW 0.

Each SPW directory therefore ends up with its own calibrator MMS (from
partition.py) *and* its own target MMS, letting the target flag + apply chain
run per-SPW in parallel. The corrected per-SPW targets are recombined along
frequency at the top level afterwards.
"""
import sys
import os

from .. import processMeerKAT
from .. import config_parser
from ..config_parser import typed_get
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


def do_partition_target(visname, target_field, spw, preavg, include_crosshand, createmms, spwname):
    basename, ext = os.path.splitext(visname)
    filebase = os.path.split(basename)[1]
    extn = 'mms' if createmms else 'ms'

    target_vis = '{0}.{1}.target.{2}'.format(filebase, spwname, extn)
    nscan = msmd.nscans() if createmms else 1
    numsubms = min(nscan, MAX_TARGET_NUMSUBMS) if createmms else 1
    chanaverage = preavg > 1
    correlation = '' if include_crosshand else 'XX,YY'

    mstransform(
        vis=visname, outputvis=target_vis, field=target_field, spw=spw,
        createmms=createmms, datacolumn='DATA',
        chanaverage=chanaverage, chanbin=preavg,
        numsubms=numsubms, separationaxis='scan',
        keepflags=True, usewtspectrum=True, antenna='*&',
        correlation=correlation,
    )
    return target_vis


def main(ctx):
    # By the time this runs (per-SPW postcal sub-chain), partition.py has set
    # [data].vis to the calibrator-only MMS and stamped the original input MS
    # into [state].orig_vis. We split the target from the original MS, then
    # switch [data].vis to the new per-SPW target so the subsequent flag steps
    # operate on the target. The calibrator MMS name is preserved in
    # [state].cal_vis so apply_to_target can still locate the caltables.
    cal_vis = typed_get(ctx.config, 'state', 'cal_vis', str, default='') \
        or typed_get(ctx.config, 'data', 'vis', str)
    orig_vis = typed_get(ctx.config, 'state', 'orig_vis', str, default='')
    visname = orig_vis or cal_vis

    spw = typed_get(ctx.config, 'crosscal', 'spw', str, default='')
    preavg = typed_get(ctx.config, 'crosscal', 'chanbin', int, default=1)
    include_crosshand = typed_get(ctx.config, 'state', 'dopol', bool, default=False)
    createmms = typed_get(ctx.config, 'crosscal', 'createmms', bool, default=True)

    if ',' in spw:
        raise RuntimeError(
            "partition_target.py expects a single per-SPW selection in [crosscal].spw, "
            "got a multi-SPW list '{0}'. It must run inside a per-SPW directory.".format(spw)
        )
    spwname = spw.replace('*:', '')

    casalog.setlogfile('logs/{SLURM_JOB_NAME}-{SLURM_JOB_ID}.casa'.format(**os.environ))

    target_field = target_field_selection(ctx.config.get('fields', {}))

    msmd.open(visname)
    target_vis = do_partition_target(
        visname, target_field, spw, preavg, include_crosshand, createmms, spwname
    )
    msmd.done()

    # Preserve the calibrator MMS name for apply_to_target, record the target,
    # and switch [data].vis to the target so the flag steps run against it.
    config_parser.overwrite_config(
        ctx.config_path,
        conf_sec='state',
        conf_dict={'cal_vis': cal_vis, 'target_vis': target_vis},
    )
    config_parser.overwrite_config(
        ctx.config_path,
        conf_sec='data',
        conf_dict={'vis': target_vis},
    )


if __name__ == '__main__':
    bookkeeping.run_script(main, logfile)
