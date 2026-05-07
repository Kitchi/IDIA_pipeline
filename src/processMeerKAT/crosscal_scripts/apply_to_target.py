#Copyright (C) 2022 Inter-University Institute for Data Intensive Astronomy
#See processMeerKAT.py for license details.

"""
Apply the combined per-SPW caltables to the monolithic target MMS.

Runs once at top level (postcal) after ``concat_caltables.py`` has joined all
per-SPW caltables. Reads the target MMS path from ``[run] target_vis`` and
the combined caltable paths from ``[run] combined_*``, then runs a single
``applycal`` so the target MMS gets a populated ``CORRECTED_DATA`` column
ready for selfcal / science imaging.
"""
import os
import sys

from .. import config_parser
from ..config_parser import validate_args as va
from .. import bookkeeping
from .. import processMeerKAT

from casatasks import *
logfile = casalog.logfile()
import casampi


def _strip(s):
    if isinstance(s, str):
        return s.strip().strip("'").strip('"')
    return s


def main(args, taskvals):
    run = taskvals.get('run', {})
    fields = bookkeeping.get_field_ids(taskvals['fields'])

    target_vis = _strip(run.get('target_vis', ''))
    if not target_vis:
        raise RuntimeError("apply_to_target: [run].target_vis is missing — partition_target.py must run first.")

    # Pull combined caltable paths set by concat_caltables.py.
    kcorr = _strip(run.get('combined_kcorr', ''))
    bpass = _strip(run.get('combined_bpass', ''))
    flux  = _strip(run.get('combined_flux', '')) or _strip(run.get('combined_gain', ''))

    gaintables = []
    gainfields = []
    if kcorr:
        gaintables.append(kcorr)
        gainfields.append(_strip(fields.kcorrfield))
    if bpass:
        gaintables.append(bpass)
        gainfields.append(_strip(fields.bpassfield))
    if flux:
        gaintables.append(flux)
        gainfields.append(_strip(fields.fluxfield))

    if not gaintables:
        raise RuntimeError("apply_to_target: no combined caltables found in [run] section.")

    casalog.setlogfile('logs/{SLURM_JOB_NAME}-{SLURM_JOB_ID}.casa'.format(**os.environ))
    casalog.post(
        "applycal target_vis={0} field={1} gaintables={2}".format(
            target_vis, fields.targetfield, gaintables
        ),
        'INFO',
    )

    applycal(
        vis=target_vis,
        field=_strip(fields.targetfield),
        selectdata=False,
        calwt=False,
        gaintable=gaintables,
        gainfield=gainfields,
        parang=False,
        interp='linear,linearflag',
    )


if __name__ == '__main__':
    bookkeeping.run_script(main, logfile)
