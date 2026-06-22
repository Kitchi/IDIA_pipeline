#Copyright (C) 2022 Inter-University Institute for Data Intensive Astronomy
#See processMeerKAT.py for license details.

"""
Apply this SPW's cross-cal solutions to this SPW's target MMS.

Runs once *inside each per-SPW directory* (not at top level), as the last link
of the per-SPW chain. Each SPW directory holds its own calibrator caltables
(in ``caltables/``) and — after ``partition_target.py`` ran per-SPW — its own
target MMS split to the same frequency range. Because every SPW directory
operates on independent MeasurementSets, all SPWs apply in parallel with no
shared-column contention, and a failure in one SPW (e.g. missing fluxscale)
fails only that SPW's job, loudly.

Both the per-SPW caltables and the per-SPW target carry a single spectral
window (id 0), so ``applycal`` index-matches caltable SPW 0 → target SPW 0 with
no spwmap or channel sub-selection needed.

IMPORTANT: the fluxscale (bootstrapped, flux-scaled gain) table is *required*.
It replaces — it is not additive to — the raw ``gcal`` gains. If it is missing
for this SPW we raise rather than silently substitute ``gcal``, which would
apply gains in relative (un-fluxscaled) units and produce a target with a
silently incorrect flux scale.
"""
import os
import sys

from .. import bookkeeping
from ..config_parser import typed_get

from casatasks import *
logfile = casalog.logfile()
import casampi

import logging
from time import gmtime
logging.Formatter.converter = gmtime
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)-15s %(levelname)s: %(message)s", level=logging.INFO)


def _strip(s):
    if isinstance(s, str):
        return s.strip().strip("'").strip('"')
    return s


def main(ctx):
    state = ctx.config.get('state', {})
    fields = ctx.fields

    # partition_target switched [data].vis to the target and stashed the
    # calibrator MMS name in [state].cal_vis — that name is what the caltable
    # filenames derive from.
    cal_vis = _strip(state.get('cal_vis', '')) or typed_get(ctx.config, 'data', 'vis', str)

    target_vis = _strip(state.get('target_vis', '')) or getattr(ctx, 'target_vis', None)
    if not target_vis:
        raise RuntimeError(
            "apply_to_target: [state].target_vis is missing — partition_target.py "
            "must run (per-SPW) before this step."
        )
    if not os.path.exists(target_vis):
        raise RuntimeError(
            "apply_to_target: target MMS '{0}' does not exist.".format(target_vis)
        )

    calfiles, caldir = bookkeeping.bookkeeping(cal_vis)
    dopol = typed_get(ctx.config, 'state', 'dopol', bool, default=False)

    casalog.setlogfile('logs/{SLURM_JOB_NAME}-{SLURM_JOB_ID}.casa'.format(**os.environ))

    # The fluxscale table is mandatory: it carries the absolute flux scale and
    # replaces the raw gains. No silent fallback to gcal.
    if not os.path.isdir(calfiles.fluxfile):
        raise RuntimeError(
            "apply_to_target: required fluxscale table '{0}' not found for this SPW. "
            "The flux-cal solve likely failed for this frequency range — refusing to "
            "apply (raw gcal gains would give a silently incorrect flux scale).".format(
                calfiles.fluxfile
            )
        )

    # Build the gaintable/gainfield lists, applying only tables that exist.
    # kcorr (delay) and bandpass are applied when present; the fluxscale gain
    # table is always applied (checked above).
    gaintables = []
    gainfields = []

    if os.path.isdir(calfiles.kcorrfile):
        gaintables.append(calfiles.kcorrfile)
        gainfields.append(_strip(fields.kcorrfield))
    if os.path.isdir(calfiles.bpassfile):
        gaintables.append(calfiles.bpassfile)
        gainfields.append(_strip(fields.bpassfield))

    gaintables.append(calfiles.fluxfile)
    gainfields.append(_strip(fields.fluxfield))

    # Polarisation tables, when this run solved for polarisation.
    if dopol:
        for calfile, field in (
            (calfiles.xdelfile, fields.xdelfield),
            (calfiles.dpolfile, fields.dpolfield),
            (calfiles.xpolfile, fields.xpolfield),
        ):
            if os.path.isdir(calfile):
                gaintables.append(calfile)
                gainfields.append(_strip(field))

    casalog.post(
        "apply_to_target: target_vis={0} field={1} gaintables={2}".format(
            target_vis, fields.targetfield, gaintables
        ),
        'INFO',
    )
    logger.info("Applying %d caltable(s) to target %s", len(gaintables), target_vis)

    applycal(
        vis=target_vis,
        field=_strip(fields.targetfield),
        selectdata=False,
        calwt=False,
        gaintable=gaintables,
        gainfield=gainfields,
        parang=dopol,
        interp='linear,linearflag',
    )


if __name__ == '__main__':
    bookkeeping.run_script(main, logfile)
