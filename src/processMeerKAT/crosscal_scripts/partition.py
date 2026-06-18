#Copyright (C) 2022 Inter-University Institute for Data Intensive Astronomy
#See processMeerKAT.py for license details.

"""
Partition the input MS into per-SPW MMSes containing only calibrator fields.

The target field is intentionally excluded — it is split off separately by
``partition_target.py`` (single monolithic MMS spanning all SPWs) so that the
calibrator solve chain and the target flag chain can run in parallel branches.
"""
import sys
import os

from .. import read_ms
from .. import config_parser
from ..config_parser import typed_get
from .. import processMeerKAT
from .. import bookkeeping

from casatasks import *
logfile=casalog.logfile()
from casatools import msmetadata
import casampi
msmd = msmetadata()

def _strip_quotes(value):
    if isinstance(value, str):
        return value.strip().strip("'").strip('"')
    return value

def cal_field_selection(fields_section):
    """Build a comma-separated calibrator-field selection from the [fields] config dict.

    Joins bpassfield + fluxfield + phasecalfield + extrafields, deduplicating
    while preserving order. The target field is intentionally excluded.
    """
    keys = ['bpassfield', 'fluxfield', 'phasecalfield', 'extrafields']
    seen = []
    for key in keys:
        raw = _strip_quotes(fields_section.get(key, ''))
        if not raw:
            continue
        for entry in str(raw).split(','):
            entry = entry.strip()
            if entry and entry not in seen:
                seen.append(entry)
    return ','.join(seen)

def do_partition(visname, spw, preavg, CPUs, include_crosshand, createmms, spwname, field=''):
    # Get the .ms bit of the filename, case independent
    basename, ext = os.path.splitext(visname)
    filebase = os.path.split(basename)[1]
    extn = 'mms' if createmms else 'ms'

    mvis = '{0}.{1}.{2}'.format(filebase,spwname,extn)
    nscan = 1 if not createmms else msmd.nscans()
    chanaverage = True if preavg > 1 else False
    correlation = '' if include_crosshand else 'XX,YY'

    mstransform(vis=visname, outputvis=mvis, field=field, spw=spw, createmms=createmms, datacolumn='DATA', chanaverage=chanaverage, chanbin=preavg,
                numsubms=nscan, separationaxis='scan', keepflags=True, usewtspectrum=True, nthreads=CPUs, antenna='*&', correlation=correlation)

    return mvis

def main(ctx):

    visname = typed_get(ctx.config, 'data', 'vis', str)
    calcrefant = typed_get(ctx.config, 'crosscal', 'calcrefant', bool, default=False)
    refant = typed_get(ctx.config, 'crosscal', 'refant', str, default='m005')
    spw = typed_get(ctx.config, 'crosscal', 'spw', str, default='')
    nspw = typed_get(ctx.config, 'crosscal', 'nspw', int, default='')
    tasks = typed_get(ctx.config, 'slurm', 'ntasks_per_node', int)
    preavg = typed_get(ctx.config, 'crosscal', 'chanbin', int, default=1)
    include_crosshand = typed_get(ctx.config, 'state', 'dopol', bool, default=False)
    createmms = typed_get(ctx.config, 'crosscal', 'createmms', bool, default=True)

    if nspw > 1:
        casalog.setlogfile('logs/{SLURM_JOB_NAME}-{SLURM_ARRAY_JOB_ID}_{SLURM_ARRAY_TASK_ID}.casa'.format(**os.environ))
    else:
        logfile=casalog.logfile()
        casalog.setlogfile('logs/{SLURM_JOB_NAME}-{SLURM_JOB_ID}.casa'.format(**os.environ))

    if ',' in spw:
        low,high,unit,dirs = config_parser.parse_spw(ctx.config_path)
        spwname = '{0:.0f}~{1:.0f}MHz'.format(min(low),max(high))
    else:
        spwname = spw.replace('*:','')

    msmd.open(visname)
    npol = msmd.ncorrforpol()[0]

    if not include_crosshand and npol == 4:
        npol = 2
    CPUs = npol if tasks*npol <= processMeerKAT.CPUS_PER_NODE_LIMIT else 1 #hard-code for number of polarisations

    cal_fields = cal_field_selection(ctx.config.get('fields', {}))
    if not cal_fields:
        raise ValueError("No calibrator fields found in [fields] section of config — partition.py needs at least one of bpassfield/fluxfield/phasecalfield to run.")

    mvis = do_partition(visname, spw, preavg, CPUs, include_crosshand, createmms, spwname, field=cal_fields)
    config_parser.overwrite_config(ctx.config_path, conf_sec='data', conf_dict={'vis': mvis})
    config_parser.overwrite_config(ctx.config_path, conf_sec='state', conf_dict={'orig_vis': visname})
    msmd.done()

if __name__ == '__main__':

    bookkeeping.run_script(main,logfile)
