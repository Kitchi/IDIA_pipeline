#Copyright (C) 2022 Inter-University Institute for Data Intensive Astronomy
#See processMeerKAT.py for license details.

"""
Concatenate the corrected per-SPW target MMSes into one full-band target.

Runs once at the top level (postcal) after every per-SPW chain has finished
applying its own caltables to its own target. Walks each per-SPW directory,
reads its ``[state].target_vis`` (the corrected, single-SPW target for that
frequency range), and concatenates them along frequency — in ascending global
SPW order — into a single multi-SPW target MMS at the top level. The combined
path is written to ``[state].target_vis`` so selfcal / science_image consume it
as the science target.
"""
import os
import sys

from .. import bookkeeping
from .. import config_parser
from ..config_parser import typed_get

from casatasks import *
logfile = casalog.logfile()
import casampi


def main(ctx):
    casalog.setlogfile('logs/{SLURM_JOB_NAME}-{SLURM_JOB_ID}.casa'.format(**os.environ))

    top_dir = os.getcwd()
    config_name = os.path.basename(ctx.config_path)
    spw_entries = bookkeeping.get_all_spw_caldirs(top_dir, config_name=config_name)
    if not spw_entries:
        raise RuntimeError(
            "concat_target: no per-SPW directories found under {0}".format(top_dir)
        )

    # Collect each SPW's corrected target MMS, in ascending global-SPW order.
    per_spw_targets = []
    missing = []
    for entry in spw_entries:
        spw_config = os.path.join(entry['dir'], config_name)
        target_vis = config_parser.get_key(spw_config, 'state', 'target_vis')
        if isinstance(target_vis, str):
            target_vis = target_vis.strip().strip("'").strip('"')
        if not target_vis:
            missing.append((entry['spw_id'], entry['dir'], 'no [state].target_vis'))
            continue
        target_path = os.path.join(entry['dir'], target_vis)
        if not os.path.exists(target_path):
            missing.append((entry['spw_id'], entry['dir'], target_path))
            continue
        per_spw_targets.append((entry['spw_id'], target_path))

    if missing:
        raise RuntimeError(
            "concat_target: missing corrected target for SPW(s): {0}. Each per-SPW "
            "apply must succeed before the targets can be recombined.".format(missing)
        )

    per_spw_targets.sort(key=lambda e: e[0])
    target_paths = [p for _, p in per_spw_targets]

    visname = typed_get(ctx.config, 'data', 'vis', str)
    base = os.path.splitext(os.path.basename(visname.strip("'")))[0]
    # Strip any per-SPW / cal suffixes down to the dataset base name.
    combined_target = os.path.join(top_dir, '{0}.target.mms'.format(base.split('.')[0]))

    if os.path.exists(combined_target):
        import shutil
        casalog.post("Removing existing combined target {0}".format(combined_target), 'INFO')
        shutil.rmtree(combined_target)

    casalog.post(
        "concat_target: concatenating {0} per-SPW targets → {1}".format(
            len(target_paths), combined_target
        ),
        'INFO',
    )

    # concat appends the input MSes' spectral windows, producing a multi-SPW
    # target spanning the full band, which tclean handles natively.
    concat(vis=target_paths, concatvis=combined_target)

    config_parser.overwrite_config(
        ctx.config_path,
        conf_sec='state',
        sec_comment='# Internal variables for pipeline execution',
        conf_dict={'target_vis': "'{0}'".format(combined_target)},
    )
    # Downstream selfcal / science_image operate on [data].vis (see
    # bookkeeping._build_selfcal_params / _build_imaging_params), so point it at
    # the concatenated full-band target.
    config_parser.overwrite_config(
        ctx.config_path,
        conf_sec='data',
        conf_dict={'vis': "'{0}'".format(combined_target)},
    )


if __name__ == '__main__':
    bookkeeping.run_script(main, logfile)
