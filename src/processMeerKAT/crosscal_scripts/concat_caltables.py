#Copyright (C) 2022 Inter-University Institute for Data Intensive Astronomy
#See processMeerKAT.py for license details.

"""
Combine per-SPW caltables into a single multi-SPW caltable per type.

Runs once at the top level (postcal) after every per-SPW calibrator chain has
finished. Walks each per-SPW subdir, locates its ``caltables/<base>.<ext>``
files, and concatenates them into ``caltables/<base>.<ext>`` at the top level
with each row's ``SPECTRAL_WINDOW_ID`` rewritten to the per-SPW dir's global
SPW index.

We roll our own row-copy via ``casatools.table`` rather than relying on CASA's
higher-level table-concat helpers, because caltable concat across CASA
versions has historically been fragile. The combined output paths are written
to the ``[state]`` config section so ``apply_to_target.py`` can find them.
"""
import os
import sys

from .. import bookkeeping
from ..config_parser import typed_get
from .. import processMeerKAT

import numpy as np
from casatasks import *
logfile = casalog.logfile()
from casatools import table
import casampi


# Caltable extensions produced by the cross-cal chain. Order matters only for
# logging — each is concatenated independently.
CAL_EXTS = ['kcal', 'bcal', 'gcal', 'fluxscale', 'pcal', 'xcal', 'xdel']
# Mapping from extension to the [state] config key holding the combined path.
RUN_KEYS = {
    'kcal': 'combined_kcorr',
    'bcal': 'combined_bpass',
    'gcal': 'combined_gain',
    'fluxscale': 'combined_flux',
    'pcal': 'combined_dpol',
    'xcal': 'combined_xpol',
    'xdel': 'combined_xdel',
}


def _expected_caltable_paths(caldir, base):
    """Return {ext: path} for each CAL_EXTS entry present under caldir."""
    found = {}
    for ext in CAL_EXTS:
        candidate = os.path.join(caldir, base + '.' + ext)
        if os.path.isdir(candidate):
            found[ext] = candidate
    return found


def concat_one_caltable(per_spw_paths, per_spw_global_ids, output_path):
    """Concatenate per-SPW caltables of one type into a single combined table.

    Parameters
    ----------
    per_spw_paths : list[str]
        Caltable paths to concatenate, in the same order as per_spw_global_ids.
    per_spw_global_ids : list[int]
        Global SPW index (in the original MS) for each input caltable.
    output_path : str
        Destination path for the combined caltable.
    """
    if len(per_spw_paths) != len(per_spw_global_ids):
        raise ValueError("per_spw_paths and per_spw_global_ids must have equal length")
    if not per_spw_paths:
        raise ValueError("Nothing to concatenate")

    if os.path.exists(output_path):
        casalog.post("Removing existing combined caltable {0}".format(output_path), 'INFO')
        import shutil
        shutil.rmtree(output_path)

    tb = table()

    # Step 1: copy the first per-SPW caltable as the template (preserves
    # column descriptors, keywords, subtable structure). We copy *with* its
    # rows then clear them; CASA tb.copy()'s norows= flag is unreliable
    # across versions.
    tb.open(per_spw_paths[0])
    tb.copy(output_path, deep=True, valuecopy=True, returnobject=False)
    tb.close()

    tb.open(output_path, nomodify=False)
    if tb.nrows() > 0:
        tb.removerows(list(range(tb.nrows())))
    tb.close()

    # Clear the SPECTRAL_WINDOW subtable too — we'll rebuild it from scratch.
    spw_subtable = os.path.join(output_path, 'SPECTRAL_WINDOW')
    if os.path.isdir(spw_subtable):
        tb.open(spw_subtable, nomodify=False)
        if tb.nrows() > 0:
            tb.removerows(list(range(tb.nrows())))
        tb.close()

    # Step 2: append rows from each per-SPW caltable, remapping
    # SPECTRAL_WINDOW_ID to the global index.
    for src_path, global_spw_id in zip(per_spw_paths, per_spw_global_ids):
        _append_rows(src_path, output_path, global_spw_id)
        if os.path.isdir(spw_subtable):
            _append_spw_rows(src_path, output_path, global_spw_id)

    casalog.post(
        "Wrote combined caltable {0} with {1} input SPW(s) → global IDs {2}".format(
            output_path, len(per_spw_paths), per_spw_global_ids
        ),
        'INFO',
    )


def _append_rows(src_path, dst_path, global_spw_id):
    """Append all rows from src_path's main table to dst_path, remapping SPECTRAL_WINDOW_ID."""
    tb = table()
    tb.open(src_path)
    src_nrows = tb.nrows()
    if src_nrows == 0:
        tb.close()
        return
    cols = tb.colnames()
    data = {col: tb.getcol(col) for col in cols}
    tb.close()

    if 'SPECTRAL_WINDOW_ID' in data:
        data['SPECTRAL_WINDOW_ID'] = np.full(src_nrows, global_spw_id, dtype=np.int32)

    tb.open(dst_path, nomodify=False)
    start = tb.nrows()
    tb.addrows(src_nrows)
    for col, vals in data.items():
        tb.putcol(col, vals, startrow=start, nrow=src_nrows)
    tb.close()


def _append_spw_rows(src_path, dst_path, global_spw_id):
    """Append SPECTRAL_WINDOW subtable rows from src→dst.

    Per-SPW caltables typically have one SPECTRAL_WINDOW row at index 0; we
    append it to the combined SPECTRAL_WINDOW subtable so the row index in
    the combined table matches global_spw_id.
    """
    src_spw = os.path.join(src_path, 'SPECTRAL_WINDOW')
    dst_spw = os.path.join(dst_path, 'SPECTRAL_WINDOW')
    tb = table()
    tb.open(src_spw)
    nrows = tb.nrows()
    if nrows == 0:
        tb.close()
        return
    cols = tb.colnames()
    data = {col: tb.getcol(col) for col in cols}
    tb.close()

    tb.open(dst_spw, nomodify=False)
    start = tb.nrows()
    # Pad with empty rows up to global_spw_id so row index == SPW ID.
    if start < global_spw_id:
        tb.addrows(global_spw_id - start)
        start = global_spw_id
    tb.addrows(nrows)
    for col, vals in data.items():
        tb.putcol(col, vals, startrow=start, nrow=nrows)
    tb.close()


def main(ctx):
    visname = typed_get(ctx.config, 'data', 'vis', str)
    base = os.path.splitext(os.path.basename(visname.strip("'")))[0]

    casalog.setlogfile('logs/{SLURM_JOB_NAME}-{SLURM_JOB_ID}.casa'.format(**os.environ))

    # Discover per-SPW dirs from the top-level run directory (CWD).
    top_dir = os.getcwd()
    spw_entries = bookkeeping.get_all_spw_caldirs(top_dir, config_name=os.path.basename(ctx.config_path))
    if not spw_entries:
        raise RuntimeError("concat_caltables: no per-SPW caldirs found under {0}".format(top_dir))

    casalog.post("Discovered {0} per-SPW caldirs: {1}".format(
        len(spw_entries), [(e['spw_id'], e['caldir']) for e in spw_entries]
    ), 'INFO')

    # Output combined caltables go in caltables/ at the top level.
    combined_caldir = os.path.join(top_dir, 'caltables')
    os.makedirs(combined_caldir, exist_ok=True)

    combined_paths = {}
    for ext in CAL_EXTS:
        sources = []
        global_ids = []
        for entry in spw_entries:
            tables = _expected_caltable_paths(entry['caldir'], base)
            if ext in tables:
                sources.append(tables[ext])
                global_ids.append(entry['spw_id'])
        if not sources:
            casalog.post("No {0} caltables found across SPWs — skipping.".format(ext), 'INFO')
            continue
        out_path = os.path.join(combined_caldir, base + '.' + ext)
        concat_one_caltable(sources, global_ids, out_path)
        combined_paths[RUN_KEYS[ext]] = "'{0}'".format(out_path)

    if not combined_paths:
        raise RuntimeError("concat_caltables produced no combined tables — check per-SPW caldirs.")

    config_parser.overwrite_config(
        ctx.config_path,
        conf_sec='state',
        sec_comment='# Internal variables for pipeline execution',
        conf_dict=combined_paths,
    )


if __name__ == '__main__':
    bookkeeping.run_script(main, logfile)
