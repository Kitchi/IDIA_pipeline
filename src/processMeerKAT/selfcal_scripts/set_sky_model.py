#Copyright (C) 2022 Inter-University Institute for Data Intensive Astronomy
#See processMeerKAT.py for license details.

#!/usr/bin/env python3
import os
from .. import bookkeeping
from .. import config_parser
from .selfcal_part2 import find_outliers
from casatasks import casalog
logfile=casalog.logfile()
casalog.setlogfile('logs/{SLURM_JOB_NAME}-{SLURM_JOB_ID}.casa'.format(**os.environ))


def main(ctx):
    params = bookkeeping._build_selfcal_params(ctx.config, ctx.config_path)
    find_outliers(**params, step='sky')


if __name__ == '__main__':

    bookkeeping.run_script(main, logfile)
