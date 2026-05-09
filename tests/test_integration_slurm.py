"""End-to-end integration test: full pipeline run through SLURM.

Run with:  pytest -m integration -v tests/test_integration_slurm.py

Requires:
  - SLURM (partition 'debug') accessible from this machine
  - The 2-channel test MS at INTEGRATION_MS
  - The py312 micromamba environment with casatasks installed
"""

import os
import subprocess
import time
from pathlib import Path

import pytest

from processMeerKAT import config_parser

INTEGRATION_MS = '/home/krishna/work/data/1770051982-sdp-l0_2026-02-04T13-34-18_itd.2chan.ms'
MICROMAMBA = '/home/krishna/.local/bin/micromamba'
RUNNER = f'{MICROMAMBA} run -n py312'

# SLURM resources available on this machine (razor: 12 CPUs, ~15 GB)
SLURM_OVERRIDES = {
    'partition': 'debug',
    'nodes': 1,
    'ntasks_per_node': 2,
    'mem': 14,
    'account': '',
    'container': '',
    'modules': [],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_submitted_ids(workdir):
    """Parse submitted_jobs.txt — newline-separated, comma-separated IDs."""
    path = Path(workdir) / 'submitted_jobs.txt'
    if not path.exists():
        return []
    ids = []
    for line in path.read_text().splitlines():
        ids.extend(j.strip() for j in line.split(',') if j.strip())
    return ids


def _poll_squeue(job_ids, timeout=1800, poll_interval=20):
    """Block until all job_ids are gone from squeue (terminal state)."""
    if not job_ids:
        raise RuntimeError('No job IDs to wait on — submitted_jobs.txt may be empty')
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        still_running = [
            jid for jid in job_ids
            if subprocess.run(
                ['squeue', '--jobs', jid, '--noheader'],
                capture_output=True, text=True,
            ).stdout.strip()
        ]
        if not still_running:
            return
        time.sleep(poll_interval)
    raise TimeoutError(
        f'Jobs {job_ids} did not reach a terminal state within {timeout}s'
    )


# ---------------------------------------------------------------------------
# Session-scoped fixture: build config, submit, wait
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def pipeline_run(tmp_path_factory):
    """Run the full pipeline against the 2-channel test MS via SLURM.

    Yields the working directory Path once all submitted jobs are done.
    """
    if not os.path.isdir(INTEGRATION_MS):
        pytest.skip(f'Integration MS not found: {INTEGRATION_MS}')

    workdir = tmp_path_factory.mktemp('idia_integration')
    old_dir = os.getcwd()
    os.chdir(workdir)

    try:
        # ---- Step 1: build config from the test MS -------------------------
        subprocess.run(
            [
                'processMeerKAT', '-B',
                '-M', INTEGRATION_MS,
                '-F', 'generic_slurm',
                '--local',
                f'--runner={RUNNER}',
                '-N', str(SLURM_OVERRIDES['nodes']),
                '-t', str(SLURM_OVERRIDES['ntasks_per_node']),
                '-m', str(SLURM_OVERRIDES['mem']),
                '-p', SLURM_OVERRIDES['partition'],
            ],
            cwd=workdir, capture_output=True, text=True, check=True,
        )

        cfg = str(workdir / 'default_config.toml')

        # ---- Step 2: apply test-cluster overrides --------------------------
        config_parser.overwrite_config(cfg, SLURM_OVERRIDES, 'slurm')
        # Force nspw=2 so we exercise the parallel DAG with 2 SPW branches
        config_parser.overwrite_config(cfg, {'nspw': 2}, 'crosscal')
        # Clear bad frequency ranges so neither SPW gets dropped
        config_parser.overwrite_config(cfg, {'badfreqranges': []}, 'crosscal')

        # ---- Step 3: generate scripts and submit to SLURM ------------------
        subprocess.run(
            ['processMeerKAT', '-R', '--submit'],
            cwd=workdir, capture_output=True, text=True, check=True,
        )

        # ---- Step 4: wait for all jobs to reach a terminal state -----------
        job_ids = _read_submitted_ids(workdir)
        _poll_squeue(job_ids, timeout=3600, poll_interval=20)

        yield workdir

    finally:
        os.chdir(old_dir)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_submitted_jobs_file_exists(pipeline_run):
    assert (pipeline_run / 'submitted_jobs.txt').exists()


@pytest.mark.integration
def test_submitted_job_ids_non_empty(pipeline_run):
    ids = _read_submitted_ids(pipeline_run)
    assert len(ids) > 0, 'submitted_jobs.txt contains no job IDs'


@pytest.mark.integration
def test_master_script_exists(pipeline_run):
    assert (pipeline_run / 'submit_pipeline.sh').exists()


@pytest.mark.integration
def test_spw_directories_created(pipeline_run):
    # spw_split creates one directory per SPW; we forced nspw=2
    spw_dirs = [
        p for p in pipeline_run.iterdir()
        if p.is_dir() and '~' in p.name and 'MHz' in p.name
    ]
    assert len(spw_dirs) == 2, (
        f'Expected 2 SPW directories, found: {[d.name for d in spw_dirs]}'
    )


@pytest.mark.integration
def test_slurm_logs_exist(pipeline_run):
    log_dir = pipeline_run / 'logs'
    assert log_dir.is_dir(), 'logs/ directory not created'
    log_files = list(log_dir.glob('*.out'))
    assert len(log_files) > 0, 'No SLURM .out files found in logs/'


@pytest.mark.integration
def test_no_severe_errors_in_logs(pipeline_run):
    log_dir = pipeline_run / 'logs'
    severe = []
    for logfile in log_dir.glob('*.out'):
        for line in logfile.read_text(errors='replace').splitlines():
            if 'SEVERE' in line and 'MPI' not in line:
                severe.append(f'{logfile.name}: {line.strip()}')
    assert not severe, 'SEVERE errors found in SLURM logs:\n' + '\n'.join(severe[:10])


@pytest.mark.integration
def test_calibration_tables_created(pipeline_run):
    # Calibration tables are CASA directories inside each SPW directory.
    # Accept any extension used by the calibration scripts.
    cal_extensions = {'.G0', '.K0', '.B0', '.gcal', '.bcal', '.kcal',
                      '.Gamp', '.Gphase', '.delays', '.bandpass'}
    caltables = [
        p for p in pipeline_run.rglob('*')
        if p.is_dir() and p.suffix in cal_extensions
    ]
    assert len(caltables) > 0, (
        'No calibration tables found — check per-SPW calibration scripts ran'
    )


@pytest.mark.integration
def test_target_partition_output_exists(pipeline_run):
    # partition_target.py creates a target MMS at the top level.
    # Accept any directory whose name contains 'target' or ends in .mms/.ms.
    target_outputs = [
        p for p in pipeline_run.iterdir()
        if p.is_dir() and ('target' in p.name.lower() or p.suffix in {'.mms', '.ms'})
    ]
    assert len(target_outputs) > 0, (
        'No target partition output found — check partition_target.py ran'
    )
