"""Tests for write_command's `python -m` invocation + default_runner prefix."""

import pytest

from processMeerKAT.slurm_jobs import (
    write_command, script_module_path, _resolve_runner,
)


# ---------------------------------------------------------------------------
# script_module_path — maps script name → dotted module path
# ---------------------------------------------------------------------------

def test_module_path_for_crosscal_script():
    assert script_module_path('partition.py') == 'processMeerKAT.crosscal_scripts.partition'


def test_module_path_for_partition_target():
    assert script_module_path('partition_target.py') == 'processMeerKAT.crosscal_scripts.partition_target'


def test_module_path_for_concat_caltables():
    assert script_module_path('concat_caltables.py') == 'processMeerKAT.crosscal_scripts.concat_caltables'


def test_module_path_for_apply_to_target():
    assert script_module_path('apply_to_target.py') == 'processMeerKAT.crosscal_scripts.apply_to_target'


def test_module_path_for_selfcal_script():
    assert script_module_path('selfcal_part1.py') == 'processMeerKAT.selfcal_scripts.selfcal_part1'


def test_module_path_for_aux_script():
    assert script_module_path('concat.py') == 'processMeerKAT.aux_scripts.concat'


def test_module_path_for_package_root_script():
    assert script_module_path('validate_input.py') == 'processMeerKAT.validate_input'
    assert script_module_path('read_ms.py') == 'processMeerKAT.read_ms'
    assert script_module_path('science_image.py') == 'processMeerKAT.science_image'


def test_module_path_returns_none_for_user_script():
    """Scripts not inside the package can't be invoked with python -m."""
    assert script_module_path('/tmp/userscript.py') is None
    assert script_module_path('totally_made_up.py') is None


def test_module_path_returns_none_for_non_python_files():
    assert script_module_path('foo.sh') is None


# ---------------------------------------------------------------------------
# _resolve_runner — picks the command prefix
# ---------------------------------------------------------------------------

def test_runner_per_script_container_wins_over_default():
    """Per-script container override beats facility default_runner."""
    runner = _resolve_runner(container='/foo/bar.sif',
                             default_runner='conda run -n env')
    assert runner == 'singularity exec /foo/bar.sif'


def test_runner_falls_back_to_default_when_no_container():
    runner = _resolve_runner(container='', default_runner='conda run -n py312')
    assert runner == 'conda run -n py312'


def test_runner_empty_when_neither_set():
    assert _resolve_runner(container='', default_runner='') == ''


# ---------------------------------------------------------------------------
# write_command — full command assembly
# ---------------------------------------------------------------------------

def test_write_command_uses_python_m_for_package_scripts(tmp_pipeline_dir):
    cmd = write_command('partition.py', '--config foo.txt',
                        mpi_wrapper='mpirun', container='', logfile=False,
                        default_runner='')
    assert 'python -m processMeerKAT.crosscal_scripts.partition --config foo.txt' in cmd
    assert 'singularity' not in cmd
    assert 'casa --nogui' not in cmd


def test_write_command_prepends_default_runner(tmp_pipeline_dir):
    cmd = write_command('partition.py', '--config foo.txt',
                        mpi_wrapper='mpirun', container='', logfile=False,
                        default_runner='conda run -n py312')
    assert cmd.startswith('conda run -n py312 mpirun python -m processMeerKAT.crosscal_scripts.partition')


def test_write_command_per_script_container_wraps_in_singularity(tmp_pipeline_dir):
    cmd = write_command('partition.py', '--config foo.txt',
                        mpi_wrapper='mpirun', container='/idia/casa.sif',
                        logfile=False, default_runner='')
    assert cmd.startswith('singularity exec /idia/casa.sif mpirun python -m processMeerKAT.crosscal_scripts.partition')


def test_write_command_per_script_container_overrides_default_runner(tmp_pipeline_dir):
    cmd = write_command('partition.py', '--config foo.txt',
                        mpi_wrapper='mpirun', container='/idia/casa.sif',
                        logfile=False, default_runner='conda run -n py312')
    assert 'singularity exec /idia/casa.sif' in cmd
    assert 'conda run' not in cmd


def test_write_command_no_p_flag_no_more(tmp_pipeline_dir):
    """`python -P` was a workaround for sys.path[0] script-dir shadowing.
    With `python -m` it's no longer needed."""
    cmd = write_command('partition.py', '--config foo.txt',
                        mpi_wrapper='mpirun', container='', logfile=False,
                        default_runner='')
    assert ' -P ' not in cmd


def test_write_command_array_job_for_partition(tmp_pipeline_dir):
    cmd = write_command('partition.py', '--config foo.txt',
                        mpi_wrapper='mpirun', container='', logfile=False,
                        default_runner='', SPWs='*:880~933MHz,*:960~1010MHz', nspw=2)
    assert 'arr=' in cmd
    assert 'SLURM_ARRAY_TASK_ID' in cmd
    assert 'cd ..' in cmd
