"""Tests for sbatch/script generation in processMeerKAT.py — no SLURM required."""

import os
import re
import pytest
import processMeerKAT as pmk
from processMeerKAT.slurm_jobs import _expand_selfcal_loops


# ---------------------------------------------------------------------------
# write_command
# ---------------------------------------------------------------------------

class TestWriteCommand:

    def test_returns_string(self, tmp_pipeline_dir):
        cmd = pmk.write_command('validate_input.py', '--config .config.tmp',
                                mpi_wrapper='srun', container='/some/container.sif',
                                casa_script=False, logfile=False)
        assert isinstance(cmd, str)

    def test_uses_python_for_non_casa(self, tmp_pipeline_dir):
        cmd = pmk.write_command('validate_input.py', '--config .config.tmp',
                                mpi_wrapper='srun', container='/some/container.sif',
                                casa_script=False, logfile=False)
        assert 'python' in cmd

    def test_uses_python_dash_m_for_package_scripts(self, tmp_pipeline_dir):
        """All scripts shipped with the package are invoked via `python -m`,
        not `casa --nogui -c` and not as a bare file path. CASA 6+ ships its
        tooling as importable Python modules, so a plain python invocation
        works in both container and bare-env modes."""
        cmd = pmk.write_command('flag_round_1.py', '--config .config.tmp',
                                mpi_wrapper='mpirun', container='/some/container.sif',
                                casa_script=True, logfile=False)
        assert 'python -m processMeerKAT.crosscal_scripts.flag_round_1' in cmd
        assert 'casa --nogui' not in cmd

    def test_singularity_exec_present(self, tmp_pipeline_dir):
        cmd = pmk.write_command('flag_round_1.py', '--config .config.tmp',
                                mpi_wrapper='mpirun', container='/some/container.sif',
                                casa_script=True, logfile=False)
        assert 'singularity exec' in cmd

    def test_container_path_in_command(self, tmp_pipeline_dir):
        container = '/my/special/container.sif'
        cmd = pmk.write_command('flag_round_1.py', '--config .config.tmp',
                                mpi_wrapper='srun', container=container,
                                casa_script=False, logfile=False)
        assert container in cmd

    def test_mpi_wrapper_in_command(self, tmp_pipeline_dir):
        cmd = pmk.write_command('flag_round_1.py', '--config .config.tmp',
                                mpi_wrapper='mpirun', container='/c.sif',
                                casa_script=False, logfile=False)
        assert 'mpirun' in cmd

    def test_xvfb_for_plot_scripts(self, tmp_pipeline_dir):
        cmd = pmk.write_command('plotcal_spw.py', '--config .config.tmp',
                                mpi_wrapper='srun', container='/c.sif',
                                casa_script=False, logfile=False, plot=True)
        assert 'xvfb-run' in cmd

    def test_no_xvfb_for_normal_scripts(self, tmp_pipeline_dir):
        cmd = pmk.write_command('flag_round_1.py', '--config .config.tmp',
                                mpi_wrapper='srun', container='/c.sif',
                                casa_script=False, logfile=False, plot=False)
        assert 'xvfb-run' not in cmd

    def test_array_job_for_partition(self, tmp_pipeline_dir):
        """Partition script with multiple SPWs should emit an array job preamble."""
        cmd = pmk.write_command('partition.py', '--config .config.tmp',
                                mpi_wrapper='mpirun', container='/c.sif',
                                casa_script=False, logfile=False,
                                SPWs='880~933MHz,960~1010MHz', nspw=2)
        assert 'arr=' in cmd
        assert 'SLURM_ARRAY_TASK_ID' in cmd

    def test_no_array_for_non_partition(self, tmp_pipeline_dir):
        cmd = pmk.write_command('flag_round_1.py', '--config .config.tmp',
                                mpi_wrapper='mpirun', container='/c.sif',
                                casa_script=False, logfile=False,
                                SPWs='880~933MHz,960~1010MHz', nspw=2)
        assert 'SLURM_ARRAY_TASK_ID' not in cmd


# ---------------------------------------------------------------------------
# write_sbatch
# ---------------------------------------------------------------------------

class TestWriteSbatch:

    def _read_sbatch(self, tmp_path, name='validate_input'):
        return (tmp_path / f'{name}.sbatch').read_text()

    def test_creates_sbatch_file(self, tmp_pipeline_dir):
        pmk.write_sbatch('validate_input.py', '--config .config.tmp',
                         nodes=1, tasks=8, mem=232,
                         name='validate_input', container='/c.sif',
                         partition='Main', time='12:00:00', account='test-acct')
        assert (tmp_pipeline_dir / 'validate_input.sbatch').exists()

    def test_sbatch_has_bash_shebang(self, tmp_pipeline_dir):
        pmk.write_sbatch('validate_input.py', '--config .config.tmp',
                         nodes=1, tasks=8, name='validate_input',
                         container='/c.sif', account='test-acct')
        content = self._read_sbatch(tmp_pipeline_dir)
        assert content.startswith('#!/bin/bash')

    def test_sbatch_account_directive(self, tmp_pipeline_dir):
        pmk.write_sbatch('validate_input.py', '--config .config.tmp',
                         nodes=1, tasks=4, name='validate_input',
                         container='/c.sif', account='my-account')
        content = self._read_sbatch(tmp_pipeline_dir)
        assert '#SBATCH --account=my-account' in content

    def test_sbatch_nodes_directive(self, tmp_pipeline_dir):
        pmk.write_sbatch('validate_input.py', '--config .config.tmp',
                         nodes=3, tasks=8, name='validate_input',
                         container='/c.sif', account='acct')
        content = self._read_sbatch(tmp_pipeline_dir)
        assert '#SBATCH --nodes=3' in content

    def test_sbatch_partition_directive(self, tmp_pipeline_dir):
        pmk.write_sbatch('validate_input.py', '--config .config.tmp',
                         nodes=1, tasks=8, name='validate_input',
                         container='/c.sif', account='acct', partition='HighMem')
        content = self._read_sbatch(tmp_pipeline_dir)
        assert '#SBATCH --partition=HighMem' in content

    def test_sbatch_time_directive(self, tmp_pipeline_dir):
        pmk.write_sbatch('validate_input.py', '--config .config.tmp',
                         nodes=1, tasks=8, name='validate_input',
                         container='/c.sif', account='acct', time='02:00:00')
        content = self._read_sbatch(tmp_pipeline_dir)
        assert '#SBATCH --time=02:00:00' in content

    def test_sbatch_module_loading(self, tmp_pipeline_dir):
        pmk.write_sbatch('validate_input.py', '--config .config.tmp',
                         nodes=1, tasks=8, name='validate_input',
                         container='/c.sif', account='acct',
                         modules=['openmpi/4.0.3'])
        content = self._read_sbatch(tmp_pipeline_dir)
        assert 'module load openmpi/4.0.3' in content

    def test_sbatch_no_module_loading_when_empty(self, tmp_pipeline_dir):
        pmk.write_sbatch('validate_input.py', '--config .config.tmp',
                         nodes=1, tasks=8, name='validate_input',
                         container='/c.sif', account='acct', modules=[])
        content = self._read_sbatch(tmp_pipeline_dir)
        assert 'module load' not in content

    def test_sbatch_exclude_directive(self, tmp_pipeline_dir):
        pmk.write_sbatch('validate_input.py', '--config .config.tmp',
                         nodes=1, tasks=8, name='validate_input',
                         container='/c.sif', account='acct', exclude='node001')
        content = self._read_sbatch(tmp_pipeline_dir)
        assert '#SBATCH --exclude=node001' in content

    def test_sbatch_no_exclude_when_empty(self, tmp_pipeline_dir):
        pmk.write_sbatch('validate_input.py', '--config .config.tmp',
                         nodes=1, tasks=8, name='validate_input',
                         container='/c.sif', account='acct', exclude='')
        content = self._read_sbatch(tmp_pipeline_dir)
        assert '--exclude' not in content

    def test_sbatch_array_directive_for_partition(self, tmp_pipeline_dir):
        pmk.write_sbatch('partition.py', '--config .config.tmp',
                         nodes=1, tasks=8, name='partition',
                         container='/c.sif', account='acct',
                         SPWs='880~933MHz,960~1010MHz,1010~1060MHz', nspw=3)
        content = self._read_sbatch(tmp_pipeline_dir, name='partition')
        assert '#SBATCH --array=' in content

    def test_sbatch_justrun_does_not_overwrite(self, tmp_pipeline_dir):
        pmk.write_sbatch('validate_input.py', '--config .config.tmp',
                         nodes=1, tasks=8, name='validate_input',
                         container='/c.sif', account='acct', time='01:00:00')
        # Now write again with different time but justrun=True
        pmk.write_sbatch('validate_input.py', '--config .config.tmp',
                         nodes=1, tasks=8, name='validate_input',
                         container='/c.sif', account='acct', time='99:00:00', justrun=True)
        content = self._read_sbatch(tmp_pipeline_dir)
        assert '01:00:00' in content
        assert '99:00:00' not in content

    def test_sbatch_runname_prefix(self, tmp_pipeline_dir):
        pmk.write_sbatch('validate_input.py', '--config .config.tmp',
                         nodes=1, tasks=8, name='validate_input',
                         runname='myrun_', container='/c.sif', account='acct')
        content = self._read_sbatch(tmp_pipeline_dir)
        assert '#SBATCH --job-name=myrun_validate_input' in content

    def test_sbatch_omp_num_threads_set(self, tmp_pipeline_dir):
        pmk.write_sbatch('validate_input.py', '--config .config.tmp',
                         nodes=1, tasks=8, name='validate_input',
                         container='/c.sif', account='acct')
        content = self._read_sbatch(tmp_pipeline_dir)
        assert 'OMP_NUM_THREADS' in content

    def test_logs_directory_created(self, tmp_pipeline_dir):
        pmk.write_sbatch('validate_input.py', '--config .config.tmp',
                         nodes=1, tasks=8, name='validate_input',
                         container='/c.sif', account='acct')
        assert (tmp_pipeline_dir / 'logs').is_dir()


# ---------------------------------------------------------------------------
# srun helper
# ---------------------------------------------------------------------------

class TestSrun:

    def _make_arg_dict(self, partition='Main', account='b03', exclude='', reservation=''):
        return {'partition': partition, 'account': account,
                'exclude': exclude, 'reservation': reservation}

    def test_returns_string(self):
        result = pmk.srun(self._make_arg_dict())
        assert isinstance(result, str)

    def test_includes_partition(self):
        result = pmk.srun(self._make_arg_dict(partition='HighMem'))
        assert '--partition=HighMem' in result

    def test_includes_account(self):
        result = pmk.srun(self._make_arg_dict(account='my-group'))
        assert '--account=my-group' in result

    def test_qos_interactive(self):
        result = pmk.srun(self._make_arg_dict(), qos=True)
        assert 'qos-interactive' in result

    def test_no_qos_when_false(self):
        result = pmk.srun(self._make_arg_dict(), qos=False)
        assert 'qos-interactive' not in result

    def test_exclude_added(self):
        result = pmk.srun(self._make_arg_dict(exclude='badnode'))
        assert '--exclude=badnode' in result

    def test_no_exclude_when_empty(self):
        result = pmk.srun(self._make_arg_dict(exclude=''))
        assert '--exclude' not in result

    def test_time_and_mem_params(self):
        result = pmk.srun(self._make_arg_dict(), qos=False, time=5, mem=8)
        assert '--time=5' in result
        assert '--mem=8GB' in result


# ---------------------------------------------------------------------------
# get_config_kwargs
# ---------------------------------------------------------------------------

class TestGetConfigKwargs:

    def test_returns_expected_keys(self, minimal_config):
        kwargs = pmk.get_config_kwargs(minimal_config, 'slurm', pmk.SLURM_CONFIG_KEYS)
        for key in pmk.SLURM_CONFIG_KEYS:
            assert key in kwargs, f"Missing key: {key}"

    def test_raises_for_missing_section(self, minimal_config):
        with pytest.raises(KeyError, match="no section"):
            pmk.get_config_kwargs(minimal_config, 'nonexistent', ['somekey'])

    def test_raises_for_missing_key(self, minimal_config):
        with pytest.raises(KeyError, match="missing from section"):
            pmk.get_config_kwargs(minimal_config, 'slurm', ['nodes', 'nonexistent_key'])

    def test_warns_for_unknown_keys(self, minimal_config, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            pmk.get_config_kwargs(minimal_config, 'crosscal', pmk.CROSSCAL_CONFIG_KEYS)
        # No unknown keys in our fixture, so no warning expected
        assert 'Unknown keys' not in caplog.text


# ---------------------------------------------------------------------------
# _expand_selfcal_loops
# ---------------------------------------------------------------------------

class TestExpandSelfcalLoops:

    def test_no_selfcal_section_returns_unchanged(self, minimal_config):
        scripts = ['validate_input.sbatch', 'flag_round_1.sbatch']
        result = _expand_selfcal_loops(minimal_config, scripts)
        assert result == scripts

    def test_scripts_without_selfcal_pair_unchanged(self, tmp_path):
        cfg = tmp_path / 'cfg.txt'
        cfg.write_text('[selfcal]\nnloops = 2\nloop = 0\n')
        scripts = ['validate_input.sbatch', 'flag_round_1.sbatch']
        result = _expand_selfcal_loops(str(cfg), scripts)
        assert result == scripts

    def test_one_loop_adds_final_clean(self, tmp_path):
        cfg = tmp_path / 'cfg.txt'
        cfg.write_text('[selfcal]\nnloops = 1\nloop = 0\n')
        scripts = ['selfcal_part1.sbatch', 'selfcal_part2.sbatch']
        result = _expand_selfcal_loops(str(cfg), scripts)
        assert result == ['selfcal_part1.sbatch', 'selfcal_part2.sbatch', 'selfcal_part1.sbatch']

    def test_two_loops_expands_correctly(self, tmp_path):
        cfg = tmp_path / 'cfg.txt'
        cfg.write_text('[selfcal]\nnloops = 2\nloop = 0\n')
        scripts = ['selfcal_part1.sbatch', 'selfcal_part2.sbatch', 'concat.sbatch']
        result = _expand_selfcal_loops(str(cfg), scripts)
        assert result == [
            'selfcal_part1.sbatch', 'selfcal_part2.sbatch',
            'selfcal_part1.sbatch', 'selfcal_part2.sbatch',
            'selfcal_part1.sbatch',
            'concat.sbatch',
        ]

    def test_non_adjacent_selfcal_pair_unchanged(self, tmp_path):
        cfg = tmp_path / 'cfg.txt'
        cfg.write_text('[selfcal]\nnloops = 2\nloop = 0\n')
        scripts = ['selfcal_part1.sbatch', 'other.sbatch', 'selfcal_part2.sbatch']
        result = _expand_selfcal_loops(str(cfg), scripts)
        assert result == scripts

    def test_start_loop_nonzero_reduces_expansion(self, tmp_path):
        cfg = tmp_path / 'cfg.txt'
        cfg.write_text('[selfcal]\nnloops = 3\nloop = 2\n')
        scripts = ['selfcal_part1.sbatch', 'selfcal_part2.sbatch']
        result = _expand_selfcal_loops(str(cfg), scripts)
        # selfcal_loops = nloops - loop = 1; same as one-loop case
        assert result == ['selfcal_part1.sbatch', 'selfcal_part2.sbatch', 'selfcal_part1.sbatch']
