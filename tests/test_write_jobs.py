"""Tests for write_jobs() — script generation without SLURM submission."""

from pathlib import Path

from processMeerKAT.slurm_jobs import write_jobs
from processMeerKAT.constants import MASTER_SCRIPT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call_nspw1(cfg):
    """Minimal write_jobs() call for a 3-script nspw=1 linear chain."""
    write_jobs(
        config=str(cfg),
        scripts=['validate_input.py', 'flag_round_1.py', 'setjy.py'],
        threadsafe=[False, True, True],
        containers=['', '', ''],
        num_precal_scripts=0,
        nodes=1, ntasks_per_node=2, mem=14,
        partition='debug', account='', reservation='',
        exclude='', modules=[], mpi_wrapper='mpirun',
        submit=False,
    )


def _call_nspw2(cfg):
    """Minimal write_jobs() call for a 2-SPW DAG with a target branch."""
    write_jobs(
        config=str(cfg),
        scripts=['partition.py', 'partition_target.py',
                 'concat_caltables.py', 'apply_to_target.py'],
        threadsafe=[True, True, False, True],
        containers=['', '', '', ''],
        num_precal_scripts=2,
        target_scripts=[
            {'script': 'validate_input.py', 'mpi': False, 'container': ''},
            {'script': 'flag_round_1.py',   'mpi': True,  'container': ''},
        ],
        nodes=1, ntasks_per_node=2, mem=14,
        partition='debug', account='', reservation='',
        exclude='', modules=[], mpi_wrapper='mpirun',
        submit=False, timestamp='2026-05-09-12-00-00',
    )


def _master(workdir):
    return (workdir / MASTER_SCRIPT).read_text()


# ---------------------------------------------------------------------------
# nspw=1: linear chain
# ---------------------------------------------------------------------------

class TestWriteJobsLinear:

    def test_sbatch_files_created(self, wj_nspw1_dir):
        _call_nspw1(wj_nspw1_dir / 'config.toml')
        assert (wj_nspw1_dir / 'validate_input.sbatch').exists()
        assert (wj_nspw1_dir / 'flag_round_1.sbatch').exists()
        assert (wj_nspw1_dir / 'setjy.sbatch').exists()

    def test_master_script_created(self, wj_nspw1_dir):
        _call_nspw1(wj_nspw1_dir / 'config.toml')
        assert (wj_nspw1_dir / MASTER_SCRIPT).exists()

    def test_first_script_no_dependency(self, wj_nspw1_dir):
        _call_nspw1(wj_nspw1_dir / 'config.toml')
        master = _master(wj_nspw1_dir)
        # First script submitted without -d flag
        assert 'IDs=$(sbatch validate_input.sbatch' in master

    def test_subsequent_scripts_chained(self, wj_nspw1_dir):
        _call_nspw1(wj_nspw1_dir / 'config.toml')
        master = _master(wj_nspw1_dir)
        assert 'sbatch -d afterok:${IDs//,/:} --kill-on-invalid-dep=yes flag_round_1.sbatch' in master
        assert 'sbatch -d afterok:${IDs//,/:} --kill-on-invalid-dep=yes setjy.sbatch' in master

    def test_script_order_in_master(self, wj_nspw1_dir):
        _call_nspw1(wj_nspw1_dir / 'config.toml')
        master = _master(wj_nspw1_dir)
        vi = master.index('validate_input.sbatch')
        fr = master.index('flag_round_1.sbatch')
        sj = master.index('setjy.sbatch')
        assert vi < fr < sj

    def test_non_mpi_script_single_task(self, wj_nspw1_dir):
        _call_nspw1(wj_nspw1_dir / 'config.toml')
        sbatch = (wj_nspw1_dir / 'validate_input.sbatch').read_text()
        assert '#SBATCH --ntasks-per-node=1' in sbatch

    def test_mpi_script_uses_ntasks(self, wj_nspw1_dir):
        _call_nspw1(wj_nspw1_dir / 'config.toml')
        sbatch = (wj_nspw1_dir / 'flag_round_1.sbatch').read_text()
        assert '#SBATCH --ntasks-per-node=2' in sbatch

    def test_sbatch_partition_directive(self, wj_nspw1_dir):
        _call_nspw1(wj_nspw1_dir / 'config.toml')
        sbatch = (wj_nspw1_dir / 'validate_input.sbatch').read_text()
        assert '#SBATCH --partition=debug' in sbatch

    def test_master_writes_submitted_jobs_file(self, wj_nspw1_dir):
        _call_nspw1(wj_nspw1_dir / 'config.toml')
        master = _master(wj_nspw1_dir)
        assert 'echo "$IDs" > submitted_jobs.txt' in master

    def test_logs_dir_created(self, wj_nspw1_dir):
        _call_nspw1(wj_nspw1_dir / 'config.toml')
        # write_sbatch creates logs/ on first call
        assert (wj_nspw1_dir / 'logs').is_dir()

    def test_python_m_invocation_in_sbatch(self, wj_nspw1_dir):
        _call_nspw1(wj_nspw1_dir / 'config.toml')
        sbatch = (wj_nspw1_dir / 'flag_round_1.sbatch').read_text()
        assert 'python -m processMeerKAT.crosscal_scripts.flag_round_1' in sbatch


# ---------------------------------------------------------------------------
# nspw=2: parallel SPW DAG
# ---------------------------------------------------------------------------

class TestWriteJobsDAG:

    def test_precal_sbatch_files_created(self, wj_nspw2_dir):
        _call_nspw2(wj_nspw2_dir / 'config.toml')
        assert (wj_nspw2_dir / 'partition.sbatch').exists()
        assert (wj_nspw2_dir / 'partition_target.sbatch').exists()

    def test_postcal_sbatch_files_created(self, wj_nspw2_dir):
        _call_nspw2(wj_nspw2_dir / 'config.toml')
        assert (wj_nspw2_dir / 'concat_caltables.sbatch').exists()
        assert (wj_nspw2_dir / 'apply_to_target.sbatch').exists()

    def test_target_sbatch_files_created_with_prefix(self, wj_nspw2_dir):
        _call_nspw2(wj_nspw2_dir / 'config.toml')
        assert (wj_nspw2_dir / 'target_validate_input.sbatch').exists()
        assert (wj_nspw2_dir / 'target_flag_round_1.sbatch').exists()

    def test_master_script_created(self, wj_nspw2_dir):
        _call_nspw2(wj_nspw2_dir / 'config.toml')
        assert (wj_nspw2_dir / MASTER_SCRIPT).exists()

    def test_master_extracts_partition_job_id(self, wj_nspw2_dir):
        _call_nspw2(wj_nspw2_dir / 'config.toml')
        master = _master(wj_nspw2_dir)
        assert 'partitionID=$(echo $allSPWIDs | cut -d , -f' in master

    def test_master_extracts_target_partition_id(self, wj_nspw2_dir):
        _call_nspw2(wj_nspw2_dir / 'config.toml')
        master = _master(wj_nspw2_dir)
        assert 'targetPartID=$(echo $allSPWIDs | cut -d , -f' in master

    def test_target_branch_depends_on_target_partition(self, wj_nspw2_dir):
        _call_nspw2(wj_nspw2_dir / 'config.toml')
        master = _master(wj_nspw2_dir)
        assert 'targetIDs=$(sbatch -d afterok:${targetPartID//,/:}' in master
        assert 'target_validate_input.sbatch' in master

    def test_target_branch_scripts_chained(self, wj_nspw2_dir):
        _call_nspw2(wj_nspw2_dir / 'config.toml')
        master = _master(wj_nspw2_dir)
        assert 'targetIDs+=,$(sbatch -d afterok:${targetIDs//,/:}' in master
        assert 'target_flag_round_1.sbatch' in master

    def test_postcal_joins_both_branches(self, wj_nspw2_dir):
        _call_nspw2(wj_nspw2_dir / 'config.toml')
        master = _master(wj_nspw2_dir)
        # First postcal script waits on per-SPW IDs AND target IDs
        assert 'afterany:${IDs//,/:}:${targetIDs//,/:}' in master
        assert 'concat_caltables.sbatch' in master

    def test_master_writes_submitted_jobs_file(self, wj_nspw2_dir):
        _call_nspw2(wj_nspw2_dir / 'config.toml')
        master = _master(wj_nspw2_dir)
        assert '> submitted_jobs.txt' in master
