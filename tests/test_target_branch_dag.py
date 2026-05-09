"""Tests for the parallel target branch added to write_spw_master."""

import os

import pytest

from processMeerKAT.slurm_jobs import write_spw_master


def _read(path):
    with open(path) as f:
        return f.read()


def test_master_includes_partition_target_id_extraction(tmp_pipeline_dir):
    out = tmp_pipeline_dir / 'master.sh'
    write_spw_master(
        str(out), config='pipeline_state.toml',
        SPWs='*:880~933MHz,*:960~1010MHz',
        precal_scripts=['partition.sbatch', 'partition_target.sbatch'],
        postcal_scripts=['concat_caltables.sbatch'],
        target_scripts=['target_validate_input.sbatch', 'target_flag_round_1.sbatch'],
        submit=False, timestamp='2026-05-06-12-00-00',
        slurm_kwargs={'account': 'b03-idia-ag', 'partition': 'Main', 'exclude': '', 'reservation': ''},
    )
    contents = _read(str(out))
    assert 'partitionID=$(echo $allSPWIDs | cut -d , -f1)' in contents
    assert 'targetPartID=$(echo $allSPWIDs | cut -d , -f2)' in contents


def test_master_emits_target_branch_dependent_on_target_partition(tmp_pipeline_dir):
    out = tmp_pipeline_dir / 'master.sh'
    write_spw_master(
        str(out), config='pipeline_state.toml',
        SPWs='*:880~933MHz,*:960~1010MHz',
        precal_scripts=['partition.sbatch', 'partition_target.sbatch'],
        postcal_scripts=['concat_caltables.sbatch'],
        target_scripts=['target_validate_input.sbatch', 'target_flag_round_1.sbatch'],
        submit=False, timestamp='2026-05-06-12-00-00',
        slurm_kwargs={'account': 'b03-idia-ag', 'partition': 'Main', 'exclude': '', 'reservation': ''},
    )
    contents = _read(str(out))
    # First target script depends on $targetPartID
    assert 'targetIDs=$(sbatch -d afterok:${targetPartID//,/:}' in contents
    assert 'target_validate_input.sbatch' in contents
    # Second chains on previous target IDs
    assert 'targetIDs+=,$(sbatch -d afterok:${targetIDs//,/:}' in contents
    assert 'target_flag_round_1.sbatch' in contents


def test_master_postcal_joins_spw_and_target_branches(tmp_pipeline_dir):
    out = tmp_pipeline_dir / 'master.sh'
    write_spw_master(
        str(out), config='pipeline_state.toml',
        SPWs='*:880~933MHz,*:960~1010MHz',
        precal_scripts=['partition.sbatch', 'partition_target.sbatch'],
        postcal_scripts=['concat_caltables.sbatch', 'apply_to_target.sbatch'],
        target_scripts=['target_validate_input.sbatch', 'target_flag_round_1.sbatch'],
        submit=False, timestamp='2026-05-06-12-00-00',
        slurm_kwargs={'account': 'b03-idia-ag', 'partition': 'Main', 'exclude': '', 'reservation': ''},
    )
    contents = _read(str(out))
    # First postcal script (concat_caltables) waits on BOTH per-SPW IDs and target IDs
    assert 'sbatch -d afterany:${IDs//,/:}:${targetIDs//,/:}' in contents
    assert 'concat_caltables.sbatch' in contents


def test_master_no_target_branch_when_target_scripts_empty(tmp_pipeline_dir):
    out = tmp_pipeline_dir / 'master.sh'
    write_spw_master(
        str(out), config='pipeline_state.toml',
        SPWs='*:880~933MHz,*:960~1010MHz',
        precal_scripts=['partition.sbatch', 'partition_target.sbatch'],
        postcal_scripts=['concat_caltables.sbatch'],
        target_scripts=[],
        submit=False, timestamp='2026-05-06-12-00-00',
        slurm_kwargs={'account': 'b03-idia-ag', 'partition': 'Main', 'exclude': '', 'reservation': ''},
    )
    contents = _read(str(out))
    assert 'targetIDs' not in contents
    # Postcal still depends on per-SPW IDs only
    assert 'sbatch -d afterany:${IDs//,/:}' in contents
    assert ':${targetIDs' not in contents


def test_master_no_target_branch_when_partition_target_missing(tmp_pipeline_dir):
    """Even with target_scripts, if partition_target.sbatch isn't in precal, skip the branch."""
    out = tmp_pipeline_dir / 'master.sh'
    write_spw_master(
        str(out), config='pipeline_state.toml',
        SPWs='*:880~933MHz,*:960~1010MHz',
        precal_scripts=['partition.sbatch'],
        postcal_scripts=['concat_caltables.sbatch'],
        target_scripts=['target_validate_input.sbatch'],
        submit=False, timestamp='2026-05-06-12-00-00',
        slurm_kwargs={'account': 'b03-idia-ag', 'partition': 'Main', 'exclude': '', 'reservation': ''},
    )
    contents = _read(str(out))
    assert 'targetPartID' not in contents
    assert 'targetIDs' not in contents


def test_master_partition_id_is_first_when_partition_first(tmp_pipeline_dir):
    """The cut field index for partitionID must reflect partition's position in precal."""
    out = tmp_pipeline_dir / 'master.sh'
    write_spw_master(
        str(out), config='pipeline_state.toml',
        SPWs='*:880~933MHz',
        precal_scripts=['partition_target.sbatch', 'partition.sbatch'],  # reversed order
        postcal_scripts=[],
        target_scripts=['target_validate_input.sbatch'],
        submit=False, timestamp='2026-05-06-12-00-00',
        slurm_kwargs={'account': 'b03-idia-ag', 'partition': 'Main', 'exclude': '', 'reservation': ''},
    )
    contents = _read(str(out))
    # partition_target is field 1, partition is field 2
    assert 'targetPartID=$(echo $allSPWIDs | cut -d , -f1)' in contents
    assert 'partitionID=$(echo $allSPWIDs | cut -d , -f2)' in contents
