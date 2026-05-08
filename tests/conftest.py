"""Pytest fixtures shared across all tests."""

import os
import pytest

TEST_MS = '/home/krishna/work/data/test_refactor/1538856059_sdp_l0.cals.1350-1450.scan1and3.ms'

MINIMAL_CONFIG = """\
data:
  vis: test.ms

fields:
  bpassfield: '0'
  fluxfield: '0'
  phasecalfield: '1'
  targetfields: '2'
  extrafields: ''

slurm:
  nodes: 1
  ntasks_per_node: 8
  plane: 1
  mem: 232
  partition: Main
  exclude: ''
  time: '12:00:00'
  submit: false
  container: /idia/software/containers/casa-6.5.0-modular.sif
  mpi_wrapper: mpirun
  name: ''
  dependencies: ''
  account: b03-idia-ag
  reservation: ''
  modules:
    - openmpi/4.0.3
  verbose: false
  precal_scripts:
    - [calc_refant.py, false, '']
    - [partition.py, true, '']
  postcal_scripts:
    - [concat.py, false, '']
    - [plotcal_spw.py, false, '']
  scripts:
    - [validate_input.py, false, '']
    - [flag_round_1.py, true, '']
    - [split.py, true, '']
  target_scripts:
    - [validate_input.py, false, '']
    - [flag_round_1.py, true, '']

crosscal:
  minbaselines: 4
  chanbin: 1
  width: 1
  timeavg: 8s
  createmms: true
  keepmms: true
  spw: '*:880~1080MHz'
  nspw: 1
  calcrefant: false
  refant: m059
  standard: Stevens-Reynolds 2016
  badants: []
  badfreqranges:
    - 933~960MHz

run:
  continue: true
  dopol: false

facility:
  name: ilifu
"""

MULTIBAND_CONFIG = """\
data:
  vis: test.ms

fields:
  bpassfield: '0'
  fluxfield: '0'
  phasecalfield: '1'
  targetfields: '2'
  extrafields: ''

slurm:
  nodes: 1
  ntasks_per_node: 8
  plane: 1
  mem: 232
  partition: Main
  exclude: ''
  time: '12:00:00'
  submit: false
  container: /idia/software/containers/casa-6.5.0-modular.sif
  mpi_wrapper: mpirun
  name: ''
  dependencies: ''
  account: b03-idia-ag
  reservation: ''
  modules:
    - openmpi/4.0.3
  verbose: false
  precal_scripts:
    - [calc_refant.py, false, '']
    - [partition.py, true, '']
  postcal_scripts:
    - [concat.py, false, '']
  scripts:
    - [validate_input.py, false, '']
    - [flag_round_1.py, true, '']

crosscal:
  minbaselines: 4
  chanbin: 1
  width: 1
  timeavg: 8s
  createmms: true
  keepmms: true
  spw: '*:880~933MHz,*:960~1010MHz,*:1010~1060MHz'
  nspw: 3
  calcrefant: false
  refant: m059
  standard: Stevens-Reynolds 2016
  badants: []
  badfreqranges:
    - 933~960MHz

run:
  continue: true
  dopol: false

facility:
  name: ilifu
"""


@pytest.fixture
def tmp_pipeline_dir(tmp_path):
    """Return a temp directory with a minimal pipeline_state.yaml so write_sbatch doesn't blow up."""
    state_file = tmp_path / 'pipeline_state.yaml'
    state_file.write_text(MINIMAL_CONFIG)
    old_dir = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(old_dir)


@pytest.fixture
def minimal_config(tmp_path):
    """Write a minimal config file to a temp dir and return its path."""
    cfg = tmp_path / 'test_config.yaml'
    cfg.write_text(MINIMAL_CONFIG)
    return str(cfg)


@pytest.fixture
def multiband_config(tmp_path):
    """Write a multi-SPW config to a temp dir and return its path."""
    cfg = tmp_path / 'test_config.yaml'
    cfg.write_text(MULTIBAND_CONFIG)
    return str(cfg)


@pytest.fixture
def test_ms():
    """Return path to the test MeasurementSet (skip if missing)."""
    if not os.path.isdir(TEST_MS):
        pytest.skip(f'Test MS not found: {TEST_MS}')
    return TEST_MS
