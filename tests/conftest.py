"""Pytest fixtures shared across all tests."""

import os
import pytest

TEST_MS = '/home/krishna/work/data/test_refactor/1538856059_sdp_l0.cals.1350-1450.scan1and3.ms'

MINIMAL_CONFIG = """\
[data]
vis = "test.ms"

[fields]
bpassfield = "0"
fluxfield = "0"
phasecalfield = "1"
targetfields = "2"
extrafields = ""

[slurm]
nodes = 1
ntasks_per_node = 8
plane = 1
mem = 232
partition = "Main"
exclude = ""
time = "12:00:00"
submit = false
runner = "singularity exec"
container = "/idia/software/containers/casa-6.5.0-modular.sif"
mpi_wrapper = "mpirun"
name = ""
dependencies = ""
account = "b03-idia-ag"
reservation = ""
modules = ["openmpi/4.0.3"]
verbose = false
precal_scripts = [
  {script = "calc_refant.py", mpi = false},
  {script = "partition.py", mpi = true},
]
postcal_scripts = [
  {script = "concat.py", mpi = false},
  {script = "plotcal_spw.py", mpi = false},
]
scripts = [
  {script = "validate_input.py", mpi = false},
  {script = "flag_round_1.py", mpi = true},
  {script = "split.py", mpi = true},
]
target_scripts = [
  {script = "validate_input.py", mpi = false},
  {script = "flag_round_1.py", mpi = true},
]

[crosscal]
minbaselines = 4
chanbin = 1
width = 1
timeavg = "8s"
createmms = true
keepmms = true
spw = "*:880~1080MHz"
nspw = 1
calcrefant = false
refant = "m059"
standard = "Stevens-Reynolds 2016"
badants = []
badfreqranges = ["933~960MHz"]
badfreq_uvrange = ""

[state]
continue = true
dopol = false

[facility]
name = "ilifu"
"""

MULTIBAND_CONFIG = """\
[data]
vis = "test.ms"

[fields]
bpassfield = "0"
fluxfield = "0"
phasecalfield = "1"
targetfields = "2"
extrafields = ""

[slurm]
nodes = 1
ntasks_per_node = 8
plane = 1
mem = 232
partition = "Main"
exclude = ""
time = "12:00:00"
submit = false
runner = "singularity exec"
container = "/idia/software/containers/casa-6.5.0-modular.sif"
mpi_wrapper = "mpirun"
name = ""
dependencies = ""
account = "b03-idia-ag"
reservation = ""
modules = ["openmpi/4.0.3"]
verbose = false
precal_scripts = [
  {script = "calc_refant.py", mpi = false},
  {script = "partition.py", mpi = true},
]
postcal_scripts = [
  {script = "concat.py", mpi = false},
]
scripts = [
  {script = "validate_input.py", mpi = false},
  {script = "flag_round_1.py", mpi = true},
]

[crosscal]
minbaselines = 4
chanbin = 1
width = 1
timeavg = "8s"
createmms = true
keepmms = true
spw = "*:880~933MHz,*:960~1010MHz,*:1010~1060MHz"
nspw = 3
calcrefant = false
refant = "m059"
standard = "Stevens-Reynolds 2016"
badants = []
badfreqranges = ["933~960MHz"]
badfreq_uvrange = ""

[state]
continue = true
dopol = false

[facility]
name = "ilifu"
"""


NSPW2_CONFIG = """\
[data]
vis = "test.ms"

[fields]
bpassfield = "0"
fluxfield = "0"
phasecalfield = "1"
targetfields = "2"
extrafields = ""

[slurm]
nodes = 1
ntasks_per_node = 2
plane = 1
mem = 14
partition = "debug"
exclude = ""
time = "12:00:00"
submit = false
runner = ""
container = ""
mpi_wrapper = "mpirun"
name = ""
dependencies = ""
account = ""
reservation = ""
modules = []
verbose = false
precal_scripts = [
  {script = "partition.py", mpi = true},
  {script = "partition_target.py", mpi = true},
]
postcal_scripts = [
  {script = "concat_caltables.py", mpi = false},
  {script = "apply_to_target.py", mpi = true},
]
scripts = [
  {script = "validate_input.py", mpi = false},
  {script = "flag_round_1.py", mpi = true},
]
target_scripts = [
  {script = "validate_input.py", mpi = false},
  {script = "flag_round_1.py", mpi = true},
]

[crosscal]
minbaselines = 4
chanbin = 1
width = 1
timeavg = "8s"
createmms = true
keepmms = true
spw = "*:880~960MHz,*:960~1080MHz"
nspw = 2
calcrefant = false
refant = "m059"
standard = "Stevens-Reynolds 2016"
badants = []
badfreqranges = []
badfreq_uvrange = ""

[state]
continue = true
dopol = false

[facility]
name = "generic_slurm"
"""


@pytest.fixture
def tmp_pipeline_dir(tmp_path):
    """Return a temp directory with a minimal pipeline_state.toml so write_sbatch doesn't blow up."""
    state_file = tmp_path / 'pipeline_state.toml'
    state_file.write_text(MINIMAL_CONFIG)
    old_dir = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(old_dir)


@pytest.fixture
def minimal_config(tmp_path):
    """Write a minimal config file to a temp dir and return its path."""
    cfg = tmp_path / 'test_config.toml'
    cfg.write_text(MINIMAL_CONFIG)
    return str(cfg)


@pytest.fixture
def multiband_config(tmp_path):
    """Write a multi-SPW config to a temp dir and return its path."""
    cfg = tmp_path / 'test_config.toml'
    cfg.write_text(MULTIBAND_CONFIG)
    return str(cfg)


@pytest.fixture
def wj_nspw1_dir(tmp_path):
    """CWD-switched tmp dir with pipeline_state.toml + config.toml for nspw=1 write_jobs tests."""
    (tmp_path / 'pipeline_state.toml').write_text(MINIMAL_CONFIG)
    (tmp_path / 'config.toml').write_text(MINIMAL_CONFIG)
    old = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(old)


@pytest.fixture
def wj_nspw2_dir(tmp_path):
    """CWD-switched tmp dir with pipeline_state.toml + config.toml for nspw=2 write_jobs tests."""
    (tmp_path / 'pipeline_state.toml').write_text(NSPW2_CONFIG)
    (tmp_path / 'config.toml').write_text(NSPW2_CONFIG)
    old = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(old)


@pytest.fixture
def test_ms():
    """Return path to the test MeasurementSet (skip if missing)."""
    if not os.path.isdir(TEST_MS):
        pytest.skip(f'Test MS not found: {TEST_MS}')
    return TEST_MS
