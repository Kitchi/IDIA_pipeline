"""Tests for SPW-related utilities in processMeerKAT.py — no CASA required."""

import os
import pytest
import processMeerKAT as pmk


# ---------------------------------------------------------------------------
# get_spw_bounds
# ---------------------------------------------------------------------------

class TestGetSpwBounds:

    def test_simple_mhz_range(self):
        result = pmk.get_spw_bounds('*:880~1080MHz')
        assert result is not None
        low, high, unit, func = result
        assert low == 880
        assert high == 1080
        assert unit == 'MHz'

    def test_float_range(self):
        result = pmk.get_spw_bounds('*:880.5~1080.5MHz')
        assert result is not None
        low, high, unit, func = result
        assert low == 880.5
        assert high == 1080.5
        assert unit == 'MHz'
        assert func is float

    def test_integer_range_uses_int_func(self):
        _, _, _, func = pmk.get_spw_bounds('*:880~1080MHz')
        assert func is int

    def test_comma_separated_returns_none(self):
        """Comma-separated SPWs should return None (caller handles them differently)."""
        result = pmk.get_spw_bounds('*:880~933MHz,*:960~1010MHz')
        assert result is None

    def test_no_colon_returns_none(self):
        result = pmk.get_spw_bounds('880~1080MHz')
        assert result is None

    def test_no_tilde_returns_none(self):
        result = pmk.get_spw_bounds('*:8801080MHz')
        assert result is None

    def test_channel_range(self):
        """Channel ranges (no unit suffix) use int func."""
        result = pmk.get_spw_bounds('*:0~63')
        assert result is not None
        low, high, unit, func = result
        assert low == 0
        assert high == 63
        assert unit == ''
        assert func is int


# ---------------------------------------------------------------------------
# linspace
# ---------------------------------------------------------------------------

class TestLinspace:

    def test_two_elements(self):
        result = pmk.linspace(0, 10, 2)
        assert result == [0, 10]

    def test_five_elements(self):
        result = pmk.linspace(0, 4, 5)
        assert result == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_same_length_as_requested(self):
        for n in [2, 3, 5, 11]:
            assert len(pmk.linspace(880, 1680, n)) == n

    def test_first_and_last(self):
        result = pmk.linspace(880, 1680, 11)
        assert result[0] == 880
        assert result[-1] == 1680


# ---------------------------------------------------------------------------
# spw_split
# ---------------------------------------------------------------------------

class TestSpwSplit:

    CFG_NAME = 'test_config.txt'

    def _make_config(self, tmp_path, spw, nspw, badfreqranges=None):
        """Helper: write a minimal config inside tmp_path and return its *basename*.
        Caller must os.chdir(tmp_path) before calling spw_split so the relative path resolves."""
        if badfreqranges is None:
            badfreqranges = []
        cfg_text = f"""\
[data]
vis = 'test.ms'

[fields]
bpassfield = '0'
fluxfield = '0'
phasecalfield = '1'
targetfields = '2'
extrafields = ''

[crosscal]
minbaselines = 4
chanbin = 1
width = 1
timeavg = '8s'
createmms = True
keepmms = True
spw = '{spw}'
nspw = {nspw}
calcrefant = False
refant = 'm059'
standard = 'Stevens-Reynolds 2016'
badants = []
badfreqranges = {badfreqranges!r}

[slurm]
nodes = 1
ntasks_per_node = 8
plane = 1
mem = 232
partition = 'Main'
exclude = ''
time = '12:00:00'
submit = False
container = '/idia/software/containers/casa-6.5.0-modular.sif'
mpi_wrapper = 'mpirun'
name = ''
dependencies = ''
account = 'b03-idia-ag'
reservation = ''
modules = ['openmpi/4.0.3']
verbose = False
precal_scripts = []
postcal_scripts = []
scripts = [('validate_input.py', False, '')]

[run]
continue = True
dopol = False
"""
        (tmp_path / self.CFG_NAME).write_text(cfg_text)
        return self.CFG_NAME  # relative path — caller must chdir first

    def test_creates_correct_number_of_directories(self, tmp_path):
        old_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            cfg = self._make_config(tmp_path, '*:880~1080MHz', 4)
            returned_nspw = pmk.spw_split('*:880~1080MHz', 4, cfg, 232, [], 'test.ms', partition=True)
            assert returned_nspw == 4
            dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
            assert len(dirs) == 4
        finally:
            os.chdir(old_dir)

    def test_directory_names_contain_frequency_range(self, tmp_path):
        old_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            cfg = self._make_config(tmp_path, '*:880~1080MHz', 2)
            pmk.spw_split('*:880~1080MHz', 2, cfg, 232, [], 'test.ms', partition=True)
            dir_names = [d.name for d in tmp_path.iterdir() if d.is_dir()]
            assert any('MHz' in n for n in dir_names)
        finally:
            os.chdir(old_dir)

    def test_each_directory_gets_config_copy(self, tmp_path):
        old_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            cfg = self._make_config(tmp_path, '*:880~1080MHz', 3)
            pmk.spw_split('*:880~1080MHz', 3, cfg, 232, [], 'test.ms', partition=True)
            spw_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
            for d in spw_dirs:
                assert (d / self.CFG_NAME).exists(), f"Config missing in {d}"
        finally:
            os.chdir(old_dir)

    def test_spw_config_has_nspw_1(self, tmp_path):
        """Each per-SPW config should have nspw=1."""
        import processMeerKAT.config_parser as config_parser
        old_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            cfg = self._make_config(tmp_path, '*:880~1080MHz', 2)
            pmk.spw_split('*:880~1080MHz', 2, cfg, 232, [], 'test.ms', partition=True)
            spw_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
            for d in spw_dirs:
                spw_cfg = str(d / self.CFG_NAME)
                assert config_parser.get_key(spw_cfg, 'crosscal', 'nspw') == 1
        finally:
            os.chdir(old_dir)

    def test_badfreqrange_removes_spw(self, tmp_path):
        """An SPW completely within a bad frequency range should be removed."""
        old_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            # 2 SPWs: 880~980MHz and 980~1080MHz. Mark 880~980MHz as bad.
            cfg = self._make_config(tmp_path, '*:880~1080MHz', 2, badfreqranges=['880~980MHz'])
            returned_nspw = pmk.spw_split('*:880~1080MHz', 2, cfg, 232, ['880~980MHz'], 'test.ms', partition=True)
            assert returned_nspw == 1
        finally:
            os.chdir(old_dir)

    def test_prebuilt_comma_separated_spws(self, tmp_path):
        """Comma-separated SPW string: each element is a separate directory."""
        old_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            spw = '*:880~933MHz,*:960~1010MHz,*:1010~1060MHz'
            cfg = self._make_config(tmp_path, spw, 3)
            returned_nspw = pmk.spw_split(spw, 3, cfg, 232, [], 'test.ms', partition=True)
            assert returned_nspw == 3
        finally:
            os.chdir(old_dir)

    def test_invalid_spw_returns_1(self, tmp_path):
        """An SPW format that can't be split returns nspw=1."""
        old_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            cfg = self._make_config(tmp_path, '*:880', 2)  # No tilde, can't split
            returned_nspw = pmk.spw_split('*:880', 2, cfg, 232, [], 'test.ms', partition=True)
            assert returned_nspw == 1
        finally:
            os.chdir(old_dir)
