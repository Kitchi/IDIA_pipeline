"""Tests for facility abstraction — no SLURM required."""

import dataclasses
import pytest

from processMeerKAT.facilities import get_facility, FACILITIES
from processMeerKAT.facilities.base import FacilityConfig, _noop_validate_account, _noop_validate_reservation
from processMeerKAT.facilities.ilifu import ILIFU
from processMeerKAT.facilities.generic_slurm import GENERIC_SLURM
import processMeerKAT as pmk
import processMeerKAT.processMeerKAT as _pmk_mod


# ---------------------------------------------------------------------------
# FacilityConfig dataclass
# ---------------------------------------------------------------------------

class TestFacilityConfigDataclass:

    def test_ilifu_has_correct_limits(self):
        assert ILIFU.total_nodes_limit == 79
        assert ILIFU.cpus_per_node_limit == 32
        assert ILIFU.mem_per_node_gb_limit == 232
        assert ILIFU.mem_per_node_gb_limit_highmem == 480

    def test_ilifu_has_name(self):
        assert ILIFU.name == 'ilifu'

    def test_generic_slurm_has_name(self):
        assert GENERIC_SLURM.name == 'generic_slurm'

    def test_noop_validators_are_defaults(self):
        f = FacilityConfig(
            total_nodes_limit=10, cpus_per_node_limit=4,
            mem_per_node_gb_limit=64, mem_per_node_gb_limit_highmem=64,
            name='test',
        )
        assert f.validate_account is _noop_validate_account
        assert f.validate_reservation is _noop_validate_reservation

    def test_noop_validate_account_returns_account(self):
        assert _noop_validate_account('myaccount', 'cfg.txt') == 'myaccount'

    def test_noop_validate_reservation_returns_none(self):
        assert _noop_validate_reservation('myres', {}, 'cfg.txt') is None

    def test_custom_validator_used(self):
        called = []
        def my_validator(account, config, parser=None):
            called.append(account)
            return account
        f = FacilityConfig(
            total_nodes_limit=10, cpus_per_node_limit=4,
            mem_per_node_gb_limit=64, mem_per_node_gb_limit_highmem=64,
            name='custom', validate_account=my_validator,
        )
        f.validate_account('grp', 'cfg.txt')
        assert called == ['grp']


# ---------------------------------------------------------------------------
# get_facility
# ---------------------------------------------------------------------------

class TestGetFacility:

    def test_known_name_returns_instance(self):
        f = get_facility('ilifu')
        assert f is ILIFU

    def test_generic_slurm_returns_instance(self):
        f = get_facility('generic_slurm')
        assert f is GENERIC_SLURM

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown facility 'badname'"):
            get_facility('badname')

    def test_override_single_field(self):
        f = get_facility('ilifu', total_nodes_limit=10)
        assert f.total_nodes_limit == 10
        # All other fields unchanged
        assert f.cpus_per_node_limit == ILIFU.cpus_per_node_limit
        assert f.default_container == ILIFU.default_container

    def test_override_preserves_validators(self):
        """Overriding a data field must not drop the Ilifu validators."""
        f = get_facility('ilifu', total_nodes_limit=5)
        assert f.validate_account is ILIFU.validate_account
        assert f.validate_reservation is ILIFU.validate_reservation

    def test_override_returns_new_instance(self):
        f = get_facility('ilifu', total_nodes_limit=5)
        assert f is not ILIFU

    def test_no_override_returns_same_instance(self):
        f = get_facility('ilifu')
        assert f is ILIFU

    def test_all_known_facilities_present(self):
        assert 'ilifu' in FACILITIES
        assert 'generic_slurm' in FACILITIES


# ---------------------------------------------------------------------------
# load_facility_from_config
# ---------------------------------------------------------------------------

class TestLoadFacilityFromConfig:

    def test_no_facility_section_returns_default(self, minimal_config, monkeypatch):
        """Config without [facility] section → current default facility returned."""
        # Strip the [facility] section from the fixture config
        text = open(minimal_config).read()
        text = '\n'.join(
            line for line in text.splitlines()
            if not line.startswith('[facility]') and "name = 'ilifu'" not in line
        )
        open(minimal_config, 'w').write(text)
        monkeypatch.setattr(_pmk_mod, '_FACILITY', ILIFU)
        result = pmk.load_facility_from_config(minimal_config)
        assert result is ILIFU

    def test_known_facility_name_loaded(self, minimal_config, monkeypatch):
        monkeypatch.setattr(_pmk_mod, '_FACILITY', ILIFU)
        result = pmk.load_facility_from_config(minimal_config)
        assert result is ILIFU

    def test_override_field_via_config(self, tmp_path, monkeypatch):
        cfg = tmp_path / 'cfg.toml'
        cfg.write_text('[facility]\nname = "ilifu"\ntotal_nodes_limit = 10\n')
        monkeypatch.setattr(_pmk_mod, '_FACILITY', ILIFU)
        result = pmk.load_facility_from_config(str(cfg))
        assert result.total_nodes_limit == 10
        assert result.cpus_per_node_limit == ILIFU.cpus_per_node_limit

    def test_unknown_facility_name_raises(self, tmp_path, monkeypatch):
        cfg = tmp_path / 'cfg.toml'
        cfg.write_text('[facility]\nname = "badname"\n')
        monkeypatch.setattr(_pmk_mod, '_FACILITY', ILIFU)
        with pytest.raises(ValueError, match="Unknown facility"):
            pmk.load_facility_from_config(str(cfg))


# ---------------------------------------------------------------------------
# validate_args uses _FACILITY limits (not constants)
# ---------------------------------------------------------------------------

class TestValidateArgsUsesFacilityLimits:

    def _base_args(self):
        return {
            'ntasks_per_node': 8,
            'nodes': 1,
            'mem': 10,
            'plane': 1,
            'MS': None,
            'nofields': True,
            'build': False,
        }

    def test_nodes_within_facility_limit_passes(self, monkeypatch):
        monkeypatch.setattr(_pmk_mod, '_FACILITY', get_facility('ilifu'))
        args = self._base_args()
        args['nodes'] = 79
        pmk.validate_args(args, 'cfg.txt')  # must not raise

    def test_nodes_exceed_facility_limit_raises(self, monkeypatch):
        monkeypatch.setattr(_pmk_mod, '_FACILITY', get_facility('ilifu'))
        args = self._base_args()
        args['nodes'] = 80
        with pytest.raises(ValueError, match="must not exceed 79"):
            pmk.validate_args(args, 'cfg.txt')

    def test_custom_facility_limit_used(self, monkeypatch):
        small_facility = get_facility('ilifu', total_nodes_limit=5)
        monkeypatch.setattr(_pmk_mod, '_FACILITY', small_facility)
        args = self._base_args()
        args['nodes'] = 6
        with pytest.raises(ValueError, match="must not exceed 5"):
            pmk.validate_args(args, 'cfg.txt')

    def test_mem_within_limit_passes(self, monkeypatch):
        monkeypatch.setattr(_pmk_mod, '_FACILITY', get_facility('ilifu'))
        args = self._base_args()
        args['mem'] = 232
        pmk.validate_args(args, 'cfg.txt')

    def test_mem_exceeds_limit_raises(self, monkeypatch):
        monkeypatch.setattr(_pmk_mod, '_FACILITY', get_facility('ilifu'))
        args = self._base_args()
        args['mem'] = 300
        with pytest.raises(ValueError, match="must not exceed 232"):
            pmk.validate_args(args, 'cfg.txt')

    def test_highmem_partition_allows_higher_mem(self, monkeypatch):
        monkeypatch.setattr(_pmk_mod, '_FACILITY', get_facility('ilifu'))
        args = self._base_args()
        args['mem'] = 300
        args['partition'] = 'HighMem'
        pmk.validate_args(args, 'cfg.txt')  # must not raise

    def test_highmem_exceeds_highmem_limit_raises(self, monkeypatch):
        monkeypatch.setattr(_pmk_mod, '_FACILITY', get_facility('ilifu'))
        args = self._base_args()
        args['mem'] = 500
        args['partition'] = 'HighMem'
        with pytest.raises(ValueError, match="must not exceed 480"):
            pmk.validate_args(args, 'cfg.txt')

    def test_facility_validation_not_called_during_validate_args(self, monkeypatch):
        """validate_args must NOT call validate_account or validate_reservation."""
        called = []
        f = dataclasses.replace(
            ILIFU,
            validate_account=lambda *a, **kw: called.append('account') or a[0],
            validate_reservation=lambda *a, **kw: called.append('reservation'),
        )
        monkeypatch.setattr(_pmk_mod, '_FACILITY', f)
        args = self._base_args()
        pmk.validate_args(args, 'cfg.txt')
        assert called == [], "Facility validation must not run during validate_args (only during -R)"
