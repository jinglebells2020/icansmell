"""Tests for sniffsniff.config — Channel, Config, load_config, default_config."""
import numpy as np
import pytest

from sniffsniff.config import Channel, Config, default_config, load_config

REPO_TOML = "sniffsniff.toml"  # resolved relative to repo root below

EXPECTED_SENSORS = ["MQ3", "MQ135", "MQ2", "MQ4", "MQ8", "MQ7"]

# The shipped sniffsniff.toml is the current dual-Uno, 9-sensor rig.
DUAL_SENSORS = ["MQ5", "MQ3", "MQ135", "MQ7", "MQ9", "MQ8", "MQ2", "MQ4", "MQ6"]


def _repo_root():
    # tests/ is directly under the repo root
    from pathlib import Path

    return Path(__file__).resolve().parent.parent


# --- default_config -----------------------------------------------------------


def test_default_config_channel_count():
    cfg = default_config()
    assert cfg.n_channels == 6
    assert len(cfg.channels) == 6


def test_default_config_sensor_order():
    cfg = default_config()
    assert cfg.sensor_names() == EXPECTED_SENSORS


def test_default_config_channel_indices_ordered():
    cfg = default_config()
    assert [c.ch for c in cfg.channels] == [0, 1, 2, 3, 4, 5]


def test_default_config_rl_array():
    cfg = default_config()
    rl = cfg.rl_array()
    assert isinstance(rl, np.ndarray)
    assert rl.shape == (6,)
    assert rl.dtype == np.float64
    np.testing.assert_array_equal(rl, np.full(6, 1000.0))


def test_default_config_board_and_array_scalars():
    cfg = default_config()
    assert cfg.bits == 10
    assert cfg.vref == 5.0
    assert cfg.vcc == 5.0


def test_default_config_timing():
    cfg = default_config()
    assert cfg.scan_hz == 20
    assert cfg.baseline_s == 15
    assert cfg.exposure_s == 45
    assert cfg.purge_s == 90
    assert cfg.plateau_s == 10


def test_default_config_features_and_baseline():
    cfg = default_config()
    assert cfg.ema_alphas == (0.1, 0.01, 0.001)
    assert cfg.max_cv == 0.05
    assert cfg.recover_tol == 0.02


def test_default_config_is_frozen():
    cfg = default_config()
    with pytest.raises(Exception):
        cfg.bits = 12  # type: ignore[misc]


def test_channel_is_frozen():
    ch = Channel(ch=0, sensor="MQ2", rl=1000.0)
    with pytest.raises(Exception):
        ch.ch = 1  # type: ignore[misc]


def test_sensor_names_is_list():
    cfg = default_config()
    names = cfg.sensor_names()
    assert isinstance(names, list)
    assert len(names) == 6


# --- load_config round-trip ---------------------------------------------------


def test_load_config_round_trips_repo_toml():
    path = _repo_root() / REPO_TOML
    assert path.exists(), f"expected {path} to exist"
    cfg = load_config(path)
    # shipped rig is the dual-Uno, 9-sensor array
    assert cfg.n_channels == 9
    assert cfg.sensor_names() == DUAL_SENSORS
    assert cfg.multi_board is True
    assert len(cfg.boards) == 2
    assert cfg.boards[0].n_channels == 6 and cfg.boards[1].n_channels == 3
    # the airflow servo is wired to Uno 2 (the 3-sensor board)
    assert cfg.boards[0].servo is False and cfg.boards[1].servo is True
    assert cfg.servo_enabled is True


def test_load_config_accepts_str_path():
    path = _repo_root() / REPO_TOML
    cfg = load_config(str(path))
    assert cfg.n_channels == 9
    assert cfg.sensor_names() == DUAL_SENSORS


def test_load_config_channel_count_matches(tmp_path):
    path = _repo_root() / REPO_TOML
    cfg = load_config(path)
    assert cfg.n_channels == 9
    assert cfg.sensor_names() == DUAL_SENSORS


# --- load_config validation ---------------------------------------------------

_VALID_HEADER = """
[board]
bits = 10
vref = 5.0

[array]
vcc = 5.0
"""

_TIMING_TAIL = """
[timing]
scan_hz = 20
baseline_s = 15
exposure_s = 45
purge_s = 90
plateau_s = 10

[features]
ema_alphas = [0.1, 0.01, 0.001]

[baseline]
max_cv = 0.05
recover_tol = 0.02
"""


def _write_toml(tmp_path, channels_block):
    text = _VALID_HEADER + channels_block + _TIMING_TAIL
    p = tmp_path / "cfg.toml"
    p.write_text(text)
    return p


def test_load_config_rejects_skipped_index(tmp_path):
    # ch values {0,1,2,3,4,6} — index 5 skipped, 6 out of range
    block = """
channels = [
  { ch = 0, sensor = "MQ2",   rl = 1000 },
  { ch = 1, sensor = "MQ3",   rl = 1000 },
  { ch = 2, sensor = "MQ4",   rl = 1000 },
  { ch = 3, sensor = "MQ7",   rl = 1000 },
  { ch = 4, sensor = "MQ8",   rl = 1000 },
  { ch = 6, sensor = "MQ135", rl = 1000 },
]
"""
    p = _write_toml(tmp_path, block)
    with pytest.raises(ValueError):
        load_config(p)


def test_load_config_rejects_duplicate_ch(tmp_path):
    # ch values {0,1,2,3,4,4} — duplicate 4, missing 5
    block = """
channels = [
  { ch = 0, sensor = "MQ2",   rl = 1000 },
  { ch = 1, sensor = "MQ3",   rl = 1000 },
  { ch = 2, sensor = "MQ4",   rl = 1000 },
  { ch = 3, sensor = "MQ7",   rl = 1000 },
  { ch = 4, sensor = "MQ8",   rl = 1000 },
  { ch = 4, sensor = "MQ135", rl = 1000 },
]
"""
    p = _write_toml(tmp_path, block)
    with pytest.raises(ValueError):
        load_config(p)


def test_load_config_accepts_valid_reordered(tmp_path):
    # ch values {0..5} but listed out of order — should be accepted and re-sorted
    block = """
channels = [
  { ch = 2, sensor = "MQ2",   rl = 1000 },
  { ch = 0, sensor = "MQ3",   rl = 1000 },
  { ch = 5, sensor = "MQ7",   rl = 1000 },
  { ch = 1, sensor = "MQ135", rl = 1000 },
  { ch = 4, sensor = "MQ8",   rl = 1000 },
  { ch = 3, sensor = "MQ4",   rl = 1000 },
]
"""
    p = _write_toml(tmp_path, block)
    cfg = load_config(p)
    assert [c.ch for c in cfg.channels] == [0, 1, 2, 3, 4, 5]
    assert cfg.sensor_names() == EXPECTED_SENSORS


def test_load_config_smaller_array_is_channel_agnostic(tmp_path):
    # 3-channel config: exercises that N is derived from the table, not hard-coded 6.
    block = """
channels = [
  { ch = 0, sensor = "MQ2", rl = 1000 },
  { ch = 1, sensor = "MQ3", rl = 2000 },
  { ch = 2, sensor = "MQ4", rl = 3000 },
]
"""
    p = _write_toml(tmp_path, block)
    cfg = load_config(p)
    assert cfg.n_channels == 3
    assert cfg.sensor_names() == ["MQ2", "MQ3", "MQ4"]
    np.testing.assert_array_equal(cfg.rl_array(), np.array([1000.0, 2000.0, 3000.0]))


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises((FileNotFoundError, OSError)):
        load_config(tmp_path / "does_not_exist.toml")


# --- servo config ------------------------------------------------------------

def test_default_config_has_servo():
    c = default_config()
    assert c.servo_enabled is True
    assert c.servo_pin == 12
    assert c.servo_fresh_air_angle == 0
    assert c.servo_sample_angle == 105


def test_load_config_servo_defaults_when_section_missing(tmp_path):
    # a config with no [servo] section falls back to disabled defaults
    toml = tmp_path / "no_servo.toml"
    toml.write_text(
        """
[board]
bits = 10
vref = 5.0
[array]
vcc = 5.0
channels = [ {ch=0,sensor="MQ3",rl=1000}, {ch=1,sensor="MQ135",rl=1000} ]
[timing]
scan_hz=20
baseline_s=15
exposure_s=45
purge_s=90
plateau_s=10
[features]
ema_alphas=[0.1,0.01,0.001]
[baseline]
max_cv=0.05
recover_tol=0.02
"""
    )
    c = load_config(toml)
    assert c.servo_enabled is False        # absent section -> disabled
    assert c.servo_fresh_air_angle == 0
    assert c.servo_sample_angle == 105


def test_default_config_has_capture_tuning():
    c = default_config()
    assert c.settle_hold_s == 3.0
    assert c.settle_max_wait_s == 30.0
    assert c.min_exposure_s == 15.0
    assert c.plateau_hold_s == 8.0
    assert c.plateau_eps == 0.005
    assert c.smooth_alpha == 0.2


# --- multi-board (dual-Uno) parsing ------------------------------------------

_DUAL_HEADER = """
[board]
bits = 10
vref = 5.0

[array]
vcc = 5.0

[[array.board]]
port = "/dev/ttyA"
servo = true
channels = [
  { ch = 0, sensor = "MQ5",   rl = 1000 },
  { ch = 1, sensor = "MQ3",   rl = 1000 },
  { ch = 2, sensor = "MQ135", rl = 1100 },
]

[[array.board]]
port = "/dev/ttyB"
channels = [
  { ch = 0, sensor = "MQ2", rl = 2000 },
  { ch = 1, sensor = "MQ4", rl = 2000 },
]
"""


def _write_dual(tmp_path, header=_DUAL_HEADER):
    p = tmp_path / "dual.toml"
    p.write_text(header + _TIMING_TAIL)
    return p


def test_multiboard_flattens_channels_in_board_order(tmp_path):
    cfg = load_config(_write_dual(tmp_path))
    assert cfg.n_channels == 5
    assert cfg.sensor_names() == ["MQ5", "MQ3", "MQ135", "MQ2", "MQ4"]
    assert [c.ch for c in cfg.channels] == [0, 1, 2, 3, 4]
    assert [c.board for c in cfg.channels] == [0, 0, 0, 1, 1]
    np.testing.assert_array_equal(
        cfg.rl_array(), np.array([1000.0, 1000.0, 1100.0, 2000.0, 2000.0])
    )


def test_multiboard_board_layout_and_ports(tmp_path):
    cfg = load_config(_write_dual(tmp_path))
    assert cfg.multi_board is True
    assert len(cfg.boards) == 2
    b0, b1 = cfg.boards
    assert (b0.port, b0.n_channels, b0.servo, b0.start) == ("/dev/ttyA", 3, True, 0)
    assert (b1.port, b1.n_channels, b1.servo, b1.start) == ("/dev/ttyB", 2, False, 3)


def test_multiboard_servo_enabled_derived_from_boards(tmp_path):
    cfg = load_config(_write_dual(tmp_path))
    assert cfg.servo_enabled is True  # board 0 has servo=true


def test_multiboard_rejects_bad_local_ch(tmp_path):
    bad = """
[board]
bits = 10
vref = 5.0
[array]
vcc = 5.0
[[array.board]]
port = "/dev/ttyA"
channels = [ {ch=0,sensor="MQ5",rl=1000}, {ch=2,sensor="MQ3",rl=1000} ]
"""
    p = tmp_path / "bad.toml"
    p.write_text(bad + _TIMING_TAIL)
    with pytest.raises(ValueError):
        load_config(p)


def test_legacy_single_board_has_one_board(tmp_path):
    # a flat [array].channels config parses as a single board (port None)
    block = """
channels = [
  { ch = 0, sensor = "MQ2", rl = 1000 },
  { ch = 1, sensor = "MQ3", rl = 1000 },
]
"""
    cfg = load_config(_write_toml(tmp_path, block))
    assert cfg.multi_board is False
    assert len(cfg.boards) == 1
    assert cfg.boards[0].port is None
    assert cfg.boards[0].n_channels == 2
    assert cfg.boards[0].start == 0


def test_default_config_has_single_board():
    cfg = default_config()
    assert cfg.multi_board is False
    assert len(cfg.boards) == 1
    assert cfg.boards[0].n_channels == 6
    assert cfg.boards[0].servo is True
