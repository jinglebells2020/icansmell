"""Tests for sniffsniff.config — Channel, Config, load_config, default_config."""
import numpy as np
import pytest

from sniffsniff.config import Channel, Config, default_config, load_config

REPO_TOML = "sniffsniff.toml"  # resolved relative to repo root below

EXPECTED_SENSORS = ["MQ2", "MQ3", "MQ4", "MQ7", "MQ8", "MQ135"]


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
    assert cfg == default_config()


def test_load_config_accepts_str_path():
    path = _repo_root() / REPO_TOML
    cfg = load_config(str(path))
    assert cfg == default_config()


def test_load_config_channel_count_matches(tmp_path):
    path = _repo_root() / REPO_TOML
    cfg = load_config(path)
    assert cfg.n_channels == 6
    assert cfg.sensor_names() == EXPECTED_SENSORS


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
  { ch = 2, sensor = "MQ4",   rl = 1000 },
  { ch = 0, sensor = "MQ2",   rl = 1000 },
  { ch = 5, sensor = "MQ135", rl = 1000 },
  { ch = 1, sensor = "MQ3",   rl = 1000 },
  { ch = 4, sensor = "MQ8",   rl = 1000 },
  { ch = 3, sensor = "MQ7",   rl = 1000 },
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
