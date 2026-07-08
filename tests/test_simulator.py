"""Tests for the synthetic MQ-array simulator.

The simulator emits frames in the exact ``(t_ms, raw[N])`` shape of the real
serial reader, produced by *inverting* the calibration path (Rs -> V_RL ->
counts), with seeded Gaussian noise for byte-reproducibility.
"""
from __future__ import annotations

import numpy as np
import pytest

from sniffsniff.config import default_config
from sniffsniff import calibrate
from sniffsniff.simulator import (
    ODOR_PROFILES,
    Simulator,
    SimulatedReader,
)


# --- ODOR_PROFILES -----------------------------------------------------------

def test_odor_profiles_present_and_shaped():
    required = {
        "clean_air",
        "coffee",
        "vinegar",
        "alcohol",
        "fresh_milk",
        "spoiled_milk",
    }
    assert required <= set(ODOR_PROFILES)
    # Each profile is now a {sensor_name: gain} mapping.
    for name, profile in ODOR_PROFILES.items():
        assert isinstance(profile, dict), f"{name} profile must be a dict"
        for sensor, gain in profile.items():
            assert isinstance(sensor, str), f"{name} keys must be sensor names"
            assert isinstance(gain, float), f"{name}[{sensor}] gain must be a float"
            assert gain >= 0.0, f"{name}[{sensor}] gain must be non-negative"


def test_clean_air_is_all_zero_gain():
    # clean_air has no per-sensor gains: an empty mapping -> zero everywhere.
    assert ODOR_PROFILES["clean_air"] == {}


def test_alcohol_dominant_on_mq3():
    names = default_config().sensor_names()
    idx = names.index("MQ3")
    g = Simulator(default_config(), seed=0)._gain("alcohol")
    assert np.argmax(g) == idx


def test_vinegar_and_spoiled_milk_high_on_mq135():
    names = default_config().sensor_names()
    idx = names.index("MQ135")
    sim = Simulator(default_config(), seed=0)
    assert np.argmax(sim._gain("vinegar")) == idx
    assert np.argmax(sim._gain("spoiled_milk")) == idx


def test_fresh_vs_spoiled_milk_differ_on_dairy_axes():
    fresh = ODOR_PROFILES["fresh_milk"]
    spoiled = ODOR_PROFILES["spoiled_milk"]
    # they must genuinely differ on the dairy axes (MQ4/MQ7/MQ8)
    assert any(fresh[s] != spoiled[s] for s in ("MQ4", "MQ7", "MQ8"))


# --- Frame shape / count / timing --------------------------------------------

def test_frame_count_matches_timing():
    cfg = default_config()
    sim = Simulator(cfg, seed=0)
    frames = sim.sniff_frames("coffee")
    expected = int(
        round((cfg.baseline_s + cfg.exposure_s + cfg.purge_s) * cfg.scan_hz)
    )
    assert len(frames) == expected


def test_frame_shapes_and_dtypes():
    cfg = default_config()
    sim = Simulator(cfg, seed=0)
    frames = sim.sniff_frames("clean_air")
    t0, raw0 = frames[0]
    assert isinstance(t0, int)
    assert raw0.shape == (cfg.n_channels,)
    assert raw0.dtype == np.int64


def test_t_ms_increments_by_step_from_zero():
    cfg = default_config()
    sim = Simulator(cfg, seed=0)
    frames = sim.sniff_frames("clean_air")
    step = round(1000 / cfg.scan_hz)  # 50 ms at 20 Hz
    assert frames[0][0] == 0
    assert frames[1][0] == step
    assert frames[5][0] == 5 * step


def test_counts_within_adc_range():
    cfg = default_config()
    sim = Simulator(cfg, seed=3, noise_counts=5.0)
    for odor in ("clean_air", "coffee", "alcohol", "spoiled_milk"):
        frames = sim.sniff_frames(odor)
        allraw = np.array([r for _, r in frames])
        assert allraw.min() >= 0
        assert allraw.max() <= (2 ** cfg.bits - 1)


# --- Determinism -------------------------------------------------------------

def test_same_seed_identical_frames():
    cfg = default_config()
    a = Simulator(cfg, seed=42).sniff_frames("vinegar")
    b = Simulator(cfg, seed=42).sniff_frames("vinegar")
    assert len(a) == len(b)
    for (ta, ra), (tb, rb) in zip(a, b):
        assert ta == tb
        assert np.array_equal(ra, rb)


def test_different_seed_differs_when_noisy():
    cfg = default_config()
    a = Simulator(cfg, seed=1, noise_counts=2.0).sniff_frames("coffee")
    b = Simulator(cfg, seed=2, noise_counts=2.0).sniff_frames("coffee")
    arr_a = np.array([r for _, r in a])
    arr_b = np.array([r for _, r in b])
    assert not np.array_equal(arr_a, arr_b)


# --- Fractional signatures ---------------------------------------------------

def _plateau_fractional(cfg, frames):
    """Fractional response at the end of exposure, using R0 from this frame set's
    own clean-air baseline window."""
    raw = np.array([r for _, r in frames], dtype=np.float64)
    rl = cfg.rl_array()
    rs = calibrate.counts_to_rs(raw, rl, cfg.vcc, cfg.vref, cfg.bits)
    n_base = int(round(cfg.baseline_s * cfg.scan_hz))
    n_exp = int(round(cfg.exposure_s * cfg.scan_hz))
    r0 = rs[:n_base].mean(axis=0)
    # last frame of exposure
    exp_end = n_base + n_exp - 1
    y = rs[exp_end] / r0 - 1.0
    return y


def test_clean_air_near_zero_fractional_at_plateau():
    cfg = default_config()
    # no noise so we can assert tight near-zero
    sim = Simulator(cfg, seed=0, noise_counts=0.0)
    frames = sim.sniff_frames("clean_air")
    y = _plateau_fractional(cfg, frames)
    assert np.allclose(y, 0.0, atol=1e-6)


def test_coffee_vs_vinegar_plateau_differ():
    cfg = default_config()
    sim = Simulator(cfg, seed=0, noise_counts=0.0)
    yc = _plateau_fractional(cfg, sim.sniff_frames("coffee"))
    yv = _plateau_fractional(cfg, sim.sniff_frames("vinegar"))
    assert not np.allclose(yc, yv)


def test_fresh_vs_spoiled_milk_plateau_differ():
    cfg = default_config()
    sim = Simulator(cfg, seed=0, noise_counts=0.0)
    yf = _plateau_fractional(cfg, sim.sniff_frames("fresh_milk"))
    ys = _plateau_fractional(cfg, sim.sniff_frames("spoiled_milk"))
    assert not np.allclose(yf, ys)


def test_odor_pulls_fractional_negative():
    """Reducing gases drop Rs below R0 during exposure -> negative fractional."""
    cfg = default_config()
    sim = Simulator(cfg, seed=0, noise_counts=0.0)
    y = _plateau_fractional(cfg, sim.sniff_frames("alcohol"))
    # MQ3 is the alcohol responder; its channel should be clearly negative
    idx = cfg.sensor_names().index("MQ3")
    assert y[idx] < -0.05


def test_unknown_odor_raises():
    cfg = default_config()
    sim = Simulator(cfg, seed=0)
    with pytest.raises(KeyError):
        sim.sniff_frames("nonsense_odor")


# --- SimulatedReader ---------------------------------------------------------

def test_simulated_reader_replays_exactly():
    cfg = default_config()
    frames = Simulator(cfg, seed=7).sniff_frames("coffee")
    reader = SimulatedReader(frames)
    replayed = list(reader.frames())
    assert len(replayed) == len(frames)
    for (ta, ra), (tb, rb) in zip(frames, replayed):
        assert ta == tb
        assert np.array_equal(ra, rb)
    reader.close()  # must not raise


def test_simulated_reader_frames_is_iterator():
    frames = [(0, np.zeros(6, dtype=np.int64)), (50, np.ones(6, dtype=np.int64))]
    reader = SimulatedReader(frames)
    it = reader.frames()
    first = next(it)
    assert first[0] == 0
    assert np.array_equal(first[1], np.zeros(6, dtype=np.int64))


# --- Custom channel count (channel-agnostic derivation) ----------------------

def test_respects_config_n_channels():
    """r_base default and frame width derive from config.n_channels, not a
    hard-coded 6 (gain vectors are still 6-aligned but sliced to N)."""
    from sniffsniff.config import Board, Config, Channel

    channels = tuple(
        Channel(ch=i, sensor=f"S{i}", rl=1000.0) for i in range(3)
    )
    cfg = Config(
        bits=10, vref=5.0, vcc=5.0, channels=channels,
        boards=(Board(port=None, n_channels=3, servo=False, start=0),),
        scan_hz=20, baseline_s=1.0, exposure_s=1.0, purge_s=1.0, plateau_s=0.5,
        ema_alphas=(0.1, 0.01, 0.001), max_cv=0.05, recover_tol=0.02,
    )
    sim = Simulator(cfg, seed=0)
    frames = sim.sniff_frames("coffee")
    assert frames[0][1].shape == (3,)


def test_tau_decay_slower_than_rise():
    """Purge desorption must be slower than exposure adsorption (spec: tau_decay > tau_rise)."""
    sim = Simulator(default_config(), seed=0)
    assert sim.tau_decay > sim.tau_rise


# --- dual-Uno 9-sensor rig ----------------------------------------------------

def _repo_toml():
    from pathlib import Path
    return Path(__file__).resolve().parents[1] / "sniffsniff.toml"


def test_simulator_9ch_rig_new_sensors_are_alive():
    """The shipped dual-Uno rig: frames are 9 wide and the new MQ5/MQ6/MQ9 sensors
    actually respond to an odor that gives them gain (they aren't dead in sim)."""
    from sniffsniff.config import load_config

    cfg = load_config(_repo_toml())
    assert cfg.n_channels == 9
    sim = Simulator(cfg, seed=0, noise_counts=0.0)
    frames = sim.sniff_frames("fresh_milk")
    assert frames[0][1].shape == (9,)

    names = cfg.sensor_names()
    n_base = int(round(cfg.baseline_s * cfg.scan_hz))
    n_exp = int(round(cfg.exposure_s * cfg.scan_hz))
    base = frames[0][1].astype(float)
    exp_end = frames[n_base + n_exp - 1][1].astype(float)
    for s in ("MQ5", "MQ6", "MQ9"):
        i = names.index(s)
        assert abs(exp_end[i] - base[i]) > 1.0, f"{s} stayed flat for fresh_milk"


def test_record_9ch_rig_yields_72d_features(tmp_path):
    """End-to-end: a 9-channel sim sniff records a 72-D feature vector (8 × 9)."""
    import dataclasses
    import warnings

    from sniffsniff.config import load_config
    from sniffsniff.record import SniffRecorder

    cfg = load_config(_repo_toml())
    cfg = dataclasses.replace(cfg, baseline_s=1, exposure_s=2, purge_s=1, plateau_s=0.5)
    frames = Simulator(cfg, seed=0).sniff_frames("coffee")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = SniffRecorder(cfg, tmp_path).process(frames, "coffee")
    assert result.features.shape == (72,)
