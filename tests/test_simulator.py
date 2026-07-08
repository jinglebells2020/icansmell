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
    for name, gain in ODOR_PROFILES.items():
        g = np.asarray(gain)
        assert g.shape == (6,), f"{name} gain must be a 6-vector"
        assert np.all(g >= 0.0), f"{name} gains must be non-negative"


def test_clean_air_is_all_zero_gain():
    assert np.allclose(ODOR_PROFILES["clean_air"], 0.0)


def test_alcohol_dominant_on_mq3_idx1():
    g = np.asarray(ODOR_PROFILES["alcohol"])
    assert np.argmax(g) == 1  # MQ3


def test_vinegar_and_spoiled_milk_high_on_mq135_idx5():
    assert np.argmax(np.asarray(ODOR_PROFILES["vinegar"])) == 5
    assert np.argmax(np.asarray(ODOR_PROFILES["spoiled_milk"])) == 5


def test_fresh_vs_spoiled_milk_differ_on_dairy_axes():
    fresh = np.asarray(ODOR_PROFILES["fresh_milk"])
    spoiled = np.asarray(ODOR_PROFILES["spoiled_milk"])
    # they must genuinely differ on idx2/idx3/idx4 (MQ4/MQ7/MQ8)
    assert not np.allclose(fresh[2:5], spoiled[2:5])


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
    # MQ3 (idx1) is the alcohol responder; it should be clearly negative
    assert y[1] < -0.05


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
    from sniffsniff.config import Config, Channel

    channels = tuple(
        Channel(ch=i, sensor=f"S{i}", rl=1000.0) for i in range(3)
    )
    cfg = Config(
        bits=10, vref=5.0, vcc=5.0, channels=channels,
        scan_hz=20, baseline_s=1.0, exposure_s=1.0, purge_s=1.0, plateau_s=0.5,
        ema_alphas=(0.1, 0.01, 0.001), max_cv=0.05, recover_tol=0.02,
    )
    sim = Simulator(cfg, seed=0)
    frames = sim.sniff_frames("coffee")
    assert frames[0][1].shape == (3,)
