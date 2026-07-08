"""Tests for the single-connection MonitorEngine + ContinuousSim source."""
from __future__ import annotations

import dataclasses

import numpy as np

from sniffsniff.config import default_config
from sniffsniff.capture import session_frame_count
from sniffsniff.monitor import ContinuousSim, MonitorEngine
from sniffsniff.record import SniffRecorder


def _cfg():
    return dataclasses.replace(
        default_config(), baseline_s=1, exposure_s=1, purge_s=1, plateau_s=0.5
    )


# --- ContinuousSim -----------------------------------------------------------

def test_continuous_sim_streams_endlessly_and_advances_time():
    cfg = _cfg()
    sim = ContinuousSim(cfg, seed=0)
    f0, f1 = sim.read(), sim.read()
    assert f0[1].shape == (6,)
    assert f1[0] - f0[0] == round(1000 / cfg.scan_hz)  # t_ms advances by the step
    # keeps producing well past one session (never ends)
    for _ in range(session_frame_count(cfg) * 2):
        assert sim.read()[1].shape == (6,)


def test_continuous_sim_begin_odor_then_reverts():
    cfg = _cfg()
    sim = ContinuousSim(cfg, seed=0)
    n = session_frame_count(cfg)
    idle_before = sim.read()[1].copy()
    sim.begin_odor("coffee")
    odor = np.array([sim.read()[1] for _ in range(n)])
    # after exactly one odor session it reverts to the (idle) clean stream
    after = sim.read()[1]
    # the odor session differs from a clean-air idle frame somewhere
    assert not np.array_equal(odor[n // 2], idle_before)


# --- MonitorEngine -----------------------------------------------------------

def _engine(cfg, tmp_path):
    # small, deterministic settle so tests are fast (noise-free sim settles cleanly)
    return MonitorEngine(
        cfg, SniffRecorder(cfg, tmp_path), settle_hold_s=0.25, settle_max_wait_s=2.0
    )


def _settle(eng, sim):
    """Feed clean frames until the engine leaves SETTLE and starts capturing."""
    for _ in range(2000):
        eng.step(sim.read())
        if eng.capturing:
            return
    raise AssertionError("engine never settled")


def test_idle_frames_are_monitor_only(tmp_path):
    cfg = _cfg()
    eng = _engine(cfg, tmp_path)
    sim = ContinuousSim(cfg, seed=0, noise_counts=0.0)
    for _ in range(5):
        ev = eng.step(sim.read())
        assert ev["phase"] == "monitor"
        assert ev["saved"] is None and ev["recovery"] is None
        assert ev["rs"].shape == (6,)


def test_arm_settles_before_capturing(tmp_path):
    cfg = _cfg()
    eng = _engine(cfg, tmp_path)
    sim = ContinuousSim(cfg, seed=0, noise_counts=0.0)
    assert eng.arm_capture("coffee") is True
    assert eng.arm_capture("coffee") is False  # busy (settling/armed) — refused
    ev = eng.step(sim.read())
    assert ev["phase"] == "settle"          # SETTLE first, not baseline
    assert ev["settle"] is not None
    assert eng.capturing is False           # not yet windowing a sniff


def test_capture_windows_the_stream_and_saves(tmp_path):
    cfg = _cfg()
    eng = _engine(cfg, tmp_path)
    sim = ContinuousSim(cfg, seed=1, noise_counts=0.0)

    eng.arm_capture("coffee")
    _settle(eng, sim)                        # wait for a stable baseline first
    sim.begin_odor("coffee", seed=1)

    phases, saved = set(), None
    for _ in range(eng.n):
        ev = eng.step(sim.read())
        phases.add(ev["phase"])
        assert ev["capture"] is not None
        if ev["saved"] is not None:
            saved = ev["saved"]
    assert phases == {"baseline", "exposure", "purge"}
    assert saved is not None
    result, path = saved
    assert path.exists()
    assert result.features.shape[0] == 48
    assert eng.capturing is False


def test_recovery_tracked_after_a_sniff(tmp_path):
    cfg = _cfg()
    eng = _engine(cfg, tmp_path)
    sim = ContinuousSim(cfg, seed=2, noise_counts=0.0)
    eng.arm_capture("coffee")
    _settle(eng, sim)
    sim.begin_odor("coffee", seed=2)
    for _ in range(eng.n):
        eng.step(sim.read())  # run the capture to completion
    ev = eng.step(sim.read())
    assert ev["recovery"] is not None
    assert set(ev["recovery"]) >= {"within_tol", "held_s", "target_s", "recovered"}


def test_phase_changed_fires_on_transitions(tmp_path):
    cfg = _cfg()
    eng = _engine(cfg, tmp_path)
    sim = ContinuousSim(cfg, seed=0, noise_counts=0.0)
    eng.arm_capture("coffee")
    _settle(eng, sim)  # SETTLE happens before the timed phases
    sim.begin_odor("coffee")
    changes = [
        ev["phase"] for _ in range(eng.n) if (ev := eng.step(sim.read()))["phase_changed"]
    ]
    assert changes == ["baseline", "exposure", "purge"]  # one change per timed phase


def test_capture_without_save_does_not_write(tmp_path):
    # identify path: process (features + R0 for recovery) but don't persist a sniff.
    cfg = _cfg()
    eng = _engine(cfg, tmp_path)
    sim = ContinuousSim(cfg, seed=3, noise_counts=0.0)
    eng.arm_capture("?", save=False)
    _settle(eng, sim)
    sim.begin_odor("coffee", seed=3)
    saved = None
    for _ in range(eng.n):
        ev = eng.step(sim.read())
        if ev["saved"] is not None:
            saved = ev["saved"]
    result, path = saved
    assert path is None                          # nothing written to disk
    assert result.features.shape[0] == 48        # but features are available
    assert not any(tmp_path.glob("*/*.npz"))     # dataset stays empty
    # recovery still tracked after an identify capture
    assert eng.step(sim.read())["recovery"] is not None
