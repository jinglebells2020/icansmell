"""Tests for the single-connection MonitorEngine + servo-driven ContinuousSim."""
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


def test_continuous_sim_presents_odor_at_sample_angle():
    """At the sample angle the array responds to the odor; fresh air recovers it."""
    cfg = _cfg()
    sim = ContinuousSim(cfg, seed=0, noise_counts=0.0, tau_rise=1.0, tau_decay=1.0)
    sim.set_odor("coffee")
    fresh = sim.read()[1].astype(float)          # clean-air baseline counts

    sim.write_command(f"S{cfg.servo_sample_angle}")   # open the sample straw
    exposed = fresh
    for _ in range(cfg.scan_hz * 8):
        exposed = sim.read()[1].astype(float)
    # reducing gas drops Rs -> the divider voltage (and thus counts) rises on every ch
    assert np.all(exposed > fresh)

    sim.write_command(f"S{cfg.servo_fresh_air_angle}")  # back to fresh air
    recovered = exposed
    for _ in range(cfg.scan_hz * 40):
        recovered = sim.read()[1].astype(float)
    assert np.all(recovered < exposed)            # relaxes back toward baseline


def test_continuous_sim_ignores_malformed_commands():
    cfg = _cfg()
    sim = ContinuousSim(cfg, seed=0, noise_counts=0.0)
    assert sim.write_command("S105") is True
    assert sim.write_command("nonsense") is False


# --- MonitorEngine -----------------------------------------------------------

def _engine(cfg, tmp_path, **kw):
    # small, deterministic settle so tests are fast (noise-free sim settles cleanly)
    kw.setdefault("settle_hold_s", 0.25)
    kw.setdefault("settle_max_wait_s", 2.0)
    return MonitorEngine(cfg, SniffRecorder(cfg, tmp_path), **kw)


def _armed(cfg, tmp_path, label="coffee", *, seed=0, save=True, odor=None, **kw):
    """An engine + servo-driven sim wired together, armed and ready to settle."""
    eng = _engine(cfg, tmp_path, **kw)
    sim = ContinuousSim(cfg, seed=seed, noise_counts=0.0)
    sim.set_odor(odor or (label if label != "?" else "coffee"))
    eng.set_airflow(sim.write_command)     # engine drives the sim's airflow straw
    eng.arm_capture(label, save=save)
    return eng, sim


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
    eng, sim = _armed(cfg, tmp_path)
    assert eng.arm_capture("coffee") is False  # busy (settling/armed) — refused
    ev = eng.step(sim.read())
    assert ev["phase"] == "settle"          # SETTLE first, not baseline
    assert ev["settle"] is not None
    assert eng.capturing is False           # not yet windowing a sniff


def test_capture_windows_the_stream_and_saves(tmp_path):
    cfg = _cfg()
    eng, sim = _armed(cfg, tmp_path, seed=1)
    _settle(eng, sim)                        # wait for a stable baseline first

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
    eng, sim = _armed(cfg, tmp_path, seed=2)
    _settle(eng, sim)
    for _ in range(eng.n):
        eng.step(sim.read())  # run the capture to completion
    ev = eng.step(sim.read())
    assert ev["recovery"] is not None
    assert set(ev["recovery"]) >= {"within_tol", "held_s", "target_s", "recovered"}


def test_phase_changed_fires_on_transitions(tmp_path):
    cfg = _cfg()
    eng, sim = _armed(cfg, tmp_path)
    _settle(eng, sim)  # SETTLE happens before the timed phases
    changes = [
        ev["phase"] for _ in range(eng.n) if (ev := eng.step(sim.read()))["phase_changed"]
    ]
    assert changes == ["baseline", "exposure", "purge"]  # one change per timed phase


def test_capture_without_save_does_not_write(tmp_path):
    # identify path: process (features + R0 for recovery) but don't persist a sniff.
    cfg = _cfg()
    eng, sim = _armed(cfg, tmp_path, label="?", seed=3, save=False)
    _settle(eng, sim)
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


def test_exposure_ends_on_plateau_before_the_cap(tmp_path):
    """Dynamic exposure: a fast-responding array plateaus well before the cap."""
    cfg = dataclasses.replace(
        default_config(), baseline_s=0.5, exposure_s=30, purge_s=0.5, plateau_s=0.5
    )
    eng = _engine(cfg, tmp_path, plateau_hold_s=0.5, smooth_alpha=0.0)
    sim = ContinuousSim(cfg, seed=0, noise_counts=0.0, tau_rise=0.5, tau_decay=0.5)
    sim.set_odor("coffee")
    eng.set_airflow(sim.write_command)
    eng.arm_capture("coffee")
    _settle(eng, sim)

    saved = None
    for _ in range(eng.n):          # eng.n is the CAP ceiling; the plateau ends sooner
        ev = eng.step(sim.read())
        if ev["saved"] is not None:
            saved = ev["saved"]
            break
    assert saved is not None
    result, _ = saved
    b_end = result.slices["baseline"][1]
    e_end = result.slices["exposure"][1]
    assert (e_end - b_end) < eng.n_exp_max       # ended on the plateau, not the cap
    assert result.features.shape[0] == 48


def test_exposure_hits_the_cap_without_a_plateau(tmp_path):
    """No plateau (still rising at the cap) → exposure ends at exactly the cap."""
    cfg = dataclasses.replace(
        default_config(), baseline_s=0.5, exposure_s=1.0, purge_s=0.5, plateau_s=0.5
    )
    # very slow rise + a long plateau requirement -> the plateau never triggers
    eng = _engine(cfg, tmp_path, plateau_hold_s=5.0, smooth_alpha=0.0)
    sim = ContinuousSim(cfg, seed=0, noise_counts=0.0, tau_rise=60.0, tau_decay=60.0)
    sim.set_odor("coffee")
    eng.set_airflow(sim.write_command)
    eng.arm_capture("coffee")
    _settle(eng, sim)

    saved = None
    for _ in range(eng.n):
        ev = eng.step(sim.read())
        if ev["saved"] is not None:
            saved = ev["saved"]
            break
    result, _ = saved
    b_end = result.slices["baseline"][1]
    e_end = result.slices["exposure"][1]
    assert (e_end - b_end) == eng.n_exp_max      # capped
