"""Tests for post-sniff baseline-recovery detection."""
from __future__ import annotations

import numpy as np

from sniffsniff.recovery import RecoveryMonitor


def _mon(tol=0.02, hz=20, hold_s=1.0):
    r0 = np.array([40000.0, 20000.0, 60000.0])
    return RecoveryMonitor(r0, tol=tol, scan_hz=hz, hold_s=hold_s), r0


def test_not_recovered_while_off_baseline():
    mon, r0 = _mon()
    st = mon.update(r0 * 0.5)  # far from baseline (50% off)
    assert st["within_tol"] is False
    assert st["recovered"] is False
    assert st["max_dev"] > 0.4


def test_recovers_after_sustained_within_tolerance():
    mon, r0 = _mon(hz=20, hold_s=1.0)  # need 20 consecutive in-tol frames
    # 19 in-tolerance frames: not yet recovered
    for _ in range(19):
        st = mon.update(r0 * 1.01)  # 1% off, within 2% tol
        assert st["within_tol"] is True
        assert st["recovered"] is False
    # 20th frame crosses the 1 s hold -> just_recovered fires once
    st = mon.update(r0 * 1.01)
    assert st["recovered"] is True
    assert st["just_recovered"] is True
    # stays recovered, but doesn't re-fire just_recovered
    st = mon.update(r0 * 1.0)
    assert st["recovered"] is True
    assert st["just_recovered"] is False


def test_hold_resets_on_excursion():
    mon, r0 = _mon(hz=20, hold_s=1.0)
    for _ in range(10):
        mon.update(r0 * 1.01)          # building up (in tol)
    st = mon.update(r0 * 1.5)          # excursion -> resets the counter
    assert st["within_tol"] is False
    assert st["held_s"] == 0.0
    # must build the full hold again from scratch
    for _ in range(19):
        assert mon.update(r0 * 1.01)["recovered"] is False
    assert mon.update(r0 * 1.01)["recovered"] is True


def test_held_seconds_progress():
    mon, r0 = _mon(hz=20, hold_s=5.0)
    for i in range(1, 41):  # 40 frames = 2 s at 20 Hz
        st = mon.update(r0 * 0.995)  # 0.5% off, within tol
    assert abs(st["held_s"] - 2.0) < 1e-9
    assert st["target_s"] == 5.0
    assert st["recovered"] is False  # only 2 s of the 5 s hold


def test_worst_channel_reported():
    mon, r0 = _mon()
    rs = r0.copy()
    rs[2] = r0[2] * 1.3  # channel 2 is 30% off
    st = mon.update(rs)
    assert st["worst_channel"] == 2
    assert abs(st["max_dev"] - 0.3) < 1e-6


def test_open_channel_is_ignored_not_blocking():
    # A railed/open channel (inf Rs) must NOT permanently block recovery — decide on
    # the finite channels so a dead sensor doesn't hang the "recovered" signal forever.
    mon, r0 = _mon()
    rs = r0.copy()
    rs[1] = np.inf  # one open channel; the other two sit at baseline
    st = mon.update(rs)  # no crash; recovery decided on the finite channels
    assert st["within_tol"] is True
    assert st["worst_channel"] != 1  # the inf channel is never the "worst finite dev"


def test_all_channels_open_is_not_recovered():
    mon, r0 = _mon()
    st = mon.update(np.array([np.inf, np.inf, np.inf]))  # nothing finite
    assert st["within_tol"] is False
    assert st["recovered"] is False


# --- StabilityMonitor (pre-baseline settle gate) -----------------------------

from sniffsniff.recovery import StabilityMonitor


def test_stability_not_settled_while_drifting():
    m = StabilityMonitor(tol=0.02, scan_hz=20, hold_s=1.0, max_wait_s=None)
    base = np.array([40000.0, 20000.0, 60000.0])
    for k in range(40):
        st = m.update(base * (1.0 + 0.01 * k))  # steadily rising — never flat
    assert st["stable"] is False
    assert st["settled"] is False


def test_stability_settles_when_flat():
    m = StabilityMonitor(tol=0.02, scan_hz=20, hold_s=1.0)  # window = 20 frames
    base = np.array([40000.0, 20000.0, 60000.0])
    # needs a full window of flat data before it can declare stable
    for _ in range(19):
        st = m.update(base)
        assert st["stable"] is False  # window not full yet
    st = m.update(base)
    assert st["stable"] is True
    assert st["settled"] is True
    assert st["max_dev"] == 0.0


def test_stability_times_out_and_proceeds():
    m = StabilityMonitor(tol=0.001, scan_hz=20, hold_s=1.0, max_wait_s=1.5)  # cap 30 frames
    base = np.array([40000.0, 20000.0, 60000.0])
    settled = False
    for k in range(30):
        st = m.update(base * (1.0 + 0.05 * (k % 3)))  # jittery — never within 0.1%
    assert st["timed_out"] is True
    assert st["settled"] is True   # proceeds anyway after max_wait
    assert st["stable"] is False


def test_stability_ignores_open_channel():
    m = StabilityMonitor(tol=0.02, scan_hz=20, hold_s=0.5)  # window 10
    base = np.array([40000.0, np.inf, 60000.0])  # channel 1 railed
    for _ in range(10):
        st = m.update(base)
    assert st["stable"] is True   # decided on the finite channels


# --- EMA smoothing (robustness to per-frame noise) ---------------------------

def test_stability_smoothing_settles_noisy_flat():
    rng = np.random.default_rng(0)
    base = np.array([40000.0, 20000.0, 60000.0])
    # ~4% per-frame noise exceeds a 2% tol raw; EMA smoothing tames it so a
    # noisy-but-flat signal still reaches 'stable'.
    m = StabilityMonitor(tol=0.02, scan_hz=20, hold_s=2.0, ema_alpha=0.1)
    settled = False
    for _ in range(600):
        if m.update(base * (1 + 0.04 * rng.standard_normal(3)))["stable"]:
            settled = True
            break
    assert settled


def test_stability_without_smoothing_stays_jittery():
    rng = np.random.default_rng(0)
    base = np.array([40000.0, 20000.0, 60000.0])
    m = StabilityMonitor(tol=0.02, scan_hz=20, hold_s=2.0)  # no smoothing
    ever = any(
        m.update(base * (1 + 0.04 * rng.standard_normal(3)))["stable"] for _ in range(200)
    )
    assert ever is False  # raw 4% noise never flat within 2%


def test_recovery_smoothing_reaches_recovered_under_noise():
    rng = np.random.default_rng(1)
    r0 = np.array([40000.0, 20000.0, 60000.0])
    m = RecoveryMonitor(r0, tol=0.02, scan_hz=20, hold_s=1.0, ema_alpha=0.1)
    recovered = False
    for _ in range(600):
        if m.update(r0 * (1 + 0.03 * rng.standard_normal(3)))["recovered"]:
            recovered = True
            break
    assert recovered  # smoothed signal at baseline is judged recovered despite noise


# --- ResponsePlateauMonitor (dynamic-exposure end: gate on growth) -----------

from pathlib import Path

from sniffsniff.recovery import ResponsePlateauMonitor

_R0 = np.array([40000.0, 20000.0, 60000.0])


def _plateau(hold_s=1.0, min_s=0.5, eps=0.005, hz=20, ema=None):
    return ResponsePlateauMonitor(_R0, hz, hold_s=hold_s, min_s=min_s, eps=eps, ema_alpha=ema)


def test_plateau_not_before_min_floor():
    # a flat (no-growth) signal still must not plateau before the min_s floor
    m = _plateau(hold_s=0.2, min_s=1.0, hz=20)  # min = 20 frames, hold = 4 frames
    for i in range(19):
        assert m.update(_R0 * 1.05)["plateaued"] is False
    assert m.update(_R0 * 1.05)["plateaued"] is True  # 20th frame clears the floor


def test_plateau_fires_after_growth_stops():
    m = _plateau(hold_s=0.5, min_s=0.25, hz=20)  # hold = 10 frames, min = 5
    # rising phase: keeps setting new highs -> never plateaus
    for g in np.linspace(0.0, 0.10, 40):
        assert m.update(_R0 * (1 + g))["plateaued"] is False
    # then hold flat: after hold_s of no new high it plateaus
    fired = False
    for _ in range(20):
        if m.update(_R0 * 1.10)["plateaued"]:
            fired = True
            break
    assert fired


def test_plateau_stays_open_while_slowly_rising():
    # a slow, steady creep keeps setting new highs (> eps per hold window) -> never
    # plateaus; this is the milk-truncation bug the growth gate exists to prevent.
    m = _plateau(hold_s=0.5, min_s=0.25, hz=20, eps=0.002)
    plateaued = False
    for i in range(400):
        g = 0.0003 * i  # ~0.03pp per frame, monotonic
        if m.update(_R0 * (1 + g))["plateaued"]:
            plateaued = True
            break
    assert plateaued is False  # still growing the whole time


def test_plateau_eps_ignores_sub_eps_jitter():
    # jitter smaller than eps must NOT count as a new high (else a plateau never fires)
    rng = np.random.default_rng(0)
    m = _plateau(hold_s=0.5, min_s=0.25, hz=20, eps=0.01, ema=0.3)
    # rise then hold at a plateau with small (<eps) noise
    for g in np.linspace(0, 0.08, 30):
        m.update(_R0 * (1 + g))
    fired = False
    for _ in range(60):
        noisy = _R0 * (1 + 0.08 + 0.001 * rng.standard_normal(3))  # ~0.1pp jitter << eps
        if m.update(noisy)["plateaued"]:
            fired = True
            break
    assert fired  # sub-eps jitter didn't keep resetting the new-high timer


def test_real_milk_curve_is_not_truncated():
    """Regression: the REAL fresh-milk exposure (captured on the rig) must ride to
    the cap, not plateau-stop early — with the SHIPPED config parameters."""
    from sniffsniff.config import default_config

    fx = Path(__file__).parent / "data" / "milk_exposure.npz"
    d = np.load(fx)
    rs_exp, t_exp, r0 = d["rs_exp"].astype(float), d["t_exp"].astype(float), d["r0"].astype(float)
    hz = float(d["eff_hz"])
    cfg = default_config()

    m = ResponsePlateauMonitor(
        r0, round(hz), hold_s=cfg.plateau_hold_s, min_s=cfg.min_exposure_s,
        eps=cfg.plateau_eps, ema_alpha=cfg.smooth_alpha,
    )
    plateau_time = None
    for k, rs in enumerate(rs_exp):
        if m.update(rs)["plateaued"]:
            plateau_time = float(t_exp[k])
            break
    # milk is still rising through the whole 60 s window; it must NOT stop early.
    # (Before the fix it ended at ~3.7 s, capturing ~20% of the response.)
    assert plateau_time is None or plateau_time >= 40.0, (
        f"milk exposure truncated at {plateau_time:.1f}s — should ride to the cap"
    )
