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
