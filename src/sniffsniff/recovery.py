"""Post-sniff baseline-recovery detection.

After a sniff, the sensors are saturated and drift back toward their clean-air
resistance. Before the next sniff you want them *recovered* — otherwise the new
sniff's R0 baseline is contaminated. :class:`RecoveryMonitor` watches the live Rs
against the previous sniff's clean-air baseline ``R0`` and reports when every
channel's ratio ``Rs/R0`` has held within ``±tol`` for a sustained window
(``hold_s`` seconds) — the "recovered, safe to sniff" condition from the design.

Pure and UI-free: feed it one live ``Rs`` vector per frame; it returns a small
status dict the CLI/TUI can render.
"""
from __future__ import annotations

from collections import deque

import numpy as np

__all__ = ["RecoveryMonitor", "StabilityMonitor", "ResponsePlateauMonitor"]


class StabilityMonitor:
    """Self-referential "sensors are at rest" detector for the pre-baseline settle.

    Unlike :class:`RecoveryMonitor` (which asks *did Rs return to a prior R0?*), this
    asks *is Rs flat right now?* — every channel stays within ``±tol`` of the recent
    rolling-window mean for the whole ``hold_s`` window. That's the right gate before
    measuring a fresh R0 (it needs no prior baseline, so it works for the first sniff).

    A ``max_wait_s`` cap makes it fall through (``timed_out``) rather than wait forever
    on a sensor that never fully settles — the capture still proceeds, just flagged.

    Parameters
    ----------
    tol:
        Fractional flatness tolerance (e.g. ``0.02`` = within ±2% of the window mean).
    scan_hz:
        Frame rate, to size the hold window and the timeout in frames.
    hold_s:
        The signal must be flat across a window this long (default 3 s).
    max_wait_s:
        Give up waiting after this long (``None`` = wait indefinitely).
    """

    def __init__(self, tol: float, scan_hz: int, hold_s: float = 3.0, max_wait_s=30.0,
                 ema_alpha=None):
        self.tol = float(tol)
        self.scan_hz = int(scan_hz)
        self.hold_s = float(hold_s)
        self.win = max(1, round(hold_s * scan_hz))
        self.max_wait = None if max_wait_s is None else max(1, round(max_wait_s * scan_hz))
        # 0 (or None) means "no smoothing" — matches the config's `smooth_alpha` note.
        self.ema_alpha = float(ema_alpha) if ema_alpha else None
        self._ema = None
        self._buf: deque = deque(maxlen=self.win)
        self._count = 0

    def update(self, rs) -> dict:
        """Feed one live ``Rs`` vector; return a status dict.

        Keys: ``stable`` (window is flat), ``settled`` (stable OR timed out — the
        engine may proceed), ``max_dev`` (largest deviation across the window, or None
        until the window fills), ``waited_s``, and ``timed_out``.
        """
        self._count += 1
        rs = np.asarray(rs, dtype=np.float64)
        if self.ema_alpha is not None:
            self._ema = (
                rs.copy() if self._ema is None
                else self.ema_alpha * rs + (1.0 - self.ema_alpha) * self._ema
            )
            rs = self._ema
        self._buf.append(rs.copy())

        stable = False
        max_dev = None
        if len(self._buf) >= self.win:
            arr = np.array(self._buf)              # (win, N)
            mean = arr.mean(axis=0)
            dev = np.abs(arr / mean - 1.0)         # per-frame, per-channel deviation
            finite = np.isfinite(dev)
            if finite.any():
                max_dev = float(dev[finite].max())
                stable = bool(np.all(dev[finite] <= self.tol))

        timed_out = self.max_wait is not None and self._count >= self.max_wait
        return {
            "stable": stable,
            "settled": bool(stable or timed_out),
            "max_dev": max_dev,
            "waited_s": self._count / self.scan_hz,
            "timed_out": bool(timed_out and not stable),
        }


class ResponsePlateauMonitor:
    """Detect when a sniff's response has *stopped growing* — a real plateau.

    The naive "is Rs flat within ±tol?" test fails on real hardware: a weak, slow
    odor (fresh milk creeps up at ~0.2 %/s) has a per-frame slope buried in sensor
    noise, so a coarse flatness window reads "plateaued" while the response is still
    climbing — and exposure ends far too early. Measured on the real rig, milk's rise
    rate sits *below* the derivative noise band, so slope alone can't decide.

    Instead, gate on *growth of the aggregate response magnitude*
    ``m = mean_ch |Rs/R0 - 1|`` (a single scalar; per-channel quantization noise
    averages out). Track the running peak of ``m``; while the response keeps setting
    new highs (gains at least ``eps``) it is still developing. Declare a plateau only
    once ``m`` has gone ``hold_s`` **without** a meaningful new high *and* at least
    ``min_s`` has elapsed. A still-rising response (milk) therefore keeps the exposure
    open until the caller's cap; one that truly flattens (a strong odor) stops early.

    Parameters
    ----------
    r0:
        Per-channel clean-air baseline resistance ``(N,)`` measured this sniff.
    scan_hz:
        Frame rate, to size ``hold_s`` / ``min_s`` in frames.
    hold_s:
        Seconds the response must go without a new high before it counts as plateaued.
    min_s:
        Minimum exposure floor — never declare a plateau before this (lets the
        response develop; replaces the old "has it moved off baseline?" guard).
    eps:
        Minimum *fractional* growth of ``m`` that counts as a genuine new high
        (e.g. ``0.005`` = 0.5 percentage-points of R0). Set well above the noise of
        ``m`` so jitter can't fake continued growth.
    ema_alpha:
        Optional EMA factor to smooth ``m`` (0/None = off).
    """

    def __init__(self, r0, scan_hz: int, *, hold_s: float, min_s: float,
                 eps: float, ema_alpha=None):
        self.r0 = np.asarray(r0, dtype=np.float64)
        self.scan_hz = int(scan_hz)
        self.hold_frames = max(1, round(float(hold_s) * scan_hz))
        self.min_frames = max(1, round(float(min_s) * scan_hz))
        self.eps = float(eps)
        self.ema_alpha = float(ema_alpha) if ema_alpha else None
        self._ema = None
        self._peak = -np.inf
        self._since_high = 0      # frames since the last meaningful new high
        self._count = 0           # frames elapsed (exposure length so far)

    def update(self, rs) -> dict:
        """Feed one live ``Rs`` vector ``(N,)``; return a status dict.

        Keys: ``plateaued`` (growth has stalled ``hold_s`` past the ``min_s`` floor),
        ``mag`` (smoothed aggregate response magnitude, fractional), ``peak``,
        ``elapsed_s`` (exposure so far), ``held_s`` (seconds since the last new high),
        ``min_s`` (the floor).
        """
        self._count += 1
        rs = np.asarray(rs, dtype=np.float64)
        frac = rs / self.r0 - 1.0
        frac = np.where(np.isfinite(frac), frac, 0.0)   # dead/open channel → 0 weight
        mag = float(np.abs(frac).mean())
        if self.ema_alpha is not None:
            self._ema = (
                mag if self._ema is None
                else self.ema_alpha * mag + (1.0 - self.ema_alpha) * self._ema
            )
            mag = self._ema

        if self._count == 1:
            # Seed only: don't let a single, still-unsmoothed first frame define the
            # peak (a frame-1 noise spike could otherwise suppress genuine later
            # growth). Begin new-high / hold tracking from the next, smoothed frame.
            return {
                "plateaued": False, "mag": mag, "peak": mag,
                "elapsed_s": self._count / self.scan_hz, "held_s": 0.0,
                "min_s": self.min_frames / self.scan_hz,
            }

        if mag > self._peak + self.eps:
            self._peak = mag
            self._since_high = 0
        else:
            self._since_high += 1

        plateaued = (
            self._count >= self.min_frames and self._since_high >= self.hold_frames
        )
        return {
            "plateaued": bool(plateaued),
            "mag": mag,
            "peak": float(self._peak),
            "elapsed_s": self._count / self.scan_hz,
            "held_s": self._since_high / self.scan_hz,
            "min_s": self.min_frames / self.scan_hz,
        }


class RecoveryMonitor:
    """Track return-to-baseline after a sniff.

    Parameters
    ----------
    r0:
        The previous sniff's per-channel clean-air baseline resistance ``(N,)``.
    tol:
        Fractional tolerance on ``Rs/R0`` (e.g. ``0.02`` = within ±2%). From
        ``config.recover_tol``.
    scan_hz:
        Frame rate, to convert the hold window to a frame count.
    hold_s:
        How long every channel must stay within ``tol`` before "recovered"
        (default 5 s — the design's "±2% for ≥5 consecutive seconds").
    """

    def __init__(self, r0, tol: float, scan_hz: int, hold_s: float = 5.0, ema_alpha=None):
        self.r0 = np.asarray(r0, dtype=np.float64)
        self.tol = float(tol)
        self.hold_frames = max(1, round(float(hold_s) * scan_hz))
        self.hold_s = float(hold_s)
        self.scan_hz = int(scan_hz)
        # 0 (or None) means "no smoothing" — matches the config's `smooth_alpha` note.
        self.ema_alpha = float(ema_alpha) if ema_alpha else None
        self._ema = None
        self._within_count = 0     # consecutive frames within tolerance
        self._recovered = False    # latched once the hold is met

    def _smooth(self, rs):
        """EMA-smooth Rs (if enabled) so per-frame noise doesn't reset the hold."""
        if self.ema_alpha is None:
            return rs
        self._ema = (
            rs.copy() if self._ema is None
            else self.ema_alpha * rs + (1.0 - self.ema_alpha) * self._ema
        )
        return self._ema

    def update(self, rs) -> dict:
        """Feed one live ``Rs`` vector ``(N,)``; return a status dict.

        Keys:
        * ``within_tol``   — all channels currently within ``±tol``.
        * ``max_dev``      — largest ``|Rs/R0 − 1|`` across channels (fraction).
        * ``worst_channel``— index of that largest deviation.
        * ``held_s``       — seconds continuously within tol (0 if not).
        * ``target_s``     — the hold target (``hold_s``).
        * ``recovered``    — the hold has been met (latched True thereafter).
        * ``just_recovered`` — True on the single frame recovery is first reached.
        """
        rs = self._smooth(np.asarray(rs, dtype=np.float64))
        # Ignore non-finite channels (open/rail) so one dead channel can't block
        # or falsely satisfy recovery; require at least one finite channel.
        ratio = rs / self.r0
        dev = np.abs(ratio - 1.0)
        finite = np.isfinite(dev)
        if not finite.any():
            self._within_count = 0
            return self._status(within=False, max_dev=float("inf"), worst=-1)

        dev_f = np.where(finite, dev, -np.inf)
        worst = int(np.argmax(dev_f))
        max_dev = float(dev[worst])
        within = bool(np.all(dev[finite] <= self.tol))

        if within:
            self._within_count += 1
        else:
            self._within_count = 0

        just = False
        if self._within_count >= self.hold_frames and not self._recovered:
            self._recovered = True
            just = True

        return self._status(within=within, max_dev=max_dev, worst=worst, just=just)

    def _status(self, *, within: bool, max_dev: float, worst: int, just: bool = False) -> dict:
        return {
            "within_tol": within,
            "max_dev": max_dev,
            "worst_channel": worst,
            "held_s": self._within_count / self.scan_hz,
            "target_s": self.hold_s,
            "recovered": self._recovered,
            "just_recovered": just,
        }

    @property
    def recovered(self) -> bool:
        """Whether recovery has been reached (latched)."""
        return self._recovered
