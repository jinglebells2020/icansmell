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

import numpy as np

__all__ = ["RecoveryMonitor"]


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

    def __init__(self, r0, tol: float, scan_hz: int, hold_s: float = 5.0):
        self.r0 = np.asarray(r0, dtype=np.float64)
        self.tol = float(tol)
        self.hold_frames = max(1, round(float(hold_s) * scan_hz))
        self.hold_s = float(hold_s)
        self.scan_hz = int(scan_hz)
        self._within_count = 0     # consecutive frames within tolerance
        self._recovered = False    # latched once the hold is met

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
        rs = np.asarray(rs, dtype=np.float64)
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
