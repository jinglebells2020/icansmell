"""Single-connection monitoring engine: continuous live view + windowed capture
+ post-sniff recovery, all off ONE frame stream.

The v1 TUI opened a fresh serial reader per sniff (each open resets the Uno), so
the sensor bars only lived during a capture. :class:`MonitorEngine` instead runs
one continuous frame loop: every frame updates the live view; a capture is just a
*window* over that same stream (no reopen, no reset); and after a sniff it tracks
return-to-baseline against that sniff's R0.

The engine is a **frame-by-frame state machine** — feed it one ``(t_ms, raw)``
frame via :meth:`step` and it returns an event dict. It owns no threads and no UI,
so it is fully deterministic to test; a caller (the TUI worker) pulls frames from a
source, calls ``step``, and renders the events.

:class:`ContinuousSim` is a frame source for ``--sim`` mode: it streams clean-air
frames while idle and the requested odor's session during a capture window, so the
same engine drives both simulator and real hardware.
"""
from __future__ import annotations

import numpy as np

from . import calibrate
from .capture import phase_of, session_frame_count
from .record import SniffRecorder, phase_slices
from .recovery import RecoveryMonitor, StabilityMonitor
from .simulator import Simulator

__all__ = ["MonitorEngine", "ContinuousSim"]


def _rs_of(raw, config) -> np.ndarray:
    """Per-frame sensor resistance for one raw counts vector."""
    return calibrate.counts_to_rs(
        np.asarray(raw), config.rl_array(), config.vcc, config.vref, config.bits
    )


class MonitorEngine:
    """Continuous-monitor + windowed-capture + recovery state machine.

    Parameters
    ----------
    config:
        The :class:`~sniffsniff.config.Config` (timing, channels, recover_tol).
    recorder:
        A :class:`~sniffsniff.record.SniffRecorder` used to process + persist a
        completed capture window.
    hold_s:
        Recovery hold window (seconds) passed to :class:`RecoveryMonitor`.
    """

    def __init__(
        self,
        config,
        recorder: SniffRecorder,
        *,
        hold_s: float = 5.0,
        settle_hold_s=None,
        settle_max_wait_s=None,
        smooth_alpha=None,
    ):
        self.config = config
        self.recorder = recorder
        self.hold_s = hold_s
        # Adaptive-capture tuning defaults come from config; constructor overrides win.
        self.settle_hold_s = config.settle_hold_s if settle_hold_s is None else settle_hold_s
        self.settle_max_wait_s = (
            config.settle_max_wait_s if settle_max_wait_s is None else settle_max_wait_s
        )
        self.smooth_alpha = config.smooth_alpha if smooth_alpha is None else smooth_alpha
        self.n = session_frame_count(config)
        self._slices = phase_slices(self.n, config)

        self._pending_label: str | None = None
        self._pending_save = True
        self._save = True
        self._settling = False
        self._stability: StabilityMonitor | None = None
        self._capturing = False
        self._label: str | None = None
        self._buf: list = []
        self._i = 0
        self._phase: str | None = None
        self._recovery: RecoveryMonitor | None = None

    # -------------------------------------------------------- control
    @property
    def capturing(self) -> bool:
        """True while actively windowing a sniff (baseline/exposure/purge)."""
        return self._capturing

    @property
    def busy(self) -> bool:
        """True while settling, armed, or capturing — anything but idle monitoring."""
        return self._settling or self._capturing or self._pending_label is not None

    def arm_capture(self, label: str, *, save: bool = True) -> bool:
        """Request a capture of ``label`` (settles, then begins on a later :meth:`step`).

        ``save=True`` persists the sniff to the dataset (a *record*); ``save=False``
        processes it but does not write it (an *identify*). Ignored (returns False)
        if a capture is already settling, armed, or running.
        """
        if self.busy:
            return False
        self._pending_label = label
        self._pending_save = bool(save)
        return True

    # -------------------------------------------------------- per-frame
    def step(self, frame) -> dict:
        """Process one ``(t_ms, raw)`` frame; return an event dict.

        Event keys (all present every call):
        * ``rs``       — ``(N,)`` live sensor resistance for the bars.
        * ``phase``    — "monitor" when idle, else "baseline"/"exposure"/"purge".
        * ``phase_changed`` — True on the frame a capture crosses into a new phase.
        * ``capture``  — ``(k, n)`` progress during a capture, else None.
        * ``saved``    — ``(SniffResult, path)`` on the frame a capture completes, else None.
        * ``recovery`` — the :class:`RecoveryMonitor` status dict while tracking, else None.
        """
        raw = frame[1]
        rs = _rs_of(raw, self.config)
        event = {
            "rs": rs,
            "phase": "monitor",
            "phase_changed": False,
            "capture": None,
            "saved": None,
            "recovery": None,
            "settle": None,
        }

        # Promote a pending request into the SETTLE phase (wait for a stable baseline
        # before we measure R0 — not a blind fixed wait).
        if not self._capturing and not self._settling and self._pending_label is not None:
            self._settling = True
            self._label = self._pending_label
            self._save = self._pending_save
            self._pending_label = None
            self._phase = None
            self._recovery = None  # a new sniff supersedes the old recovery track
            self._stability = StabilityMonitor(
                self.config.recover_tol, self.config.scan_hz,
                hold_s=self.settle_hold_s, max_wait_s=self.settle_max_wait_s,
                ema_alpha=self.smooth_alpha,
            )

        if self._settling:
            st = self._stability.update(rs)
            event["phase"] = "settle"
            event["settle"] = st
            if self._phase != "settle":
                event["phase_changed"] = True
                self._phase = "settle"
            if st["settled"]:
                # sensors are at rest — begin the timed capture on the next frame
                self._settling = False
                self._capturing = True
                self._buf = []
                self._i = 0
                self._phase = None
            return event

        if self._capturing:
            self._buf.append(frame)
            phase = phase_of(self._i, self._slices)
            event["phase"] = phase
            event["capture"] = (self._i + 1, self.n)
            if phase != self._phase:
                event["phase_changed"] = True
                self._phase = phase
            self._i += 1
            if self._i >= self.n:
                result = self.recorder.process(self._buf, self._label)
                path = self.recorder.save(result) if self._save else None
                event["saved"] = (result, path)
                self._recovery = RecoveryMonitor(
                    result.r0, self.config.recover_tol, self.config.scan_hz,
                    hold_s=self.hold_s, ema_alpha=self.smooth_alpha,
                )
                self._capturing = False
        elif self._recovery is not None:
            event["recovery"] = self._recovery.update(rs)

        return event


class ContinuousSim:
    """A never-ending simulated frame source for the monitoring engine.

    Streams clean-air frames while idle; when :meth:`begin_odor` is called it emits
    exactly one odor session (``session_frame_count`` frames) then reverts to clean
    air. ``t_ms`` advances monotonically. Mirrors a real device that streams
    continuously while airflow (servo/operator) decides what the sensors smell.
    """

    def __init__(self, config, seed: int = 0, noise_counts: float = 1.0):
        self.config = config
        self._seed = seed
        self._step_ms = round(1000 / config.scan_hz)
        self._t = 0
        # A clean-air session we cycle through for the idle stream.
        self._clean = Simulator(config, seed=seed, noise_counts=noise_counts).sniff_frames(
            "clean_air"
        )
        self._clean_i = 0
        self._noise = noise_counts
        self._odor: list | None = None
        self._odor_i = 0

    def begin_odor(self, label: str, seed: int | None = None) -> None:
        """Emit ``label``'s odor session over the next reads, then revert to clean."""
        s = self._seed if seed is None else seed
        self._odor = Simulator(self.config, seed=s, noise_counts=self._noise).sniff_frames(
            label
        )
        self._odor_i = 0

    def read(self) -> tuple[int, np.ndarray]:
        """Return the next ``(t_ms, raw)`` frame (never ends)."""
        if self._odor is not None:
            raw = self._odor[self._odor_i][1]
            self._odor_i += 1
            if self._odor_i >= len(self._odor):
                self._odor = None
        else:
            raw = self._clean[self._clean_i][1]
            self._clean_i = (self._clean_i + 1) % len(self._clean)
        t = self._t
        self._t += self._step_ms
        return (t, raw)

    def frames(self):
        """Endless iterator over :meth:`read` (interface parity with readers)."""
        while True:
            yield self.read()

    def close(self) -> None:  # interface parity
        return None
