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

:class:`ContinuousSim` is a servo-driven frame source for ``--sim`` mode: it streams
a continuously-relaxing array that responds to airflow commands (``S<angle>``) just
like the real device, so the *same* engine — including its dynamic, plateau-gated
exposure and its airflow driving — behaves identically on simulator and hardware.
"""
from __future__ import annotations

import numpy as np

from . import calibrate
from .record import SniffRecorder
from .recovery import RecoveryMonitor, ResponsePlateauMonitor, StabilityMonitor
from .simulator import ODOR_PROFILES

__all__ = ["MonitorEngine", "ContinuousSim"]


def _odor_gain(config, label: str) -> np.ndarray:
    """Per-channel odor gain ``(N,)`` for ``label`` (0 for clean air / unknown)."""
    profile = ODOR_PROFILES.get(label, {})
    return np.array(
        [profile.get(name, 0.0) for name in config.sensor_names()], dtype=np.float64
    )


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
        plateau_hold_s=None,
        min_exposure_s=None,
        plateau_eps=None,
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
        self.plateau_hold_s = (
            config.plateau_hold_s if plateau_hold_s is None else plateau_hold_s
        )
        self.min_exposure_s = (
            config.min_exposure_s if min_exposure_s is None else min_exposure_s
        )
        self.plateau_eps = config.plateau_eps if plateau_eps is None else plateau_eps
        self.smooth_alpha = config.smooth_alpha if smooth_alpha is None else smooth_alpha
        # Fixed baseline & purge lengths; exposure is DYNAMIC (baseline..plateau/cap).
        hz = config.scan_hz
        self.n_base = max(1, round(config.baseline_s * hz))
        self.n_exp_max = max(1, round(config.exposure_s * hz))   # exposure cap
        self.n_purge = max(1, round(config.purge_s * hz))
        self.n_plateau = max(1, round(config.plateau_s * hz))    # trailing feature window

        self._pending_label: str | None = None
        self._pending_save = True
        self._save = True
        self._settling = False
        self._stability: StabilityMonitor | None = None
        self._capturing = False
        self._label: str | None = None
        self._buf: list = []
        self._phase: str | None = None
        self._recovery: RecoveryMonitor | None = None
        # per-capture dynamic-exposure state (reset when a capture begins)
        self._plateau: ResponsePlateauMonitor | None = None
        self._exp_end: int | None = None
        self._r0_est: np.ndarray | None = None
        # optional airflow sink the engine drives per phase (fresh vs sample straw)
        self._airflow = None

    @property
    def n(self) -> int:
        """Upper-bound frames for one full capture (baseline + exposure cap + purge).

        Exposure is dynamic, so a real capture is usually *shorter* than this; ``n``
        is the ceiling used for progress display and as a settle-independent cap.
        """
        return self.n_base + self.n_exp_max + self.n_purge

    def set_airflow(self, fn) -> None:
        """Wire a servo-command sink ``fn("S<angle>")`` that the engine drives per phase.

        Called immediately with the fresh-air angle so the rig starts in clean air;
        thereafter the engine sends the sample angle for exposure and the fresh-air
        angle for baseline/purge. ``fn=None`` disables airflow driving (manual rig).
        """
        self._airflow = fn
        if fn is not None:
            fn(f"S{self.config.servo_fresh_air_angle}")

    def _command_airflow(self, phase: str) -> None:
        if self._airflow is None:
            return
        angle = (
            self.config.servo_sample_angle
            if phase == "exposure"
            else self.config.servo_fresh_air_angle
        )
        self._airflow(f"S{angle}")

    def _dynamic_slices(self, exp_end: int, total: int) -> dict:
        """Half-open phase slices for a capture whose exposure ended at ``exp_end``."""
        n_base = self.n_base
        plat_start = max(n_base, exp_end - self.n_plateau)
        return {
            "baseline": (0, n_base),
            "exposure": (n_base, exp_end),
            "purge": (exp_end, total),
            "plateau": (plat_start, exp_end),
        }

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
        * ``settle``   — the settle :class:`StabilityMonitor` status while settling, else None.
        * ``plateau``  — the :class:`ResponsePlateauMonitor` status during exposure, else None.
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
            "plateau": None,
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
                # sensors are at rest — begin the windowed capture on the next frame
                self._settling = False
                self._capturing = True
                self._buf = []
                self._phase = None
                self._plateau = None
                self._exp_end = None
                self._r0_est = None
            return event

        if self._capturing:
            self._buf.append(frame)
            n_so_far = len(self._buf)

            if n_so_far <= self.n_base:
                phase = "baseline"                       # fixed-length clean-air window
            elif self._exp_end is None:
                phase = "exposure"                       # dynamic — hold until it stops growing
                if self._plateau is None:
                    # entering exposure: fix R0 from the baseline window and start the
                    # growth-plateau detector (min floor + hold; the cap bounds it).
                    base = np.array(
                        [_rs_of(f[1], self.config) for f in self._buf[: self.n_base]]
                    )
                    self._r0_est = base.mean(axis=0)
                    # Clamp the min-exposure floor to the cap: if a config sets
                    # exposure_s < min_exposure_s, the cap bounds the sniff anyway, so
                    # keep the plateau gate *live* up to the cap rather than silently
                    # inert (min_frames > cap would make `plateaued` unreachable).
                    min_s = min(self.min_exposure_s, self.n_exp_max / self.config.scan_hz)
                    self._plateau = ResponsePlateauMonitor(
                        self._r0_est, self.config.scan_hz,
                        hold_s=self.plateau_hold_s, min_s=min_s,
                        eps=self.plateau_eps, ema_alpha=self.smooth_alpha,
                    )
                st = self._plateau.update(rs)
                event["plateau"] = st                    # live exposure feedback for the UI
                exp_len = n_so_far - self.n_base
                if st["plateaued"] or exp_len >= self.n_exp_max:
                    self._exp_end = n_so_far             # this frame is the last exposure frame
            else:
                phase = "purge"                          # fixed-length recovery window

            event["phase"] = phase
            total = (self._exp_end + self.n_purge) if self._exp_end is not None else self.n
            event["capture"] = (n_so_far, total)
            if phase != self._phase:
                event["phase_changed"] = True
                self._phase = phase
                self._command_airflow(phase)             # fresh vs sample straw

            if self._exp_end is not None and n_so_far >= self._exp_end + self.n_purge:
                slices = self._dynamic_slices(self._exp_end, n_so_far)
                result = self.recorder.process(self._buf, self._label, slices=slices)
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
    """A servo-driven continuous airflow simulator — a drop-in frame source.

    Models a continuously-relaxing sensor array whose airflow is chosen by the
    servo, exactly like real hardware: at the SAMPLE angle the resistances relax
    toward the current odor's target (rise time constant ``tau_rise``); at the
    FRESH angle they relax back toward clean air (``tau_decay``). This lets the
    engine's *dynamic* exposure (hold until plateau) behave identically in sim and
    on hardware.

    Drive it with :meth:`write_command` (``"S<angle>"`` — the same command the app
    sends the device) and :meth:`set_odor` to choose which odor the sample presents.
    """

    def __init__(self, config, seed: int = 0, noise_counts: float = 1.0,
                 r_base=None, tau_rise: float = 5.0, tau_decay: float = 20.0):
        self.config = config
        n = config.n_channels
        self.r_base = (
            np.asarray(r_base, dtype=np.float64)
            if r_base is not None
            else np.linspace(20000.0, 60000.0, n)
        )
        self.tau_rise = float(tau_rise)
        self.tau_decay = float(tau_decay)
        self.noise = float(noise_counts)
        self._rng = np.random.default_rng(seed)
        self._dt = 1.0 / config.scan_hz
        self._step_ms = round(1000 / config.scan_hz)
        self._t = 0
        self._rs = self.r_base.copy()          # current resistance state
        self._odor: str | None = None
        self._at_sample = False
        self._sample_angle = config.servo_sample_angle
        self._target = self.r_base.copy()

    def set_odor(self, label) -> None:
        """Choose the odor presented while airflow is at the sample angle."""
        self._odor = label
        self._retarget()

    def write_command(self, text) -> bool:
        """React to an ``S<angle>`` airflow command (sample angle → present odor)."""
        try:
            angle = int(str(text).strip().lstrip("Ss").strip())
        except (ValueError, TypeError):
            return False
        self._at_sample = angle == self._sample_angle
        self._retarget()
        return True

    def _retarget(self) -> None:
        if self._at_sample and self._odor:
            gain = _odor_gain(self.config, self._odor)
            self._target = self.r_base / (1.0 + gain)  # reducing gas drops Rs
        else:
            self._target = self.r_base

    def read(self) -> tuple[int, np.ndarray]:
        """Advance the relaxation one frame and return ``(t_ms, raw counts)``."""
        tau = self.tau_rise if self._at_sample else self.tau_decay
        alpha = 1.0 - np.exp(-self._dt / tau)
        self._rs = self._rs + (self._target - self._rs) * alpha
        rl = self.config.rl_array()
        v_rl = rl * self.config.vcc / (self._rs + rl)  # invert Rs -> V_RL
        full_scale = float(2 ** self.config.bits - 1)
        counts = v_rl * full_scale / self.config.vref
        if self.noise > 0.0:
            counts = counts + self._rng.normal(0.0, self.noise, size=counts.shape)
        counts = np.clip(np.rint(counts), 0, full_scale).astype(np.int64)
        t = self._t
        self._t += self._step_ms
        return (t, counts)

    def frames(self):
        """Endless iterator over :meth:`read` (interface parity with readers)."""
        while True:
            yield self.read()

    def close(self) -> None:  # interface parity
        return None
