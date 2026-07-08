"""Synthetic MQ-array simulator — a drop-in for the real serial reader.

The simulator emits frames in the exact ``(t_ms, raw[N])`` shape produced by
:class:`sniffsniff.serialio.SerialReader`, so the whole Python pipeline can be
developed and tested with no hardware. Frames are produced by *inverting* the
calibration path (Rs -> V_RL -> counts), so the simulator exercises the real
:mod:`sniffsniff.calibrate` math rather than shortcutting it.

Per-sensor model (see the Foundation design spec):

* Clean-air baseline resistance ``r_base[ch]`` (deterministic per channel).
* Per-odor multiplicative gain ``g[ch]`` (reducing gases drop ``Rs``): during
  exposure ``Rs`` relaxes toward ``r_base / (1 + g)`` with a fast rise time
  constant; during purge it relaxes back toward ``r_base`` with a slower decay
  time constant (``tau_decay > tau_rise``, modelling slow desorption).
* Additive Gaussian noise in *counts* from a seeded ``numpy.random.Generator``,
  making every session byte-reproducible for a given seed.

The gain vectors in :data:`ODOR_PROFILES` are fixed 6-vectors aligned to the
canonical sensor order ``[MQ2, MQ3, MQ4, MQ7, MQ8, MQ135]``. Everything else is
channel-count-agnostic: the frame width and default ``r_base`` derive from
``config.n_channels``, and gain vectors are sliced to ``N`` channels.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from . import calibrate
from .config import Config

__all__ = ["ODOR_PROFILES", "Simulator", "SimulatedReader"]


# Gain vectors aligned to canonical channel order:
#   idx 0=MQ2, 1=MQ3, 2=MQ4, 3=MQ7, 4=MQ8, 5=MQ135
# Chosen for separability:
#   * alcohol dominant on MQ3 (idx1)
#   * vinegar & spoiled_milk dominant on MQ135 (idx5)
#   * coffee broad / moderate across the array
#   * fresh vs spoiled milk pulled apart via MQ4/MQ7/MQ8 (idx2/3/4)
ODOR_PROFILES: dict[str, np.ndarray] = {
    "clean_air": np.zeros(6, dtype=np.float64),
    # broad responder, moderate everywhere, MQ2/MQ135 a touch higher
    "coffee": np.array([0.9, 0.6, 0.4, 0.3, 0.35, 0.7], dtype=np.float64),
    # acetic acid: strong on MQ135 (VOC/ammonia), modest MQ2/MQ3
    "vinegar": np.array([0.4, 0.5, 0.2, 0.2, 0.2, 1.4], dtype=np.float64),
    # ethanol: dominant on MQ3, some MQ2 smoke/VOC bleed
    "alcohol": np.array([0.6, 1.6, 0.3, 0.2, 0.3, 0.4], dtype=np.float64),
    # fresh milk: mild, leans on the methane axis (MQ4), low MQ135
    "fresh_milk": np.array([0.3, 0.2, 0.7, 0.5, 0.2, 0.3], dtype=np.float64),
    # spoiled milk: MQ135 punchline + H2/CO shift (MQ8/MQ7), less MQ4 than fresh
    "spoiled_milk": np.array([0.5, 0.3, 0.3, 0.8, 0.7, 1.2], dtype=np.float64),
}


def _default_r_base(n: int) -> np.ndarray:
    """Deterministic clean-air resistances spread across 20k..60k ohms, shape (N,).

    For ``n == 1`` returns the low end; otherwise an even linear spread so
    channels have distinct baselines (deterministic, no randomness).
    """
    return np.linspace(20000.0, 60000.0, n, dtype=np.float64)


class Simulator:
    """Generate reproducible synthetic sniff sessions for a given config."""

    def __init__(
        self,
        config: Config,
        seed: int = 0,
        r_base: np.ndarray | None = None,
        noise_counts: float = 1.0,
    ) -> None:
        self.config = config
        self.seed = int(seed)
        self.noise_counts = float(noise_counts)

        n = config.n_channels
        if r_base is None:
            r_base = _default_r_base(n)
        r_base = np.asarray(r_base, dtype=np.float64)
        if r_base.shape != (n,):
            raise ValueError(
                f"r_base must have shape ({n},) to match config.n_channels; "
                f"got {r_base.shape}"
            )
        self.r_base = r_base

        # Relaxation time constants (seconds). Decay is slower than rise to model
        # slow desorption during purge.
        self.tau_rise = 5.0
        self.tau_decay = 20.0

    def _gain(self, odor: str) -> np.ndarray:
        """Per-channel gain for ``odor``, sliced/broadcast to ``N`` channels.

        Gain vectors are canonical 6-vectors; for an ``N``-channel config we take
        the first ``N`` entries (``N <= 6``) or tile if a larger array is ever
        configured, keeping the simulator channel-count-agnostic.
        """
        if odor not in ODOR_PROFILES:
            raise KeyError(f"unknown odor {odor!r}; known: {sorted(ODOR_PROFILES)}")
        base = ODOR_PROFILES[odor]
        n = self.config.n_channels
        if n <= base.shape[0]:
            return base[:n].astype(np.float64)
        reps = int(np.ceil(n / base.shape[0]))
        return np.tile(base, reps)[:n].astype(np.float64)

    def sniff_frames(self, odor: str) -> list[tuple[int, np.ndarray]]:
        """Simulate a full three-phase session for ``odor``.

        Phases at ``config.scan_hz``: ``baseline_s`` clean air, ``exposure_s``
        odor, ``purge_s`` clean air. Returns a list of ``(t_ms, raw[N])`` frames
        with ``t_ms`` starting at 0 and stepping by ``round(1000/scan_hz)``.
        """
        cfg = self.config
        gain = self._gain(odor)

        n_base = int(round(cfg.baseline_s * cfg.scan_hz))
        n_exp = int(round(cfg.exposure_s * cfg.scan_hz))
        n_purge = int(round(cfg.purge_s * cfg.scan_hz))
        n_total = n_base + n_exp + n_purge

        dt = 1.0 / cfg.scan_hz  # seconds per frame

        exposure_target = self.r_base / (1.0 + gain)  # Rs drops during exposure
        purge_target = self.r_base                    # relaxes back to baseline

        # Discrete relaxation coefficients: rs += (target - rs) * (1 - exp(-dt/tau))
        alpha_rise = 1.0 - np.exp(-dt / self.tau_rise)
        alpha_decay = 1.0 - np.exp(-dt / self.tau_decay)

        rs = np.empty((n_total, cfg.n_channels), dtype=np.float64)
        cur = self.r_base.copy()
        for k in range(n_total):
            if k < n_base:
                target, a = self.r_base, 0.0  # steady clean air
            elif k < n_base + n_exp:
                target, a = exposure_target, alpha_rise
            else:
                target, a = purge_target, alpha_decay
            cur = cur + (target - cur) * a
            rs[k] = cur

        # Invert calibration: Rs -> V_RL -> counts.
        #   Rs = rl*(vcc - v_rl)/v_rl  =>  v_rl = rl*vcc / (Rs + rl)
        rl = cfg.rl_array()
        v_rl = rl * cfg.vcc / (rs + rl)
        full_scale = float(2 ** cfg.bits - 1)
        counts = v_rl * full_scale / cfg.vref

        # Seeded Gaussian noise in counts.
        rng = np.random.default_rng(self.seed)
        if self.noise_counts > 0.0:
            counts = counts + rng.normal(0.0, self.noise_counts, size=counts.shape)

        counts = np.clip(np.rint(counts), 0, full_scale).astype(np.int64)

        step = round(1000 / cfg.scan_hz)
        return [(int(k * step), counts[k]) for k in range(n_total)]


class SimulatedReader:
    """Replay a pre-generated frame list with the :class:`SerialReader` interface."""

    def __init__(self, frames: list[tuple[int, np.ndarray]]) -> None:
        self._frames = frames

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield the stored frames in order (identical shape to SerialReader)."""
        for frame in self._frames:
            yield frame

    def close(self) -> None:
        """No-op; present for interface parity with :class:`SerialReader`."""
        return None
