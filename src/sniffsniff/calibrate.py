"""Pure, stateless sensor-calibration math for the sniffsniff e-nose.

The chain, per channel::

    V_RL = counts * VREF / (2**BITS - 1)     # ADC counts -> load-resistor volts
    Rs   = RL * (VCC - V_RL) / V_RL          # sensor resistance (guard V_RL <= EPS)
    r    = Rs / R0                           # ratio vs clean-air baseline R0
    y    = r - 1                             # fractional response (dimensionless)

Every function is numpy-vectorized and accepts either a python scalar or an
``np.ndarray``; nothing hard-codes the channel count ``N`` — it flows from the
shape of the inputs. Computed results are float64.

Div-by-zero guard: where ``V_RL <= EPS`` the load-resistor voltage is treated as
zero (sensor open / rail) and ``Rs`` is ``+inf`` rather than a NaN, so the
feature/record layer can flag the channel instead of being NaN-poisoned.
"""
from __future__ import annotations

import numpy as np

EPS: float = 1e-9

__all__ = [
    "EPS",
    "counts_to_volts",
    "volts_to_rs",
    "rs_to_ratio",
    "ratio_to_fractional",
    "counts_to_rs",
    "counts_to_fractional",
]


def counts_to_volts(counts, vref, bits) -> np.ndarray:
    """Convert raw ADC counts to load-resistor volts.

    ``V_RL = counts * vref / (2**bits - 1)``. Vectorizes over ``counts``;
    ``vref`` and ``bits`` are scalars (or broadcastable). Returns float64.
    """
    counts = np.asarray(counts, dtype=np.float64)
    full_scale = float(2 ** bits - 1)
    return counts * (float(vref) / full_scale)


def volts_to_rs(v_rl, rl, vcc) -> np.ndarray:
    """Convert load-resistor volts to sensor resistance ``Rs``.

    ``Rs = rl * (vcc - v_rl) / v_rl``. Where ``v_rl <= EPS`` the channel is
    treated as open/rail and ``Rs`` is ``+inf`` (no div-by-zero, no NaN).
    Broadcasts ``v_rl`` against ``rl``; ``vcc`` is a scalar (or broadcastable).
    Returns float64.
    """
    v_rl = np.asarray(v_rl, dtype=np.float64)
    rl = np.asarray(rl, dtype=np.float64)
    vcc = np.asarray(vcc, dtype=np.float64)

    open_mask = v_rl <= EPS
    # Avoid a divide-by-zero warning by computing on a safe denominator, then
    # overwriting the guarded entries with +inf.
    safe_v = np.where(open_mask, 1.0, v_rl)
    rs = rl * (vcc - v_rl) / safe_v
    rs = np.where(open_mask, np.inf, rs)
    # np.where on all-scalar inputs yields a 0-d array; return it as float64
    # ndarray for a consistent, broadcastable result type.
    return np.asarray(rs, dtype=np.float64)


def rs_to_ratio(rs, r0) -> np.ndarray:
    """Ratio of sensor resistance to clean-air baseline: ``r = rs / r0``.

    ``inf`` in ``rs`` propagates to ``inf`` in the ratio. Returns float64.
    """
    rs = np.asarray(rs, dtype=np.float64)
    r0 = np.asarray(r0, dtype=np.float64)
    return rs / r0


def ratio_to_fractional(r) -> np.ndarray:
    """Fractional response ``y = r - 1`` (0 in clean air). Returns float64."""
    r = np.asarray(r, dtype=np.float64)
    return r - 1.0


def counts_to_rs(counts, rl, vcc, vref, bits) -> np.ndarray:
    """Compose counts -> volts -> Rs in one step. Returns float64."""
    v_rl = counts_to_volts(counts, vref, bits)
    return volts_to_rs(v_rl, rl, vcc)


def counts_to_fractional(counts, r0, rl, vcc, vref, bits) -> np.ndarray:
    """Full chain counts -> fractional response ``y``. Returns float64."""
    rs = counts_to_rs(counts, rl, vcc, vref, bits)
    r = rs_to_ratio(rs, r0)
    return ratio_to_fractional(r)
