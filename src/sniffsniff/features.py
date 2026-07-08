"""Pure feature extraction — one sniff's fractional curve to an ``N*8``-D vector.

Matches the UCI-style 8-feature design: 2 steady-state (peak, plateau_mean) and
6 transient EMA features (rise/decay for three alphas). The module is
channel-count-agnostic: ``N`` is inferred from ``y.shape[1]`` and never hard-coded.

Feature ordering is fixed and sensor-major: sensor 0's 8 features, then sensor
1's 8, and so on, so downstream code has a stable column layout.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# The 8 per-sensor feature names, in fixed order.
FEATURE_BASE_NAMES: tuple[str, ...] = (
    "peak",
    "plateau_mean",
    "ema_rise_a0",
    "ema_rise_a1",
    "ema_rise_a2",
    "ema_decay_a0",
    "ema_decay_a1",
    "ema_decay_a2",
)


def feature_names(sensor_names: list[str]) -> list[str]:
    """Return the ``len(sensor_names) * 8`` feature column names, sensor-major.

    Each name is ``"<sensor>__<base>"``; sensor 0's 8 names come first, then
    sensor 1's, and so on.
    """
    return [f"{s}__{b}" for s in sensor_names for b in FEATURE_BASE_NAMES]


def _ema(dy: np.ndarray, alpha: float) -> np.ndarray:
    """Column-wise EMA over the difference sequence ``dy`` (shape ``(T, N)``).

    ``ema[0] = 0`` (all channels); ``ema[k] = (1-a)*ema[k-1] + a*dy[k]``.
    Returns an array of the same shape as ``dy``.
    """
    ema = np.zeros_like(dy)
    for k in range(1, dy.shape[0]):
        ema[k] = (1.0 - alpha) * ema[k - 1] + alpha * dy[k]
    return ema


def extract_features(
    y: np.ndarray,
    *,
    exposure: tuple[int, int],
    purge: tuple[int, int],
    plateau: tuple[int, int],
    ema_alphas: Sequence[float],
) -> np.ndarray:
    """Extract the ``N*8``-D feature vector from a sniff's fractional curve.

    Parameters
    ----------
    y:
        Fractional response, shape ``(T, N)``. ``N`` (channel count) is inferred
        from ``y.shape[1]``.
    exposure, purge, plateau:
        ``(start, end)`` half-open slices into the time axis ``T``.
    ema_alphas:
        Exactly three smoothing factors.

    Returns
    -------
    np.ndarray
        Shape ``(N*8,)`` float64, sensor-major ordering.
    """
    alphas = list(ema_alphas)
    if len(alphas) != 3:
        raise ValueError(f"ema_alphas must contain exactly 3 floats; got {len(alphas)}")

    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError(f"y must be 2-D (T, N); got shape {y.shape}")
    T, N = y.shape

    exp_lo, exp_hi = exposure
    pur_lo, pur_hi = purge
    plat_lo, plat_hi = plateau

    # dy per column over the FULL curve: dy[0] = 0, dy[k] = y[k] - y[k-1].
    dy = np.zeros_like(y)
    dy[1:] = y[1:] - y[:-1]

    # Steady-state features over the requested slices.
    peak = np.abs(y[exp_lo:exp_hi]).max(axis=0)              # (N,)
    plateau_mean = y[plat_lo:plat_hi].mean(axis=0)           # (N,)

    # Transient EMA features, one rise/decay per alpha.
    ema_rise = np.empty((3, N), dtype=np.float64)
    ema_decay = np.empty((3, N), dtype=np.float64)
    for i, a in enumerate(alphas):
        ema = _ema(dy, a)
        ema_rise[i] = ema[exp_lo:exp_hi].max(axis=0)
        ema_decay[i] = ema[pur_lo:pur_hi].min(axis=0)

    # Assemble sensor-major: for each sensor, the 8 features in FEATURE_BASE_NAMES order.
    # Per-sensor block matrix, shape (N, 8).
    blocks = np.column_stack(
        [
            peak,               # peak
            plateau_mean,       # plateau_mean
            ema_rise[0],        # ema_rise_a0
            ema_rise[1],        # ema_rise_a1
            ema_rise[2],        # ema_rise_a2
            ema_decay[0],       # ema_decay_a0
            ema_decay[1],       # ema_decay_a1
            ema_decay[2],       # ema_decay_a2
        ]
    )
    return blocks.reshape(N * 8).astype(np.float64)
