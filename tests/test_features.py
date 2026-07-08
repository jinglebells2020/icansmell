"""Tests for :mod:`sniffsniff.features` — pure feature extraction.

Hand-computed EMA / peak / plateau reference values are checked against the
implementation. The module MUST be channel-count-agnostic: N is inferred from
``y.shape[1]``, never hard-coded to 6.
"""
import numpy as np
import pytest

from sniffsniff.features import (
    FEATURE_BASE_NAMES,
    extract_features,
    feature_names,
)

ALPHAS = (0.1, 0.01, 0.001)


def test_feature_base_names_shape():
    assert FEATURE_BASE_NAMES == (
        "peak",
        "plateau_mean",
        "ema_rise_a0",
        "ema_rise_a1",
        "ema_rise_a2",
        "ema_decay_a0",
        "ema_decay_a1",
        "ema_decay_a2",
    )
    assert len(FEATURE_BASE_NAMES) == 8


def test_feature_names_length_and_ordering():
    sensors = ["MQ2", "MQ3", "MQ4", "MQ7", "MQ8", "MQ135"]
    names = feature_names(sensors)
    assert len(names) == 48
    # Sensor-major: first 8 belong to sensor 0, next 8 to sensor 1, ...
    assert names[0] == "MQ2__peak"
    assert names[1] == "MQ2__plateau_mean"
    assert names[7] == "MQ2__ema_decay_a2"
    assert names[8] == "MQ3__peak"
    assert names[15] == "MQ3__ema_decay_a2"
    assert names[40] == "MQ135__peak"
    assert names[47] == "MQ135__ema_decay_a2"
    # Full construction check.
    expected = [f"{s}__{b}" for s in sensors for b in FEATURE_BASE_NAMES]
    assert names == expected


def test_feature_names_three_channels():
    names = feature_names(["a", "b", "c"])
    assert len(names) == 24
    assert names[0] == "a__peak"
    assert names[23] == "c__ema_decay_a2"


def _make_step_curve(n_channels: int):
    """Column 0: 0 during baseline, +2.0 during exposure, decaying in purge.

    T = 12. exposure slice = (2, 7), purge slice = (7, 12),
    plateau slice = (4, 7) (last 3 exposure samples, all == 2.0).
    """
    T = 12
    y = np.zeros((T, n_channels), dtype=np.float64)
    col0 = np.array([0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.5, 1.0, 0.5, 0.2, 0.0])
    y[:, 0] = col0
    return y


def test_peak_and_plateau_step_curve():
    y = _make_step_curve(6)
    exposure = (2, 7)
    purge = (7, 12)
    plateau = (4, 7)  # samples at t=4,5,6 are all exactly 2.0
    feats = extract_features(
        y, exposure=exposure, purge=purge, plateau=plateau, ema_alphas=ALPHAS
    )
    # Sensor 0's block is feats[0:8].
    peak = feats[0]
    plateau_mean = feats[1]
    assert peak == pytest.approx(2.0)
    assert plateau_mean == pytest.approx(2.0)


def test_ema_rise_and_decay_hand_computed():
    """Hand-traced EMA over a known dy sequence for the three alphas.

    y col0 (T=12): [0,0,1,2,2,2,2,1.5,1,0.5,0.2,0]
    dy[0]=0, dy[k]=y[k]-y[k-1].
    ema_a[k] = (1-a)*ema_a[k-1] + a*dy[k], ema_a[0]=0.
    exposure=(2,7): ema_rise = max ema over t in {2,3,4,5,6}
    purge=(7,12):   ema_decay = min ema over t in {7,8,9,10,11}
    Reference values computed independently below.
    """
    y = _make_step_curve(6)
    exposure = (2, 7)
    purge = (7, 12)
    plateau = (4, 7)
    feats = extract_features(
        y, exposure=exposure, purge=purge, plateau=plateau, ema_alphas=ALPHAS
    )
    # Sensor 0 block layout: [peak, plateau_mean,
    #   ema_rise_a0, ema_rise_a1, ema_rise_a2,
    #   ema_decay_a0, ema_decay_a1, ema_decay_a2]
    ema_rise_a0 = feats[2]
    ema_rise_a1 = feats[3]
    ema_rise_a2 = feats[4]
    ema_decay_a0 = feats[5]
    ema_decay_a1 = feats[6]
    ema_decay_a2 = feats[7]

    # Hand-computed references (see module-level derivation comment).
    assert ema_rise_a0 == pytest.approx(0.19, abs=1e-12)
    assert ema_rise_a1 == pytest.approx(0.0199, abs=1e-12)
    assert ema_rise_a2 == pytest.approx(0.001999, abs=1e-12)

    assert ema_decay_a0 == pytest.approx(-0.07496623009999999, abs=1e-12)
    assert ema_decay_a1 == pytest.approx(-0.001162355630884392, abs=1e-12)
    assert ema_decay_a2 == pytest.approx(-1.2141137304682318e-05, abs=1e-12)


def test_ema_independent_reference():
    """Recompute the EMA features with an independent loop and cross-check."""
    y = _make_step_curve(6)
    exposure = (2, 7)
    purge = (7, 12)
    plateau = (4, 7)
    feats = extract_features(
        y, exposure=exposure, purge=purge, plateau=plateau, ema_alphas=ALPHAS
    )

    col = y[:, 0]
    dy = np.zeros_like(col)
    dy[1:] = col[1:] - col[:-1]
    for i, a in enumerate(ALPHAS):
        ema = np.zeros_like(col)
        for k in range(1, len(col)):
            ema[k] = (1 - a) * ema[k - 1] + a * dy[k]
        exp_rise = ema[exposure[0]:exposure[1]].max()
        exp_decay = ema[purge[0]:purge[1]].min()
        assert feats[2 + i] == pytest.approx(exp_rise, abs=1e-12)
        assert feats[5 + i] == pytest.approx(exp_decay, abs=1e-12)


def test_output_shape_six_channels():
    y = _make_step_curve(6)
    feats = extract_features(
        y, exposure=(2, 7), purge=(7, 12), plateau=(4, 7), ema_alphas=ALPHAS
    )
    assert feats.shape == (48,)
    assert feats.dtype == np.float64


def test_output_shape_three_channels():
    """Channel-agnosticism: 3 columns -> 24 features."""
    y = _make_step_curve(3)
    feats = extract_features(
        y, exposure=(2, 7), purge=(7, 12), plateau=(4, 7), ema_alphas=ALPHAS
    )
    assert feats.shape == (24,)
    assert feats.dtype == np.float64


def test_peak_uses_absolute_value():
    """peak is max |y| over exposure — a large negative excursion should win."""
    y = np.zeros((6, 2), dtype=np.float64)
    # exposure slice (1, 5): put a -3.0 dip in column 0.
    y[:, 0] = [0.0, 1.0, -3.0, 2.0, 1.0, 0.0]
    feats = extract_features(
        y, exposure=(1, 5), purge=(5, 6), plateau=(4, 5), ema_alphas=ALPHAS
    )
    assert feats[0] == pytest.approx(3.0)


def test_extract_requires_three_alphas():
    y = _make_step_curve(6)
    with pytest.raises((ValueError, AssertionError, IndexError)):
        extract_features(
            y, exposure=(2, 7), purge=(7, 12), plateau=(4, 7), ema_alphas=(0.1, 0.01)
        )
