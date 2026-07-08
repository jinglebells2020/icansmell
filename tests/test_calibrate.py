"""Tests for calibrate.py — pure counts→V→Rs→ratio→fractional conversions.

All expected values are hand-computed closed-form so a regression in the math
is caught immediately. The functions must be numpy-vectorized and accept either
python scalars or ``np.ndarray`` inputs, with no hard-coded channel count.
"""
import numpy as np
import pytest

from sniffsniff import calibrate
from sniffsniff.calibrate import (
    EPS,
    counts_to_fractional,
    counts_to_rs,
    counts_to_volts,
    ratio_to_fractional,
    rs_to_ratio,
    volts_to_rs,
)


# --------------------------------------------------------------------------
# counts_to_volts:  V = counts * vref / (2**bits - 1)
# --------------------------------------------------------------------------
def test_counts_to_volts_full_scale():
    # 1023 counts on a 10-bit ADC at vref=5.0 -> exactly full scale 5.0 V
    assert counts_to_volts(1023, 5.0, 10) == pytest.approx(5.0)


def test_counts_to_volts_zero():
    assert counts_to_volts(0, 5.0, 10) == pytest.approx(0.0)


def test_counts_to_volts_half_scale():
    # 511.5 counts -> 2.5 V (linear); use a mid count to check linearity.
    # 511 * 5 / 1023 = 2.4975562...
    assert counts_to_volts(511, 5.0, 10) == pytest.approx(511 * 5.0 / 1023)


def test_counts_to_volts_array_vectorizes():
    counts = np.array([0, 1023, 511, 1023, 0, 1023], dtype=np.int64)
    out = counts_to_volts(counts, 5.0, 10)
    assert out.shape == (6,)
    assert out.dtype == np.float64
    expected = counts.astype(np.float64) * 5.0 / 1023.0
    np.testing.assert_allclose(out, expected)


def test_counts_to_volts_12bit():
    # channel-count-agnostic AND bit-depth-agnostic: 4095 counts, 12-bit, vref 3.3
    assert counts_to_volts(4095, 3.3, 12) == pytest.approx(3.3)


# --------------------------------------------------------------------------
# volts_to_rs:  Rs = rl * (vcc - v_rl) / v_rl ;  v_rl <= EPS -> inf
# --------------------------------------------------------------------------
def test_volts_to_rs_hand_case():
    # rl=10k, vcc=5, v_rl=1.0 -> Rs = 10000*(5-1)/1 = 40000
    assert volts_to_rs(1.0, 10000.0, 5.0) == pytest.approx(40000.0)


def test_volts_to_rs_zero_volts_is_inf():
    assert volts_to_rs(0.0, 10000.0, 5.0) == np.inf


def test_volts_to_rs_below_eps_is_inf():
    # anything <= EPS is treated as the sensor reading open/rail -> +inf
    assert volts_to_rs(EPS, 10000.0, 5.0) == np.inf
    assert volts_to_rs(EPS / 2, 10000.0, 5.0) == np.inf


def test_volts_to_rs_half_supply():
    # v_rl at half of vcc -> Rs == rl
    assert volts_to_rs(2.5, 10000.0, 5.0) == pytest.approx(10000.0)


def test_volts_to_rs_array_with_zero_entry():
    v = np.array([1.0, 0.0, 2.5, 0.5, 5.0, 4.0], dtype=np.float64)
    rl = np.array([10000.0, 10000.0, 10000.0, 1000.0, 1000.0, 1000.0])
    out = volts_to_rs(v, rl, 5.0)
    assert out.shape == (6,)
    assert out.dtype == np.float64
    # index 1 (v=0) -> inf
    assert out[1] == np.inf
    assert out[0] == pytest.approx(40000.0)
    assert out[2] == pytest.approx(10000.0)
    # v=0.5, rl=1000: 1000*(5-0.5)/0.5 = 9000
    assert out[3] == pytest.approx(9000.0)
    # v=5.0, rl=1000: 1000*(5-5)/5 = 0
    assert out[4] == pytest.approx(0.0)


def test_volts_to_rs_scalar_rl_broadcasts_over_array():
    v = np.array([1.0, 2.5], dtype=np.float64)
    out = volts_to_rs(v, 10000.0, 5.0)
    np.testing.assert_allclose(out, [40000.0, 10000.0])


# --------------------------------------------------------------------------
# rs_to_ratio:  r = rs / r0
# --------------------------------------------------------------------------
def test_rs_to_ratio_unity():
    assert rs_to_ratio(40000.0, 40000.0) == pytest.approx(1.0)


def test_rs_to_ratio_array():
    rs = np.array([40000.0, 20000.0, 80000.0])
    r0 = np.array([40000.0, 40000.0, 40000.0])
    out = rs_to_ratio(rs, r0)
    np.testing.assert_allclose(out, [1.0, 0.5, 2.0])
    assert out.dtype == np.float64


def test_rs_to_ratio_inf_stays_inf():
    assert rs_to_ratio(np.inf, 40000.0) == np.inf


# --------------------------------------------------------------------------
# ratio_to_fractional:  y = r - 1
# --------------------------------------------------------------------------
def test_ratio_to_fractional_zero_at_unity():
    assert ratio_to_fractional(1.0) == pytest.approx(0.0)


def test_ratio_to_fractional_array():
    r = np.array([1.0, 0.5, 2.0])
    out = ratio_to_fractional(r)
    np.testing.assert_allclose(out, [0.0, -0.5, 1.0])
    assert out.dtype == np.float64


# --------------------------------------------------------------------------
# counts_to_rs:  compose counts->volts->rs
# --------------------------------------------------------------------------
def test_counts_to_rs_hand_case():
    # We want v_rl = 1.0 V. With vref=5, bits=10: counts = 1.0 * 1023 / 5 = 204.6
    # Use vref chosen so a round count gives v_rl=1.0 exactly:
    # vref=5, bits=10, counts such that counts*5/1023 = 1.0 -> counts = 204.6 (non-int).
    # Instead verify by composition against the primitives directly.
    counts = 204.6
    v = counts_to_volts(counts, 5.0, 10)
    expected = volts_to_rs(v, 10000.0, 5.0)
    got = counts_to_rs(counts, 10000.0, 5.0, 5.0, 10)
    assert got == pytest.approx(expected)
    assert got == pytest.approx(40000.0)


def test_counts_to_rs_zero_counts_is_inf():
    # 0 counts -> 0 V -> inf Rs
    assert counts_to_rs(0, 10000.0, 5.0, 5.0, 10) == np.inf


def test_counts_to_rs_array_vectorizes():
    counts = np.array([204.6, 511.5, 0.0], dtype=np.float64)
    rl = np.array([10000.0, 10000.0, 10000.0])
    out = counts_to_rs(counts, rl, 5.0, 5.0, 10)
    assert out.shape == (3,)
    assert out[0] == pytest.approx(40000.0)  # v=1.0 -> 40k
    assert out[1] == pytest.approx(10000.0)  # v=2.5 -> rl
    assert out[2] == np.inf                  # v=0 -> inf


# --------------------------------------------------------------------------
# counts_to_fractional: full chain counts -> y
# --------------------------------------------------------------------------
def test_counts_to_fractional_clean_air_is_zero():
    # A count whose Rs equals R0 must give fractional response ~0.
    # Pick counts so Rs=40000 with rl=10000: v_rl=1.0 -> counts=204.6
    counts = 204.6
    r0 = 40000.0
    y = counts_to_fractional(counts, r0, 10000.0, 5.0, 5.0, 10)
    assert y == pytest.approx(0.0, abs=1e-9)


def test_counts_to_fractional_reducing_gas_negative():
    # A reducing gas drops Rs below R0 -> ratio < 1 -> fractional negative.
    # v_rl=2.5 -> Rs=10000 (rl=10000); r0=40000 -> r=0.25 -> y=-0.75
    counts = 511.5  # 511.5*5/1023 = 2.5 V
    y = counts_to_fractional(counts, 40000.0, 10000.0, 5.0, 5.0, 10)
    assert y == pytest.approx(-0.75)


def test_counts_to_fractional_array_vectorizes():
    # (6,) input -> (6,) output, mixed channels including one that is clean-air.
    counts = np.array([204.6, 511.5, 204.6, 511.5, 204.6, 0.0], dtype=np.float64)
    rl = np.full(6, 10000.0)
    r0 = np.full(6, 40000.0)
    y = counts_to_fractional(counts, r0, rl, 5.0, 5.0, 10)
    assert y.shape == (6,)
    assert y.dtype == np.float64
    np.testing.assert_allclose(y[0], 0.0, atol=1e-9)   # clean air
    np.testing.assert_allclose(y[1], -0.75)            # reducing
    np.testing.assert_allclose(y[5], np.inf)           # 0 counts -> inf Rs -> inf ratio -> inf y


def test_eps_value():
    assert calibrate.EPS == 1e-9


def test_channel_agnostic_arbitrary_n():
    # Nothing hard-codes 6: an N=3 and N=8 array both work.
    for n in (1, 3, 8, 16):
        counts = np.full(n, 204.6)
        rl = np.full(n, 10000.0)
        r0 = np.full(n, 40000.0)
        y = counts_to_fractional(counts, r0, rl, 5.0, 5.0, 10)
        assert y.shape == (n,)
        np.testing.assert_allclose(y, 0.0, atol=1e-9)
