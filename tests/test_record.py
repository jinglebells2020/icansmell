"""Tests for record.py — phase segmentation, R0/CV, and dataset writing."""
from __future__ import annotations

import csv
from dataclasses import replace

import numpy as np
import pytest

from sniffsniff.config import default_config
from sniffsniff.record import (
    SniffRecorder,
    SniffResult,
    baseline_cv,
    compute_r0,
    compute_rs_series,
    phase_slices,
)
from sniffsniff.simulator import Simulator


def test_phase_slices_match_config_timing():
    cfg = default_config()  # scan_hz=20, baseline=15s, exposure=45s, purge=90s, plateau=10s
    n = 3000  # full session length (300 + 900 + 1800)
    s = phase_slices(n, cfg)
    assert s["baseline"] == (0, 300)
    assert s["exposure"] == (300, 1200)
    assert s["purge"] == (1200, 3000)
    # plateau = last plateau_s (10s -> 200 frames) of exposure
    assert s["plateau"] == (1000, 1200)


def test_phase_slices_half_open_partition():
    cfg = default_config()
    n = 3000
    s = phase_slices(n, cfg)
    # baseline, exposure, purge tile [0, n) with no gaps/overlaps
    assert s["baseline"][0] == 0
    assert s["baseline"][1] == s["exposure"][0]
    assert s["exposure"][1] == s["purge"][0]
    assert s["purge"][1] == n
    # plateau lives within exposure
    lo, hi = s["plateau"]
    assert s["exposure"][0] <= lo < hi <= s["exposure"][1]


def test_phase_slices_clamped_to_short_session():
    cfg = default_config()
    n = 400  # shorter than baseline+exposure (300 + 900)
    s = phase_slices(n, cfg)
    for key in ("baseline", "exposure", "purge", "plateau"):
        lo, hi = s[key]
        assert 0 <= lo <= hi <= n


def test_compute_r0_and_cv_known_array():
    # 2 channels, 4 baseline frames.
    rs = np.array(
        [
            [10.0, 100.0],
            [12.0, 100.0],
            [10.0, 100.0],
            [12.0, 100.0],
        ]
    )
    r0 = compute_r0(rs)
    assert r0.shape == (2,)
    np.testing.assert_allclose(r0, [11.0, 100.0])

    cv = baseline_cv(rs)
    assert cv.shape == (2,)
    # ch0: mean 11, std(ddof=0)=1 -> cv = 1/11 ; ch1: constant -> cv 0
    np.testing.assert_allclose(cv, [1.0 / 11.0, 0.0])


def test_compute_rs_series_matches_calibrate():
    cfg = default_config()
    raw = np.full((5, cfg.n_channels), 512, dtype=np.int64)
    rs = compute_rs_series(raw, cfg)
    assert rs.shape == (5, cfg.n_channels)
    assert rs.dtype == np.float64
    # cross-check against a direct calibrate call
    from sniffsniff import calibrate

    expected = calibrate.counts_to_rs(
        raw, cfg.rl_array(), cfg.vcc, cfg.vref, cfg.bits
    )
    np.testing.assert_allclose(rs, expected)


def test_recorder_process_produces_48d(tmp_path):
    cfg = default_config()
    sim = Simulator(cfg, seed=1, noise_counts=0.5)
    frames = sim.sniff_frames("coffee")
    rec = SniffRecorder(cfg, tmp_path)
    result = rec.process(frames, "coffee")
    assert isinstance(result, SniffResult)
    assert result.features.shape == (48,)
    assert result.features.dtype == np.float64
    assert result.raw.shape[1] == cfg.n_channels
    assert result.fractional.shape == result.raw.shape
    assert result.r0.shape == (cfg.n_channels,)
    assert result.label == "coffee"
    # baseline fractional ~ 0 (fractional = rs/r0 - 1 over baseline)
    b_lo, b_hi = result.slices["baseline"]
    np.testing.assert_allclose(
        result.fractional[b_lo:b_hi].mean(axis=0), 0.0, atol=1e-6
    )


def test_recorder_record_writes_npz_and_manifest(tmp_path):
    cfg = default_config()
    sim = Simulator(cfg, seed=2, noise_counts=0.5)
    rec = SniffRecorder(cfg, tmp_path)

    frames = sim.sniff_frames("vinegar")
    path = rec.record(frames, "vinegar")
    assert path.exists()
    assert path.suffix == ".npz"
    assert path.parent.name == "vinegar"

    manifest = tmp_path / "manifest.csv"
    assert manifest.exists()
    with manifest.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["label"] == "vinegar"
    assert rows[0]["id"] == "vinegar_0000"
    assert int(rows[0]["n_samples"]) == len(frames)

    # reload npz and assert round-trip
    data = np.load(path, allow_pickle=False)
    assert data["features"].shape == (48,)
    result = rec.process(frames, "vinegar")
    np.testing.assert_array_equal(data["raw"], result.raw)
    np.testing.assert_allclose(data["fractional"], result.fractional)
    np.testing.assert_allclose(data["r0"], result.r0)
    np.testing.assert_allclose(data["features"], result.features)
    # meta is a JSON string
    import json

    meta = json.loads(str(data["meta"]))
    assert meta["label"] == "vinegar"
    assert meta["id"] == "vinegar_0000"
    assert len(meta["feature_names"]) == 48


def test_recorder_ids_increment(tmp_path):
    cfg = default_config()
    sim = Simulator(cfg, seed=3, noise_counts=0.5)
    rec = SniffRecorder(cfg, tmp_path)
    frames = sim.sniff_frames("alcohol")

    p0 = rec.record(frames, "alcohol")
    p1 = rec.record(frames, "alcohol")
    assert p0.stem == "alcohol_0000"
    assert p1.stem == "alcohol_0001"

    # different label restarts the sequence
    p_other = rec.record(sim.sniff_frames("coffee"), "coffee")
    assert p_other.stem == "coffee_0000"

    manifest = tmp_path / "manifest.csv"
    with manifest.open() as f:
        rows = list(csv.DictReader(f))
    assert [r["id"] for r in rows] == ["alcohol_0000", "alcohol_0001", "coffee_0000"]


def test_recorder_channel_agnostic(tmp_path):
    # drop to a 3-channel config; record must still work and yield 3*8=24 feats.
    cfg = default_config()
    cfg3 = replace(cfg, channels=cfg.channels[:3])
    assert cfg3.n_channels == 3
    sim = Simulator(cfg3, seed=0, noise_counts=0.3)
    rec = SniffRecorder(cfg3, tmp_path)
    result = rec.process(sim.sniff_frames("coffee"), "coffee")
    assert result.features.shape == (24,)
