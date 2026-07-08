"""Tests for sniffsniff.dataset — Dataset container, simulate_dataset, load_dataset."""
from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from sniffsniff.config import default_config
from sniffsniff.dataset import Dataset, load_dataset, simulate_dataset
from sniffsniff.record import SniffRecorder
from sniffsniff.simulator import Simulator


# ----------------------------------------------------------------------------
# Dataset dataclass
# ----------------------------------------------------------------------------
def test_dataset_classes_sorted_unique():
    X = np.zeros((4, 48), dtype=np.float64)
    y = np.array(["b", "a", "b", "c"])
    ds = Dataset(X=X, y=y, feature_names=[f"f{i}" for i in range(48)], ids=list("wxyz"))
    assert ds.classes == ["a", "b", "c"]


def test_dataset_is_frozen():
    X = np.zeros((1, 48), dtype=np.float64)
    ds = Dataset(X=X, y=np.array(["a"]), feature_names=[f"f{i}" for i in range(48)], ids=["a_0000"])
    with pytest.raises(Exception):
        ds.X = np.ones((1, 48))  # frozen dataclass -> FrozenInstanceError


# ----------------------------------------------------------------------------
# simulate_dataset
# ----------------------------------------------------------------------------
def test_simulate_dataset_shape_and_classes():
    cfg = default_config()
    ds = simulate_dataset(cfg, ["coffee", "vinegar", "alcohol"], reps=4, seed=0)
    assert isinstance(ds, Dataset)
    assert ds.X.shape == (12, 48)
    assert ds.X.dtype == np.float64
    assert ds.y.shape == (12,)
    assert ds.classes == ["alcohol", "coffee", "vinegar"]
    assert len(ds.feature_names) == 48
    assert len(ds.ids) == 12


def test_simulate_dataset_ids_unique_and_named():
    cfg = default_config()
    ds = simulate_dataset(cfg, ["coffee", "vinegar"], reps=3, seed=7)
    assert len(set(ds.ids)) == len(ds.ids)  # unique
    assert "coffee_0000" in ds.ids
    assert "coffee_0002" in ds.ids
    assert "vinegar_0001" in ds.ids


def test_simulate_dataset_feature_names_match_config():
    from sniffsniff import features

    cfg = default_config()
    ds = simulate_dataset(cfg, ["coffee"], reps=1, seed=0)
    assert ds.feature_names == features.feature_names(cfg.sensor_names())


def test_simulate_dataset_deterministic():
    cfg = default_config()
    a = simulate_dataset(cfg, ["coffee", "vinegar", "alcohol"], reps=4, seed=0)
    b = simulate_dataset(cfg, ["coffee", "vinegar", "alcohol"], reps=4, seed=0)
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.y, b.y)
    assert a.ids == b.ids


def test_simulate_dataset_seed_changes_data():
    cfg = default_config()
    a = simulate_dataset(cfg, ["coffee"], reps=2, seed=0)
    b = simulate_dataset(cfg, ["coffee"], reps=2, seed=1)
    assert not np.array_equal(a.X, b.X)


def test_simulate_dataset_reps_vary_within_odor():
    cfg = default_config()
    ds = simulate_dataset(cfg, ["coffee"], reps=2, seed=0)
    # each rep uses a distinct simulator seed, so the two coffee reps differ
    assert not np.array_equal(ds.X[0], ds.X[1])


def test_simulate_dataset_all_finite():
    cfg = default_config()
    ds = simulate_dataset(cfg, ["coffee", "vinegar", "alcohol"], reps=4, seed=0)
    assert np.all(np.isfinite(ds.X))


def test_simulate_dataset_y_matches_odor_order():
    cfg = default_config()
    ds = simulate_dataset(cfg, ["coffee", "vinegar"], reps=2, seed=0)
    # odor-major layout: coffee, coffee, vinegar, vinegar
    assert list(ds.y) == ["coffee", "coffee", "vinegar", "vinegar"]


def test_simulate_dataset_suppresses_noisy_baseline_warning():
    cfg = default_config()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any leaked UserWarning would fail the test
        simulate_dataset(cfg, ["coffee", "vinegar"], reps=2, seed=0)


def test_simulate_dataset_matches_manual_pipeline():
    """A single sample must equal SniffRecorder.process(...).features for its seed."""
    cfg = default_config()
    ds = simulate_dataset(cfg, ["coffee", "vinegar"], reps=1, seed=0)
    # vinegar is odor index 1, rep 0 -> seed 0 + 1*1000 + 0 = 1000
    frames = Simulator(cfg, seed=1000, noise_counts=1.0).sniff_frames("vinegar")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = SniffRecorder(cfg, ".").process(frames, "vinegar").features
    idx = ds.ids.index("vinegar_0000")
    np.testing.assert_array_equal(ds.X[idx], expected)


# ----------------------------------------------------------------------------
# load_dataset
# ----------------------------------------------------------------------------
def test_load_dataset_roundtrips_recorder_output(tmp_path):
    cfg = default_config()
    rec = SniffRecorder(cfg, tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f1 = Simulator(cfg, seed=1, noise_counts=1.0).sniff_frames("coffee")
        r1 = rec.process(f1, "coffee")
        rec.save(r1)
        f2 = Simulator(cfg, seed=2, noise_counts=1.0).sniff_frames("vinegar")
        r2 = rec.process(f2, "vinegar")
        rec.save(r2)

    ds = load_dataset(tmp_path)
    assert isinstance(ds, Dataset)
    assert ds.X.shape == (2, 48)
    assert ds.X.dtype == np.float64
    assert ds.classes == ["coffee", "vinegar"]
    assert len(ds.feature_names) == 48
    assert set(ds.ids) == {"coffee_0000", "vinegar_0000"}

    # features must match the recorder's exactly (id-aligned)
    ci = ds.ids.index("coffee_0000")
    np.testing.assert_array_equal(ds.X[ci], r1.features)
    vi = ds.ids.index("vinegar_0000")
    np.testing.assert_array_equal(ds.X[vi], r2.features)


def test_load_dataset_label_from_dir(tmp_path):
    cfg = default_config()
    rec = SniffRecorder(cfg, tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for od in ("coffee", "alcohol"):
            frames = Simulator(cfg, seed=3, noise_counts=1.0).sniff_frames(od)
            rec.record(frames, od)
    ds = load_dataset(tmp_path)
    for lbl, fid in zip(ds.y, ds.ids):
        assert fid.startswith(lbl)


def test_load_dataset_feature_names_length(tmp_path):
    cfg = default_config()
    rec = SniffRecorder(cfg, tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec.record(Simulator(cfg, seed=1).sniff_frames("coffee"), "coffee")
    ds = load_dataset(tmp_path)
    assert len(ds.feature_names) == 48


def test_load_dataset_skips_nonfinite_with_warning(tmp_path):
    cfg = default_config()
    rec = SniffRecorder(cfg, tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # a clean, finite sniff
        rec.record(Simulator(cfg, seed=1).sniff_frames("coffee"), "coffee")

    # craft a NaN sniff written into a label dir directly
    bad_dir = tmp_path / "coffee"
    bad_feats = np.zeros(48, dtype=np.float64)
    bad_feats[5] = np.nan  # open channel -> non-finite feature
    meta = {
        "label": "coffee",
        "id": "coffee_9999",
        "feature_names": [f"c{i}" for i in range(48)],
    }
    np.savez_compressed(
        bad_dir / "coffee_9999.npz",
        features=bad_feats,
        meta=json.dumps(meta),
    )

    with pytest.warns(UserWarning):
        ds = load_dataset(tmp_path)

    # the bad sniff is dropped; only the finite one survives
    assert ds.X.shape == (1, 48)
    assert "coffee_9999" not in ds.ids
    assert "coffee_0000" in ds.ids
    assert np.all(np.isfinite(ds.X))


def test_load_dataset_empty_dir_returns_empty(tmp_path):
    ds = load_dataset(tmp_path)
    assert ds.X.shape[0] == 0
    assert list(ds.y) == []
    assert ds.ids == []
