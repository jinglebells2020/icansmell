"""Tests for the non-UI SniffController orchestration layer (all --sim)."""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from sniffsniff.capture import session_frame_count
from sniffsniff.config import default_config
from sniffsniff.tui.controller import SniffController


def _fast_config():
    """A short-timing config so sessions record quickly under test."""
    return dataclasses.replace(
        default_config(),
        baseline_s=1,
        exposure_s=2,
        purge_s=1,
        plateau_s=0.5,
    )


def _controller(tmp_path):
    cfg = _fast_config()
    return SniffController(
        cfg,
        out_dir=tmp_path,
        use_sim=True,
        seed=0,
        model_path=str(tmp_path / "model.joblib"),
    ), cfg


def test_record_one_writes_npz_and_reports_phases(tmp_path):
    ctrl, cfg = _controller(tmp_path)

    phases: list[str] = []
    frame_hits: list[int] = []

    path = ctrl.record_one(
        "coffee",
        on_phase=lambda phase, k, n: phases.append(phase),
        on_frame=lambda k, n, phase, frame: frame_hits.append(k),
    )

    assert path.exists()
    assert path.suffix == ".npz"
    assert path.parent.name == "coffee"
    assert path.parent.parent == tmp_path

    assert phases == ["baseline", "exposure", "purge"]
    assert len(frame_hits) == session_frame_count(cfg)


def test_record_many_increments_ids_and_differs(tmp_path):
    ctrl, _ = _controller(tmp_path)

    saved: list[tuple] = []
    paths = ctrl.record_many(
        "coffee", 3, on_saved=lambda p, i: saved.append((p, i))
    )

    assert len(paths) == 3
    assert len(saved) == 3
    assert [i for _, i in saved] == [0, 1, 2]

    stems = [p.stem for p in paths]
    assert stems == ["coffee_0000", "coffee_0001", "coffee_0002"]

    # Per-rep seed offset => the recorded sniffs differ.
    feats = []
    for p in paths:
        with np.load(p, allow_pickle=False) as data:
            feats.append(np.asarray(data["features"]))
    assert not np.allclose(feats[0], feats[1])
    assert not np.allclose(feats[1], feats[2])


def test_dataset_counts(tmp_path):
    ctrl, _ = _controller(tmp_path)

    assert ctrl.dataset_counts() == {}

    ctrl.record_many("coffee", 2)
    ctrl.record_many("vinegar", 3)

    assert ctrl.dataset_counts() == {"coffee": 2, "vinegar": 3}


def test_fit_and_has_model(tmp_path):
    ctrl, _ = _controller(tmp_path)

    assert ctrl.has_model() is False

    ctrl.record_many("coffee", 3)
    ctrl.record_many("vinegar", 3)

    mean, std = ctrl.fit()
    assert 0.0 <= mean <= 1.0
    assert 0.0 <= std <= 1.0
    assert ctrl.has_model() is True


def test_fit_single_class_raises(tmp_path):
    ctrl, _ = _controller(tmp_path)
    ctrl.record_many("coffee", 3)
    with pytest.raises(ValueError):
        ctrl.fit()


def test_identify_after_fit(tmp_path):
    ctrl, _ = _controller(tmp_path)
    ctrl.record_many("coffee", 3)
    ctrl.record_many("vinegar", 3)
    ctrl.fit()

    result = ctrl.identify()

    ds_classes = list(ctrl.dataset_counts().keys())
    assert result["label"] in ds_classes
    assert isinstance(result["proba"], float)
    assert isinstance(result["novelty"], float)
    assert isinstance(result["threshold"], float)
    assert isinstance(result["is_novel"], bool)

    # geometry must be JSON-serializable.
    json.dumps(result["geometry"])


def test_connected_property(tmp_path):
    ctrl, _ = _controller(tmp_path)
    assert ctrl.connected is True  # sim is always "connected"


def test_rs_of_one_frame(tmp_path):
    ctrl, cfg = _controller(tmp_path)
    raw = np.full((cfg.n_channels,), 400, dtype=np.int64)
    rs = ctrl.rs_of(raw)
    assert rs.shape == (cfg.n_channels,)
    assert np.all(np.isfinite(rs))


def test_stream_stops(tmp_path):
    ctrl, _ = _controller(tmp_path)
    seen: list[int] = []

    def on_frame(k, frame):
        seen.append(k)

    # Stop after 5 frames.
    ctrl.stream(on_frame, lambda: len(seen) >= 5, odor="coffee")
    assert len(seen) == 5
