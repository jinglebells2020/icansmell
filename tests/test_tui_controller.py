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


# --- v2 guided-training controller additions --------------------------------

from sniffsniff.tui.controller import CLASSIFIERS, DEFAULT_LABELS, GOOD_REPS


def test_default_classifier_is_knn(tmp_path):
    ctrl, _ = _controller(tmp_path)
    assert ctrl.classifier == "knn"


def test_known_labels_includes_defaults_and_recorded(tmp_path):
    ctrl, _ = _controller(tmp_path)
    # With nothing recorded, exactly the defaults, in order.
    assert ctrl.known_labels() == list(DEFAULT_LABELS)

    ctrl.record_many("coffee", 1)  # a default
    # A novel label the simulator doesn't know: record real sim frames but
    # persist them under the new label directly via the recorder.
    from sniffsniff.record import SniffRecorder
    from sniffsniff.simulator import Simulator

    frames = Simulator(ctrl.config, seed=1).sniff_frames("coffee")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        SniffRecorder(ctrl.config, ctrl.out_dir).record(frames, "garlic")

    known = ctrl.known_labels()
    # Defaults preserved, in order, first.
    assert known[: len(DEFAULT_LABELS)] == list(DEFAULT_LABELS)
    # Recorded-but-novel labels appended (once), after the defaults.
    assert "garlic" in known
    assert known.count("garlic") == 1
    assert known.count("coffee") == 1  # not duplicated even though recorded


def test_cycle_classifier_advances_and_wraps(tmp_path):
    ctrl, _ = _controller(tmp_path)
    seq = [ctrl.cycle_classifier() for _ in range(len(CLASSIFIERS) + 1)]
    # First call moves off "knn" to the next, and after a full cycle we wrap.
    assert seq[: len(CLASSIFIERS)] == CLASSIFIERS[1:] + CLASSIFIERS[:1]
    assert seq[len(CLASSIFIERS)] == seq[0]
    assert ctrl.classifier == seq[-1]


def test_ready_to_fit_needs_two_labels(tmp_path):
    ctrl, _ = _controller(tmp_path)
    assert ctrl.ready_to_fit() is False
    ctrl.record_many("coffee", 1)
    assert ctrl.ready_to_fit() is False  # only one class
    ctrl.record_many("vinegar", 1)
    assert ctrl.ready_to_fit() is True


def test_fit_uses_self_classifier(tmp_path):
    ctrl, _ = _controller(tmp_path)
    ctrl.record_many("coffee", 3)
    ctrl.record_many("vinegar", 3)
    ctrl.classifier = "lda"

    import sniffsniff.tui.controller as ctrl_mod

    seen = {}
    orig_cva = ctrl_mod.cross_val_accuracy

    def _spy_cva(X, y, *, classifier="knn", groups=None, **kw):
        seen["classifier"] = classifier
        return orig_cva(X, y, classifier=classifier, groups=groups, **kw)

    ctrl_mod.cross_val_accuracy = _spy_cva
    try:
        ctrl.fit()
    finally:
        ctrl_mod.cross_val_accuracy = orig_cva
    assert seen["classifier"] == "lda"


def test_delete_last_removes_most_recent(tmp_path):
    ctrl, _ = _controller(tmp_path)
    ctrl.record_many("coffee", 3)
    assert ctrl.dataset_counts()["coffee"] == 3

    deleted = ctrl.delete_last("coffee")
    assert deleted is not None
    assert deleted.stem == "coffee_0002"
    assert ctrl.dataset_counts().get("coffee", 0) == 2

    # dataset still loads cleanly (manifest valid).
    from sniffsniff.dataset import load_dataset

    ds = load_dataset(ctrl.out_dir)
    assert ds.y.tolist().count("coffee") == 2


def test_delete_last_unknown_is_none(tmp_path):
    ctrl, _ = _controller(tmp_path)
    ctrl.record_many("coffee", 1)
    assert ctrl.delete_last("banana") is None


def test_clear_empties_dataset(tmp_path):
    ctrl, _ = _controller(tmp_path)
    ctrl.record_many("coffee", 2)
    ctrl.record_many("vinegar", 1)
    removed = ctrl.clear()
    assert removed == 3
    assert ctrl.dataset_counts() == {}


def test_next_step_not_connected(tmp_path):
    ctrl, _ = _controller(tmp_path)
    ctrl.use_sim = False
    ctrl.port = "/dev/cu.nope_xyz"
    assert not ctrl.connected
    msg = ctrl.next_step()
    assert "connect" in msg.lower() or "simulator" in msg.lower()


def test_next_step_needs_more_labels(tmp_path):
    ctrl, _ = _controller(tmp_path)
    # No data yet: coach nudges toward recording another label.
    msg = ctrl.next_step()
    assert "label" in msg.lower() or "record" in msg.lower()

    ctrl.record_many("coffee", GOOD_REPS)
    # One label only — still needs a second class.
    msg = ctrl.next_step()
    assert "label" in msg.lower() or "another" in msg.lower()


def test_next_step_collect_more_below_target(tmp_path):
    ctrl, _ = _controller(tmp_path)
    ctrl.record_many("coffee", GOOD_REPS)
    ctrl.record_many("vinegar", 1)  # below GOOD_REPS
    msg = ctrl.next_step()
    # Ready to fit (2 classes), but coach still nudges to collect more.
    assert ctrl.ready_to_fit()
    assert "vinegar" in msg or "collect" in msg.lower() or "more" in msg.lower()


def test_next_step_enough_data(tmp_path):
    ctrl, _ = _controller(tmp_path)
    ctrl.record_many("coffee", GOOD_REPS)
    ctrl.record_many("vinegar", GOOD_REPS)
    msg = ctrl.next_step()
    assert "enough" in msg.lower() or ("f" in msg.lower() and "train" in msg.lower())


def test_next_step_trained(tmp_path):
    ctrl, _ = _controller(tmp_path)
    ctrl.record_many("coffee", GOOD_REPS)
    ctrl.record_many("vinegar", GOOD_REPS)
    ctrl.fit()
    assert ctrl.has_model()
    msg = ctrl.next_step()
    assert "trained" in msg.lower() or "identify" in msg.lower()
