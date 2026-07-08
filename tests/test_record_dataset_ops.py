"""Tests for additive dataset-management ops in record.py.

Exercises :func:`delete_last_sniff` and :func:`clear_dataset` against a real
dataset written by :class:`SniffRecorder` + the simulator, and confirms
:func:`load_dataset` reflects the mutations.
"""
from __future__ import annotations

import csv
import warnings
from dataclasses import replace

from sniffsniff.config import default_config
from sniffsniff.dataset import load_dataset
from sniffsniff.record import SniffRecorder, delete_last_sniff, clear_dataset
from sniffsniff.simulator import Simulator


def _short_config():
    """Small-timing config so hand-built sniffs are cheap (80 frames total)."""
    return replace(
        default_config(),
        baseline_s=1.0, exposure_s=2.0, purge_s=1.0, plateau_s=0.5,
    )


def _record_sniffs(cfg, out_dir, label, n, seed0):
    """Record ``n`` sniffs of ``label`` into ``out_dir`` via the simulator."""
    rec = SniffRecorder(cfg, out_dir)
    paths = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(n):
            sim = Simulator(cfg, seed=seed0 + k, noise_counts=1.0)
            frames = sim.sniff_frames(label)
            paths.append(rec.record(frames, label))
    return paths


def _build_dataset(tmp_path):
    """3 coffee + 2 vinegar sniffs in tmp_path; return the out_dir Path."""
    cfg = _short_config()
    _record_sniffs(cfg, tmp_path, "coffee", 3, seed0=100)
    _record_sniffs(cfg, tmp_path, "vinegar", 2, seed0=200)
    return tmp_path


def _manifest_rows(out_dir):
    with (out_dir / "manifest.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def _label_counts(out_dir):
    counts: dict[str, int] = {}
    for row in _manifest_rows(out_dir):
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    return counts


def test_delete_last_sniff_removes_highest_id(tmp_path):
    out_dir = _build_dataset(tmp_path)

    # Sanity: fully populated manifest + files.
    assert _label_counts(out_dir) == {"coffee": 3, "vinegar": 2}
    target = out_dir / "coffee" / "coffee_0002.npz"
    assert target.exists()

    deleted = delete_last_sniff(out_dir, "coffee")

    assert deleted == target
    assert not target.exists(), "the highest-id coffee npz must be deleted"

    # Manifest drops exactly that row; coffee now 2, vinegar still 2.
    assert _label_counts(out_dir) == {"coffee": 2, "vinegar": 2}
    ids = [r["id"] for r in _manifest_rows(out_dir)]
    assert "coffee_0002" not in ids
    assert ids == ["coffee_0000", "coffee_0001", "vinegar_0000", "vinegar_0001"]

    # load_dataset reflects the new counts.
    ds = load_dataset(out_dir)
    y = ds.y.tolist()
    assert y.count("coffee") == 2
    assert y.count("vinegar") == 2
    assert "coffee_0002" not in ds.ids


def test_delete_last_sniff_unknown_label_is_noop(tmp_path):
    out_dir = _build_dataset(tmp_path)
    before_rows = _manifest_rows(out_dir)

    result = delete_last_sniff(out_dir, "banana")

    assert result is None
    assert _manifest_rows(out_dir) == before_rows, "manifest must be unchanged"
    assert _label_counts(out_dir) == {"coffee": 3, "vinegar": 2}
    # All npz files still present.
    assert len(list(out_dir.glob("*/*.npz"))) == 5


def test_clear_dataset_removes_everything(tmp_path):
    out_dir = _build_dataset(tmp_path)
    # Drop the most-recent coffee first, leaving 4 sniffs.
    delete_last_sniff(out_dir, "coffee")
    assert len(list(out_dir.glob("*/*.npz"))) == 4

    removed = clear_dataset(out_dir)

    assert removed == 4
    assert list(out_dir.glob("*/*.npz")) == []
    assert not (out_dir / "manifest.csv").exists()

    ds = load_dataset(out_dir)
    assert ds.y.tolist() == []
    assert ds.ids == []


def test_clear_dataset_missing_dir_returns_zero(tmp_path):
    missing = tmp_path / "nope"
    assert clear_dataset(missing) == 0


def test_clear_dataset_leaves_non_dataset_files(tmp_path):
    out_dir = _build_dataset(tmp_path)
    keep = out_dir / "coffee" / "notes.txt"
    keep.write_text("keep me")

    removed = clear_dataset(out_dir)

    assert removed == 5
    assert keep.exists(), "non-.npz files must be left untouched"
    assert not (out_dir / "manifest.csv").exists()
