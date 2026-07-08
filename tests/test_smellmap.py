"""Tests for smellmap.render_map (Milestone 2, smellmap.py).

The smell map is the optional matplotlib visualisation of the PCA scatter.
All data is synthesised deterministically via ``dataset.simulate_dataset`` and
a fitted :class:`SmellModel`. Rendering runs headless (Agg backend) so the test
works without a display; if matplotlib is not installed the whole module skips.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from sniffsniff.config import default_config
from sniffsniff.dataset import simulate_dataset
from sniffsniff.model import SmellModel
from sniffsniff.smellmap import render_map


ODORS = ["coffee", "vinegar", "alcohol"]


@pytest.fixture(scope="module")
def dataset():
    return simulate_dataset(default_config(), ODORS, reps=8, seed=1)


@pytest.fixture(scope="module")
def model(dataset):
    return SmellModel(n_components=2, classifier="knn").fit(dataset.X, dataset.y)


def _assert_nonempty_png(path) -> None:
    from pathlib import Path

    p = Path(path)
    assert p.exists()
    assert p.stat().st_size > 0
    # PNG magic number.
    with p.open("rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"


def test_render_map_writes_nonempty_png(model, dataset, tmp_path):
    out = tmp_path / "m.png"
    returned = render_map(model, dataset, path=out)
    assert returned == str(out)
    _assert_nonempty_png(out)


def test_render_map_with_new_sample(model, dataset, tmp_path):
    # One raw 48-D feature vector (a coffee sniff on a fresh seed).
    sample_ds = simulate_dataset(default_config(), ["coffee"], reps=1, seed=999)
    new_sample = sample_ds.X[0]
    assert new_sample.shape == (48,)

    out = tmp_path / "m_new.png"
    returned = render_map(model, dataset, new_sample=new_sample, path=out)
    assert returned == str(out)
    _assert_nonempty_png(out)


def test_render_map_without_dataset_plots_centroids(model, tmp_path):
    out = tmp_path / "m_centroids.png"
    returned = render_map(model, dataset=None, path=out)
    assert returned == str(out)
    _assert_nonempty_png(out)


def test_render_map_default_path_returns_tempfile(model, dataset):
    from pathlib import Path

    returned = render_map(model, dataset)
    try:
        assert isinstance(returned, str)
        _assert_nonempty_png(returned)
    finally:
        Path(returned).unlink(missing_ok=True)


def test_render_map_uses_agg_backend(model, dataset, tmp_path):
    import matplotlib

    render_map(model, dataset, path=tmp_path / "agg.png")
    # render_map forces the headless Agg backend.
    assert matplotlib.get_backend().lower() == "agg"
