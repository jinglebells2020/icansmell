"""Tests for geometry.py (Milestone 2) — axis interpretation + geometry serialization.

All test data is synthesised via ``dataset.simulate_dataset`` on the default
config so the whole suite is deterministic for fixed seeds.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from sniffsniff.config import default_config
from sniffsniff.dataset import simulate_dataset
from sniffsniff.geometry import axis_interpretation, serialize_geometry
from sniffsniff.model import SmellModel


ODORS = ["coffee", "vinegar", "alcohol"]


@pytest.fixture(scope="module")
def dataset():
    return simulate_dataset(default_config(), ODORS, reps=8, seed=1)


@pytest.fixture(scope="module")
def model(dataset):
    m = SmellModel(n_components=2, classifier="knn").fit(dataset.X, dataset.y)
    # The caller attaches the dataset's column labels so geometry can name axes
    # with real feature names (the model itself is fit on a bare numpy array).
    m.feature_names_ = dataset.feature_names
    return m


@pytest.fixture(scope="module")
def coffee_sample():
    # A fresh coffee sniff (different seed) as the (48,) raw feature vector.
    ds = simulate_dataset(default_config(), ["coffee"], reps=1, seed=999)
    return ds.X[0]


# ------------------------------------------------------------ axis_interpretation

def test_axis_interpretation_keys_are_PC1_to_PCk(model):
    axes = axis_interpretation(model)
    assert set(axes.keys()) == {"PC1", "PC2"}


def test_axis_interpretation_references_real_feature_names(model, dataset):
    axes = axis_interpretation(model, top_k=3)
    for value in axes.values():
        assert isinstance(value, str)
        # Each token looks like "NAME(+)" or "NAME(-)"; the NAME is a real column.
        tokens = [t.strip() for t in value.split(",")]
        assert len(tokens) == 3
        for tok in tokens:
            assert tok.endswith("(+)") or tok.endswith("(-)")
            name = tok[:-3]
            assert name in dataset.feature_names


def test_axis_interpretation_top_k_controls_count(model):
    axes = axis_interpretation(model, top_k=5)
    for value in axes.values():
        assert len([t for t in value.split(",")]) == 5


# ------------------------------------------------------------ serialize_geometry

def test_serialize_geometry_has_core_keys(model):
    geo = serialize_geometry(model)
    assert set(["pca", "axis_interpretation", "known_clusters", "novelty"]).issubset(
        geo.keys()
    )


def test_serialize_geometry_is_json_serializable(model):
    geo = serialize_geometry(model)
    # Must not raise (no numpy scalars).
    text = json.dumps(geo)
    assert isinstance(text, str)
    # round-trips
    assert json.loads(text)["pca"]["map_components"] == 2


def test_serialize_geometry_pca_block(model):
    geo = serialize_geometry(model)
    pca = geo["pca"]
    assert pca["map_components"] == 2
    assert pca["working_components"] == 2  # this fixture fits with n_components=2
    assert isinstance(pca["map_explained_variance_ratio"], list)
    assert len(pca["map_explained_variance_ratio"]) == 2
    assert all(isinstance(v, float) for v in pca["map_explained_variance_ratio"])
    assert isinstance(pca["total_explained_variance"], float)


def test_map_stays_2d_while_working_space_is_higher():
    # Decoupling: classifier/novelty use k>2 PCs, but the map (centroids, coords,
    # axes) stays 2-D so the picture and the LLM's spatial story remain 2-D.
    ds = simulate_dataset(default_config(), ODORS, reps=8, seed=3)
    m = SmellModel(n_components=5, classifier="knn").fit(ds.X, ds.y)
    m.feature_names_ = list(ds.feature_names)
    geo = serialize_geometry(m, new_sample=ds.X[0])
    assert m.n_components_ == 5                      # full working space
    assert geo["pca"]["working_components"] == 5
    assert geo["pca"]["map_components"] == 2
    assert len(geo["axis_interpretation"]) == 2      # only the 2 map axes named
    for entry in geo["known_clusters"].values():
        assert len(entry["centroid"]) == 2           # map centroid is 2-D
    assert len(geo["new_sample"]["pca_coords"]) == 2 # map coords are 2-D


def test_serialize_geometry_known_clusters(model):
    geo = serialize_geometry(model)
    clusters = geo["known_clusters"]
    assert set(clusters.keys()) == set(ODORS)
    for label, entry in clusters.items():
        assert isinstance(entry["centroid"], list)
        assert len(entry["centroid"]) == 2  # n_components
        assert all(isinstance(c, float) for c in entry["centroid"])
        assert isinstance(entry["radius"], float)
        assert isinstance(entry["n"], int)
        assert entry["n"] == 8
        # no distance without a new_sample.
        assert "distance" not in entry


def test_serialize_geometry_novelty_block_no_sample(model):
    geo = serialize_geometry(model)
    nov = geo["novelty"]
    assert set(nov.keys()) == {"min_mahalanobis", "threshold", "is_novel", "nearest"}
    assert isinstance(nov["threshold"], float)


def test_serialize_geometry_new_sample_block(model, coffee_sample):
    geo = serialize_geometry(model, new_sample=coffee_sample)
    assert "new_sample" in geo
    ns = geo["new_sample"]
    assert isinstance(ns["pca_coords"], list)
    assert len(ns["pca_coords"]) == 2
    assert set(ns["predicted"].keys()) == {"label", "proba"}
    assert isinstance(ns["predicted"]["label"], str)
    assert isinstance(ns["predicted"]["proba"], float)
    assert isinstance(ns["top_features_z"], dict)
    assert len(ns["top_features_z"]) >= 1
    for name, z in ns["top_features_z"].items():
        assert isinstance(name, str)
        assert isinstance(z, float)
    # JSON serializable with a sample too.
    json.dumps(geo)


def test_serialize_geometry_new_sample_adds_distance(model, coffee_sample):
    geo = serialize_geometry(model, new_sample=coffee_sample)
    for entry in geo["known_clusters"].values():
        assert "distance" in entry
        assert isinstance(entry["distance"], float)


def test_nearest_cluster_to_coffee_is_coffee(model, coffee_sample):
    geo = serialize_geometry(model, new_sample=coffee_sample)
    clusters = geo["known_clusters"]
    nearest = min(clusters, key=lambda k: clusters[k]["distance"])
    assert nearest == "coffee"
    # novelty.nearest agrees with the min-mahalanobis cluster.
    assert geo["novelty"]["nearest"] in ODORS


def test_novelty_nearest_is_coffee_for_coffee_sample(model, coffee_sample):
    geo = serialize_geometry(model, new_sample=coffee_sample)
    nov = geo["novelty"]
    assert nov["nearest"] == "coffee"
    assert isinstance(nov["min_mahalanobis"], float)
    assert isinstance(nov["is_novel"], bool)
