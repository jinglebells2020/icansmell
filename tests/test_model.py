"""Tests for the SmellModel + cross_val_accuracy (Milestone 2, model.py).

All test data is synthesised via ``dataset.simulate_dataset`` on the default
config so the whole suite is deterministic for fixed seeds.
"""
from __future__ import annotations

import numpy as np
import pytest

from sniffsniff.config import default_config
from sniffsniff.dataset import simulate_dataset
from sniffsniff.model import SmellModel, cross_val_accuracy


ODORS = ["coffee", "vinegar", "alcohol", "spoiled_milk"]


@pytest.fixture(scope="module")
def dataset():
    return simulate_dataset(default_config(), ODORS, reps=8, seed=1)


def test_fit_returns_self_and_stores_geometry(dataset):
    model = SmellModel(n_components=2, classifier="knn")
    returned = model.fit(dataset.X, dataset.y)
    assert returned is model

    # classes_ is sorted unique labels.
    assert model.classes_ == sorted(ODORS)

    # loadings_ is (k, 48).
    assert model.loadings_.shape == (2, dataset.X.shape[1])
    assert model.loadings_.shape[1] == 48

    # explained_variance_ratio_ has length 2.
    assert len(model.explained_variance_ratio_) == 2

    # per-class stats present for every class.
    for label in model.classes_:
        assert label in model.centroids_
        assert model.centroids_[label].shape == (2,)
        assert isinstance(model.radii_[label], float)
        assert model.counts_[label] == 8
        assert model.cov_inv_[label].shape == (2, 2)

    assert model.novelty_threshold_ > 0.0


def test_transform_shape(dataset):
    model = SmellModel(n_components=2).fit(dataset.X, dataset.y)
    scores = model.transform(dataset.X)
    assert scores.shape == (dataset.X.shape[0], 2)


def test_predict_recovers_training_labels(dataset):
    model = SmellModel(n_components=2, classifier="knn").fit(dataset.X, dataset.y)
    preds = model.predict(dataset.X)
    assert preds.shape == (dataset.X.shape[0],)
    acc = float(np.mean(preds == dataset.y))
    assert acc > 0.9


def test_predict_proba_shape_and_columns(dataset):
    model = SmellModel(n_components=2, classifier="knn").fit(dataset.X, dataset.y)
    classes, proba = model.predict_proba(dataset.X)
    assert classes == model.classes_
    assert proba.shape == (dataset.X.shape[0], len(model.classes_))
    # rows are (near) probability distributions.
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_mahalanobis_shape(dataset):
    model = SmellModel(n_components=2).fit(dataset.X, dataset.y)
    d = model.mahalanobis(dataset.X)
    assert d.shape == (dataset.X.shape[0], len(model.classes_))
    assert np.all(d >= 0.0)


def test_cross_val_accuracy_high(dataset):
    mean, std = cross_val_accuracy(dataset.X, dataset.y, classifier="knn")
    assert 0.0 <= mean <= 1.0
    assert std >= 0.0
    assert mean > 0.85


def test_save_load_predicts_identically(dataset, tmp_path):
    model = SmellModel(n_components=2, classifier="knn").fit(dataset.X, dataset.y)
    path = tmp_path / "model.joblib"
    model.save(path)
    loaded = SmellModel.load(path)

    assert np.array_equal(model.predict(dataset.X), loaded.predict(dataset.X))
    assert np.array_equal(model.transform(dataset.X), loaded.transform(dataset.X))
    assert loaded.classes_ == model.classes_


def test_novelty_flags_unseen_odor():
    cfg = default_config()
    # Train on 3 odors, hold out an unseen one.
    train = simulate_dataset(cfg, ["coffee", "vinegar", "alcohol"], reps=8, seed=2)
    model = SmellModel(n_components=2, classifier="knn").fit(train.X, train.y)

    # An in-distribution fresh sample of a trained odor (fresh seed).
    indist = simulate_dataset(cfg, ["coffee"], reps=3, seed=555)
    # An unseen odor sample.
    unseen = simulate_dataset(cfg, ["fresh_milk"], reps=3, seed=777)

    nov_indist = model.novelty(indist.X)
    nov_unseen = model.novelty(unseen.X)

    assert nov_indist.shape == (3,)
    assert nov_unseen.shape == (3,)

    # Core novelty signal (contract: "at least assert its novelty distance is
    # smaller"): the unseen odor sits farther from every known cluster than a
    # fresh in-distribution sniff of a trained odor does.
    assert nov_unseen.mean() > nov_indist.mean()

    # The in-distribution sniff is never flagged novel; the unseen odor is never
    # flagged novel less often than the in-distribution one.
    assert not bool(model.is_novel(indist.X).any())
    assert model.is_novel(unseen.X).mean() >= model.is_novel(indist.X).mean()


def test_is_novel_matches_threshold(dataset):
    model = SmellModel(n_components=2).fit(dataset.X, dataset.y)
    nov = model.novelty(dataset.X)
    flags = model.is_novel(dataset.X)
    assert np.array_equal(flags, nov > model.novelty_threshold_)


def test_other_classifiers_fit_and_predict(dataset):
    for clf in ("svm", "rf", "lda"):
        model = SmellModel(n_components=2, classifier=clf).fit(dataset.X, dataset.y)
        preds = model.predict(dataset.X)
        assert preds.shape == (dataset.X.shape[0],)
        classes, proba = model.predict_proba(dataset.X)
        assert proba.shape == (dataset.X.shape[0], len(model.classes_))


def test_invalid_classifier_raises(dataset):
    with pytest.raises((ValueError, KeyError)):
        SmellModel(classifier="bogus").fit(dataset.X, dataset.y)


# --- classifier-space augmentation (chemistry-informed discriminating axes) ----

def test_augment_default_is_pure_pca_backward_compatible(dataset):
    """Default (no augment) must behave exactly like the old pure-PCA model."""
    m = SmellModel(n_components=2, classifier="knn").fit(dataset.X, dataset.y)
    assert m.augment_cols_ == []                       # nothing grafted
    # transform() (map + novelty space) is pure PCA of width n_components_
    assert m.transform(dataset.X).shape[1] == m.n_components_


def test_augment_grafts_named_columns_onto_classifier_only(dataset):
    names = list(dataset.feature_names)
    m = SmellModel(n_components=2, classifier="knn", augment_features=("MQ3",))
    m.fit(dataset.X, dataset.y, feature_names=names)
    # resolves exactly the MQ3 columns
    expected = [i for i, nm in enumerate(names) if nm.startswith("MQ3__")]
    assert m.augment_cols_ == expected and len(expected) == 8
    # novelty/map space is UNCHANGED (still pure PCA), only the classifier sees more
    assert m.transform(dataset.X).shape[1] == m.n_components_
    assert m._clf_input(dataset.X).shape[1] == m.n_components_ + len(expected)
    # still a working classifier
    assert m.predict(dataset.X).shape == (dataset.X.shape[0],)


def test_augment_requires_feature_names(dataset):
    m = SmellModel(classifier="knn", augment_features=("MQ3",))
    with pytest.raises(ValueError):
        m.fit(dataset.X, dataset.y)  # no feature_names -> can't resolve columns


def test_augment_save_load_round_trips(dataset, tmp_path):
    names = list(dataset.feature_names)
    m = SmellModel(n_components=2, classifier="knn", augment_features=("MQ3",))
    m.fit(dataset.X, dataset.y, feature_names=names)
    path = tmp_path / "aug.joblib"
    m.save(path)
    loaded = SmellModel.load(path)
    assert np.array_equal(m.predict(dataset.X), loaded.predict(dataset.X))


def test_cross_val_accuracy_augment_runs(dataset):
    mean, std = cross_val_accuracy(
        dataset.X, dataset.y, classifier="knn",
        augment_features=("MQ3",), feature_names=list(dataset.feature_names),
    )
    assert 0.0 <= mean <= 1.0 and std >= 0.0
