"""The smell map model — StandardScaler → PCA → classifier + Mahalanobis novelty.

:class:`SmellModel` reduces the 48-D feature vectors to a low-dimensional PCA
"map", classifies a sniff against known odors, and scores how *novel* a sniff is
relative to every known cluster (per-class Mahalanobis distance in PCA space).

The math, in PCA space with ``k = n_components``:

* per class ``c``: centroid ``μ_c`` and inverse covariance ``Σ_c⁻¹`` (via
  :func:`numpy.linalg.pinv`; the pooled covariance is used when a class has too
  few members to estimate its own).
* Mahalanobis ``D_c(x) = sqrt((x−μ_c)ᵀ Σ_c⁻¹ (x−μ_c))``.
* ``novelty(x) = min_c D_c(x)``; ``x`` is novel when that exceeds
  ``sqrt(chi2.ppf(alpha, df=k))``.

:func:`cross_val_accuracy` reports honest, cross-validated accuracy with the full
scale→PCA→classify pipeline rebuilt inside every fold.
"""
from __future__ import annotations

import numpy as np
import joblib
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneOut,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

__all__ = ["SmellModel", "cross_val_accuracy"]


def _make_classifier(name: str, n_samples: int):
    """Build a fresh classifier estimator for ``name`` (contract defaults)."""
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=min(3, n_samples))
    if name == "svm":
        return SVC(kernel="rbf", probability=True)
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=0)
    if name == "lda":
        return LinearDiscriminantAnalysis()
    raise ValueError(
        f"unknown classifier {name!r}; expected one of "
        "'knn', 'svm', 'rf', 'lda'."
    )


class SmellModel:
    """A fitted (or fittable) smell map: scaler → PCA → classifier + novelty.

    Parameters
    ----------
    n_components:
        Number of PCA components ``k`` (the map dimensionality).
    classifier:
        One of ``{"knn", "svm", "rf", "lda"}``.
    novelty_alpha:
        Chi-squared confidence level for the novelty threshold; the threshold is
        ``sqrt(chi2.ppf(novelty_alpha, df=n_components))``.
    """

    def __init__(
        self,
        n_components: int = 2,
        classifier: str = "knn",
        novelty_alpha: float = 0.975,
    ):
        self.n_components = int(n_components)
        self.classifier = str(classifier)
        self.novelty_alpha = float(novelty_alpha)

    # ------------------------------------------------------------------ fit
    def fit(self, X, y) -> "SmellModel":
        """Fit scaler, PCA, classifier, and per-class Mahalanobis statistics."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=str)
        n_samples = X.shape[0]

        self.scaler_ = StandardScaler().fit(X)
        scaled = self.scaler_.transform(X)

        self.pca_ = PCA(n_components=self.n_components, random_state=0).fit(scaled)
        scores = self.pca_.transform(scaled)

        self.clf_ = _make_classifier(self.classifier, n_samples)
        self.clf_.fit(scores, y)

        self.classes_ = sorted(set(y.tolist()))
        self.loadings_ = self.pca_.components_  # (k, 48)
        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_

        # Pooled covariance (fallback for classes too small to estimate their own).
        if scores.shape[0] > 1:
            pooled_cov = np.cov(scores.T)
            pooled_cov = np.atleast_2d(pooled_cov)
        else:
            pooled_cov = np.eye(self.n_components)
        pooled_cov_inv = np.linalg.pinv(pooled_cov)

        self.centroids_: dict[str, np.ndarray] = {}
        self.radii_: dict[str, float] = {}
        self.counts_: dict[str, int] = {}
        self.cov_inv_: dict[str, np.ndarray] = {}

        for label in self.classes_:
            member = scores[y == label]
            centroid = member.mean(axis=0)
            dists = np.linalg.norm(member - centroid, axis=1)
            self.centroids_[label] = centroid
            self.radii_[label] = float(dists.std())
            self.counts_[label] = int(member.shape[0])

            # Own covariance only if the class has enough members; else pooled.
            if member.shape[0] >= self.n_components + 1:
                cov = np.atleast_2d(np.cov(member.T))
                self.cov_inv_[label] = np.linalg.pinv(cov)
            else:
                self.cov_inv_[label] = pooled_cov_inv

        self.novelty_threshold_ = float(
            np.sqrt(chi2.ppf(self.novelty_alpha, df=self.n_components))
        )
        return self

    # ------------------------------------------------------------- inference
    def transform(self, X) -> np.ndarray:
        """PCA scores ``(n, k)`` for ``X`` (scaler then PCA)."""
        X = np.asarray(X, dtype=np.float64)
        return self.pca_.transform(self.scaler_.transform(X))

    def predict(self, X) -> np.ndarray:
        """Predicted labels ``(n,)`` for ``X``."""
        return self.clf_.predict(self.transform(X))

    def predict_proba(self, X) -> tuple[list[str], np.ndarray]:
        """``(classes, (n, C))`` class-probability estimates, columns = ``classes_``."""
        proba = self.clf_.predict_proba(self.transform(X))
        # Reorder columns to self.classes_ (clf_.classes_ may differ / be a subset).
        clf_classes = list(self.clf_.classes_)
        out = np.zeros((proba.shape[0], len(self.classes_)), dtype=np.float64)
        for j, label in enumerate(self.classes_):
            if label in clf_classes:
                out[:, j] = proba[:, clf_classes.index(label)]
        return self.classes_, out

    def mahalanobis(self, X) -> np.ndarray:
        """Per-class Mahalanobis distance ``(n, C)`` in PCA space."""
        scores = self.transform(X)
        out = np.empty((scores.shape[0], len(self.classes_)), dtype=np.float64)
        for j, label in enumerate(self.classes_):
            delta = scores - self.centroids_[label]
            cov_inv = self.cov_inv_[label]
            # Row-wise quadratic form (δ Σ⁻¹ δᵀ), diagonal only.
            d2 = np.einsum("ni,ij,nj->n", delta, cov_inv, delta)
            out[:, j] = np.sqrt(np.clip(d2, 0.0, None))
        return out

    def novelty(self, X) -> np.ndarray:
        """Min over classes of the per-class Mahalanobis distance, ``(n,)``."""
        return self.mahalanobis(X).min(axis=1)

    def is_novel(self, X) -> np.ndarray:
        """Boolean ``(n,)``: ``novelty(X) > novelty_threshold_``."""
        return self.novelty(X) > self.novelty_threshold_

    # ------------------------------------------------------------ persistence
    def save(self, path) -> None:
        """Persist the whole fitted model with joblib."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path) -> "SmellModel":
        """Load a model previously written by :meth:`save`."""
        return joblib.load(path)


def cross_val_accuracy(
    X,
    y,
    *,
    n_components: int = 2,
    classifier: str = "knn",
    groups=None,
) -> tuple[float, float]:
    """Cross-validated accuracy (mean, std) of the scale→PCA→classify pipeline.

    Uses :class:`StratifiedKFold` (``n_splits = min(5, min_class_count)``), falling
    back to :class:`LeaveOneOut` when the rarest class has fewer than 3 members. If
    ``groups`` (e.g. sniff ids) is given, :class:`GroupKFold` prevents same-group
    leakage across folds. The full pipeline is rebuilt inside every fold.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=str)
    n_samples = X.shape[0]

    _, counts = np.unique(y, return_counts=True)
    min_class_count = int(counts.min())

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=0)),
            ("clf", _make_classifier(classifier, n_samples)),
        ]
    )

    if groups is not None:
        n_groups = len(set(np.asarray(groups).tolist()))
        cv = GroupKFold(n_splits=min(5, n_groups))
    elif min_class_count < 3:
        cv = LeaveOneOut()
    else:
        cv = StratifiedKFold(
            n_splits=min(5, min_class_count), shuffle=True, random_state=0
        )

    scores = cross_val_score(
        pipeline, X, y, cv=cv, groups=groups, scoring="accuracy"
    )
    return float(scores.mean()), float(scores.std())
