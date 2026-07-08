"""Serialize a fitted :class:`~sniffsniff.model.SmellModel`'s geometry for M3.

The Milestone 3 LLM reasoner never sees numpy arrays — it consumes the plain
JSON blob produced here. Two entry points:

* :func:`axis_interpretation` names each PCA axis from its dominant feature
  loadings, e.g. ``{"PC1": "MQ3__peak(+), MQ2__peak(+), MQ135__peak(-)"}``.
* :func:`serialize_geometry` emits the full M3 schema (PCA summary, axis
  interpretation, known cluster centroids/radii/counts, optional new-sample
  position + prediction + top z-scored features, and novelty) as pure
  ``float``/``int``/``list``/``str`` so :func:`json.dumps` just works.
"""
from __future__ import annotations

import numpy as np

__all__ = ["axis_interpretation", "serialize_geometry"]


def axis_interpretation(model, *, top_k: int = 3) -> dict[str, str]:
    """Name each PCA component from its ``top_k`` dominant feature loadings.

    For principal component ``i`` the ``top_k`` features with the largest
    ``|loadings_[i]|`` are picked (ties broken by column order) and formatted as
    ``"NAME(+)"`` / ``"NAME(-)"`` (sign of the loading), joined by ``", "``.

    Returns a dict keyed ``"PC1" .. "PCk"`` mapping to those human-readable
    strings; the names are exactly the model's ``feature_names_`` columns.
    """
    loadings = np.asarray(model.loadings_)  # (k, 48)
    names = _feature_names(model, loadings.shape[1])

    out: dict[str, str] = {}
    for i in range(loadings.shape[0]):
        row = loadings[i]
        # Indices of the top_k features by absolute loading (largest first).
        order = np.argsort(np.abs(row))[::-1][:top_k]
        tokens = []
        for j in order:
            sign = "+" if row[j] >= 0 else "-"
            tokens.append(f"{names[j]}({sign})")
        out[f"PC{i + 1}"] = ", ".join(tokens)
    return out


def serialize_geometry(model, *, new_sample=None) -> dict:
    """Serialize the model geometry (and an optional new sample) to M3 JSON.

    The returned dict is fully JSON-serializable (plain python scalars/lists):

    * ``pca``: ``n_components`` and ``explained_variance_ratio``.
    * ``axis_interpretation``: per-PC dominant-feature strings.
    * ``known_clusters``: per label ``{centroid, radius, n[, distance]}`` in PCA
      space; ``distance`` (Euclidean from the new sample's PCA coords to the
      centroid) is present only when ``new_sample`` is given.
    * ``new_sample`` (only when given): ``pca_coords``, ``predicted{label, proba}``
      and ``top_features_z`` (the standardized features with the largest ``|z|``).
    * ``novelty``: ``min_mahalanobis``, ``threshold``, ``is_novel`` and the
      ``nearest`` cluster (min per-class Mahalanobis).

    ``new_sample`` is a single ``(48,)`` raw feature vector.
    """
    evr = [float(v) for v in np.asarray(model.explained_variance_ratio_).tolist()]

    geo: dict = {
        "pca": {
            "n_components": int(model.n_components),
            "explained_variance_ratio": evr,
        },
        "axis_interpretation": axis_interpretation(model),
        "known_clusters": {},
        "novelty": {},
    }

    # ---------------------------------------------------------- new sample prep
    sample_scores = None
    if new_sample is not None:
        x = np.asarray(new_sample, dtype=np.float64).reshape(1, -1)
        sample_scores = model.transform(x)[0]  # (k,)

    # ---------------------------------------------------------- known clusters
    for label in model.classes_:
        centroid = np.asarray(model.centroids_[label], dtype=np.float64)
        entry = {
            "centroid": [float(c) for c in centroid.tolist()],
            "radius": float(model.radii_[label]),
            "n": int(model.counts_[label]),
        }
        if sample_scores is not None:
            entry["distance"] = float(
                np.linalg.norm(sample_scores - centroid)
            )
        geo["known_clusters"][label] = entry

    # ---------------------------------------------------------------- new sample
    if new_sample is not None:
        classes, proba = model.predict_proba(x)
        proba_row = np.asarray(proba)[0]
        pred_idx = int(np.argmax(proba_row))
        pred_label = str(classes[pred_idx])
        pred_proba = float(proba_row[pred_idx])

        names = _feature_names(model, x.shape[1])
        z = model.scaler_.transform(x)[0]  # standardized features
        top = np.argsort(np.abs(z))[::-1][:3]
        top_features_z = {str(names[j]): float(z[j]) for j in top}

        geo["new_sample"] = {
            "pca_coords": [float(c) for c in sample_scores.tolist()],
            "predicted": {"label": pred_label, "proba": pred_proba},
            "top_features_z": top_features_z,
        }

    # ------------------------------------------------------------------ novelty
    if new_sample is not None:
        maha = model.mahalanobis(x)[0]  # (C,)
        nearest_idx = int(np.argmin(maha))
        min_maha = float(maha[nearest_idx])
        nearest = str(model.classes_[nearest_idx])
        is_novel = bool(min_maha > float(model.novelty_threshold_))
        geo["novelty"] = {
            "min_mahalanobis": min_maha,
            "threshold": float(model.novelty_threshold_),
            "is_novel": is_novel,
            "nearest": nearest,
        }
    else:
        geo["novelty"] = {
            "min_mahalanobis": None,
            "threshold": float(model.novelty_threshold_),
            "is_novel": None,
            "nearest": None,
        }

    return geo


def _feature_names(model, n: int) -> list[str]:
    """Best-effort feature-name column labels of length ``n``.

    Prefers ``model.feature_names_`` if the model carries one; otherwise falls
    back to positional ``feature_0 .. feature_{n-1}`` so serialization never
    crashes on a model fitted without names.
    """
    names = getattr(model, "feature_names_", None)
    if names is not None and len(names) == n:
        return list(names)
    return [f"feature_{i}" for i in range(n)]
