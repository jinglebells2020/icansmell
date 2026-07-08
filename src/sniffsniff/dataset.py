"""Labeled 48-D feature datasets for the smell map.

Two entry points feed the Milestone 2 model with identical ``(X[n,48], y[n])``
material:

* :func:`load_dataset` reads a directory of Milestone 1 recordings
  (``<label>/*.npz`` written by :class:`sniffsniff.record.SniffRecorder`) and
  stacks their feature vectors, dropping any sniff whose features are non-finite
  (an open/rail channel) with a warning.
* :func:`simulate_dataset` synthesises a labeled dataset straight from the
  :class:`~sniffsniff.simulator.Simulator` + :class:`~sniffsniff.record.SniffRecorder`
  pipeline — no disk round-trip — so tests and demos get a reproducible dataset
  for a given seed.

Both return the same immutable :class:`Dataset` container.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import features
from .config import Config
from .record import SniffRecorder
from .simulator import Simulator

__all__ = ["Dataset", "load_dataset", "simulate_dataset"]


@dataclass(frozen=True)
class Dataset:
    """A labeled set of 48-D feature vectors, one row per sniff.

    ``X`` is ``(n, 48)`` float64, ``y`` is ``(n,)`` odor-label strings,
    ``feature_names`` is the length-48 column layout, and ``ids`` is the length-n
    list of sniff ids (used for split-by-sniff / leakage control downstream).
    """

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    ids: list[str]

    @property
    def classes(self) -> list[str]:
        """Sorted unique labels present in ``y``."""
        return sorted(set(self.y.tolist()))


def load_dataset(data_dir) -> Dataset:
    """Load every ``<label>/*.npz`` recording under ``data_dir`` into a Dataset.

    Each ``.npz`` (written by :class:`sniffsniff.record.SniffRecorder`) carries a
    ``features`` array and a JSON ``meta`` blob holding the sniff ``id``,
    ``label``, and ``feature_names``. Files are visited in sorted order for
    determinism. A sniff whose feature vector contains any non-finite value
    (an open/rail channel) is skipped with a :class:`UserWarning` rather than
    poisoning the matrix.

    Returns
    -------
    Dataset
        Stacked ``X``/``y``/``ids`` and the shared ``feature_names``. If the
        directory holds no usable recordings, ``X`` is ``(0, 0)`` and the label
        list, id list, and feature-name list are empty.
    """
    data_dir = Path(data_dir)

    rows: list[np.ndarray] = []
    labels: list[str] = []
    ids: list[str] = []
    feature_names: list[str] = []

    # Deterministic traversal: label dirs, then files within each.
    for npz_path in sorted(data_dir.glob("*/*.npz")):
        with np.load(npz_path, allow_pickle=False) as data:
            feats = np.asarray(data["features"], dtype=np.float64)
            meta = json.loads(str(data["meta"]))

        # Label from meta, falling back to the parent directory name.
        label = str(meta.get("label", npz_path.parent.name))
        sniff_id = str(meta.get("id", npz_path.stem))

        if not np.all(np.isfinite(feats)):
            warnings.warn(
                f"skipping sniff {sniff_id!r} ({npz_path}): features contain "
                "non-finite values (open/rail channel).",
                stacklevel=2,
            )
            continue

        if not feature_names:
            names = meta.get("feature_names")
            if names is not None:
                feature_names = list(names)

        rows.append(feats)
        labels.append(label)
        ids.append(sniff_id)

    if rows:
        X = np.vstack(rows).astype(np.float64)
    else:
        X = np.empty((0, 0), dtype=np.float64)

    return Dataset(
        X=X,
        y=np.array(labels, dtype=str),
        feature_names=feature_names,
        ids=ids,
    )


def simulate_dataset(
    config: Config,
    odors: list[str],
    reps: int,
    *,
    seed: int,
    noise_counts: float = 1.0,
) -> Dataset:
    """Synthesise a labeled 48-D dataset from the simulator + recorder pipeline.

    For each odor ``oi`` and rep ``r`` a fresh :class:`~sniffsniff.simulator.Simulator`
    with seed ``seed + oi*1000 + r`` generates a full three-phase session, which
    :meth:`SniffRecorder.process` turns into a feature vector (no disk write).
    The distinct per-rep seeds make reps vary while keeping the whole dataset
    byte-reproducible for a given ``seed``. Rows are laid out odor-major, with
    ids ``f"{odor}_{r:04d}"``.

    The recorder's "baseline too noisy" :class:`UserWarning` is suppressed during
    this bulk generation (it is expected noise on synthetic baselines, not a data
    fault worth surfacing per-sniff).
    """
    recorder = SniffRecorder(config, ".")
    feature_col_names = features.feature_names(config.sensor_names())

    rows: list[np.ndarray] = []
    labels: list[str] = []
    ids: list[str] = []

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="baseline too noisy",
            category=UserWarning,
        )
        for oi, odor in enumerate(odors):
            for r in range(reps):
                sim = Simulator(
                    config,
                    seed=seed + oi * 1000 + r,
                    noise_counts=noise_counts,
                )
                frames = sim.sniff_frames(odor)
                result = recorder.process(frames, odor)
                rows.append(np.asarray(result.features, dtype=np.float64))
                labels.append(odor)
                ids.append(f"{odor}_{r:04d}")

    if rows:
        X = np.vstack(rows).astype(np.float64)
    else:
        X = np.empty((0, len(feature_col_names)), dtype=np.float64)

    return Dataset(
        X=X,
        y=np.array(labels, dtype=str),
        feature_names=feature_col_names,
        ids=ids,
    )
