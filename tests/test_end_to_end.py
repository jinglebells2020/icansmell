"""End-to-end sanity: simulate multiple odors, record, and check separability.

Not a trained model — just a geometric sanity check that same-odor 48-D vectors
cluster closer (Euclidean) than different-odor ones, plus a CLI smoke test.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from sniffsniff import cli
from sniffsniff.config import default_config
from sniffsniff.record import SniffRecorder
from sniffsniff.simulator import Simulator


def _feature_matrix(labels, reps, tmp_path):
    cfg = default_config()
    rec = SniffRecorder(cfg, tmp_path)
    vecs = {}
    for label in labels:
        rows = []
        for r in range(reps):
            # vary the seed per rep so noise differs, same odor chemistry
            sim = Simulator(cfg, seed=100 + r, noise_counts=0.5)
            result = rec.process(sim.sniff_frames(label), label)
            rows.append(result.features)
        vecs[label] = np.array(rows)
    return vecs


def test_within_class_closer_than_across_class(tmp_path):
    labels = ["coffee", "vinegar", "alcohol"]
    reps = 4
    vecs = _feature_matrix(labels, reps, tmp_path)

    # normalize per-feature so no single large-magnitude feature dominates the
    # Euclidean distance (a standard-scale sanity, not a trained transform).
    allv = np.vstack([vecs[l] for l in labels])
    mu = allv.mean(axis=0)
    sd = allv.std(axis=0)
    sd[sd == 0] = 1.0

    def norm(m):
        return (m - mu) / sd

    within = []
    for label in labels:
        m = norm(vecs[label])
        for i, j in itertools.combinations(range(reps), 2):
            within.append(np.linalg.norm(m[i] - m[j]))

    across = []
    for a, b in itertools.combinations(labels, 2):
        ma, mb = norm(vecs[a]), norm(vecs[b])
        for i in range(reps):
            for j in range(reps):
                across.append(np.linalg.norm(ma[i] - mb[j]))

    mean_within = float(np.mean(within))
    mean_across = float(np.mean(across))
    assert mean_within < mean_across, (mean_within, mean_across)


def test_cli_record_sim_creates_files(tmp_path):
    out = tmp_path / "data"
    rc = cli.main(["record", "--sim", "--label", "coffee", "--out", str(out)])
    assert rc == 0
    npzs = list(out.glob("coffee/*.npz"))
    assert len(npzs) == 1
    assert (out / "manifest.csv").exists()


def test_cli_simulate_runs(capsys):
    rc = cli.main(["simulate", "--odor", "vinegar"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "vinegar" in captured.out


def test_cli_stream_sim_runs(capsys):
    rc = cli.main(["stream", "--sim"])
    assert rc == 0
